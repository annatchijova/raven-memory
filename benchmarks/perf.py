#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — Performance benchmark harness (plan 4.2).

Measures, per corpus size, with deterministic dummy embeddings (seeded):
  - store throughput (memories/s)
  - first recall latency (includes the lazy KDTree rebuild)
  - warm recall latency p50 / p95 (with STDP turn context — the real path)
  - database size and audit-log growth per recall

Run:
    python benchmarks/perf.py                       # default sizes
    python benchmarks/perf.py --sizes 1000,5000 --queries 50
    python benchmarks/perf.py --json results.json

Results are environment-dependent: compare only runs from the same machine.
"""

import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raven.memory_engine import AdaptiveMemoryEngine  # noqa: E402

DIM = 384
TOPICS = 40          # hot topics so auto-INHIBITORY links exist, like real use
CLAIMS_PER_TOPIC = 3


def emb(rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(DIM).astype(np.float32)
    e /= np.linalg.norm(e) + 1e-10
    return e


def near(base: np.ndarray, rng: np.random.Generator, noise: float = 0.05) -> np.ndarray:
    e = base + rng.standard_normal(DIM).astype(np.float32) * noise
    e /= np.linalg.norm(e) + 1e-10
    return e


def bench_size(n: int, queries: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    tmp = Path(tempfile.mkdtemp()) / f"bench_{n}.db"
    engine = AdaptiveMemoryEngine(db_path=tmp)

    # ---- populate -----------------------------------------------------------
    stored_embs = []
    t0 = time.perf_counter()
    for i in range(n):
        e = emb(rng)
        stored_embs.append(e)
        metadata = {}
        if i % 10 == 0:   # every 10th memory competes on a hot topic
            metadata = {
                "topic": f"topic_{i % TOPICS}",
                "claim": f"claim_{(i // TOPICS) % CLAIMS_PER_TOPIC}",
            }
        engine.store(
            f"benchmark memory number {i} with some content to fingerprint "
            f"and enough words to look like an actual stored document",
            e,
            layer="episodic" if i % 3 else "semantic",
            metadata=metadata,
        )
    store_s = time.perf_counter() - t0

    # ---- first recall (lazy KDTree rebuild included) ------------------------
    q0 = near(stored_embs[0], rng)
    t0 = time.perf_counter()
    engine.recall(q0, query_text="warmup", top_k=5, hops=2)
    first_recall_ms = (time.perf_counter() - t0) * 1000

    # ---- warm recalls with STDP context (the realistic hot path) ------------
    audit_rows_before = _audit_rows(tmp)
    latencies = []
    turn: list = []
    for qi in range(queries):
        base = stored_embs[rng.integers(0, n)]
        q = near(base, rng)
        t0 = time.perf_counter()
        results, _ = engine.recall(
            q, query_text=f"query {qi}", top_k=5, hops=2,
            current_turn_memories=turn[-20:] or None,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        turn.extend(r.memory.memory_id for r in results)
        turn = turn[-40:]

    db_bytes = tmp.stat().st_size
    audit_bytes_per_recall = (
        (db_bytes - _db_size_without_audit(tmp)) / max(_audit_rows(tmp), 1)
    )

    return {
        "n": n,
        "store_total_s": round(store_s, 2),
        "store_per_s": round(n / store_s, 1),
        "first_recall_ms": round(first_recall_ms, 1),
        "recall_p50_ms": round(statistics.median(latencies), 2),
        "recall_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2),
        "db_mb": round(db_bytes / 1e6, 2),
        "audit_rows": _audit_rows(tmp),
        "audit_bytes_per_row": int(audit_bytes_per_recall),
        "cell_links": _link_rows(tmp),
    }


def _audit_rows(db: Path) -> int:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]


def _link_rows(db: Path) -> int:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM cell_links").fetchone()[0]


def _db_size_without_audit(db: Path) -> int:
    # Approximation: audit payload dominates row size; measure via SQL length sums.
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(memories_retrieved) + LENGTH(cells_activated) "
            "+ COALESCE(LENGTH(query_embedding),0)), 0) FROM audit_log"
        ).fetchone()
    return db.stat().st_size - (row[0] or 0)


def main():
    parser = argparse.ArgumentParser(description="raven-memory performance benchmark")
    parser.add_argument("--sizes", default="1000,5000",
                        help="comma-separated corpus sizes (default 1000,5000)")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", default="", help="also write results to this JSON file")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    results = []
    print(f"\n🦅 raven-memory perf benchmark — sizes={sizes}, queries={args.queries}\n")
    header = (f"{'n':>7} {'store/s':>9} {'1st recall':>11} "
              f"{'p50 ms':>8} {'p95 ms':>8} {'DB MB':>7} {'B/audit':>8} {'links':>7}")
    print(header)
    print("-" * len(header))
    for n in sizes:
        r = bench_size(n, args.queries, args.seed)
        results.append(r)
        print(f"{r['n']:>7} {r['store_per_s']:>9} {r['first_recall_ms']:>10.1f}ms "
              f"{r['recall_p50_ms']:>8} {r['recall_p95_ms']:>8} {r['db_mb']:>7} "
              f"{r['audit_bytes_per_row']:>8} {r['cell_links']:>7}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nJSON → {args.json}")


if __name__ == "__main__":
    main()
