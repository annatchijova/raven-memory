#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — Recall-quality benchmark: field dynamics vs. plain top-k
(plan 4.1).

Methodology (honest by construction):
  The corpus is SYNTHETIC with known ground truth — clustered embeddings
  where relevance is defined by the generating cluster, plus contradiction
  pairs (same topic, different claim) with one side user-validated. This
  measures the FIELD MECHANICS (state boosts, INHIBITORY suppression, the
  rescue rule) against a brute-force cosine top-k baseline over the exact
  same vectors. It does NOT measure semantic embedding quality — that is a
  property of the embedding model, not of the field. Benchmarks over public
  conversational-memory datasets (LongMemEval, LoCoMo) require real
  embeddings and are future work.

Conditions:
  baseline — brute-force cosine top-k over all active memories (what most
             "memory" layers ship).
  field    — full raven-memory recall (states × hops × links × rescue).

Metrics:
  recall@5            do field dynamics hurt plain retrieval? (parity check)
  validated-first     contradiction queries where the user-validated claim
                      outranks the contradicted one
  suppression         contradiction queries where the unvalidated claim is
                      absent from top-k while the validated one is present
  rescue              queries where a REINFORCED truth challenged by an
                      unvalidated claim still appears in top-k

Run:  python benchmarks/quality.py [--seed 42] [--clusters 20]
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raven.memory_engine import AdaptiveMemoryEngine, MemoryState  # noqa: E402

DIM = 384
PER_CLUSTER = 10
TOP_K = 5


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)


def perturb(base: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    return unit(base + rng.standard_normal(DIM).astype(np.float32) * noise)


def baseline_topk(query: np.ndarray, entries, k: int = TOP_K):
    """Brute-force cosine top-k over active memories — the industry default."""
    active = [e for e in entries if e.state != MemoryState.FORGOTTEN]
    sims = [(float(np.dot(query, unit(e.embedding))), e) for e in active]
    sims.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in sims[:k]]


def build_corpus(engine, rng, n_clusters):
    """Clustered memories with known relevance + contradiction pairs."""
    centroids = [unit(rng.standard_normal(DIM).astype(np.float32))
                 for _ in range(n_clusters)]
    cluster_members = {c: [] for c in range(n_clusters)}
    for c, centroid in enumerate(centroids):
        for i in range(PER_CLUSTER):
            m = engine.store(
                f"documento del tema {c}, variante {i}, con contenido suficiente "
                f"para parecer un texto real almacenado en el campo",
                perturb(centroid, rng, 0.15),
            )
            cluster_members[c].append(m.memory_id)

    # One contradiction pair per cluster: claim A (validated) vs claim B.
    pairs = []
    for c, centroid in enumerate(centroids):
        anchor = perturb(centroid, rng, 0.05)
        m_true = engine.store(
            f"afirmación validada sobre el tema {c}: el sistema es determinista",
            perturb(anchor, rng, 0.02),
            metadata={"topic": f"tema_{c}", "claim": "A"},
        )
        m_false = engine.store(
            f"afirmación no verificada sobre el tema {c}: el sistema usa ML",
            perturb(anchor, rng, 0.02),
            metadata={"topic": f"tema_{c}", "claim": "B"},
        )
        engine.reinforce(m_true.memory_id)
        pairs.append((anchor, m_true.memory_id, m_false.memory_id))

    return centroids, cluster_members, pairs


def main():
    parser = argparse.ArgumentParser(description="raven-memory quality benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--queries-per-cluster", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tmp = Path(tempfile.mkdtemp()) / "quality.db"
    engine = AdaptiveMemoryEngine(db_path=tmp)

    centroids, members, pairs = build_corpus(engine, rng, args.clusters)
    all_entries = engine._db.load_memories()

    # ---- Metric 1: recall@5 on cluster queries (parity check) --------------
    rec_base, rec_field = [], []
    for c, centroid in enumerate(centroids):
        relevant = set(members[c])
        for _ in range(args.queries_per_cluster):
            q = perturb(centroid, rng, 0.10)
            b = baseline_topk(q, all_entries)
            f, _ = engine.recall(q, top_k=TOP_K, hops=2)
            denom = min(TOP_K, len(relevant))
            rec_base.append(len({e.memory_id for e in b} & relevant) / denom)
            rec_field.append(
                len({r.memory.memory_id for r in f} & relevant) / denom)

    # ---- Metrics 2-4: contradiction pairs ----------------------------------
    val_first_base = val_first_field = 0
    suppressed_field = suppressed_base = 0
    rescued = 0
    for anchor, true_id, false_id in pairs:
        q = perturb(anchor, rng, 0.03)

        b_ids = [e.memory_id for e in baseline_topk(q, all_entries)]
        f_res, _ = engine.recall(q, top_k=TOP_K, hops=2)
        f_ids = [r.memory.memory_id for r in f_res]

        def ranks_first(ids):
            if true_id in ids and false_id in ids:
                return ids.index(true_id) < ids.index(false_id)
            return true_id in ids and false_id not in ids

        val_first_base += ranks_first(b_ids)
        val_first_field += ranks_first(f_ids)
        suppressed_base += (true_id in b_ids and false_id not in b_ids)
        suppressed_field += (true_id in f_ids and false_id not in f_ids)
        rescued += (true_id in f_ids)

    n_pairs = len(pairs)
    print(f"\n🦅 raven-memory quality benchmark — seed={args.seed}, "
          f"{args.clusters} clusters × {PER_CLUSTER}, {n_pairs} contradiction pairs\n")
    rows = [
        ("recall@5 (cluster queries)",
         f"{np.mean(rec_base):.3f}", f"{np.mean(rec_field):.3f}"),
        ("validated claim ranks first",
         f"{val_first_base}/{n_pairs}", f"{val_first_field}/{n_pairs}"),
        ("contradiction suppressed from top-k",
         f"{suppressed_base}/{n_pairs}", f"{suppressed_field}/{n_pairs}"),
        ("validated truth present (rescue)",
         "n/a", f"{rescued}/{n_pairs}"),
    ]
    w = max(len(r[0]) for r in rows)
    print(f"{'metric':<{w}}  {'baseline top-k':>15}  {'raven field':>12}")
    print("-" * (w + 32))
    for name, b, f in rows:
        print(f"{name:<{w}}  {b:>15}  {f:>12}")
    print()


if __name__ == "__main__":
    main()
