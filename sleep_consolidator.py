#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — Sleep Consolidator
Offline memory consolidation: groups similar episodic NEUTRAL memories
and merges them into consolidated semantic nodes.

Run:
    python sleep_consolidator.py                   # consolidate with default threshold
    python sleep_consolidator.py --threshold 0.90  # stricter grouping
    python sleep_consolidator.py --dry-run         # preview only

Authors: Anna Tchijova + Claude (VIGÍA AI Collective)
License: Apache 2.0
"""

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Same canonical hash scheme as the engine — the consolidation entry must be
# verifiable by memory_engine.verify_audit_chain() like any recall entry.
from memory_engine import compute_audit_hash

DB_PATH = Path("raven_memory.db")


def _connect(db_path: Path) -> sqlite3.Connection:
    """
    P0: busy timeout so a concurrent API server doesn't make consolidation
    crash with 'database is locked'. WAL mode is a persistent DB property
    already set by the engine at init.
    """
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# ============================================================
# DATA LOADING
# ============================================================

def load_episodic_neutral(db_path: Path) -> List[Dict]:
    """Load all episodic NEUTRAL memories from DB."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT memory_id, content, embedding, metadata, synaptic_links "
            "FROM memories WHERE layer='episodic' AND state='NEUTRAL'"
        ).fetchall()

    memories = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
        memories.append({
            "memory_id": r["memory_id"],
            "content": r["content"],
            "embedding": emb,
            "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
            "synaptic_links": json.loads(r["synaptic_links"]) if r["synaptic_links"] else {},
            "recall_count": 0,
        })
    return memories


def get_recall_counts(db_path: Path, memory_ids: List[str]) -> Dict[str, int]:
    """Count how often each memory was retrieved from the audit log."""
    counts = {mid: 0 for mid in memory_ids}
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT memories_retrieved FROM audit_log WHERE operation='recall'"
        ).fetchall()
    for (raw,) in rows:
        if not raw:
            continue
        try:
            for item in json.loads(raw):
                mid = item.get("memory_id", "")
                if mid in counts:
                    counts[mid] += 1
        except Exception:
            pass
    return counts


# ============================================================
# CLUSTERING
# ============================================================

def cluster_by_similarity(memories: List[Dict], threshold: float) -> List[List[Dict]]:
    """
    Group memories whose cosine similarity exceeds `threshold`.

    Two regimes:
      - Small corpora (< AGGLOMERATIVE_MAX): agglomerative clustering with a
        precomputed cosine-distance matrix — highest quality.
      - Large corpora: a greedy single-pass clusterer that never materialises
        an n×n matrix. cosine_distances() + average-linkage agglomerative is
        O(n²) memory and worse in time; at tens of thousands of episodic
        memories the matrix alone is hundreds of MB and the consolidator would
        effectively never finish. The greedy pass is O(n·clusters) and good
        enough for redundancy collapse, which is what consolidation needs.
    """
    if len(memories) < 2:
        return [[m] for m in memories]

    AGGLOMERATIVE_MAX = 2000

    if len(memories) <= AGGLOMERATIVE_MAX:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_distances

        embeddings = np.vstack([m["embedding"] for m in memories])
        dist_matrix = cosine_distances(embeddings)
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1.0 - threshold,
        )
        labels = model.fit_predict(dist_matrix)
        groups: Dict[int, List[Dict]] = {}
        for mem, label in zip(memories, labels):
            groups.setdefault(int(label), []).append(mem)
        return list(groups.values())

    # ---- greedy fallback for large corpora ----
    def _unit(v):
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v

    normed = [_unit(m["embedding"]) for m in memories]
    centroids: List[np.ndarray] = []     # running mean (unnormalised) per cluster
    members: List[List[int]] = []
    for i, v in enumerate(normed):
        best, best_sim = -1, -1.0
        for ci, c in enumerate(centroids):
            cu = _unit(c)
            sim = float(np.dot(v, cu))
            if sim > best_sim:
                best_sim, best = sim, ci
        if best >= 0 and best_sim >= threshold:
            members[best].append(i)
            k = len(members[best])
            centroids[best] = centroids[best] + (v - centroids[best]) / k
        else:
            centroids.append(v.copy())
            members.append([i])
    return [[memories[i] for i in grp] for grp in members]


# ============================================================
# MERGE
# ============================================================

def merge_group(group: List[Dict], recall_counts: Dict[str, int]) -> Dict:
    """
    Merge a cluster of memories into a single consolidated node.
    Embedding = weighted average by recall frequency.
    Summary = most information-dense sentences from the cluster.
    """
    weights = [max(1, recall_counts.get(m["memory_id"], 1)) for m in group]
    total_w = sum(weights)

    # Weighted embedding
    embs = np.vstack([m["embedding"] for m in group])
    weighted_emb = np.average(embs, axis=0, weights=weights).astype(np.float32)
    norm = np.linalg.norm(weighted_emb)
    if norm > 1e-10:
        weighted_emb /= norm

    # Extractive summary: pick sentences with highest uniqueness × length
    sentence_scores: Dict[str, float] = {}
    for m in group:
        sents = [s.strip() for s in m["content"].replace("?", ".").replace("!", ".").split(".") if s.strip()]
        for sent in sents:
            words = len(sent.split())
            presence = sum(1 for g in group if sent.lower() in g["content"].lower())
            sentence_scores[sent] = words * presence  # reward length + cross-memory appearance

    top_sents = sorted(sentence_scores, key=sentence_scores.__getitem__, reverse=True)[:3]
    summary = ". ".join(top_sents).strip()
    if summary and summary[-1] not in ".!?":
        summary += "."

    # P1: if every sentence in the cluster is too short to score (or the
    # contents have no sentence delimiters), the extractive summary comes
    # out empty — a consolidated node with empty content is unrecallable
    # and breaks content_hash semantics. Fall back to the longest source.
    if not summary:
        longest = max(group, key=lambda m: len(m["content"]))
        summary = longest["content"][:300].strip() or "(empty consolidated node)"

    # Union of synaptic links (keep highest weight; exclude intra-group)
    group_ids = {m["memory_id"] for m in group}
    merged_links: Dict[str, float] = {}
    for m in group:
        for lid, w in m["synaptic_links"].items():
            if lid not in group_ids:
                merged_links[lid] = max(merged_links.get(lid, 0.0), w)

    return {
        "summary": summary,
        "embedding": weighted_emb,
        "synaptic_links": merged_links,
        "metadata": {
            "consolidated_from": [m["memory_id"] for m in group],
            "original_count": len(group),
            "recall_counts": {m["memory_id"]: recall_counts.get(m["memory_id"], 0) for m in group},
            "total_weight": total_w,
            "consolidation_ts": time.time(),
        },
    }


# ============================================================
# DB OPERATIONS
# ============================================================

def apply_consolidation(db_path: Path, node: Dict) -> Tuple[str, int]:
    """
    Apply one consolidation atomically: insert the merged node, delete its
    sources, and cascade-clean their cell_links — ALL in one transaction.

    P0: the previous flow committed the INSERT and the DELETE separately.
    A crash between the two left the merged node AND its sources alive:
    duplicated content, double-counted recalls, and a spectral field built
    over a space that contains both. BEGIN IMMEDIATE takes the write lock
    up front so a concurrent engine write can't interleave either.

    Cell-link cascade rationale: without it, the BFS would follow links
    pointing to empty cells (ghost activations). Synaptic_links in other
    memories may still reference deleted IDs, but recall() guards against
    missing targets, so those dirty keys decay naturally via LTD.
    """
    ids: List[str] = node["metadata"]["consolidated_from"]
    content_hash = hashlib.sha256(node["summary"].encode()).hexdigest()
    memory_id = f"cons_{content_hash[:16]}_{int(time.time() * 1000)}"

    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")

        row = conn.execute("SELECT COALESCE(MAX(cell_id), -1) FROM memories").fetchone()
        # Re-read MAX inside the IMMEDIATE transaction to minimize TOCTOU window.
        # The UNIQUE constraint on cell_id (enforced at DB level) is the hard guard.
        cell_id = (row[0] if row[0] is not None else -1) + 1

        conn.execute(
            """INSERT INTO memories
            (memory_id, layer, content, content_hash, embedding, state, cell_id,
             created_at, session_id, author_id, metadata, synaptic_links,
             last_activation, recall_count, fingerprint)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                memory_id, "semantic", node["summary"], content_hash,
                node["embedding"].tobytes(), "NEUTRAL", cell_id,
                time.time(), "consolidation", "system",
                json.dumps(node["metadata"]),
                json.dumps(node["synaptic_links"]),
                0.0, 0, None,
            ),
        )

        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT cell_id FROM memories WHERE memory_id IN ({placeholders})", ids
        ).fetchall()
        cell_ids = [r[0] for r in rows]

        conn.execute(f"DELETE FROM memories WHERE memory_id IN ({placeholders})", ids)

        if cell_ids:
            cp = ",".join("?" * len(cell_ids))
            conn.execute(
                f"DELETE FROM cell_links "
                f"WHERE from_cell_id IN ({cp}) OR to_cell_id IN ({cp})",
                cell_ids + cell_ids,
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return memory_id, cell_id


def _get_last_audit_hash(db_path: Path) -> str:
    """Read the most recent audit_hash so consolidation continues the chain."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT audit_hash FROM audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if (row and row[0]) else "0" * 64


def _verify_chain_tail(db_path: Path, n: int = 25) -> bool:
    """
    P0: check linkage of the last n audit entries BEFORE appending.
    Chaining silently onto a broken chain launders the break — every entry
    after it looks valid and the original tamper point gets buried. We still
    append (consolidation must be logged), but the operator is warned loudly
    and the break stays discoverable via verify_audit_chain().
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT audit_hash, prev_hash FROM audit_log ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
    for i in range(len(rows) - 1):
        if rows[i][1] != rows[i + 1][0]:   # prev_hash[i] != audit_hash[i+1]
            return False
    return True


def log_consolidation(db_path: Path, processed: int, merged: int, created: int):
    # P0-5: chain consolidation into the existing audit log instead of
    # restarting with prev_hash="0"*64, which would break the forensic chain.
    if not _verify_chain_tail(db_path):
        print("   ⚠️  AUDIT CHAIN BROKEN before this consolidation — "
              "appending anyway, but run verify_audit_chain() and investigate.")

    prev_hash = _get_last_audit_hash(db_path)

    # P0: same canonical scheme as the engine. ts is captured ONCE and the
    # hashed fields are exactly the stored columns, so this entry verifies
    # under memory_engine.verify_audit_chain() like any recall entry.
    ts = time.time()
    query_text = f"Processed {processed}, merged {merged} → {created} consolidated nodes"
    cells_activated: List[int] = []
    results_payload = {"processed": processed, "merged": merged, "created": created}

    audit_hash = compute_audit_hash(
        ts, "consolidation", query_text, cells_activated, results_payload, prev_hash,
    )

    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO audit_log
            (timestamp, operation, query_text, query_embedding, cells_activated,
             memories_retrieved, total_candidates, filtered_by_state,
             filtered_by_estilometria, filtered_by_inhibitory, synaptic_activated,
             returned_to_agent, audit_hash, prev_hash)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ts, "consolidation",
                query_text,
                None, json.dumps(cells_activated),
                json.dumps(results_payload),
                processed, 0, 0, 0, 0, created,
                audit_hash, prev_hash,
            ),
        )
        conn.commit()


# ============================================================
# MAIN
# ============================================================

def main():
    # P0-2 design note: this script touches SQLite directly while the engine
    # may be running in another process. After consolidation, restart the engine
    # so its in-memory KDTree, _points, and _topic_index reflect the new state.
    # Future versions will expose a /consolidate endpoint to do this atomically.
    parser = argparse.ArgumentParser(description="raven-memory Sleep Consolidator")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for clustering (default: 0.85)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be merged without making changes")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="Path to SQLite database")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return

    print(f"\n🌙 Sleep Consolidator")
    print(f"   DB path   : {db_path}")
    print(f"   Threshold : {args.threshold}")
    print(f"   Dry-run   : {args.dry_run}")

    # Load
    memories = load_episodic_neutral(db_path)
    print(f"\n📥  Loaded {len(memories)} episodic NEUTRAL memories")

    if len(memories) < 2:
        print("✅  Nothing to consolidate (< 2 memories).")
        return

    # Recall counts
    counts = get_recall_counts(db_path, [m["memory_id"] for m in memories])
    for m in memories:
        m["recall_count"] = counts.get(m["memory_id"], 0)

    # Cluster
    groups = cluster_by_similarity(memories, args.threshold)
    mergeable = [g for g in groups if len(g) > 1]
    singletons = len(groups) - len(mergeable)

    print(f"🔬  Formed {len(groups)} clusters:")
    print(f"     • {len(mergeable)} mergeable groups")
    print(f"     • {singletons} singletons (unchanged)")

    if not mergeable:
        print("✅  No groups meet the merge threshold.")
        return

    # Preview
    print()
    for i, group in enumerate(mergeable, 1):
        total_recalls = sum(counts.get(m["memory_id"], 0) for m in group)
        print(f"  Group {i} ({len(group)} memories, {total_recalls} total recalls):")
        for m in group:
            print(f"    [{counts.get(m['memory_id'], 0):3d} recalls]  {m['content'][:70]}…")

    if args.dry_run:
        print("\n🔍  Dry-run — no changes made.")
        return

    # Merge
    size_before = db_path.stat().st_size
    total_merged = 0
    total_created = 0

    for group in mergeable:
        node = merge_group(group, counts)
        # P0: insert + delete + cell-link cascade in ONE transaction.
        mem_id, cell_id = apply_consolidation(db_path, node)
        total_merged += len(group)
        total_created += 1
        print(f"   ✅  {len(group)} → {mem_id[:24]}… (cell {cell_id})")

    size_after = db_path.stat().st_size
    log_consolidation(db_path, len(memories), total_merged, total_created)

    print(f"\n☀️  Consolidation complete.")
    print(f"   Merged   : {total_merged} memories → {total_created} nodes")
    print(f"   DB size  : {size_before / 1024:.1f} KB → {size_after / 1024:.1f} KB")

    # ---- Spectral field rebuild ------------------------------------------------
    # After consolidation the active-memory set changed: old episodic clusters
    # were replaced by consolidated semantic nodes. The spectral field must be
    # rebuilt so future recalls use eigen-modes that reflect the new field state.
    # This is an offline step — no running engine to notify.
    print("\n   Spectral field rebuild...")
    try:
        from spectral import build_and_persist_spectral_field
        spec = build_and_persist_spectral_field(db_path)
        if spec and spec.is_built:
            print(f"   {spec.summary()}")
        else:
            print("   Spectral rebuild skipped (fewer than 2 active memories)")
    except ImportError:
        print("   spectral module not found — skipping spectral rebuild")
    except Exception as _e:
        print(f"   Spectral rebuild error: {_e}")


if __name__ == "__main__":
    main()
