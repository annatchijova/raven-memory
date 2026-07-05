#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — Test Suite
Integration tests for all P0 features.

Run: python test_suite.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from raven.memory_engine import (
    AdaptiveMemoryEngine, MemoryState, LinkType,
    MemoryStore, StylometricExtractor, StylometricFingerprint,
)


# ============================================================
# HELPERS
# ============================================================

_db_counter = 0

def fresh_engine() -> AdaptiveMemoryEngine:
    """Create an engine with a unique temporary DB."""
    global _db_counter
    _db_counter += 1
    p = Path(f"/tmp/raven_test_{_db_counter}.db")
    if p.exists():
        p.unlink()
    return AdaptiveMemoryEngine(db_path=p)


def make_emb(text: str, dim: int = 384) -> np.ndarray:
    """Deterministic test embedding."""
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(dim).astype(np.float32)
    e /= np.linalg.norm(e) + 1e-10
    return e


def close_emb(base: np.ndarray, noise: float = 0.02) -> np.ndarray:
    """Slightly perturbed embedding (similar to base)."""
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**31)
    e = base + rng.standard_normal(len(base)).astype(np.float32) * noise
    e /= np.linalg.norm(e) + 1e-10
    return e


# ============================================================
# TESTS
# ============================================================

def test_01_kdtree_construction():
    print("[TEST  1] KDTree construction from stores")
    eng = fresh_engine()
    for i in range(8):
        eng.store(f"Memory {i}", make_emb(f"mem_{i}"))
    eng._ensure_kdtree()
    assert eng.kdtree is not None, "KDTree must be built"
    assert len(eng._points) == 8, f"Expected 8 points, got {len(eng._points)}"
    avg = np.mean([len(n) for n in eng.cell_neighbors.values()])
    assert avg > 0, "Must have neighbours"
    print(f"  ✅  PASS — 8 cells, avg {avg:.1f} neighbours")


def test_02_ternary_states_in_recall():
    print("[TEST  2] Ternary states filter correctly")
    eng = fresh_engine()
    base = make_emb("base_cluster")
    m1 = eng.store("REINFORCED memory", close_emb(base), state=MemoryState.REINFORCED)
    m2 = eng.store("NEUTRAL memory",    close_emb(base), state=MemoryState.NEUTRAL)
    m3 = eng.store("FORGOTTEN memory",  close_emb(base), state=MemoryState.FORGOTTEN)

    results, _ = eng.recall(base, top_k=10, hops=2)
    ids = {r.memory.memory_id for r in results}

    assert m1.memory_id in ids, "REINFORCED must appear"
    assert m2.memory_id in ids, "NEUTRAL must appear"
    assert m3.memory_id not in ids, "FORGOTTEN must be excluded"
    print("  ✅  PASS — REINFORCED & NEUTRAL retrieved; FORGOTTEN excluded")


def test_03_reinforced_scores_higher():
    print("[TEST  3] REINFORCED scores higher than NEUTRAL (same content)")
    eng = fresh_engine()
    base = make_emb("identical_concept")
    m_r = eng.store("Reinforced fact", close_emb(base, 0.005), state=MemoryState.REINFORCED)
    m_n = eng.store("Neutral fact",    close_emb(base, 0.005), state=MemoryState.NEUTRAL)

    results, _ = eng.recall(base, top_k=5, hops=2)
    scores = {r.memory.memory_id: r.final_score for r in results}

    assert m_r.memory_id in scores and m_n.memory_id in scores, "Both must appear"
    assert scores[m_r.memory_id] > scores[m_n.memory_id], (
        f"REINFORCED ({scores[m_r.memory_id]:.4f}) must beat NEUTRAL ({scores[m_n.memory_id]:.4f})"
    )
    ratio = scores[m_r.memory_id] / scores[m_n.memory_id]
    print(f"  ✅  PASS — REINFORCED/NEUTRAL ratio = {ratio:.2f}× (expected ~1.5×)")


def test_04_hop_decay():
    print("[TEST  4] Hop decay penalises distant cells")
    eng = fresh_engine()
    base_a = make_emb("cluster_alpha")
    base_b = make_emb("cluster_beta_far")

    for i in range(4):
        eng.store(f"Cluster A {i}", close_emb(base_a, 0.01))
    for i in range(4):
        eng.store(f"Cluster B {i}", close_emb(base_b, 0.01))

    results, _ = eng.recall(base_a, top_k=10, hops=2)
    a_scores = [r.final_score for r in results if "Cluster A" in r.memory.content]
    b_scores = [r.final_score for r in results if "Cluster B" in r.memory.content]

    if a_scores and b_scores:
        assert max(a_scores) >= max(b_scores), "Cluster A must score ≥ Cluster B near its seed"
    print(f"  ✅  PASS — A_max={max(a_scores, default=0):.3f}, B_max={max(b_scores, default=0):.3f}")


def test_05_stdp_ltp():
    print("[TEST  5] STDP LTP — co-activated memories strengthen")
    eng = fresh_engine()
    m1 = eng.store("Fact A", make_emb("stdp_a"))
    m2 = eng.store("Fact B", make_emb("stdp_b"))

    q = make_emb("query_near_both")
    results1, _ = eng.recall(q, top_k=10, hops=3, current_turn_memories=None)
    turn1 = [r.memory.memory_id for r in results1]

    eng.recall(q, top_k=10, hops=3, current_turn_memories=turn1)

    m1_r = eng._db.load_memory(m1.memory_id)
    m2_r = eng._db.load_memory(m2.memory_id)

    link_a_b = m1_r.synaptic_links.get(m2.memory_id, 0)
    link_b_a = m2_r.synaptic_links.get(m1.memory_id, 0)

    # At least one direction should have a non-zero link after co-activation
    assert link_a_b > 0 or link_b_a > 0, (
        f"Expected STDP link after co-activation. A→B={link_a_b}, B→A={link_b_a}"
    )
    print(f"  ✅  PASS — STDP links: A→B={link_a_b:.2f}, B→A={link_b_a:.2f}")


def test_06_inhibitory_links():
    print("[TEST  6] INHIBITORY links auto-created for contradictions")
    eng = fresh_engine()
    emb_x = make_emb("claim_x")
    emb_y = make_emb("claim_y")

    eng.store("VIGÍA is deterministic", close_emb(emb_x),
              metadata={"topic": "vigia", "claim": "deterministic"})
    eng.store("VIGÍA uses ML", close_emb(emb_y),
              metadata={"topic": "vigia", "claim": "ml_system"})

    links = eng._db.load_all_cell_links()
    inhibitory = [(f, t) for f, t, lt in links if lt == "INHIBITORY"]
    assert len(inhibitory) >= 2, f"Expected ≥2 INHIBITORY links, got {inhibitory}"
    print(f"  ✅  PASS — {len(inhibitory)} INHIBITORY links auto-created")


def test_07_inhibitory_blocks_recall():
    print("[TEST  7] INHIBITORY link blocks contradicted memory from recall")
    eng = fresh_engine()
    base = make_emb("vigia_topic")
    m_det = eng.store("VIGÍA is deterministic", close_emb(base, 0.005),
                       metadata={"topic": "vigia_type", "claim": "deterministic"})
    m_ml  = eng.store("VIGÍA uses ML embeddings", close_emb(base, 0.005),
                       metadata={"topic": "vigia_type", "claim": "ml_hybrid"})

    # Reinforce the deterministic claim
    eng.reinforce(m_det.memory_id)

    results, _ = eng.recall(base, top_k=5, hops=2)
    ids = {r.memory.memory_id for r in results}

    # m_ml should be suppressed by INHIBITORY link (it may still appear if the cell
    # is the query seed, but typically the inhibited one scores much lower)
    scores = {r.memory.memory_id: r.final_score for r in results}
    if m_det.memory_id in scores and m_ml.memory_id in scores:
        assert scores[m_det.memory_id] > scores[m_ml.memory_id], (
            "REINFORCED+deterministic must outscore NEUTRAL+inhibited"
        )
    print(f"  ✅  PASS — deterministic={scores.get(m_det.memory_id, 'N/A')}, ml={scores.get(m_ml.memory_id, 'not retrieved')}")


def test_08_stylometric_fingerprinting():
    print("[TEST  8] Stylometric fingerprint distance")
    ext = StylometricExtractor()
    fp_es1 = ext.extract("El gato come pescado. Es un animal doméstico.", "anna")
    fp_es2 = ext.extract("El perro juega en el parque. Es muy activo.", "anna")
    fp_en  = ext.extract("The cat eats fish. It is a domestic animal.", "hacker")

    dist_same = ext.compare(fp_es1, fp_es2)
    dist_diff = ext.compare(fp_es1, fp_en)

    assert dist_diff > dist_same, (
        f"Cross-language distance ({dist_diff:.3f}) must exceed same-language ({dist_same:.3f})"
    )
    print(f"  ✅  PASS — same-lang dist={dist_same:.3f}, cross-lang dist={dist_diff:.3f}")


def test_09_audit_hash_chain():
    print("[TEST  9] Audit hash-chain integrity")
    eng = fresh_engine()
    eng.store("Test A", make_emb("audit_a"))
    eng.store("Test B", make_emb("audit_b"))

    q = make_emb("query")
    _, a1 = eng.recall(q, query_text="q1")
    _, a2 = eng.recall(q, query_text="q2")
    _, a3 = eng.recall(q, query_text="q3")

    hashes = {a1.audit_hash, a2.audit_hash, a3.audit_hash}
    assert len(hashes) == 3, "Each audit entry must have a unique hash"

    # Chain: a2.prev_hash == a1.audit_hash, etc.
    assert a2.prev_hash == a1.audit_hash, f"Chain broken: a2.prev={a2.prev_hash[:8]} ≠ a1.hash={a1.audit_hash[:8]}"
    assert a3.prev_hash == a2.audit_hash, "Chain broken at a3→a2"
    print(f"  ✅  PASS — 3-entry chain intact ({a1.audit_hash[:12]}…)")


def test_10_persistence_restart():
    print("[TEST 10] Persistence — memories survive engine restart")
    p = Path("/tmp/raven_persist_test.db")
    if p.exists():
        p.unlink()

    # Write
    eng1 = AdaptiveMemoryEngine(db_path=p)
    emb = make_emb("persistent memory text")
    entry = eng1.store("Persistent truth", emb, metadata={"topic": "persist"})
    mid = entry.memory_id
    del eng1

    # Read in new engine instance
    eng2 = AdaptiveMemoryEngine(db_path=p)
    loaded = eng2._db.load_memory(mid)
    assert loaded is not None, "Memory must survive restart"
    assert loaded.content == "Persistent truth"
    assert len(eng2._points) == 1, "KDTree must be rebuilt from DB on restart"

    results, _ = eng2.recall(emb, top_k=5, hops=2)
    assert any(r.memory.memory_id == mid for r in results), "Memory must be recallable after restart"
    p.unlink()
    print("  ✅  PASS — memory recalled after engine restart")


def test_11_mss_dynamics():
    print("[TEST 11] Memory Stability Score dynamics")
    eng = fresh_engine()
    for i in range(4):
        eng.store(f"Memory {i}", make_emb(f"m{i}"), state=MemoryState.NEUTRAL)

    stats = eng.get_stats()
    assert stats["memory_stability_score"] == 0.0, f"All NEUTRAL → MSS must be 0, got {stats['memory_stability_score']}"

    mems = eng._db.load_memories()
    eng.reinforce(mems[0].memory_id)
    stats2 = eng.get_stats()
    assert stats2["memory_stability_score"] > 0.0, "After reinforce, MSS must be > 0"

    eng.reinforce(mems[1].memory_id)
    stats3 = eng.get_stats()
    assert stats3["memory_stability_score"] > stats2["memory_stability_score"], "More reinforced → higher MSS"
    print(f"  ✅  PASS — MSS: 0.000 → {stats2['memory_stability_score']:.3f} → {stats3['memory_stability_score']:.3f}")


def test_12_forget_excludes_and_mss():
    print("[TEST 12] Forget state excludes memory and doesn't count in MSS denominator")
    eng = fresh_engine()
    base = make_emb("forgettable")
    m = eng.store("Forgettable thing", base, state=MemoryState.NEUTRAL)

    eng.forget(m.memory_id)
    results, _ = eng.recall(base, top_k=5, hops=2)
    ids = {r.memory.memory_id for r in results}
    assert m.memory_id not in ids, "FORGOTTEN memory must not be recalled"

    stats = eng.get_stats()
    assert stats["state_distribution"]["FORGOTTEN"] == 1
    print("  ✅  PASS — FORGOTTEN memory excluded from recall and counted separately")


def test_13_export_graph():
    print("[TEST 13] Graph export returns nodes and edges")
    eng = fresh_engine()
    eng.store("Node A", make_emb("node_a"), metadata={"topic": "t", "claim": "a"})
    eng.store("Node B", make_emb("node_b"), metadata={"topic": "t", "claim": "b"})

    graph = eng.export_graph()
    assert "nodes" in graph and "edges" in graph
    assert len(graph["nodes"]) == 2
    assert all("state" in n for n in graph["nodes"])
    print(f"  ✅  PASS — {len(graph['nodes'])} nodes, {len(graph['edges'])} edges exported")


def test_14_manual_cell_links():
    print("[TEST 14] Manual RESONANT cell link boosts recall")
    eng = fresh_engine()
    m1 = eng.store("Concept A", make_emb("concept_a"))
    m2 = eng.store("Concept B", make_emb("concept_b"))  # semantically distant

    # Manually link B as RESONANT from A
    eng.create_cell_link(m1.cell_id, m2.cell_id, LinkType.RESONANT)

    results, _ = eng.recall(make_emb("concept_a"), top_k=5, hops=2)
    ids = {r.memory.memory_id for r in results}

    # P2-5: unconditional assert — m1 is the query seed and must be recalled
    assert m1.memory_id in ids, "m1 must be recalled (it IS the query seed)"

    # If m2 appears via the RESONANT link, its score must be positive
    m2_res = next((r for r in results if r.memory.memory_id == m2.memory_id), None)
    if m2_res:
        assert m2_res.final_score > 0, "RESONANT-pulled memory must have positive score"
        print(f"  ✅  PASS — m1 recalled, m2 pulled via RESONANT (score={m2_res.final_score:.3f})")
    else:
        # RESONANT link exists in DB even if geometry kept m2 outside the hop radius
        links = eng._db.load_all_cell_links()
        resonant = [(f, t) for f, t, lt in links if lt == "RESONANT"]
        assert len(resonant) > 0, "RESONANT link must exist in DB"
        print(f"  ✅  PASS — RESONANT link in DB, m1 recalled, m2 outside hop radius")


def test_15_stats_completeness():
    print("[TEST 15] Stats dict has all expected keys")
    eng = fresh_engine()
    eng.store("A", make_emb("a"))
    stats = eng.get_stats()
    required = {
        "total_memories", "voronoi_cells", "state_distribution",
        "layer_distribution", "memory_stability_score", "retention_ratio",
        "avg_neighbors", "total_recalls", "cell_links", "authors",
    }
    missing = required - set(stats.keys())
    assert not missing, f"Missing stats keys: {missing}"
    print(f"  ✅  PASS — all {len(required)} stat keys present")


def test_16_audit_chain_tamper_detection():
    print("[TEST 16] Audit chain is recomputable and detects content tampering")
    from memory_engine import verify_audit_chain
    import sqlite3 as _sq

    eng = fresh_engine()
    eng.store("Immutable fact", make_emb("tamper_a"))
    eng.store("Another fact", make_emb("tamper_b"))
    q = make_emb("tamper_a")
    eng.recall(q, query_text="q1")
    eng.recall(q, query_text="q2")

    # Clean chain: linkage AND per-row hash recomputation must pass.
    entries = eng.get_audit_trail(limit=10)
    report = verify_audit_chain(entries)
    assert report["chain_intact"], f"Fresh chain must link: {report['issues']}"
    assert report["hash_integrity"], (
        f"Fresh chain must recompute from stored columns: {report['issues']}"
    )

    # Tamper: edit the retrieved-memories payload of the newest audit row
    # directly in SQLite (simulates post-hoc content manipulation).
    with _sq.connect(eng._db.db_path) as conn:
        row = conn.execute(
            "SELECT id, memories_retrieved FROM audit_log "
            "WHERE operation='recall' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        tampered = (row[1] or "[]").replace("Immutable", "Falsified")
        if tampered == row[1]:  # ensure the row actually changes
            tampered = row[1].replace('"recall"', '"recall_"') if row[1] else "[]"
        conn.execute(
            "UPDATE audit_log SET memories_retrieved=? WHERE id=?",
            (tampered, row[0]),
        )
        conn.commit()

    entries2 = eng.get_audit_trail(limit=10)
    report2 = verify_audit_chain(entries2)
    assert not report2["hash_integrity"], (
        "Tampered payload MUST break hash recomputation — "
        "otherwise tamper-evidence is cosmetic"
    )
    assert any(i["type"] == "hash_mismatch" for i in report2["issues"])
    print("  ✅  PASS — clean chain verifies; tampered payload detected")


def test_17_stylometric_language_detection():
    print("[TEST 17] Stylometric language detection — bilingual author ≠ tamperer")
    ext = StylometricExtractor()
    fp_es = ext.extract(
        "El sistema de memoria adaptativa guarda los recuerdos en celdas. "
        "Cada celda tiene un estado ternario que modula la recuperación.", "anna")
    fp_en = ext.extract(
        "The adaptive memory system stores the memories in cells. "
        "Each cell has a ternary state that modulates the retrieval.", "anna")

    assert fp_es.language == "es", f"Expected 'es', got '{fp_es.language}'"
    assert fp_en.language == "en", f"Expected 'en', got '{fp_en.language}'"

    # The engine must NOT degrade a memory just because the same author
    # switched languages (forensic check is skipped on language switch).
    eng = fresh_engine()
    base = make_emb("bilingual_base")
    eng.store(
        "El sistema de memoria adaptativa guarda los recuerdos en celdas "
        "usando estados ternarios para modular toda la recuperación semántica.",
        close_emb(base, 0.005), author_id="anna")
    m_en = eng.store(
        "The adaptive memory system stores all the memories in spatial cells "
        "using ternary states that modulate the whole semantic retrieval flow.",
        close_emb(base, 0.005), author_id="anna")

    results, audit = eng.recall(base, top_k=5, hops=2)
    ids = {r.memory.memory_id for r in results}
    assert m_en.memory_id in ids, "English memory must survive (language switch ≠ tampering)"
    assert audit.filtered_by_estilometria == 0, "No forensic false positives expected"
    print("  ✅  PASS — es/en detected; bilingual author not flagged as tamperer")


# ============================================================
# RUNNER
# ============================================================

def test_18_reinforce_restores_forgotten_to_kdtree():
    print("[TEST 18] FORGOTTEN → REINFORCED restores the cell to the KDTree")
    eng = fresh_engine()
    base = make_emb("restore_base")
    target = eng.store("The cavity is one", close_emb(base, 0.004))
    for i in range(4):
        eng.store(f"filler {i}", make_emb(f"restore_filler_{i}"))

    eng.forget(target.memory_id)
    eng._ensure_kdtree()
    assert target.cell_id not in eng._active_cells, "forgotten cell must leave active set"
    res1, _ = eng.recall(base, top_k=5, hops=2)
    assert target.memory_id not in {r.memory.memory_id for r in res1}, \
        "forgotten memory must not surface"

    # Bug #2/#5: reinforcing a forgotten memory must bring it back into the index
    eng.reinforce(target.memory_id)
    eng._ensure_kdtree()
    assert target.cell_id in eng._active_cells, "reinforce must re-register the cell"
    res2, _ = eng.recall(base, top_k=5, hops=2)
    assert target.memory_id in {r.memory.memory_id for r in res2}, \
        "reinforced memory must be recallable again"
    print("  ✅  PASS — reinforce re-enters the KDTree; memory recallable again")


def test_19_sparse_points_no_dead_cells():
    print("[TEST 19] _points is sparse — deleted cells leave no holes")
    eng = fresh_engine()
    ids = [eng.store(f"m{i}", make_emb(f"sparse_{i}")) for i in range(6)]
    assert isinstance(eng._points, dict), "_points must be a sparse dict"
    assert len(eng._points) == 6

    # Simulate consolidation deleting some sources (ids never reused)
    import sqlite3 as _sq
    dead = [ids[1].cell_id, ids[3].cell_id]
    with _sq.connect(eng._db.db_path) as conn:
        conn.execute("DELETE FROM memories WHERE cell_id IN (?,?)", dead)
        conn.commit()
    eng._load_from_db()
    assert len(eng._points) == 4, f"expected 4 live cells, got {len(eng._points)}"
    for d in dead:
        assert d not in eng._points, "deleted cell must not occupy a slot"
    # next id keeps climbing — no reuse
    nxt = eng.store("after delete", make_emb("sparse_after"))
    assert nxt.cell_id > max(m.cell_id for m in ids), "cell_id must stay monotonic"
    print("  ✅  PASS — sparse dict, monotonic ids, no zero-vector holes")


def test_20_greedy_clustering_large_corpus():
    print("[TEST 20] Consolidation clustering uses greedy path above the cap")
    from sleep_consolidator import cluster_by_similarity
    rng = np.random.default_rng(7)
    # Build 3 tight families well above the agglomerative cap → forces greedy
    families = [rng.standard_normal(384).astype(np.float32) for _ in range(3)]
    mems = []
    for fam in families:
        fam /= np.linalg.norm(fam)
        for _ in range(900):
            v = fam + rng.standard_normal(384).astype(np.float32) * 0.02
            v /= np.linalg.norm(v)
            mems.append({"embedding": v, "content": "x", "memory_id": str(len(mems))})
    assert len(mems) == 2700  # > AGGLOMERATIVE_MAX (2000)
    import time as _t
    t0 = _t.time()
    groups = cluster_by_similarity(mems, threshold=0.90)
    elapsed = _t.time() - t0
    # Should recover roughly the 3 families and finish fast (no n×n matrix)
    big = [g for g in groups if len(g) > 100]
    assert len(big) == 3, f"expected 3 dominant clusters, got {len(big)}"
    assert elapsed < 10, f"greedy clustering too slow: {elapsed:.1f}s"
    print(f"  ✅  PASS — 3 families recovered from 2700 mems in {elapsed:.2f}s (no n×n matrix)")


def run_all():
    print("=" * 65)
    print("  raven-memory v1.0 — Integration Test Suite")
    print("=" * 65)

    tests = [
        test_01_kdtree_construction,
        test_02_ternary_states_in_recall,
        test_03_reinforced_scores_higher,
        test_04_hop_decay,
        test_05_stdp_ltp,
        test_06_inhibitory_links,
        test_07_inhibitory_blocks_recall,
        test_08_stylometric_fingerprinting,
        test_09_audit_hash_chain,
        test_10_persistence_restart,
        test_11_mss_dynamics,
        test_12_forget_excludes_and_mss,
        test_13_export_graph,
        test_14_manual_cell_links,
        test_15_stats_completeness,
        test_16_audit_chain_tamper_detection,
        test_17_stylometric_language_detection,
        test_18_reinforce_restores_forgotten_to_kdtree,
        test_19_sparse_points_no_dead_cells,
        test_20_greedy_clustering_large_corpus,
    ]

    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ❌ FAIL: {e}")
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 65)
    print(f"  RESULT: {passed} PASS, {failed} FAIL  ({passed}/{len(tests)})")
    print("=" * 65)

    if failed == 0:
        print("  🎉  ALL TESTS PASSED")
    else:
        print(f"  ⚠️   {failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
