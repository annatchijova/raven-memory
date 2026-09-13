#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — causal intervention probes (docs/INTERVENTION_DESIGN.md).

The ten gates the design had to clear before the primitive ships:

  1  _recall_core() has no side effects
  2  baseline and intervention share one snapshot and one `now`
  3  stage="field" only in v1
  4  unknown mode/stage → hard failure, never a silent Δ=0
  5  resolved target ids inside the sealed v4 payload
  6  intervention=None preserves historical hashes
  7  deterministic (score, memory_id) ordering
  8  diagnostics distinguish suppression / inhibition / unreachability / filtering
  9  the probe touches no STDP, activations, states, links or stylometric enforcement
 10  after any number of probes, future observable behaviour is unchanged

Run: pytest tests/test_intervention.py -q
"""

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raven.memory_engine import (
    AdaptiveMemoryEngine,
    ExclusionReason,
    InterventionError,
    InterventionSpec,
    LinkType,
    MemoryState,
    compute_audit_hash,
    verify_audit_chain,
)
from raven import intervention as iv


def make_emb(text: str, dim: int = 384) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(dim).astype(np.float32)
    e /= np.linalg.norm(e) + 1e-10
    return e


def cluster_emb(base: str, k: int, jitter: float = 0.15) -> np.ndarray:
    """A vector near `base` — builds a genuine neighbourhood, not noise."""
    b = make_emb(base)
    n = make_emb(f"{base}::{k}")
    e = b + jitter * n
    return (e / np.linalg.norm(e)).astype(np.float32)


def build_field(tmp_path, n: int = 12) -> AdaptiveMemoryEngine:
    eng = AdaptiveMemoryEngine(db_path=tmp_path / "probe.db")
    for i in range(n):
        eng.store(f"cluster alpha document number {i}", cluster_emb("alpha", i))
    for i in range(n // 2):
        eng.store(f"cluster beta document number {i}", cluster_emb("beta", i))
    return eng


def portable_state(engine) -> dict:
    """Persistent state keyed by cell_id instead of memory_id.

    memory_id embeds the wall-clock of its store() call, so two independently
    built fields never share ids even when they are otherwise identical. cell_id
    is the stable structural identity, and content_hash pins the payload.
    """
    all_mems = engine.list_memories(limit=100000)
    id_to_cell = {m.memory_id: m.cell_id for m in all_mems}
    # last_activation is wall-clock, so two runs never produce identical values.
    # What must be identical is WHICH memories were activated and in what
    # ORDER — that is the part a probe could corrupt.
    order = {m.cell_id: i for i, m in enumerate(
        sorted(all_mems, key=lambda x: (x.last_activation, x.cell_id)))}
    mems = []
    for m in sorted(all_mems, key=lambda x: x.cell_id):
        mems.append({
            "cell_id": m.cell_id,
            "content_hash": m.content_hash,
            "state": m.state.name,
            "activated": m.last_activation > 0,
            "activation_rank": order[m.cell_id],
            "recall_count": m.recall_count,
            "synaptic_links": sorted(
                (id_to_cell.get(k, k), round(v, 9))
                for k, v in m.synaptic_links.items()
            ),
        })
    with sqlite3.connect(engine._db.db_path) as conn:
        links = [tuple(r) for r in conn.execute(
            "SELECT from_cell_id, to_cell_id, link_type FROM cell_links "
            "ORDER BY from_cell_id, to_cell_id")]
        alerts = conn.execute("SELECT COUNT(*) FROM forensic_alerts").fetchone()[0]
    return {"memories": mems, "links": links, "alerts": alerts}


def db_fingerprint(engine) -> dict:
    """Everything persistent EXCEPT the audit log, whose whole job is to grow."""
    with sqlite3.connect(engine._db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        mems = [dict(r) for r in conn.execute(
            "SELECT memory_id, state, cell_id, synaptic_links, last_activation, "
            "recall_count, content_hash FROM memories ORDER BY cell_id")]
        links = [dict(r) for r in conn.execute(
            "SELECT from_cell_id, to_cell_id, link_type FROM cell_links "
            "ORDER BY from_cell_id, to_cell_id")]
        alerts = conn.execute("SELECT COUNT(*) FROM forensic_alerts").fetchone()[0]
    return {"memories": mems, "links": links, "alerts": alerts}


# ============================================================
# Gate 4 — closed spec: unknown input fails, never returns Δ=0
# ============================================================

@pytest.mark.parametrize("bad", [
    {"mode": "nope", "targets": []},
    {"mode": "suppress", "targets": [], "stage": "nonsense"},
    {"mode": "suppress", "targets": [], "unexpected_key": 1},
    {"mode": "suppress"},
])
def test_unknown_intervention_input_fails_closed(bad):
    """A silently ignored intervention produces Δ=0, which is indistinguishable
    from the real finding 'no causal influence'. It must raise instead."""
    with pytest.raises(InterventionError):
        InterventionSpec.from_dict(bad)


def test_reserved_modes_are_rejected_with_their_reason():
    """excite/stimulate are named in the design but need plasticity authority
    a read-only probe does not have — rejected, not silently downgraded."""
    with pytest.raises(InterventionError, match="reserved but not implemented"):
        InterventionSpec.from_dict({"mode": "excite", "targets": []})


# ============================================================
# Gate 3 — v1 implements stage="field" only
# ============================================================

def test_readout_stage_is_refused_until_implemented():
    with pytest.raises(InterventionError, match="not implemented in v1"):
        InterventionSpec.suppress(["m1"], stage="readout")


def test_duplicate_targets_are_refused():
    """A repeated target makes the treatment population ambiguous."""
    with pytest.raises(InterventionError, match="duplicate"):
        InterventionSpec.suppress(["m1", "m1"])


def test_unresolvable_targets_fail_closed(tmp_path):
    eng = build_field(tmp_path)
    q = make_emb("alpha")
    with pytest.raises(InterventionError, match="could not be resolved"):
        eng.intervene(q, InterventionSpec.suppress(["mem_does_not_exist"]))


def test_forgotten_target_is_refused_not_silently_skipped(tmp_path):
    eng = build_field(tmp_path)
    victim = eng.list_memories(limit=100)[0]
    eng.forget(victim.memory_id)
    with pytest.raises(InterventionError, match="inactive"):
        eng.intervene(make_emb("alpha"), InterventionSpec.suppress([victim.memory_id]))


# ============================================================
# Gate 2 — the null intervention is exactly the baseline
# ============================================================

def test_empty_intervention_is_an_exact_null(tmp_path):
    """Negative control: with nothing suppressed, Δ must be exactly zero —
    not 'small'. If this ever drifts, every non-zero Δ is suspect."""
    eng = build_field(tmp_path)
    res = eng.intervene(cluster_emb("alpha", 3), InterventionSpec.suppress([]))

    assert [r.memory.memory_id for r in res.baseline] == \
           [r.memory.memory_id for r in res.perturbed]
    assert res.delta["rank_displacement"] == 0
    assert res.delta["score_delta"] == {}
    assert res.delta["top_k"]["disappeared"] == []
    assert res.delta["top_k"]["appeared"] == []
    assert res.delta["disappeared_from_field"] == []
    assert not res.changed


def test_baseline_branch_matches_a_plain_recall(tmp_path):
    """The probe's baseline arm must be the same computation recall() runs —
    otherwise the counterfactual is measured against the wrong reference."""
    eng = build_field(tmp_path)
    q = cluster_emb("alpha", 2)
    probe = eng.intervene(q, InterventionSpec.suppress([]))
    plain, _ = eng.recall(q)

    assert [r.memory.memory_id for r in probe.baseline] == \
           [r.memory.memory_id for r in plain]
    # recency_bonus moves with wall-clock, so scores are compared at the
    # resolution the audit itself seals.
    for a, b in zip(probe.baseline, plain):
        assert round(a.final_score, 4) == round(b.final_score, 4)


# ============================================================
# Gate 7 — deterministic ordering, gate 2 — deterministic Δ
# ============================================================

def test_delta_is_deterministic_across_repeated_probes(tmp_path):
    eng = build_field(tmp_path)
    q = cluster_emb("alpha", 1)
    target = [r.memory.memory_id for r in eng.recall(q, top_k=3)[0]][0]
    spec = InterventionSpec.suppress([target])

    d1 = eng.intervene(q, spec).delta
    d2 = eng.intervene(q, spec).delta
    assert d1 == d2


def test_ranking_breaks_ties_deterministically(tmp_path):
    """Sorting on final_score alone left ties to insertion order, which shifts
    when the active cell set changes — exactly what a probe does.

    The field is built with genuinely tied memories on purpose: a fixture that
    happens to contain no ties would make this test pass vacuously."""
    eng = AdaptiveMemoryEngine(db_path=tmp_path / "ties.db")
    shared = make_emb("identical-vector")
    for i in range(6):
        eng.store(f"distinct content number {i} sharing one embedding", shared.copy())

    results, _ = eng.recall(shared, top_k=20)
    tied = [(a, b) for a, b in zip(results, results[1:])
            if round(a.final_score, 12) == round(b.final_score, 12)]
    assert tied, "fixture produced no ties — the assertion below would be vacuous"
    for a, b in tied:
        assert a.memory.memory_id < b.memory.memory_id


# ============================================================
# Gates 1 + 9 — the probe writes nothing but its audit row
# ============================================================

def test_probe_leaves_persistent_state_untouched(tmp_path):
    eng = build_field(tmp_path)
    q = cluster_emb("alpha", 4)
    ids = [r.memory.memory_id for r in eng.recall(q, top_k=5)[0]]

    before = db_fingerprint(eng)
    audit_before = len(eng.get_audit_trail(limit=10000))

    for mem_id in ids[:3]:
        eng.intervene(q, InterventionSpec.suppress([mem_id]),
                      current_turn_memories=ids)

    after = db_fingerprint(eng)
    assert after == before, "a probe mutated persistent state"

    audit_after = eng.get_audit_trail(limit=10000)
    assert len(audit_after) == audit_before + 3
    assert sum(1 for e in audit_after
               if e["operation"] == "recall_intervention") == 3


def test_probe_does_not_touch_stdp_or_activations(tmp_path):
    """current_turn_memories drives STDP and activation writes on a real
    recall. The probe takes the same argument and must ignore its effects."""
    eng = build_field(tmp_path)
    q = cluster_emb("alpha", 6)
    ids = [r.memory.memory_id for r in eng.recall(q, top_k=5)[0]]

    links_before = {m.memory_id: dict(m.synaptic_links)
                    for m in eng.list_memories(limit=1000)}
    act_before = {m.memory_id: m.last_activation
                  for m in eng.list_memories(limit=1000)}

    eng.intervene(q, InterventionSpec.suppress([ids[0]]), current_turn_memories=ids)

    links_after = {m.memory_id: dict(m.synaptic_links)
                   for m in eng.list_memories(limit=1000)}
    act_after = {m.memory_id: m.last_activation
                 for m in eng.list_memories(limit=1000)}
    assert links_after == links_before
    assert act_after == act_before


# ============================================================
# Gates 5 + 6 — audit binding and historical hashes
# ============================================================

def test_intervention_is_sealed_with_resolved_target_ids(tmp_path):
    eng = build_field(tmp_path)
    q = cluster_emb("alpha", 5)
    victim = eng.recall(q, top_k=1)[0][0].memory
    res = eng.intervene(q, InterventionSpec.suppress([victim.memory_id]))

    sealed = res.audit.intervention
    assert sealed["mode"] == "suppress" and sealed["stage"] == "field"
    # The RESOLVED population, not the selector: a selector's meaning drifts
    # with the field, so it cannot identify the treatment that was applied.
    assert sealed["targets"] == [
        {"memory_id": victim.memory_id, "cell_id": victim.cell_id}
    ]

    row = [e for e in eng.get_audit_trail(limit=10)
           if e["operation"] == "recall_intervention"][0]
    assert json.loads(row["intervention"])["targets"] == sealed["targets"]
    assert verify_audit_chain(eng.get_audit_trail(limit=10000))["hash_integrity"]


def test_tampering_with_a_sealed_spec_breaks_verification(tmp_path):
    eng = build_field(tmp_path)
    q = cluster_emb("alpha", 5)
    victim = eng.recall(q, top_k=1)[0][0].memory
    eng.intervene(q, InterventionSpec.suppress([victim.memory_id]))

    with sqlite3.connect(eng._db.db_path) as conn:
        row = conn.execute(
            "SELECT id, intervention FROM audit_log "
            "WHERE operation='recall_intervention' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        forged = json.loads(row[1])
        forged["targets"] = []          # "we never suppressed anything"
        conn.execute("UPDATE audit_log SET intervention=? WHERE id=?",
                     (json.dumps(forged, sort_keys=True, separators=(",", ":")), row[0]))
        conn.commit()

    report = verify_audit_chain(eng.get_audit_trail(limit=10000))
    assert not report["hash_integrity"], \
        "rewriting the sealed treatment population must break the chain"


def _v3_scheme_hash(ts, operation, query, cells, results, prev, qemb_hash):
    """The pre-v4 payload, rebuilt independently of the production function.

    Comparing compute_audit_hash() against itself proves nothing: a bug that
    writes `"intervention": null` into every payload would keep both sides
    equal and the test green while silently invalidating every chain ever
    written. The oracle has to come from outside.
    """
    payload = json.dumps(
        {
            "ts": round(float(ts), 6),
            "op": operation,
            "query": query,
            "cells": cells,
            "results": results,
            "qemb_sha256": qemb_hash,
        },
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256((payload + prev).encode("utf-8")).hexdigest()


def test_absent_intervention_reproduces_the_pre_v4_payload():
    """The v4 key is omitted, never serialized as null — otherwise every
    historical chain would stop verifying the moment the column landed."""
    ts, prev = 1234567890.5, "0" * 64
    expected_v3 = _v3_scheme_hash(ts, "recall", "q", [1, 2], [], prev, "ab")

    assert compute_audit_hash(ts, "recall", "q", [1, 2], [], prev,
                              qemb_hash="ab") == expected_v3
    assert compute_audit_hash(ts, "recall", "q", [1, 2], [], prev, qemb_hash="ab",
                              intervention=None) == expected_v3

    with_spec = compute_audit_hash(ts, "recall", "q", [1, 2], [], prev,
                                   qemb_hash="ab", intervention={"mode": "suppress"})
    assert with_spec != expected_v3


def test_existing_chain_still_verifies_after_v4_migration(tmp_path):
    eng = build_field(tmp_path, n=6)
    for i in range(3):
        eng.recall(cluster_emb("alpha", i))
    eng.intervene(cluster_emb("alpha", 1), InterventionSpec.suppress([]))
    for i in range(2):
        eng.recall(cluster_emb("beta", i))

    report = verify_audit_chain(eng.get_audit_trail(limit=10000))
    assert report["chain_intact"] and report["hash_integrity"], report["issues"]


# ============================================================
# Gate 8 — exclusion provenance
# ============================================================

def test_diagnostics_name_the_mechanism_of_absence(tmp_path):
    eng = build_field(tmp_path)
    q = cluster_emb("alpha", 7)
    victim = eng.recall(q, top_k=1)[0][0].memory

    res = eng.intervene(q, InterventionSpec.suppress([victim.memory_id]))
    assert res.perturbed_exclusions[victim.memory_id] == \
        ExclusionReason.DIRECT_SUPPRESSION

    # A memory that never existed is not "unreachable" — it is not in the field.
    assert eng.absence_reason(res.perturbed_exclusions, "mem_nonexistent") == \
        ExclusionReason.NOT_IN_FIELD


def test_inhibition_and_unreachability_are_different_reasons(tmp_path):
    """The rescue-rule oracle depends on telling these two apart; an
    `excluded == True` boolean cannot."""
    eng = AdaptiveMemoryEngine(db_path=tmp_path / "inhib.db")
    a = eng.store("the system is deterministic end to end",
                  cluster_emb("claim", 0),
                  metadata={"topic": "engine", "claim": "deterministic"})
    b = eng.store("the system relies on machine learning inference",
                  cluster_emb("claim", 1),
                  metadata={"topic": "engine", "claim": "ml"})
    far = eng.store("entirely unrelated subject matter here",
                    make_emb("far-away-topic"))

    res = eng.intervene(cluster_emb("claim", 0), InterventionSpec.suppress([]),
                        top_k=10, hops=1)
    reasons = {
        b.memory_id: eng.absence_reason(res.baseline_exclusions, b.memory_id),
        far.memory_id: eng.absence_reason(res.baseline_exclusions, far.memory_id),
    }
    assert reasons[b.memory_id] == ExclusionReason.INHIBITED
    assert reasons[far.memory_id] in (
        ExclusionReason.UNREACHABLE, ExclusionReason.BELOW_TOP_K,
    )
    assert a.memory_id not in res.baseline_exclusions


# ============================================================
# Suppression semantics — stage "field" means seed, relay and readout
# ============================================================

def test_suppressed_cell_cannot_seed_the_search(tmp_path):
    eng = build_field(tmp_path)
    q = cluster_emb("alpha", 0)
    seed = eng.recall(q, top_k=1)[0][0].memory
    res = eng.intervene(q, InterventionSpec.suppress([seed.memory_id]), top_k=10)
    assert seed.memory_id not in [r.memory.memory_id for r in res.perturbed_scored]
    assert res.perturbed, "silencing the seed must not empty the whole field"


def test_suppressing_every_cell_returns_an_honest_empty(tmp_path):
    eng = build_field(tmp_path, n=4)
    everything = [m.memory_id for m in eng.list_memories(limit=1000)]
    res = eng.intervene(make_emb("alpha"), InterventionSpec.suppress(everything))
    assert res.perturbed == []
    assert res.baseline, "the baseline arm must be unaffected"


def test_suppression_blocks_the_synaptic_bypass(tmp_path):
    """Synaptic pull appends memories without going through the BFS — the one
    path a silenced cell could re-enter the result set through."""
    eng = build_field(tmp_path)
    q = cluster_emb("alpha", 1)
    ids = [r.memory.memory_id for r in eng.recall(q, top_k=5)[0]]
    for _ in range(8):   # drive several LTP rounds past the 0.5 pull threshold
        eng.recall(q, top_k=5, current_turn_memories=ids)

    res = eng.intervene(q, InterventionSpec.suppress(ids[:2]),
                        top_k=10, current_turn_memories=ids)
    survivors = {r.memory.memory_id for r in res.perturbed_scored}
    assert not (set(ids[:2]) & survivors)


# ============================================================
# Ablation runners and the invariant fuzzer
# ============================================================

def test_single_and_pairwise_ablation_find_a_dependence(tmp_path):
    eng = build_field(tmp_path)
    q = cluster_emb("alpha", 3)
    top = [r.memory.memory_id for r in eng.recall(q, top_k=6)[0]]
    watched = top[0]

    singles = iv.single_cell_ablation(eng, q, top[1:4], top_k=10)
    pairs = iv.pairwise_ablation(eng, q, top[1:4], top_k=10)
    verdict = iv.classify_dependence(watched, singles, pairs)

    assert verdict["verdict"] in (
        iv.REDUNDANT_PATHS, iv.SINGLE_NECESSARY,
        iv.MULTIPLE_NECESSARY, iv.NO_DEPENDENCE,
    )
    # Silencing the watched memory itself must always drop it — this is the
    # runner's own positive control.
    assert iv.single_cell_ablation(eng, q, [watched], top_k=10)[0].dropped(watched)


def test_random_subset_ablation_is_reproducible(tmp_path):
    eng = build_field(tmp_path)
    q = cluster_emb("alpha", 2)
    pool = [m.memory_id for m in eng.list_memories(limit=8)]
    a = iv.random_subset_ablation(eng, q, pool, size=2, trials=5, seed=7, top_k=10)
    b = iv.random_subset_ablation(eng, q, pool, size=2, trials=5, seed=7, top_k=10)
    assert [x.targets for x in a] == [x.targets for x in b]
    assert [x.result.delta for x in a] == [x.result.delta for x in b]


def test_rescue_rule_survives_targeted_fuzzing(tmp_path):
    """Silencing only unvalidated cells must never let a validated truth be
    dropped *by inhibition*. Losing it to topology is legal and is recorded
    separately — the distinction is the whole point of the oracle."""
    eng = AdaptiveMemoryEngine(db_path=tmp_path / "fuzz.db")
    truth = eng.store("the engine is deterministic and auditable end to end",
                      cluster_emb("topic", 0),
                      metadata={"topic": "engine", "claim": "deterministic"})
    for i in range(1, 10):
        eng.store(f"the engine actually uses opaque inference variant {i}",
                  cluster_emb("topic", i),
                  metadata={"topic": "engine", "claim": f"ml-{i}"})
    eng.reinforce(truth.memory_id)

    report = iv.fuzz_rescue_rule(eng, cluster_emb("topic", 0),
                                 subset_size=2, trials=15, seed=3,
                                 top_k=10, hops=2)
    assert report["trials"] > 0, report.get("note")
    assert report["violations"] == [], report["violations"]


def test_fuzzer_oracle_does_not_confuse_topology_with_inhibition(tmp_path):
    """Negative control for the oracle itself: a REINFORCED memory removed by
    silencing its own path is an UNREACHABLE exit, never a rescue violation."""
    eng = AdaptiveMemoryEngine(db_path=tmp_path / "topo.db")
    hub = eng.store("hub document bridging the neighbourhood", cluster_emb("hub", 0))
    leaf = eng.store("leaf document reachable only through the hub",
                     make_emb("isolated-leaf-vector"))
    eng.reinforce(leaf.memory_id)
    eng.create_cell_link(hub.cell_id, leaf.cell_id, LinkType.RESONANT)
    for i in range(1, 6):
        eng.store(f"filler document {i}", cluster_emb("hub", i))

    res = eng.intervene(cluster_emb("hub", 0), InterventionSpec.suppress([hub.memory_id]),
                        top_k=10, hops=1)
    if leaf.memory_id not in {r.memory.memory_id for r in res.perturbed_scored}:
        reason = eng.absence_reason(res.perturbed_exclusions, leaf.memory_id)
        assert reason != ExclusionReason.INHIBITED, (
            "a topology-driven exit must not be reported as an inhibition "
            "violation — that is the false positive the oracle exists to avoid"
        )


# ============================================================
# Gate 10 — future observable behaviour is unchanged
# ============================================================

def test_probes_do_not_change_subsequent_observable_behaviour(tmp_path):
    """The strongest gate: not 'the rows look equal', but 'the field behaves
    identically afterwards'. Two fields built identically; one is probed
    heavily between recalls, the other is not. Every later recall must agree,
    and the persistent state must match — the audit log excepted, since
    recording that the observation happened is the probe's entire output."""
    def run(with_probes: bool):
        eng = build_field(tmp_path / ("p" if with_probes else "c"), n=10)
        queries = [cluster_emb("alpha", i) for i in range(4)]
        observed = []
        turn = None
        for q in queries:
            res, audit = eng.recall(q, top_k=5, current_turn_memories=turn)
            turn = [r.memory.memory_id for r in res]
            # Compare by content, not memory_id: the id carries a store-time
            # timestamp that differs between two independently built fields.
            observed.append([(r.memory.content, round(r.final_score, 4)) for r in res])
            if with_probes:
                pool = [m.memory_id for m in eng.list_memories(limit=6)]
                for mem_id in pool[:3]:
                    eng.intervene(q, InterventionSpec.suppress([mem_id]),
                                  top_k=5, current_turn_memories=turn)
                iv.pairwise_ablation(eng, q, pool[:3], top_k=5)
        return observed, portable_state(eng), eng

    (tmp_path / "p").mkdir()
    (tmp_path / "c").mkdir()
    probed_obs, probed_state, probed_eng = run(True)
    control_obs, control_state, control_eng = run(False)

    assert probed_obs == control_obs, "probing changed a later recall"
    assert probed_state == control_state, "probing changed persistent state"

    # The real recalls must be the same entries in the same order; only the
    # chain linkage differs, because the probes are deliberately recorded.
    def real_entries(eng):
        rows = [e for e in reversed(eng.get_audit_trail(limit=10000))
                if e["operation"] == "recall"]
        out = []
        for e in rows:
            mems = json.loads(e["memories_retrieved"])
            out.append((
                e["query_text"], e["cells_activated"],
                [(m["content_hash"], m["final_score"], m["source"]) for m in mems],
                e["total_candidates"], e["returned_to_agent"],
            ))
        return out

    assert real_entries(probed_eng) == real_entries(control_eng)
    assert verify_audit_chain(probed_eng.get_audit_trail(limit=10000))["chain_intact"]
