#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — StateWitness: what it sees, and what it still cannot.

Order of business, deliberately: prove the instrument terminates on a cyclic
engine-owned graph WITHOUT losing the cycle; prove it fails closed on state it
cannot represent; then invert exactly the three blind spots the old digest's
autopsy (tests/test_digest_coverage.py) froze as green — and only those.

Still green, still honest:
  * transient mutation S0 → S1 → S0 — unreachable by ANY before/after
    comparison; needs a MutationJournal, not a better witness.

Not attempted here: semantic witnesses for _db and kdtree. They are excluded
with written justifications, and recursing into scipy or sqlite internals is
the wrong shape of answer — the claim is about RAVEN, not about a library's
representation.

No existing gate is replaced. The old digest keeps its autopsy.

Run: pytest tests/test_state_witness.py -q
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from raven.memory_engine import AdaptiveMemoryEngine, InterventionSpec, LinkType
from state_witness import StateWitness, UnrepresentableState, compare, take_witness
from test_intervention_properties import make_emb, near

# Excluded with reasons. _db and kdtree are NOT "too hard": they need SEMANTIC
# witnesses (audit head, row fingerprints; indexed cell ids, vector fingerprint,
# dirty flag) rather than structural recursion into a library's internals.
ENGINE_EXCLUSIONS = {
    "_db": "storage handle — needs a semantic DBWitness (audit head, table "
           "fingerprints), not structural recursion into sqlite internals",
    "_lock": "synchronisation primitive; holds no recall state and its "
             "internals are not engine-owned",
    "kdtree": "library index — needs a semantic KDTreeWitness (indexed cell "
              "ids, source-vector fingerprint, dirty state), not a walk "
              "through scipy's representation",
    "_spectral": "optional SVD field; when present it carries its own "
                 "determinism self-test and is absent in these fixtures",
}


@pytest.fixture
def engine(tmp_path):
    eng = AdaptiveMemoryEngine(db_path=tmp_path / "witness.db")
    for c in ("alpha", "beta"):
        for i in range(8):
            eng.store(f"{c} documento numero {i} con texto de cuerpo", near(c, i))
    t = eng.store("the engine is deterministic end to end", near("claim", 0),
                  metadata={"topic": "engine", "claim": "det"})
    eng.store("the engine relies on opaque inference", near("claim", 1),
              metadata={"topic": "engine", "claim": "ml"})
    far = eng.store("a distant document", make_emb("far-region"))
    eng.create_cell_link(t.cell_id, far.cell_id, LinkType.RESONANT)
    eng.reinforce(t.memory_id)
    eng.recall(near("alpha", 0))
    eng._ensure_kdtree()
    return eng


def witness(eng) -> StateWitness:
    return take_witness(eng, exclusions=ENGINE_EXCLUSIONS)


# ============================================================
# 1 — cycles: terminate, and keep the back-edge
# ============================================================

def test_cyclic_engine_owned_graph_terminates_and_keeps_the_back_edge(engine):
    """A naive __dict__ descent raised RecursionError on the real object graph.
    Cycle detection is a correctness requirement — but detecting a cycle must
    not mean dropping it: the back-edge IS state."""
    class Node:
        def __init__(self, name):
            self.name = name
            self.peer = None

    a, b = Node("a"), Node("b")
    a.peer, b.peer = b, a                 # a → b → a
    engine.cyclic = a

    w = witness(engine)                   # must terminate

    ids = {n.witness_id for n in w.nodes.values()}
    assert id(a) in ids and id(b) in ids
    edges = {(p, lab, c) for p, lab, c in w.edges}
    assert (id(a), ("attr", "peer"), id(b)) in edges
    assert (id(b), ("attr", "peer"), id(a)) in edges, "the back-edge was dropped"

    # And the two independent claims are computable over a cyclic graph.
    assert w.value_signature()
    assert w.alias_signature()


def test_self_referential_container_terminates(engine):
    d = {}
    d["self"] = d
    engine.selfref = d
    w = witness(engine)
    assert (id(d), ("key", ("str", "'self'")), id(d)) in set(w.edges)
    sig = w.value_signature()
    assert any("<cycle>" in str(v) for v in sig.values())


# ============================================================
# 2 — the acceptance criterion: never summarise, always name
# ============================================================

def test_unrepresentable_engine_state_fails_with_path_and_type(engine):
    """The criterion agreed before a line was written: an engine-owned mutable
    the witness has no policy for must FAIL, naming path and type. This is the
    exact class of lie that ("opaque", type_name) produced."""
    class ExoticState:
        __slots__ = ("payload",)
        def __init__(self):
            self.payload = 1

    engine.exotic = ExoticState()
    with pytest.raises(UnrepresentableState) as exc:
        witness(engine)
    assert exc.value.path == "engine.exotic"
    assert exc.value.runtime_type == "ExoticState"
    assert "do not let it be summarised away" in str(exc.value)


def test_exclusions_require_written_justification(engine):
    with pytest.raises(ValueError, match="written justification"):
        take_witness(engine, exclusions={"_db": "meh"})


def test_declared_exclusions_are_recorded_in_the_witness(engine):
    w = witness(engine)
    paths = {p for p, _t, _why in w.exclusions}
    assert paths == {f"engine.{k}" for k in ENGINE_EXCLUSIONS}
    for _p, _t, why in w.exclusions:
        assert len(why) >= 20


# ============================================================
# 3 — the inversions: exactly three, and only three
# ============================================================

def test_INVERTED_opaque_object_mutated_during_a_probe_is_now_seen(engine):
    """Was KNOWN_BLINDNESS under structural_digest(). Now detected."""
    class EvilState:
        def __init__(self):
            self.counter = 0

    engine.evil = EvilState()
    original = engine._recall_core

    def spy(*args, **kwargs):
        engine.evil.counter += 1
        return original(*args, **kwargs)

    engine._recall_core = spy
    before = witness(engine)
    engine.intervene(near("alpha", 0), InterventionSpec.suppress([]), top_k=5)
    after = witness(engine)

    assert engine.evil.counter > 0
    result = compare(before, after)
    assert not result["persistent_value_state_equal"], "the witness still cannot see it"
    assert any("evil" in p for p in result["value_diff_paths"])


def test_INVERTED_rolling_stylometric_window_is_now_seen(engine):
    """The one that mattered most: live state feeding a forensic decision,
    which the old gate declared invisible."""
    key = list(engine._author_profiles)[0]
    profile = engine._author_profiles[key]
    assert profile._samples

    before = witness(engine)
    profile._samples[0].avg_sentence_length = 999.0
    after = witness(engine)

    result = compare(before, after)
    assert not result["persistent_value_state_equal"]
    assert any("_author_profiles" in p and "avg_sentence_length" in p
               for p in result["value_diff_paths"]), result["value_diff_paths"][:5]


def test_INVERTED_identity_topology_change_is_now_seen(engine):
    """Two attributes sharing one object, then one replaced by an equal-valued
    copy. The value is unchanged and the topology is not — and the witness now
    reports those as two separate answers instead of one green."""
    shared = {"k": [1, 2, 3]}
    engine.alias_a = shared
    engine.alias_b = shared

    before = witness(engine)
    engine.alias_b = copy.deepcopy(shared)
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"], (
        "the copy has equal value — reporting a value change here would be a "
        "false positive, and would hide what actually changed"
    )
    assert not result["persistent_alias_topology_equal"], "identity topology change missed"
    assert not result["persistent_root_identity_equal"]


def test_value_claim_on_the_aliased_path_is_not_vacuous(engine):
    """Control for the test above: its `value_state_equal is True` only means
    something if the same path CAN report a value change. Replace the copy with
    a differing one and the value claim must flip — otherwise that assertion
    was measuring nothing."""
    shared = {"k": [1, 2, 3]}
    engine.alias_a = shared
    engine.alias_b = shared

    before = witness(engine)
    engine.alias_b = {"k": [1, 2, 999]}
    after = witness(engine)

    result = compare(before, after)
    assert not result["persistent_value_state_equal"]
    assert not result["persistent_alias_topology_equal"]


# ============================================================
# 4 — what stays green, and why
# ============================================================

def test_STILL_BLIND_transient_mutation(engine):
    """S0 → S1 → S0 is unreachable by any before/after comparison, however good
    the canonicalisation. An observer during S1 saw a different engine.

    This needs a MutationJournal instrumenting write surfaces, whose claim
    would be bounded to "no mutation through instrumented surfaces" — never
    "no mutation". The witness must not claim to cover this.
    """
    before = witness(engine)
    victim = next(iter(engine._active_cells))
    engine._active_cells.discard(victim)
    engine._active_cells.add(victim)
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"] and result["persistent_alias_topology_equal"], (
        "a before/after witness somehow observed S1 — if this ever fails, the "
        "instrument is doing something other than what it claims"
    )


# ============================================================
# 5 — positive controls
# ============================================================

def test_witness_sees_a_persistent_representable_mutation(engine):
    before = witness(engine)
    engine._active_cells.discard(next(iter(engine._active_cells)))
    after = witness(engine)
    result = compare(before, after)
    assert not result["persistent_value_state_equal"]


def test_witness_is_stable_with_no_mutation(engine):
    """Without this, every assertion above could be an artefact of an unstable
    instrument rather than a fact about the engine."""
    a, b = witness(engine), witness(engine)
    result = compare(a, b)
    assert result["persistent_value_state_equal"]
    assert result["persistent_alias_topology_equal"]
    assert result["persistent_root_identity_equal"]


def test_a_probe_leaves_no_trace_under_the_new_witness(engine):
    """The purity claim restated with the better instrument — reported as two
    separate answers, and NOT called `pure`."""
    pool = [m.memory_id for m in engine.list_memories(limit=5)]
    q = near("claim", 0)

    before = witness(engine)
    engine.intervene(q, InterventionSpec.suppress([]), top_k=8)
    for t in pool:
        engine.intervene(q, InterventionSpec.suppress([t]), top_k=8, hops=2)
    for i in range(len(pool) - 1):
        engine.intervene(q, InterventionSpec.suppress(pool[i:i + 2]), top_k=8)
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"], result["value_diff_paths"]
    assert result["persistent_alias_topology_equal"], (
        result["alias_only_before"], result["alias_only_after"],
    )
