#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — metamorphic properties and purity of the intervention primitive.

Separate from tests/test_intervention.py on purpose: that file pins the v1
contract, this one is the adversarial pass over it. These are properties the
primitive must hold no matter what the classification API ends up meaning.

  sham:            I = ∅           ⇒  R_I(q) = R_0(q)     in every observable
  commutativity:   suppress(A ∪ B) =  suppress(B ∪ A)
  idempotence:     suppress(A ∪ A) =  suppress(A)         (refused, not collapsed)
  purity:          a probe leaves no trace in ANY engine attribute

The sham is the cruel one, and it is not tautological: both branches run the
same code over the same shared engine structures, so if the core mutated
anything it touches — a set, a dict, a cached list, an ndarray — the second
branch would see a different field than the first. Comparing the FULL core
output (scores, diagnostics, exclusion provenance, counters) makes the sham
double as an aliasing detector.

Run: pytest tests/test_intervention_properties.py -q
"""

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raven.memory_engine import (
    AdaptiveMemoryEngine,
    InterventionError,
    InterventionSpec,
    LinkType,
)


def make_emb(text: str, dim: int = 384) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(dim).astype(np.float32)
    e /= np.linalg.norm(e) + 1e-10
    return e


def near(base: str, k: int, jitter: float = 0.25) -> np.ndarray:
    e = make_emb(base) + jitter * make_emb(f"{base}::{k}")
    return (e / np.linalg.norm(e)).astype(np.float32)


@pytest.fixture
def field(tmp_path):
    """A field with clusters, a contradiction pair and an explicit link, so the
    probe exercises inhibition, rescue and resonance — not just plain scoring."""
    eng = AdaptiveMemoryEngine(db_path=tmp_path / "props.db")
    for c in ("alpha", "beta"):
        for i in range(10):
            eng.store(f"{c} cluster document number {i} with body text", near(c, i))
    t = eng.store("the engine is deterministic end to end", near("claim", 0),
                  metadata={"topic": "engine", "claim": "deterministic"})
    eng.store("the engine relies on opaque inference", near("claim", 1),
              metadata={"topic": "engine", "claim": "ml"})
    far = eng.store("a distant document reached only by an explicit link",
                    make_emb("distant-region"))
    eng.create_cell_link(t.cell_id, far.cell_id, LinkType.RESONANT)
    eng.reinforce(t.memory_id)
    return eng


# ----------------------------------------------------------
# Canonicalisation
# ----------------------------------------------------------

def canon_result(r) -> tuple:
    """Every computed field of a RecallResult, unrounded. Same inputs through
    the same code path must give bit-identical floats; rounding here would hide
    exactly the drift these properties exist to catch."""
    return (
        r.memory.memory_id, r.cell_id, r.memory.state.name, r.source,
        repr(r.base_score), repr(r.state_boost), repr(r.hop_decay),
        repr(r.synaptic_boost), repr(r.recency_bonus), repr(r.final_score),
        r.hop_distance, repr(r.resonance_score), repr(r.coherence_score),
    )


def canon_outcome(o) -> dict:
    """The complete pure output of _recall_core — nothing omitted."""
    return {
        "top": [canon_result(r) for r in o.top],
        "scored": [canon_result(r) for r in o.scored],
        "activated_cells": sorted(o.activated_cells),
        "inhibited_cells": sorted(o.inhibited_cells),
        "query_cell": o.query_cell,
        "total_candidates": o.total_candidates,
        "f_state": o.f_state,
        "f_estilo": o.f_estilo,
        "f_inhib": o.f_inhib,
        "synaptic_count": o.synaptic_count,
        "exclusions": sorted(o.exclusions.items()),
        "pending_alerts": sorted(
            (a.memory_id, a.action_taken, repr(a.mismatch_score))
            for a in o.pending_alerts
        ),
        "enforce_forget": sorted(o.enforce_forget),
        "stylo_notices": sorted((m, repr(d)) for m, d in o.stylo_notices),
    }


def canon_probe(res) -> dict:
    """Everything an InterventionResult exposes, minus the fields that ARE the
    intervention (spec, resolved targets) — those legitimately differ between
    two specs naming the same set in a different order."""
    return {
        "baseline": [canon_result(r) for r in res.baseline],
        "perturbed": [canon_result(r) for r in res.perturbed],
        "baseline_scored": [canon_result(r) for r in res.baseline_scored],
        "perturbed_scored": [canon_result(r) for r in res.perturbed_scored],
        "delta": res.delta,
        "baseline_exclusions": sorted(res.baseline_exclusions.items()),
        "perturbed_exclusions": sorted(res.perturbed_exclusions.items()),
    }


def pinned_now(field) -> float:
    """A fixed `now` at or after every timestamp in the field.

    It must not be in the PAST relative to last_activation: `recency_bonus` is
    exp(-ln2 * age / 24h), so a negative age makes it grow without bound rather
    than decay. See the note in docs/ — that is a property of the engine, not of
    these tests, and it is why this helper exists instead of a constant.
    """
    mems = field.list_memories(limit=100000)
    latest = max([m.last_activation for m in mems] + [m.created_at for m in mems])
    return latest + 3600.0


# intervene() samples the clock once per call, and recency_bonus decays with
# `now`. Two probes therefore cannot have bit-identical absolute scores — the
# drift is real, bounded, and NOT impurity. Everything the probe actually
# measures (delta, ordering, exclusions) is clock-free because both branches of
# a single probe share one `now`, so the recency term cancels in the
# subtraction. The properties below assert exactly that, rather than rounding
# the difference away.
CLOCK_FREE_FIELDS = (
    "memory_id", "cell_id", "state", "source", "base_score", "state_boost",
    "hop_decay", "synaptic_boost", "hop_distance", "resonance_score",
    "coherence_score",
)


def _result_fields(r) -> dict:
    return {
        "memory_id": r.memory.memory_id, "cell_id": r.cell_id,
        "state": r.memory.state.name, "source": r.source,
        "base_score": r.base_score, "state_boost": r.state_boost,
        "hop_decay": r.hop_decay, "synaptic_boost": r.synaptic_boost,
        "hop_distance": r.hop_distance, "resonance_score": r.resonance_score,
        "coherence_score": r.coherence_score,
        "recency_bonus": r.recency_bonus, "final_score": r.final_score,
    }


def assert_results_equivalent(xs, ys, drift: float, label: str):
    """Exact on every clock-free field; `recency_bonus` and the `final_score`
    that carries it may differ by at most `drift`."""
    assert len(xs) == len(ys), f"{label}: different result counts"
    for x, y in zip(xs, ys):
        fx, fy = _result_fields(x), _result_fields(y)
        for k in CLOCK_FREE_FIELDS:
            assert fx[k] == fy[k], f"{label}: {k} differs ({fx[k]!r} vs {fy[k]!r})"
        for k in ("recency_bonus", "final_score"):
            assert abs(fx[k] - fy[k]) <= drift, (
                f"{label}: {k} drifted {abs(fx[k]-fy[k]):.3e} > {drift:.3e} — "
                "larger than the clock can explain"
            )


def assert_probes_equivalent(p, q, drift: float = 1e-5, label: str = ""):
    assert_results_equivalent(p.baseline, q.baseline, drift, f"{label} baseline")
    assert_results_equivalent(p.perturbed, q.perturbed, drift, f"{label} perturbed")
    assert_results_equivalent(p.baseline_scored, q.baseline_scored, drift,
                              f"{label} baseline_scored")
    assert_results_equivalent(p.perturbed_scored, q.perturbed_scored, drift,
                              f"{label} perturbed_scored")
    # The delta is clock-free and is compared EXACTLY — it is what the probe
    # claims to measure.
    assert p.delta == q.delta, f"{label}: delta differs"
    assert sorted(p.baseline_exclusions.items()) == sorted(q.baseline_exclusions.items())
    assert sorted(p.perturbed_exclusions.items()) == sorted(q.perturbed_exclusions.items())


def core_at(field, q, cells, **kw) -> dict:
    """Run the pure core at a PINNED `now`, so metamorphic properties can be
    checked bit-for-bit with no clock term in the way."""
    field._ensure_kdtree()
    base = dict(now=pinned_now(field), top_k=8, hops=2, layer_filter=None,
                current_turn_memories=None)
    base.update(kw)
    return canon_outcome(field._recall_core(q, suppressed=frozenset(cells), **base))


def cells_of(field, memory_ids):
    return [c for _, c in field._resolve_targets(InterventionSpec.suppress(memory_ids))]


def _canon_value(v):
    """Structural canonicalisation of an arbitrary engine attribute."""
    if isinstance(v, np.ndarray):
        return ("ndarray", v.dtype.str, v.shape, hashlib.sha256(v.tobytes()).hexdigest())
    if isinstance(v, dict):
        return ("dict", sorted((repr(k), _canon_value(x)) for k, x in v.items()))
    if isinstance(v, (set, frozenset)):
        return ("set", sorted(repr(x) for x in v))
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, [_canon_value(x) for x in v])
    if isinstance(v, (str, int, float, bool, type(None))):
        return (type(v).__name__, repr(v))
    return ("opaque", type(v).__name__)


OPAQUE_ATTRS = {"_db", "_lock", "kdtree", "stylometric", "_spectral"}


def structural_digest(engine) -> dict:
    """Every in-memory attribute of the engine, walked generically.

    Enumerating vars() rather than a hand-written list means an attribute added
    later is covered automatically — a hand-written list silently stops testing
    whatever it forgets.
    """
    out = {}
    for name, value in sorted(vars(engine).items()):
        if name in OPAQUE_ATTRS:
            out[name] = ("opaque", type(value).__name__)
            continue
        out[name] = _canon_value(value)
    # The author profiles hold rolling fingerprints; digest their content too.
    out["_author_profile_counts"] = sorted(
        (repr(k), p.count) for k, p in engine._author_profiles.items()
    )
    return out


# ============================================================
# Property 1 — sham stimulation
# ============================================================

def test_sham_intervention_is_identical_in_every_observable(field):
    """I = ∅ must reproduce the baseline exactly: not the same ids, not the
    same top-k — the same everything, floats included."""
    q = near("alpha", 3)
    res = field.intervene(q, InterventionSpec.suppress([]), top_k=8, hops=2)

    assert [canon_result(r) for r in res.baseline] == \
           [canon_result(r) for r in res.perturbed]
    assert [canon_result(r) for r in res.baseline_scored] == \
           [canon_result(r) for r in res.perturbed_scored]
    assert sorted(res.baseline_exclusions.items()) == \
           sorted(res.perturbed_exclusions.items())
    assert res.delta["rank_displacement"] == 0
    assert res.delta["score_delta"] == {}
    assert res.delta["disappeared_from_field"] == []
    assert res.delta["appeared_in_field"] == []


def test_repeated_core_calls_are_bit_identical(field):
    """The sham's real teeth: two successive runs of the pure core over the
    same shared engine structures. Any mutation the core performed on a set,
    dict, cached list or ndarray it merely READ would surface here as drift in
    the second run — including in the diagnostics, which no other test reads
    in full."""
    field._ensure_kdtree()
    q = near("claim", 0)
    kw = dict(now=pinned_now(field), top_k=8, hops=2, layer_filter=None,
              current_turn_memories=None)

    first = canon_outcome(field._recall_core(q, suppressed=frozenset(), **kw))
    for _ in range(5):
        assert canon_outcome(field._recall_core(q, suppressed=frozenset(), **kw)) == first


def test_sham_interleaved_with_real_suppression_stays_clean(field):
    """A sham after a real probe must still equal a sham before it: a
    suppressing run must not leave residue that a later null run can see."""
    q = near("alpha", 1)
    targets = [r.memory.memory_id for r in field.recall(q, top_k=3)[0]]

    before = field.intervene(q, InterventionSpec.suppress([]), top_k=8)
    for t in targets:
        field.intervene(q, InterventionSpec.suppress([t]), top_k=8)
    after = field.intervene(q, InterventionSpec.suppress([]), top_k=8)

    assert_probes_equivalent(before, after, label="sham before/after")


# ============================================================
# Property 2 — commutativity
# ============================================================

def test_order_independence_holds_by_construction_not_by_luck(field):
    """suppress(A ∪ B) == suppress(B ∪ A).

    Honest framing, established by a negative control: the behavioural half of
    this property CANNOT fail. `_resolve_targets` sorts by cell_id and the core
    takes a `FrozenSet`, so ordering is normalised away twice before it can
    reach any logic — deliberately breaking the suppression to be
    order-dependent still leaves the behavioural comparison green.

    So the assertions that carry weight here are the two normalisation points
    themselves, plus the sealed-treatment test below, which does go red when the
    sort is removed. The end-to-end comparison is kept as a consistency check,
    not as evidence.
    """
    import inspect
    from typing import get_type_hints
    sig = inspect.signature(field._recall_core)
    assert sig.parameters["suppressed"].default == frozenset(), \
        "the core's suppressed parameter must default to an unordered collection"
    assert "FrozenSet" in str(sig.parameters["suppressed"].annotation) \
        or "frozenset" in str(sig.parameters["suppressed"].annotation).lower(), \
        "an ordered type here would let call order reach the propagation logic"

    q = near("alpha", 2)
    a, b, c = [r.memory.memory_id for r in field.recall(q, top_k=3)[0]]

    # Resolution must normalise order — this is what the sealed test guards.
    for perm in ([a, b], [b, a]):
        resolved = field._resolve_targets(InterventionSpec.suppress(perm))
        assert resolved == sorted(resolved, key=lambda t: t[1])

    # Exact, at a pinned clock: the property with nothing in the way.
    assert core_at(field, q, cells_of(field, [a, b])) == \
           core_at(field, q, cells_of(field, [b, a]))
    for perm in ([a, b, c], [c, b, a], [b, a, c], [c, a, b]):
        assert core_at(field, q, cells_of(field, perm)) == \
               core_at(field, q, cells_of(field, [a, b, c]))

    # And end to end through intervene(), where only the clock term may move.
    assert_probes_equivalent(
        field.intervene(q, InterventionSpec.suppress([a, b]), top_k=8),
        field.intervene(q, InterventionSpec.suppress([b, a]), top_k=8),
        label="commutativity",
    )


def test_sealed_treatment_is_order_independent(field):
    """Set semantics must hold in the EVIDENCE too: the same population named
    in two orders has to seal to the same targets, or the audit records the
    call's spelling instead of the treatment."""
    q = near("alpha", 2)
    a, b = [r.memory.memory_id for r in field.recall(q, top_k=2)[0]]

    ab = field.intervene(q, InterventionSpec.suppress([a, b]), top_k=8)
    ba = field.intervene(q, InterventionSpec.suppress([b, a]), top_k=8)
    assert ab.audit.intervention["targets"] == ba.audit.intervention["targets"]
    assert ab.resolved_targets == ba.resolved_targets


# ============================================================
# Property 3 — idempotence
# ============================================================

def test_duplicate_targets_are_refused_rather_than_collapsed(field):
    """suppress(A ∪ A) == suppress(A) holds here in the stronger form: the
    ambiguous spelling is refused outright. Silently deduplicating would make
    the request and the treatment two different things, and only one of them
    gets sealed."""
    q = near("alpha", 2)
    a = field.recall(q, top_k=1)[0][0].memory.memory_id
    with pytest.raises(InterventionError, match="duplicate"):
        InterventionSpec.suppress([a, a])


def test_probing_the_same_set_twice_is_stable(field):
    """Idempotence in the observable sense: the same treatment applied again
    over an unchanged field yields the identical result."""
    q = near("beta", 4)
    a, b = [r.memory.memory_id for r in field.recall(q, top_k=2)[0]]
    spec = InterventionSpec.suppress([a, b])
    cells = cells_of(field, [a, b])

    # Exact at a pinned clock.
    fixed = [core_at(field, q, cells) for _ in range(3)]
    assert fixed[0] == fixed[1] == fixed[2]

    # Through intervene(): the measured delta must be EXACTLY stable across
    # repeats even though absolute scores carry the clock.
    runs = [field.intervene(q, spec, top_k=8) for _ in range(3)]
    assert runs[0].delta == runs[1].delta == runs[2].delta
    assert_probes_equivalent(runs[0], runs[1], label="repeat 0/1")
    assert_probes_equivalent(runs[1], runs[2], label="repeat 1/2")


# ============================================================
# Property 4 — purity against shared mutable state (aliasing)
# ============================================================

def test_probes_leave_no_trace_in_any_engine_attribute(field):
    """Gate 10 proved behaviour is unchanged. This proves the mechanism: every
    in-memory attribute of the engine, walked generically, is structurally
    identical after a battery of probes.

    The core aliases engine structures rather than copying them
    (`all_cell_links = self._cell_links_index`, `self.cell_neighbors.get(...)`),
    so "no SQL was executed" would not have been enough.
    """
    field._ensure_kdtree()      # exclude the lazy rebuild from the comparison
    q = near("claim", 0)
    pool = [m.memory_id for m in field.list_memories(limit=6)]

    before = structural_digest(field)
    field.intervene(q, InterventionSpec.suppress([]), top_k=8)
    for t in pool:
        field.intervene(q, InterventionSpec.suppress([t]), top_k=8, hops=2)
    for i in range(len(pool) - 1):
        field.intervene(q, InterventionSpec.suppress(pool[i:i + 2]), top_k=8, hops=2)
    field.intervene(near("alpha", 0), InterventionSpec.suppress(pool[:3]),
                    top_k=8, hops=3, current_turn_memories=pool)
    after = structural_digest(field)

    differing = [k for k in before if before[k] != after[k]]
    assert not differing, f"probe mutated engine attributes: {differing}"


def test_digest_actually_sees_engine_mutation(field):
    """Positive control for the instrument itself. If structural_digest() could
    not detect a real change, the purity test above would be decorative."""
    field._ensure_kdtree()
    before = structural_digest(field)
    victim = field.list_memories(limit=1)[0]
    field.forget(victim.memory_id)          # a genuine in-memory mutation
    after = structural_digest(field)
    assert before != after
    assert "_active_cells" in [k for k in before if before[k] != after[k]]
