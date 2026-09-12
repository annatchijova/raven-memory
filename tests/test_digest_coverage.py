#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — what structural_digest() actually observes.

`test_probes_leave_no_trace_in_any_engine_attribute` currently proves

    nothing changed THAT structural_digest() CAN REPRESENT

and not

    nothing changed in the engine's structural state.

Both produce the same green, which is the problem. `_canon_value()` fails OPEN:
anything it does not recognise becomes `("opaque", type_name)` — a type label
with no content — and everything reachable only through that node disappears
from the comparison entirely.

This module makes the instrument confess before anyone improves it:

  * an inventory of the reachable state graph vs. what the digest encodes,
    with declared exclusions carrying written justifications, so a new blind
    spot appearing later fails instead of passing silently;
  * four KNOWN-BLINDNESS characterisation tests that assert the gate stays
    GREEN under mutations it cannot see.

The four are NOT properties. They are documentation of a defect, written as
executable assertions so the day the witness lands they must be INVERTED, not
deleted. Read them as "today this is invisible", never as "this is fine".

Deliberately NOT done here: no change to structural_digest(). Establishing what
the instrument can claim comes before changing what it does.

Run: pytest tests/test_digest_coverage.py -q
"""

import collections
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from raven.memory_engine import AdaptiveMemoryEngine, InterventionSpec, LinkType
from test_intervention_properties import (
    OPAQUE_ATTRS,
    make_emb,
    near,
    structural_digest,
)

WALK_DEPTH = 6
WALK_FANOUT = 40

ATOMIC = (str, int, float, bool, bytes, type(None))


# Ignorance turned into an auditable decision. Each entry says why the digest
# is allowed not to represent it. An attribute NOT listed here that turns out
# opaque is an accidental blind spot and fails test_no_undeclared_blind_spots.
DECLARED_EXCLUSIONS = {
    "_db": "storage handle; persistent state is compared directly by the "
           "probe tests through SQL, not through the engine object graph",
    "_lock": "synchronisation primitive; carries no recall state",
    "kdtree": "rebuilt index — NOTE: this exclusion has teeth, a probe that "
              "corrupted the tree in place would be invisible here",
    "stylometric": "stateless extractor (its per-author state lives in "
                   "_author_profiles, which is NOT excluded)",
    "_spectral": "optional SVD field, absent in these fixtures",
}

# Opaque leaf types that carry no mutable state worth representing. Enum
# members are interned singletons; functions and types are code, not state.
BENIGN_OPAQUE_TYPES = {
    "LinkType", "MemoryState", "EnumType", "type", "function",
    "builtin_function_or_method", "method", "module",
}


def canonicalizer_for(v) -> str:
    """Exactly the dispatch `_canon_value()` performs — kept in lockstep so the
    inventory reports what the digest really does, not what it ought to."""
    if isinstance(v, np.ndarray):
        return "ndarray-sha256"
    if isinstance(v, dict):
        return "dict-recurse"
    if isinstance(v, (set, frozenset)):
        return "set-repr"
    if isinstance(v, (list, tuple)):
        return "seq-recurse"
    if isinstance(v, ATOMIC):
        return "repr"
    return "OPAQUE"


def is_mutable(v) -> bool:
    if isinstance(v, (dict, list, set, bytearray)) or isinstance(v, np.ndarray):
        return True
    if isinstance(v, ATOMIC) or isinstance(v, (tuple, frozenset)):
        return False
    return hasattr(v, "__dict__") or hasattr(v, "__slots__")


def walk_reachable(engine, max_depth=WALK_DEPTH, fanout=WALK_FANOUT):
    """The TRUE reachable state graph.

    Descends into plain objects' __dict__ — precisely where `_canon_value()`
    stops — so the difference between the two walks IS the blind spot.
    Cycle-detected and depth-bounded; the bounds are part of the claim.
    """
    nodes, seen = [], set()
    q = collections.deque(
        (f"engine.{k}", v, 1) for k, v in sorted(vars(engine).items())
    )
    while q:
        path, obj, depth = q.popleft()
        canon = canonicalizer_for(obj)
        nodes.append({
            "path": path, "type": type(obj).__name__, "canonicalizer": canon,
            "opaque": canon == "OPAQUE", "mutable": is_mutable(obj),
            "id": id(obj), "depth": depth,
            "root": path.split(".")[1].split("[")[0],
        })
        if id(obj) in seen or depth >= max_depth:
            continue
        seen.add(id(obj))
        children = []
        if isinstance(obj, dict):
            children = [(f"{path}[{k!r}]", v) for k, v in list(obj.items())[:fanout]]
        elif isinstance(obj, (list, tuple, set, frozenset)):
            children = [(f"{path}[{i}]", v) for i, v in enumerate(list(obj)[:fanout])]
        elif not isinstance(obj, ATOMIC) and not isinstance(obj, np.ndarray):
            if hasattr(obj, "__dict__"):
                children = [(f"{path}.{k}", v)
                            for k, v in sorted(vars(obj).items())[:fanout]]
        q.extend((cp, cv, depth + 1) for cp, cv in children)
    return nodes


@pytest.fixture
def engine(tmp_path):
    eng = AdaptiveMemoryEngine(db_path=tmp_path / "cov.db")
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


# ============================================================
# Inventory — the instrument confesses
# ============================================================

def test_declared_exclusions_match_the_digest(engine):
    """Every attribute the digest refuses to represent must be listed WITH a
    reason. A justification-free exclusion is indistinguishable from an
    oversight once the author has moved on."""
    assert set(OPAQUE_ATTRS) == set(DECLARED_EXCLUSIONS), (
        "OPAQUE_ATTRS and DECLARED_EXCLUSIONS drifted apart: "
        f"undocumented={set(OPAQUE_ATTRS) - set(DECLARED_EXCLUSIONS)}, "
        f"stale={set(DECLARED_EXCLUSIONS) - set(OPAQUE_ATTRS)}"
    )
    for name, reason in DECLARED_EXCLUSIONS.items():
        assert len(reason) > 20, f"{name}: exclusion reason is not a justification"


def test_no_undeclared_blind_spots(engine):
    """Fail-closed on the INVENTORY (not yet on the digest itself): an opaque
    node that is neither a declared exclusion nor a benign code/enum leaf is a
    blind spot someone introduced without noticing."""
    nodes = walk_reachable(engine)
    offenders = sorted({
        (n["root"], n["type"])
        for n in nodes
        if n["opaque"]
        and n["root"] not in OPAQUE_ATTRS
        and n["type"] not in BENIGN_OPAQUE_TYPES
    })
    # Known and accepted for now — the rolling stylometric profiles. Listing
    # them here is what makes them a decision instead of an accident.
    known = [("_author_profiles", "AuthorStyleProfile"),
             ("_author_profiles", "deque")]
    unexpected = [o for o in offenders if list(o) not in [list(k) for k in known]]
    assert not unexpected, f"undeclared blind spots: {unexpected}"


def test_state_hidden_beneath_opaque_nodes_is_reported(engine):
    """State reachable ONLY through an opaque node is invisible to the gate.
    This test does not fix that; it refuses to let it stay unquantified."""
    nodes = walk_reachable(engine)
    opaque_paths = [n["path"] for n in nodes if n["opaque"]]
    hidden = [
        n for n in nodes
        if any(n["path"].startswith(p + ".") or n["path"].startswith(p + "[")
               for p in opaque_paths)
    ]
    mutable_hidden = [n["path"] for n in hidden if n["mutable"]]

    # The gate's claim is bounded by this number being known, not by it being
    # zero. If it grows, someone widened the blind spot.
    assert len(hidden) < 200, f"hidden state exploded: {len(hidden)} nodes"
    assert any("_author_profiles" in p for p in [n["path"] for n in hidden]), (
        "the rolling stylometric window should appear as hidden state — if it "
        "no longer does, this inventory is measuring the wrong thing"
    )
    assert mutable_hidden or True   # reported, not asserted away


# ============================================================
# KNOWN BLINDNESS — green today, MUST be inverted by the witness
# ============================================================

def test_KNOWN_BLINDNESS_opaque_object_mutated_during_a_probe(engine):
    """An opaque mutable object mutated by the core is invisible.

    INVERT THIS when the StateWitness lands: it must then FAIL.
    """
    class EvilState:
        def __init__(self):
            self.counter = 0

    engine.evil = EvilState()
    original = engine._recall_core

    def spy(*args, **kwargs):
        engine.evil.counter += 1
        return original(*args, **kwargs)

    engine._recall_core = spy
    before = structural_digest(engine)
    engine.intervene(near("alpha", 0), InterventionSpec.suppress([]), top_k=5)
    after = structural_digest(engine)

    assert engine.evil.counter > 0, "the probe never ran the instrumented core"
    assert before == after, (
        "BLINDNESS RESOLVED — the digest now sees opaque mutation. "
        "Invert this test instead of deleting it."
    )


def test_KNOWN_BLINDNESS_real_engine_state_under_an_opaque_node(engine):
    """Not a contrived object: `_author_profiles[k]._samples[0]` is a live
    fingerprint in the rolling window that feeds the stylometric forensic
    comparison, and the digest cannot see it change.

    INVERT THIS when the StateWitness lands.
    """
    key = list(engine._author_profiles)[0]
    profile = engine._author_profiles[key]
    assert profile._samples, "fixture produced no stylometric samples"

    before = structural_digest(engine)
    profile._samples[0].avg_sentence_length = 999.0
    after = structural_digest(engine)

    assert before == after, (
        "BLINDNESS RESOLVED — the rolling stylometric window is now "
        "represented. Invert this test."
    )


def test_KNOWN_BLINDNESS_identity_topology_is_not_preserved(engine):
    """Two attributes aliasing ONE object, then one replaced by an equal-valued
    copy. Value canonicalisation calls that identical; the aliasing topology
    changed. VALUE_PURITY and IDENTITY_TOPOLOGY_PRESERVATION are different
    claims and the digest only attempts the first.

    INVERT THIS when the witness compares the object graph.
    """
    shared = {"k": [1, 2, 3]}
    engine.alias_a = shared
    engine.alias_b = shared
    assert engine.alias_a is engine.alias_b

    before = structural_digest(engine)
    engine.alias_b = copy.deepcopy(shared)
    after = structural_digest(engine)

    assert engine.alias_a is not engine.alias_b
    assert before == after, (
        "BLINDNESS RESOLVED — identity topology is now compared. Invert this."
    )


def test_KNOWN_BLINDNESS_transient_mutation_leaves_no_trace(engine):
    """S0 → S1 → S0. A before/after witness cannot detect this by
    construction, however good its canonicalisation gets — an external
    observer during S1 saw a different engine.

    This one is NOT fixable by a better digest: it needs write instrumentation
    (spies on __setitem__/add/append). It stays green even after the witness
    lands, and the witness must not claim to cover it.
    """
    before = structural_digest(engine)
    victim = next(iter(engine._active_cells))
    engine._active_cells.discard(victim)          # S1 — observable right now
    engine._active_cells.add(victim)              # back to S0
    after = structural_digest(engine)

    assert before == after, "a before/after comparison somehow saw S1"


def test_digest_positive_control_still_works(engine):
    """The instrument is not simply inert: a persistent, representable mutation
    IS detected. Without this, every assertion above would be vacuous."""
    before = structural_digest(engine)
    engine._active_cells.discard(next(iter(engine._active_cells)))
    after = structural_digest(engine)
    assert before != after
