#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — StateWitness after the identity-boundary rebuild.

The adversarial round on the first version (0cf0cd1) landed three defects:

  1. interned value atoms polluted the identity graph — attaching ONE alias
     renamed 19 unrelated edges, because `1` inside a fresh dict IS the `1` in
     `_active_cells`;
  2. observation executed the observed object's code (`repr`, potentially
     `__eq__`);
  3. dict keys went through `repr()`, leaking heap addresses and hiding a second
     identity graph inside the edge labels.

All three were frozen there as green KNOWN_DEFECT characterisations. This file
is their INVERSION: each now asserts the corrected semantics, and the tests that
survived the attack unchanged are kept alongside so the two categories stay
distinguishable.

One limit is unchanged and still stated: S0 → S1 → S0 is out of reach for any
before/after comparison.

Run: pytest tests/test_state_witness_attacks.py -q
"""

import copy
import enum
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from raven.memory_engine import AdaptiveMemoryEngine, LinkType
from state_witness import (
    OWNED_MUTABLE_CONTAINER,
    OWNED_OBJECT,
    UnrepresentableState,
    compare,
    descends,
    is_mutable,
    take_witness,
    tracks_identity,
)
from test_intervention_properties import near
from test_state_witness import ENGINE_EXCLUSIONS


@pytest.fixture
def engine(tmp_path):
    eng = AdaptiveMemoryEngine(db_path=tmp_path / "attack.db")
    for i in range(4):
        eng.store(f"documento numero {i} con texto de cuerpo", near("c", i))
    eng.recall(near("c", 0))
    eng._ensure_kdtree()
    return eng


def witness(eng):
    return take_witness(eng, exclusions=ENGINE_EXCLUSIONS)


# ============================================================
# The contract that makes the fix a design and not a patch
# ============================================================

def test_identity_tracking_is_a_separate_concept_from_descendability():
    """They agree on every classification that exists today except one, and the
    immutable container is exactly the case that keeps them from being the same
    predicate. Collapsing them would make a coincidence into architecture."""
    assert tracks_identity(OWNED_MUTABLE_CONTAINER) and descends(OWNED_MUTABLE_CONTAINER)
    assert tracks_identity(OWNED_OBJECT) and descends(OWNED_OBJECT)
    assert descends("OWNED_IMMUTABLE_CONTAINER")
    assert not tracks_identity("OWNED_IMMUTABLE_CONTAINER")
    assert not tracks_identity("VALUE_TERMINAL")
    assert not tracks_identity("IDENTITY_TERMINAL")


def test_no_engine_owned_mutable_escapes_identity_tracking(engine):
    """The invariant that forbids an accidental third category.

    Every supported engine-owned mutable must either participate in identity
    tracking, or be excluded as a semantic subsystem with its own witness. What
    must never exist is "mutable, represented by value only, identity ignored" —
    that is precisely the shape the blind spot had.
    """
    w = witness(engine)
    offenders = [
        (w.canonical_paths[wid], n.runtime_type, n.classification)
        for wid, n in w.nodes.items()
        if n.mutable and not n.tracks_identity
    ]
    assert not offenders, f"mutable state outside identity tracking: {offenders}"


def test_mutable_masquerading_as_a_value_object_fails_closed(engine):
    """__slots__ implies neither immutability nor value semantics, so it is not
    a membership criterion for the safe-value registry. A mutable type that
    merely looks value-like must fail closed rather than slip into the value
    domain and vanish from the identity graph."""
    class MutableValueLike:
        __slots__ = ("x",)

        def __init__(self):
            self.x = 1

    # Assert the CLASSIFICATION, not merely that something downstream raised.
    # A negative control showed the object still failed closed when __slots__ was
    # (wrongly) treated as value-semantic, because a second guard caught the
    # missing canonicaliser. Defence in depth is welcome; a test that cannot
    # tell which layer held is not.
    from state_witness import UNSUPPORTED_MUTABLE, classify
    assert classify(MutableValueLike()) == UNSUPPORTED_MUTABLE, (
        "__slots__ became a membership criterion for the value domain"
    )

    engine.sneaky_value = MutableValueLike()
    with pytest.raises(UnrepresentableState) as exc:
        witness(engine)
    assert exc.value.runtime_type == "MutableValueLike"


# ============================================================
# INVERSION 1 — interned atoms no longer manufacture topology
# ============================================================

def test_INVERTED_adding_an_alias_no_longer_churns_unrelated_edges(engine):
    """Was KNOWN_DEFECT: 19 unrelated edges renamed. Value terminals are no
    longer identity nodes, so interned atoms cannot fabricate sharing."""
    # The property, stated without a fragile path filter: attaching an alias
    # must not perturb the topology of state that was already there. The
    # pristine engine's edges are the reference.
    pristine = witness(engine).alias_signature()

    shared = {"payload": [1, 2, 3]}          # small ints — still interned
    engine.zzz_long_attribute_name = shared
    before = witness(engine)
    engine.a = shared                         # one new alias to the same object
    after = witness(engine)

    ba, aa = before.alias_signature(), after.alias_signature()
    assert pristine <= ba, "attaching the object already perturbed prior state"
    assert pristine <= aa, (
        "aliasing perturbed pre-existing engine topology: "
        f"{sorted(pristine - aa, key=repr)[:3]}"
    )

    # RESIDUAL, stated rather than asserted away: canonical-path naming renames
    # the aliased object's own subtree, because it genuinely acquired a shorter
    # shortest path. So the reported delta is not minimal — every changed edge
    # must at least NAME the aliased object. That is the sense in which this
    # claim compares a rooted RENDERING of the topology and not the topology.
    churn = ba ^ aa
    assert churn, "aliasing a mutable must be visible somewhere"
    for parent, _label, child in churn:
        assert parent in ("engine", "engine.a", "engine.zzz_long_attribute_name") \
            or parent.startswith(("engine.a|", "engine.zzz_long_attribute_name|")), \
            f"churn reached unrelated state: {parent}"
        assert child.startswith(("engine.a", "engine.zzz_long_attribute_name")), \
            f"churn reached unrelated state: {child}"


@pytest.mark.parametrize("label,value", [
    ("interned int", 7),
    ("interned str", "shared-string-value"),
    ("enum member", LinkType.RESONANT),
    ("path", Path("/tmp/x")),
    ("float", 3.5),
])
def test_aliasing_a_value_terminal_does_not_change_topology(engine, label, value):
    """Sharing an int, a string, an enum member or a path says nothing about
    engine state. Adding such an alias must move the value claim and leave the
    topology claim untouched."""
    engine.v1 = value
    before = witness(engine)
    engine.v2 = value                         # same object, second reference
    after = witness(engine)

    result = compare(before, after)
    assert not result["persistent_value_state_equal"], \
        f"{label}: a new attribute is new state and must show up by value"
    assert result["persistent_alias_topology_equal"], \
        f"{label}: a value terminal fabricated topology"


def test_aliasing_a_mutable_does_change_topology(engine):
    """The other half: for a mutable, sharing IS engine topology."""
    engine.m1 = {"k": [1]}
    before = witness(engine)
    engine.m2 = engine.m1
    after = witness(engine)
    assert not compare(before, after)["persistent_alias_topology_equal"]


# ============================================================
# INVERSION 2 — observation executes nothing the object defines
# ============================================================

def test_INVERTED_observation_executes_no_observed_object_code(engine):
    """Was KNOWN_DEFECT: 5 `__repr__` calls. An instrument that runs arbitrary
    behaviour while looking at state can itself mutate it."""
    calls = []

    class Sneaky:
        def __init__(self):
            self.data = 1

        def __repr__(self):
            calls.append("__repr__")
            return "Sneaky()"

        def __str__(self):
            calls.append("__str__")
            return "Sneaky()"

        def __eq__(self, other):
            calls.append("__eq__")
            return False

        def __lt__(self, other):
            calls.append("__lt__")
            return False

        def __hash__(self):
            calls.append("__hash__")
            return 1

    engine.plain = Sneaky()
    engine.in_list = [Sneaky(), Sneaky()]
    engine.in_dict_value = {"k": Sneaky()}
    witness(engine)

    assert calls == [], f"observation executed observed code: {calls}"


def test_observation_does_not_trigger_properties_or_descriptors(engine):
    """`vars()` rather than `getattr`, so a property cannot fire during
    observation — and a property that mutates would otherwise make the
    instrument the source of the change it reports."""
    fired = []

    class WithProperty:
        def __init__(self):
            self._v = 1

        @property
        def computed(self):
            fired.append("property")
            self._v += 1                      # a property that mutates
            return self._v

    engine.prop_holder = WithProperty()
    witness(engine)
    assert fired == []
    assert engine.prop_holder._v == 1, "observation mutated through a property"


# ============================================================
# INVERSION 3 — keys are inert tokens, addresses never leak
# ============================================================

def test_INVERTED_no_heap_address_reaches_the_signature(engine):
    class Colour(enum.Enum):
        RED = "red"

    engine.keys_ok = {
        "s": 1, 17: 2, b"bytes": 3, (1, "t"): 4,
        Colour.RED: 5, Path("/tmp/k"): 6, 2.5: 7, None: 8, True: 9,
    }
    sig = witness(engine).value_signature()
    leaking = [p for p, v in sig.items() if "0x" in str(v)]
    assert leaking == [], f"heap addresses leaked at {leaking}"


def test_INVERTED_equal_by_value_keys_compare_as_the_same_state(engine):
    """A tuple key rebuilt with equal contents is the same state, and used to
    read as different because its `repr` carried an address."""
    engine.dk = {(1, "a"): "v"}
    before = witness(engine)
    engine.dk = {(1, "a"): "v"}
    after = witness(engine)
    assert compare(before, after)["persistent_value_state_equal"]


def test_INVERTED_unsupported_dict_key_fails_closed(engine):
    """A key outside the value domain is refused, naming the path, rather than
    stringified into the signature. The second identity graph that used to hide
    inside the edge labels cannot form."""
    class KeyObj:
        def __hash__(self):
            return 1

        def __eq__(self, other):
            return self is other

    engine.dk = {KeyObj(): "v"}
    with pytest.raises(UnrepresentableState) as exc:
        witness(engine)
    assert "dict key" in str(exc.value)
    assert exc.value.runtime_type == "KeyObj"


def test_set_members_are_sorted_as_tokens_not_as_objects(engine):
    """Sorting the objects would invoke their comparison protocol. Members are
    canonicalised to inert tokens first, and the TOKENS are ordered."""
    engine.s = {3, 1, 2, "a", "b"}
    a, b = witness(engine), witness(engine)
    assert compare(a, b)["persistent_value_state_equal"]

    class UnorderableMember:
        __slots__ = ()

        def __lt__(self, other):
            raise AssertionError("the witness compared observed objects")

        def __hash__(self):
            return 7

        def __eq__(self, other):
            return self is other

    engine.s2 = {UnorderableMember()}
    with pytest.raises(UnrepresentableState):
        witness(engine)                        # unsupported member, not a crash


# ============================================================
# Orthogonality of the two claims
# ============================================================

def test_changing_only_a_terminal_value_leaves_topology_untouched(engine):
    """The cleanest demonstration that the two oracles are independent."""
    engine.holder = {"n": 1}
    before = witness(engine)
    engine.holder["n"] = 2                    # same objects, different value
    after = witness(engine)

    result = compare(before, after)
    assert not result["persistent_value_state_equal"]
    assert result["persistent_alias_topology_equal"]
    assert result["persistent_root_identity_equal"]


def test_reconstruction_preserving_values_but_destroying_sharing(engine):
    """A deepcopy-like rebuild: every value survives, every mutable identity
    relationship does not."""
    inner = {"k": [1, 2]}
    engine.r1 = inner
    engine.r2 = inner
    before = witness(engine)
    rebuilt = copy.deepcopy(inner)
    engine.r1, engine.r2 = rebuilt, copy.deepcopy(inner)
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"]
    assert not result["persistent_alias_topology_equal"]


def test_reconstruction_reassigning_only_equivalent_terminals(engine):
    """The control for the test above: rebuilding only value terminals changes
    neither claim."""
    engine.t1 = "a-string-value"
    engine.t2 = 12345
    before = witness(engine)
    engine.t1 = "a-string" + "-value"
    engine.t2 = 12000 + 345
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"]
    assert result["persistent_alias_topology_equal"]


def test_root_identity_lives_on_the_comparator_not_the_snapshot():
    """It is a relation between two captures of one process, not canonicalisable
    state, so it must not be a field of the snapshot."""
    from state_witness import StateComparison, StateSnapshot
    assert "persistent_root_identity_equal" in StateComparison.__dataclass_fields__
    assert "persistent_root_identity_equal" not in StateSnapshot.__dataclass_fields__


# ============================================================
# Survived the first attack, still holds
# ============================================================

def test_cycle_topology_is_discriminated_with_identical_payloads(engine):
    class N:
        def __init__(self):
            self.tag = "t"
            self.peer = None

    a, b = N(), N()
    a.peer, b.peer = b, a
    engine.g = a
    before = witness(engine)
    b.peer = b
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"], result["value_diff_paths"][:3]
    assert not result["persistent_alias_topology_equal"]


def test_alias_merge_is_detected(engine):
    engine.p = {"k": [1]}
    engine.q = {"k": [1]}
    before = witness(engine)
    engine.q = engine.p
    after = witness(engine)
    result = compare(before, after)
    assert result["persistent_value_state_equal"]
    assert not result["persistent_alias_topology_equal"]


def test_alias_multiplicity_is_preserved(engine):
    shared = {"k": [7]}
    engine.a1 = engine.b1 = engine.c1 = shared
    before = witness(engine)
    engine.c1 = copy.deepcopy(shared)
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"]
    assert not result["persistent_alias_topology_equal"]
    ib, ia = before.root_identity(), after.root_identity()
    assert ib["a1"] == ib["b1"] == ib["c1"]
    assert ia["a1"] == ia["b1"] != ia["c1"]


def test_witness_is_stable_under_allocation_noise(engine):
    class Plain:
        def __init__(self):
            self.v = 1

    engine.obj = Plain()
    before = witness(engine)
    garbage = [Plain() for _ in range(5000)]
    del garbage
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"]
    assert result["persistent_alias_topology_equal"]


def test_unsupported_mutable_buried_deep_still_fails(engine):
    class Exotic:
        __slots__ = ("payload",)

        def __init__(self):
            self.payload = 1

    engine.deep = {"a": [{"b": Exotic()}]}
    with pytest.raises(UnrepresentableState) as exc:
        witness(engine)
    assert exc.value.runtime_type == "Exotic"
    assert "deep" in exc.value.path


def test_unsupported_mutable_reachable_by_two_aliases_still_fails(engine):
    class Exotic:
        __slots__ = ("payload",)

        def __init__(self):
            self.payload = 1

    shared = Exotic()
    engine.first = shared
    engine.second = shared
    with pytest.raises(UnrepresentableState):
        witness(engine)


def test_STILL_BLIND_transient_mutation(engine):
    """Unchanged and still stated: no before/after comparison reaches S1."""
    before = witness(engine)
    victim = next(iter(engine._active_cells))
    engine._active_cells.discard(victim)
    engine._active_cells.add(victim)
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"]
    assert result["persistent_alias_topology_equal"]
