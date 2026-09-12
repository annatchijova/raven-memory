#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — adversarial round on StateWitness.

The instrument is NOT promotable to a purity oracle for the engine, and this
file is the evidence for that judgement rather than a claim about it. Three
defects are frozen here as KNOWN_DEFECT characterisations — green today,
asserting that the flaw is present — and must be INVERTED when fixed, never
deleted. Alongside them, the claims that DID survive the attack, so the two
categories are never confused again.

Nothing is fixed here. The first defect in particular needs a decision about
the representation of the alias graph, not a patch.

Run: pytest tests/test_state_witness_attacks.py -q
"""

import copy
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from raven.memory_engine import AdaptiveMemoryEngine
from state_witness import UnrepresentableState, compare, take_witness
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
# DEFECT 1 — interned value atoms pollute the identity graph
# ============================================================

def test_KNOWN_DEFECT_interned_atoms_pollute_the_alias_graph(engine):
    """Attaching ONE new alias renames unrelated edges across the whole graph.

    Root cause, confirmed: CPython interns small integers, so the `1` inside a
    freshly attached dict IS the `1` inside `_active_cells` — the same object.
    The graph therefore records "these two paths share an object" for value
    atoms, which is not engine topology at all. Canonical-path naming then
    turns that into a global rename: the new alias wins the canonical path for
    the interned atoms, and every edge naming them is relabelled.

    The fix is NOT a better naming scheme. Value terminals must not participate
    in the identity graph at all.

    INVERT when that lands: the unrelated-edge count must go to zero.
    """
    shared = {"payload": [1, 2, 3]}          # small ints — interned
    engine.zzz_long_attribute_name = shared
    before = witness(engine)
    engine.a = shared                         # one new alias, same object
    after = witness(engine)

    ba, aa = before.alias_signature(), after.alias_signature()

    def unrelated(sig):
        return {t for t in sig if "engine.a" not in t[0] and "engine.a" not in t[2]}

    churn = unrelated(ba) ^ unrelated(aa)
    assert churn, (
        "DEFECT RESOLVED — adding an alias no longer churns unrelated edges. "
        "Invert this test instead of deleting it."
    )
    # The pollution is specifically through interned atoms reachable elsewhere.
    assert any("_active_cells" in t[0] or "_active_cells" in t[2] for t in churn), (
        f"churn is no longer via interned atoms: {sorted(churn)[:3]}"
    )


def test_interning_is_the_mechanism_not_the_naming(engine):
    """Control that isolates the cause: the identical attack with a payload
    holding no interned atoms churns far less. If this ever reports the same
    magnitude as the test above, the diagnosis was wrong."""
    def churn_for(payload):
        eng = AdaptiveMemoryEngine(db_path=Path(tempfile.mkdtemp()) / "m.db")
        for i in range(4):
            eng.store(f"documento numero {i} con texto", near("c", i))
        eng.recall(near("c", 0))
        eng._ensure_kdtree()
        shared = {"payload": payload}
        eng.zzz_long_attribute_name = shared
        b = witness(eng)
        eng.a = shared
        a = witness(eng)

        def unrelated(sig):
            return {t for t in sig
                    if "engine.a" not in t[0] and "engine.a" not in t[2]}
        return len(unrelated(b.alias_signature()) ^ unrelated(a.alias_signature()))

    with_ints = churn_for([1, 2, 3])
    without = churn_for(["no-interned-small-ints-here"])
    assert with_ints > without, (with_ints, without)


# ============================================================
# DEFECT 2 — observation executes the observed object's code
# ============================================================

def test_KNOWN_DEFECT_observation_executes_observed_object_code(engine):
    """A purity instrument that runs arbitrary object behaviour while looking
    at state can itself be a source of mutation.

    `repr()` is called on set members and dict keys, and a comparison-driven
    sort can reach `__eq__`. Nothing stops either from having side effects.

    INVERT when observation is made behaviour-free (identity- and type-based
    canonicalisation for non-atomic keys and members).
    """
    calls = []

    class Sneaky:
        def __init__(self):
            self.data = 1

        def __repr__(self):
            calls.append("__repr__")
            return "Sneaky()"

        def __eq__(self, other):
            calls.append("__eq__")
            return False

        def __hash__(self):
            return 1

    engine.sneaky_in_set = {Sneaky()}
    engine.sneaky_as_key = {Sneaky(): 1}
    witness(engine)

    assert calls, (
        "DEFECT RESOLVED — observation no longer executes observed code. "
        "Invert this test."
    )
    assert "__repr__" in calls


# ============================================================
# DEFECT 3 — repr-based keys leak heap addresses
# ============================================================

def test_KNOWN_DEFECT_dict_keys_through_repr_leak_heap_addresses(engine):
    """Dict keys and set members are canonicalised through `repr()`. A default
    `__repr__` embeds the object's address, so the signature is not reproducible
    across processes, and two equal-but-distinct keys read as different states.

    INVERT when keys are canonicalised through the same classification as
    everything else, with the supported key types stated explicitly.
    """
    class KeyObj:
        def __init__(self, n):
            self.n = n

        def __hash__(self):
            return hash(self.n)

        def __eq__(self, other):
            return isinstance(other, KeyObj) and self.n == other.n

    engine.dk = {KeyObj(1): "v"}
    sig = witness(engine).value_signature()
    leaking = [p for p, v in sig.items() if "0x" in str(v)]
    assert leaking, (
        "DEFECT RESOLVED — no heap address reaches the signature. Invert this."
    )

    # And the consequence: equal-by-value keys are reported as different state.
    before = witness(engine)
    engine.dk = {KeyObj(1): "v"}               # equal key, distinct object
    after = witness(engine)
    assert not compare(before, after)["persistent_value_state_equal"], (
        "equal-by-value keys now compare equal — invert this half too"
    )


def test_KNOWN_DEFECT_key_objects_are_absent_from_the_graph(engine):
    """A non-scalar dict key is a reachable object with its own identity, and
    it never becomes a node. There is a second, hidden identity graph living
    inside the edge labels.

    INVERT when keys are walked under the same ownership classification.
    """
    class KeyObj:
        def __init__(self, n):
            self.n = n

        def __hash__(self):
            return hash(self.n)

        def __eq__(self, other):
            return isinstance(other, KeyObj) and self.n == other.n

    engine.dk = {KeyObj(1): "v"}
    w = witness(engine)
    assert not any("KeyObj" in n.runtime_type for n in w.nodes.values()), (
        "DEFECT RESOLVED — key objects are now nodes. Invert this."
    )


# ============================================================
# What SURVIVED the attack — real properties
# ============================================================

def test_cycle_topology_is_discriminated_with_identical_payloads(engine):
    """`A → B → A` versus `A → B → B`, same payload on both nodes.

    The earlier claim was only "it terminates and keeps the back-edge", which
    is not discrimination. This is: values identical, topology different,
    reported as two separate answers.
    """
    class N:
        def __init__(self):
            self.tag = "t"
            self.peer = None

    a, b = N(), N()
    a.peer, b.peer = b, a                      # A → B → A
    engine.g = a
    before = witness(engine)
    b.peer = b                                 # A → B → B
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"], result["value_diff_paths"][:3]
    assert not result["persistent_alias_topology_equal"]


def test_alias_merge_is_detected(engine):
    """The inverse of the split already covered: two equal independent objects
    becoming one shared object. Easy to build a comparator that catches one
    direction and normalises the other away."""
    engine.p = {"k": [1]}
    engine.q = {"k": [1]}
    before = witness(engine)
    engine.q = engine.p
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"]
    assert not result["persistent_alias_topology_equal"]


def test_alias_multiplicity_is_preserved(engine):
    """3 → 2+1. "Shared or not" is not enough; which paths belong to which
    equivalence class has to survive."""
    shared = {"k": [7]}
    engine.a1 = engine.b1 = engine.c1 = shared
    before = witness(engine)
    engine.c1 = copy.deepcopy(shared)
    after = witness(engine)

    result = compare(before, after)
    assert result["persistent_value_state_equal"]
    assert not result["persistent_alias_topology_equal"]

    ids_before = before.root_identity()
    ids_after = after.root_identity()
    assert ids_before["a1"] == ids_before["b1"] == ids_before["c1"]
    assert ids_after["a1"] == ids_after["b1"] != ids_after["c1"]


def test_witness_is_stable_under_allocation_noise(engine):
    """Any detail accidentally derived from allocation order would surface as
    instability when the heap moves between two captures of the same state."""
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
    """Fail-closed must not be a property of the top level only."""
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
    """Reached twice, refused once — a second path must not let it slip past
    the visited check unrepresented."""
    class Exotic:
        __slots__ = ("payload",)

        def __init__(self):
            self.payload = 1

    shared = Exotic()
    engine.first = shared
    engine.second = shared
    with pytest.raises(UnrepresentableState):
        witness(engine)


def test_path_objects_are_value_semantic_by_decision(engine):
    """`PosixPath` has __slots__, so the mutability heuristic read it as unknown
    mutable state and the witness failed closed on something inert. That was a
    real accidental control, and the resolution is a stated decision rather
    than a loosened heuristic: an unknown Python TYPE is not the same thing as
    unknown mutable STATE, and only the latter must fail."""
    engine.some_path = Path("/tmp/whatever")
    w = witness(engine)                        # must not raise
    node = w.nodes[id(engine.some_path)]
    assert node.policy == "TERMINAL_BY_VALUE"
    assert node.mutable is False
    assert "0x" not in str(node.canonical_value)
