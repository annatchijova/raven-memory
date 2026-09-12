#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StateWitness — a bounded, fail-closed record of engine-owned runtime state.

Successor instrument to `structural_digest()`, whose autopsy is in
tests/test_digest_coverage.py. Two things that autopsy established drive this
design:

  * `("opaque", type_name)` for anything unrecognised is a lie shaped like a
    result. Here an engine-owned mutable object the walker cannot represent
    raises `UnrepresentableState(path, type)`. Fail closed, never summarise.

  * An unrestricted "every reachable Python object" walk is not state, it is
    interpreter reachability — instance → Enum → class → functions → globals →
    modules. The naive version raised RecursionError on
    `LinkType.__objclass__`. The boundary here is SEMANTIC, not mechanical.

It answers two questions that the old digest conflated into one green:

    value_signature   — what the state IS
    alias_signature   — which references point at the SAME object

so that "two attributes stopped sharing one object, but the copies are equal"
is reported as value-equal and topology-changed, instead of as identical.

What it CANNOT do, by construction: observe S0 → S1 → S0. No before/after
comparison can. That needs a MutationJournal instrumenting write surfaces, and
its claim would be bounded to "no mutation through instrumented surfaces",
never "no mutation".

CONFIRMED DEFECTS (tests/test_state_witness_attacks.py), not yet fixed:

  1. The alias graph is polluted by interned value atoms. `id(1)` inside a
     freshly attached dict IS the `1` inside `_active_cells`, so the graph
     records "sharing" that is not engine topology. Combined with canonical-path
     naming, attaching one new alias renamed 19 UNRELATED edges. The fix is not
     a better naming scheme: value terminals must not participate in the
     identity graph at all.
  2. Observation executes the observed object's code. `repr()` runs on set
     members and dict keys, and a comparison-driven sort can run `__eq__`. A
     purity instrument that executes arbitrary object behaviour can itself be
     a source of mutation.
  3. Dict keys and set members are canonicalised through `repr()`, so a default
     `__repr__` embeds a heap address in the signature. Stable within a process
     for the same object, but not reproducible across processes, and two
     equal-but-distinct key objects compare as different states.

Until 1 is fixed, `persistent_alias_topology_equal` compares a rooted,
path-labelled RENDERING of the topology, not the topology.

Not a pytest module — an instrument used by tests.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import hashlib
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

MAX_DEPTH = 12
MAX_FANOUT = 512


class UnrepresentableState(AssertionError):
    """An engine-owned mutable object the witness has no policy for.

    Deliberately an AssertionError: a purity claim over state the instrument
    could not represent is the exact failure mode this whole exercise exists to
    remove, so it fails the test rather than degrading the report.
    """

    def __init__(self, path: str, obj: Any):
        super().__init__(
            f"no witness policy for engine-owned mutable state at {path!r} "
            f"(type {type(obj).__name__}). Add a DESCEND rule, a "
            f"TERMINAL_BY_IDENTITY justification, or an explicit exclusion — "
            f"do not let it be summarised away."
        )
        self.path, self.runtime_type = path, type(obj).__name__


DESCEND = "DESCEND"
TERMINAL_BY_VALUE = "TERMINAL_BY_VALUE"
TERMINAL_BY_IDENTITY = "TERMINAL_BY_IDENTITY"
EXCLUDED = "EXCLUDED"

_VALUE_ATOMS = (type(None), bool, int, float, str, bytes, complex)

# Immutable value objects that happen to use __slots__, so the mutability
# heuristic would misread them as unknown mutable state and fail closed on
# something inert. Each entry is a DECISION, not a convenience: these carry
# value semantics for our purpose and have no engine-owned interior.
#   PurePath — a path is a value; its parts are derived, not mutable state.
#   Decimal / datetime / date / time / timedelta / UUID — immutable scalars.
# "Unknown Python type" must not be equivalent to "unknown mutable state":
# the fail-closed requirement is about engine-owned MUTABLE state.
import datetime as _dt
import decimal as _dec
import pathlib as _pl
import uuid as _uuid

_VALUE_SEMANTIC_TYPES = (
    _pl.PurePath, _dec.Decimal, _dt.datetime, _dt.date, _dt.time,
    _dt.timedelta, _uuid.UUID,
)
_IDENTITY_TYPES = (
    types.FunctionType, types.BuiltinFunctionType, types.MethodType,
    types.ModuleType, type,
)
_ORDERED_CONTAINERS = (list, tuple, collections.deque)
_UNORDERED_CONTAINERS = (set, frozenset)


@dataclass
class WitnessNode:
    witness_id: int
    runtime_type: str
    canonical_value: Any
    mutable: bool
    policy: str


@dataclass
class StateWitness:
    nodes: Dict[int, WitnessNode] = field(default_factory=dict)
    edges: List[Tuple[int, str, int]] = field(default_factory=list)
    roots: Dict[str, int] = field(default_factory=dict)
    exclusions: List[Tuple[str, str, str]] = field(default_factory=list)
    truncations: List[str] = field(default_factory=list)
    _canon_path: Dict[int, str] = field(default_factory=dict)

    # ---- the two independent claims ----

    def value_signature(self) -> Dict[str, Any]:
        """Value at every reachable path.

        Indexed by path rather than by node, so two paths that share one object
        and two paths holding equal copies produce the SAME signature. That is
        the point: this claim is about what the state is, not about how it is
        wired.
        """
        out: Dict[str, Any] = {}
        stack: List[Tuple[str, int, Tuple[int, ...]]] = [
            (f"engine.{name}", wid, ()) for name, wid in sorted(self.roots.items())
        ]
        children = collections.defaultdict(list)
        for parent, label, child in self.edges:
            children[parent].append((label, child))
        while stack:
            path, wid, seen = stack.pop()
            node = self.nodes[wid]
            if wid in seen:
                out[path] = ("<cycle>", node.runtime_type)
                continue
            out[path] = (node.runtime_type, node.canonical_value)
            if len(seen) >= MAX_DEPTH:
                out[path] = ("<depth-bound>", node.runtime_type)
                continue
            for label, child in sorted(children.get(wid, []), key=lambda t: t[0]):
                stack.append((f"{path}{label}", child, seen + (wid,)))
        return out

    def alias_signature(self) -> Set[Tuple[str, str, str]]:
        """Which references point at the same object.

        Nodes are named by their canonical path (shortest, then lexicographic)
        rather than by `id()`, which is not comparable across two witnesses.
        Sharing shows up as two different parents naming one canonical child.
        """
        return {
            (self._canon_path[p], label, self._canon_path[c])
            for p, label, c in self.edges
        }

    def root_identity(self) -> Dict[str, str]:
        return {name: self._canon_path[wid] for name, wid in self.roots.items()}


def _canonical_value(obj: Any, policy: str) -> Any:
    """Value canonicalisation ONLY.

    Kept deliberately separate from graph construction: a set is
    order-independent as a value, a list is not, and a dict's key→child
    association must survive — none of which should be decided by whatever the
    traversal happens to find convenient.
    """
    if policy == TERMINAL_BY_IDENTITY:
        return ("identity", getattr(obj, "__qualname__", None) or repr(obj))
    if isinstance(obj, np.ndarray):
        return ("ndarray", obj.dtype.str, obj.shape,
                hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest())
    if isinstance(obj, enum.Enum):
        return ("enum", type(obj).__name__, obj.name, repr(obj.value))
    if isinstance(obj, _VALUE_SEMANTIC_TYPES):
        return ("value-semantic", type(obj).__name__, str(obj))
    if isinstance(obj, _VALUE_ATOMS):
        return repr(obj)
    if isinstance(obj, dict):
        # Shape only — the children carry the values, and the key→child
        # association is preserved by the edge labels.
        return ("dict", sorted(repr(k) for k in obj))
    if isinstance(obj, _UNORDERED_CONTAINERS):
        # Order-independent by definition. Members are recorded as sorted
        # reprs here AND as edges, so a set of mutables is still walked.
        return (type(obj).__name__, sorted(repr(x) for x in obj))
    if isinstance(obj, _ORDERED_CONTAINERS):
        return (type(obj).__name__, len(obj))     # order lives in edge labels
    if dataclasses.is_dataclass(obj) or hasattr(obj, "__dict__"):
        return ("object", type(obj).__name__, sorted(vars(obj).keys()))
    return ("value", repr(obj))


def _classify(obj: Any) -> str:
    if isinstance(obj, _VALUE_SEMANTIC_TYPES):
        return TERMINAL_BY_VALUE
    if isinstance(obj, enum.Enum):
        return TERMINAL_BY_VALUE          # kills the __objclass__ recursion
    if isinstance(obj, _VALUE_ATOMS):
        return TERMINAL_BY_VALUE
    if isinstance(obj, np.ndarray):
        return TERMINAL_BY_VALUE
    if isinstance(obj, _IDENTITY_TYPES):
        return TERMINAL_BY_IDENTITY
    if isinstance(obj, (dict, list, tuple, set, frozenset, collections.deque)):
        return DESCEND
    if dataclasses.is_dataclass(obj) or hasattr(obj, "__dict__"):
        return DESCEND
    return "UNKNOWN"


def _is_mutable(obj: Any) -> bool:
    if isinstance(obj, _VALUE_SEMANTIC_TYPES):
        return False
    if isinstance(obj, (dict, list, set, bytearray, collections.deque)):
        return True
    if isinstance(obj, np.ndarray):
        return True
    if isinstance(obj, _VALUE_ATOMS) or isinstance(obj, (tuple, frozenset)):
        return False
    if isinstance(obj, enum.Enum) or isinstance(obj, _IDENTITY_TYPES):
        return False
    return hasattr(obj, "__dict__") or hasattr(obj, "__slots__")


def _children(obj: Any) -> List[Tuple[str, Any]]:
    """Edge labels carry the structure the value canonicalisation drops:
    dict keys, sequence positions. A set's members get repr-labels so the
    edge set stays order-independent, matching its value semantics."""
    if isinstance(obj, dict):
        return [(f"[{k!r}]", v) for k, v in list(obj.items())[:MAX_FANOUT]]
    if isinstance(obj, _ORDERED_CONTAINERS):
        return [(f"[{i}]", v) for i, v in enumerate(list(obj)[:MAX_FANOUT])]
    if isinstance(obj, _UNORDERED_CONTAINERS):
        return [(f"{{{x!r}}}", x) for x in sorted(obj, key=repr)[:MAX_FANOUT]]
    if dataclasses.is_dataclass(obj) or hasattr(obj, "__dict__"):
        return [(f".{k}", v) for k, v in sorted(vars(obj).items())[:MAX_FANOUT]]
    return []


def take_witness(
    engine,
    exclusions: Optional[Dict[str, str]] = None,
    max_depth: int = MAX_DEPTH,
) -> StateWitness:
    """Record engine-owned runtime state.

    `exclusions` maps an attribute name to a written justification. An
    exclusion without one is refused — an unjustified exclusion is
    indistinguishable from an oversight six months later.
    """
    exclusions = exclusions or {}
    for name, why in exclusions.items():
        if not why or len(why) < 20:
            raise ValueError(f"exclusion {name!r} needs a written justification")

    w = StateWitness()
    queue: collections.deque = collections.deque()

    for name, value in sorted(vars(engine).items()):
        if name in exclusions:
            w.exclusions.append((f"engine.{name}", type(value).__name__,
                                 exclusions[name]))
            continue
        queue.append((f"engine.{name}", value, 0, None, name))

    visited: Set[int] = set()
    while queue:
        path, obj, depth, parent_wid, label = queue.popleft()
        wid = id(obj)

        if parent_wid is None:
            w.roots[label] = wid
        else:
            w.edges.append((parent_wid, label, wid))

        # Shortest path wins; ties broken lexicographically. Stable across
        # witnesses, unlike id().
        prev = w._canon_path.get(wid)
        if prev is None or (len(path), path) < (len(prev), prev):
            w._canon_path[wid] = path

        if wid in visited:
            # Back-edge: recorded above, not re-walked. This is what makes a
            # cyclic engine-owned structure terminate WITHOUT losing the cycle.
            continue
        visited.add(wid)

        policy = _classify(obj)
        mutable = _is_mutable(obj)
        if policy == "UNKNOWN":
            if mutable:
                raise UnrepresentableState(path, obj)
            policy = TERMINAL_BY_VALUE

        w.nodes[wid] = WitnessNode(
            witness_id=wid, runtime_type=type(obj).__name__,
            canonical_value=_canonical_value(obj, policy),
            mutable=mutable, policy=policy,
        )

        if policy != DESCEND:
            continue
        if depth >= max_depth:
            w.truncations.append(path)
            continue
        for clabel, cvalue in _children(obj):
            queue.append((f"{path}{clabel}", cvalue, depth + 1, wid, clabel))

    return w


def compare(before: StateWitness, after: StateWitness) -> Dict[str, Any]:
    """Two claims, reported separately. Neither is called `pure`."""
    bv, av = before.value_signature(), after.value_signature()
    ba, aa = before.alias_signature(), after.alias_signature()
    return {
        # Named so that no one can later collapse them into `pure`. Each is
        # scoped to PERSISTENT state observed between two captures, over the
        # SUPPORTED subset of engine-owned state — semantic subsystems (_db,
        # kdtree) are excluded and transient mutation is out of reach entirely.
        "persistent_value_state_equal": bv == av,
        "persistent_alias_topology_equal": ba == aa,
        "value_diff_paths": sorted(
            p for p in set(bv) | set(av) if bv.get(p) != av.get(p)
        )[:40],
        "alias_only_before": sorted(ba - aa)[:40],
        "alias_only_after": sorted(aa - ba)[:40],
        "persistent_root_identity_equal":
            before.root_identity() == after.root_identity(),
    }
