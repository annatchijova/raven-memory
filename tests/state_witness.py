#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StateWitness — a bounded, fail-closed record of engine-owned runtime state.

Successor to `structural_digest()`, whose autopsy is in
tests/test_digest_coverage.py. Rebuilt after the adversarial round in
tests/test_state_witness_attacks.py landed three defects on its first version.

Three rules carry the design.

1. SEMANTIC IDENTITY BOUNDARY BEFORE REPRESENTATION.
   The first version asked CPython whether two ones are the same one. Small
   integers are interned, so `1` inside a freshly attached dict IS the `1` in
   `_active_cells`, and the identity graph recorded "sharing" that is not
   engine topology — then canonical-path naming turned one new alias into a
   rename of 19 unrelated edges. Value terminals are NOT identity nodes.
   `tracks_identity()` says which classifications are, as a concept separate
   from descendability: today they coincide, and that coincidence is not
   allowed to become architecture by accident.

2. OBSERVATION EXECUTES NOTHING THE OBSERVED OBJECT DEFINES.
   No `__repr__`, `__str__`, `__eq__`, `__lt__`, `__hash__`, no properties or
   descriptors, no `getattr` on observed state — `vars()` only. An instrument
   that runs arbitrary behaviour while looking at state can itself mutate it.
   Dispatch is by EXACT type, because a `str` or `dict` subclass can override
   the protocols that would otherwise be assumed safe.

3. FAIL CLOSED, NEVER SUMMARISE.
   An engine-owned mutable with no policy raises `UnrepresentableState`. An
   unsupported dict key raises. "Unknown Python type" is not "unknown mutable
   state": inert value objects are supported through an explicit registry with
   written reasons, never through a `__slots__` heuristic — `__slots__` implies
   neither immutability nor value semantics.

Two independent claims, and the vocabulary is deliberately unusable as `pure`:

    persistent_value_state_equal       what the state IS
    persistent_alias_topology_equal    which references share a mutable object
    persistent_root_identity_equal     (comparator only — see StateComparison)

Out of reach by construction: S0 → S1 → S0. No before/after comparison sees it.
That needs a MutationJournal over instrumented write surfaces, whose claim would
be bounded to "no mutation through instrumented surfaces".

Not a pytest module — an instrument used by tests.
"""

from __future__ import annotations

import collections
import datetime as _dt
import decimal as _dec
import enum
import hashlib
import pathlib as _pl
import types
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

MAX_DEPTH = 12
MAX_FANOUT = 512


class UnrepresentableState(AssertionError):
    """Engine-owned state the witness has no policy for.

    An AssertionError on purpose: a purity claim over state the instrument could
    not represent is the exact failure this whole exercise exists to remove, so
    it fails the test instead of degrading the report.
    """

    def __init__(self, path: str, obj: Any, detail: str = ""):
        self.path = path
        self.runtime_type = type(obj).__name__
        super().__init__(
            f"no witness policy for engine-owned state at {path!r} "
            f"(type {self.runtime_type}){(': ' + detail) if detail else ''}. "
            "Add a classification rule, register it as a safe value type with a "
            "written reason, or exclude it as a semantic subsystem with its own "
            "witness — do not let it be summarised away."
        )


# ----------------------------------------------------------
# Classification — the semantic ownership boundary
# ----------------------------------------------------------

VALUE_TERMINAL = "VALUE_TERMINAL"
IDENTITY_TERMINAL = "IDENTITY_TERMINAL"
OWNED_MUTABLE_CONTAINER = "OWNED_MUTABLE_CONTAINER"
OWNED_IMMUTABLE_CONTAINER = "OWNED_IMMUTABLE_CONTAINER"
OWNED_OBJECT = "OWNED_OBJECT"
EXCLUDED_SEMANTIC_SUBSYSTEM = "EXCLUDED_SEMANTIC_SUBSYSTEM"
UNSUPPORTED_MUTABLE = "UNSUPPORTED_MUTABLE"

_DESCENDS = frozenset({
    OWNED_MUTABLE_CONTAINER, OWNED_IMMUTABLE_CONTAINER, OWNED_OBJECT,
})
_TRACKS_IDENTITY = frozenset({OWNED_MUTABLE_CONTAINER, OWNED_OBJECT})


def descends(classification: str) -> bool:
    """Whether the walker looks inside."""
    return classification in _DESCENDS


def tracks_identity(classification: str) -> bool:
    """Whether "the same object" is a meaningful statement about this node.

    Deliberately a separate predicate from `descends()`. The two agree on every
    classification that exists today — an immutable container is the one that
    splits them — and keeping them separate stops a coincidence from hardening
    into architecture. Sharing an interned int, a string or an enum member says
    nothing about engine state; sharing a dict does.
    """
    return classification in _TRACKS_IDENTITY


# Exact types only. `isinstance` would accept a subclass that overrides the very
# protocols these canonicalisers rely on being builtin.
def _atom(obj: Any) -> Any:
    return (type(obj).__name__, repr(obj))      # exact builtin repr, not user code


def _canon_path(obj: _pl.PurePath) -> Any:
    # PurePath.__str__ is stdlib, not user-defined; a path IS a value and its
    # parts are derived rather than mutable state.
    return ("path", type(obj).__name__, str(obj))


def _canon_decimal(obj) -> Any:
    return ("decimal", str(obj))


def _canon_datetimeish(obj) -> Any:
    return ("datetimeish", type(obj).__name__, obj.isoformat()
            if hasattr(type(obj), "isoformat") else str(obj))


def _canon_timedelta(obj) -> Any:
    return ("timedelta", obj.days, obj.seconds, obj.microseconds)


def _canon_uuid(obj) -> Any:
    return ("uuid", obj.hex)


def _canon_ndarray(obj: np.ndarray) -> Any:
    return ("ndarray", obj.dtype.str, obj.shape,
            hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest())


# Registry, not a heuristic. Every entry is a decision with a reason recorded in
# SAFE_VALUE_REASONS. __slots__ is NOT a membership criterion: it implies
# neither immutability nor value semantics, and an unregistered __slots__ object
# fails closed.
SAFE_VALUE_TYPES: Dict[type, Callable[[Any], Any]] = {
    type(None): _atom, bool: _atom, int: _atom, float: _atom,
    str: _atom, bytes: _atom, complex: _atom,
    _pl.PurePath: _canon_path, _pl.PurePosixPath: _canon_path,
    _pl.PureWindowsPath: _canon_path, _pl.Path: _canon_path,
    _pl.PosixPath: _canon_path, _pl.WindowsPath: _canon_path,
    _dec.Decimal: _canon_decimal,
    _dt.datetime: _canon_datetimeish, _dt.date: _canon_datetimeish,
    _dt.time: _canon_datetimeish, _dt.timedelta: _canon_timedelta,
    _uuid.UUID: _canon_uuid,
    np.ndarray: _canon_ndarray,
}

SAFE_VALUE_REASONS = {
    "builtin atoms": "immutable, and their repr is C-level for the exact type",
    "paths": "a path is a value; its parts are derived, not mutable state",
    "decimal/datetime/uuid": "immutable scalars with stdlib canonical forms",
    "ndarray": "mutable buffer, but identity is irrelevant to retrieval state; "
               "contents are fingerprinted by SHA-256 over the exact bytes",
}

_IDENTITY_TYPES = (
    types.FunctionType, types.BuiltinFunctionType, types.MethodType,
    types.ModuleType, types.WrapperDescriptorType,
    types.MethodWrapperType, types.MethodDescriptorType, type,
)

_MUTABLE_CONTAINERS = (dict, list, set, bytearray, collections.deque,
                       collections.OrderedDict, collections.defaultdict)
_IMMUTABLE_CONTAINERS = (tuple, frozenset)


def classify(obj: Any) -> str:
    """One classification, used by BOTH claims.

    Value and identity may REPRESENT a node differently, but they must not
    decide independently what belongs to the state — two incompatible notions of
    ownership inside one walker is how a blind spot gets reintroduced.
    """
    t = type(obj)
    if t in SAFE_VALUE_TYPES:
        return VALUE_TERMINAL
    if isinstance(obj, enum.Enum):
        return VALUE_TERMINAL            # interned members; kills __objclass__
    if isinstance(obj, _IDENTITY_TYPES):
        return IDENTITY_TERMINAL
    if t in _MUTABLE_CONTAINERS:
        return OWNED_MUTABLE_CONTAINER
    if t in _IMMUTABLE_CONTAINERS:
        return OWNED_IMMUTABLE_CONTAINER
    if hasattr(t, "__dict__") and hasattr(obj, "__dict__"):
        return OWNED_OBJECT
    return UNSUPPORTED_MUTABLE


def is_mutable(classification: str) -> bool:
    return classification in (OWNED_MUTABLE_CONTAINER, OWNED_OBJECT)


def canonical_value(obj: Any, classification: str, path: str) -> Any:
    """Inert token for a node. Executes nothing the observed object defines."""
    if classification == VALUE_TERMINAL:
        t = type(obj)
        if t in SAFE_VALUE_TYPES:
            return SAFE_VALUE_TYPES[t](obj)
        if isinstance(obj, enum.Enum):
            return ("enum", type(obj).__name__, obj.name)
        raise UnrepresentableState(path, obj, "value terminal without canonicaliser")
    if classification == IDENTITY_TERMINAL:
        # Name from the type, never repr(): a metaclass could define one.
        return ("code", type(obj).__name__,
                getattr(type(obj), "__name__", "?"))
    if classification == OWNED_MUTABLE_CONTAINER:
        t = type(obj)
        if t is bytearray:
            return ("bytearray", bytes(obj).hex())
        if issubclass(t, dict):
            return ("dict", t.__name__, len(obj))
        if t in (set,):
            return ("set", len(obj))
        return (t.__name__, len(obj))
    if classification == OWNED_IMMUTABLE_CONTAINER:
        return (type(obj).__name__, len(obj))
    if classification == OWNED_OBJECT:
        return ("object", type(obj).__name__, sorted(vars(obj).keys()))
    raise UnrepresentableState(path, obj)


def key_token(key: Any, path: str) -> Any:
    """A dict key as an inert token, never `repr(key)`.

    The first version turned keys into strings, which executed user `__repr__`
    and embedded heap addresses in the signature — two of the three confirmed
    defects at once. A key outside the supported value domain fails closed
    rather than being stringified.
    """
    t = type(key)
    if t in SAFE_VALUE_TYPES:
        return SAFE_VALUE_TYPES[t](key)
    if isinstance(key, enum.Enum):
        return ("enum", type(key).__name__, key.name)
    if t is tuple:
        return ("tuple", tuple(key_token(k, path) for k in key))
    if t is frozenset:
        return ("frozenset", tuple(sorted(
            (key_token(k, path) for k in key), key=lambda tok: repr(tok)
        )))
    raise UnrepresentableState(
        path, key, "unsupported dict key type — keys must be value-semantic"
    )


@dataclass
class WitnessNode:
    witness_id: int
    runtime_type: str
    classification: str
    canonical_value: Any
    mutable: bool
    tracks_identity: bool


@dataclass
class StateSnapshot:
    """What was observed. Carries no cross-snapshot relation — `root_identity`
    belongs to the comparator, since it is a statement about two captures of the
    same process rather than canonicalisable state."""
    nodes: Dict[int, WitnessNode] = field(default_factory=dict)
    edges: List[Tuple[int, Any, int]] = field(default_factory=list)
    roots: Dict[str, int] = field(default_factory=dict)
    exclusions: List[Tuple[str, str, str]] = field(default_factory=list)
    truncations: List[str] = field(default_factory=list)
    canonical_paths: Dict[int, str] = field(default_factory=dict)
    engine_id: Optional[int] = None

    def value_signature(self) -> Dict[str, Any]:
        """Value at every reachable path. Indexed by PATH, so sharing one object
        and holding equal copies give the same answer — that is what keeps this
        claim independent of the wiring."""
        out: Dict[str, Any] = {}
        children: Dict[int, List[Tuple[Any, int]]] = collections.defaultdict(list)
        for parent, label, child in self.edges:
            children[parent].append((label, child))
        stack: List[Tuple[str, int, Tuple[int, ...]]] = [
            (f"engine.{name}", wid, (self.engine_id,) if self.engine_id else ())
            for name, wid in sorted(self.roots.items())
        ]
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
            for label, child in sorted(children.get(wid, []), key=lambda t: repr(t[0])):
                stack.append((f"{path}|{label!r}", child, seen + (wid,)))
        return out

    def alias_signature(self) -> Set[Tuple[str, Any, str]]:
        """Which references share a MUTABLE object.

        Edges are kept only when both endpoints track identity, so interned
        atoms, enum members and code objects cannot manufacture topology. Nodes
        are named by canonical path because `id()` is not comparable across two
        snapshots.
        """
        keep = {wid for wid, n in self.nodes.items() if n.tracks_identity}
        return {
            (self.canonical_paths[p], label, self.canonical_paths[c])
            for p, label, c in self.edges
            if p in keep and c in keep
        }

    def root_identity(self) -> Dict[str, str]:
        return {
            name: self.canonical_paths[wid]
            for name, wid in self.roots.items()
            if self.nodes[wid].tracks_identity
        }


# Backwards-compatible alias for the earlier name.
StateWitness = StateSnapshot


@dataclass
class StateComparison:
    persistent_value_state_equal: bool
    persistent_alias_topology_equal: bool
    persistent_root_identity_equal: bool
    value_diff_paths: List[str]
    alias_only_before: List[Any]
    alias_only_after: List[Any]

    def __getitem__(self, k):          # dict-style access for test readability
        return getattr(self, k)


def _children(obj: Any, classification: str, path: str) -> List[Tuple[Any, Any]]:
    """Edge labels carry what the value token drops: dict keys and positions.

    Set members are canonicalised to inert tokens FIRST and the TOKENS are
    sorted — never the objects — so no comparison protocol of an observed object
    participates in observation.
    """
    if classification == OWNED_OBJECT:
        return [(("attr", k), v) for k, v in sorted(vars(obj).items())[:MAX_FANOUT]]
    t = type(obj)
    if issubclass(t, dict):
        return [(("key", key_token(k, path)), v)
                for k, v in list(obj.items())[:MAX_FANOUT]]
    if t in (list, tuple, collections.deque):
        return [(("idx", i), v) for i, v in enumerate(list(obj)[:MAX_FANOUT])]
    if t in (set, frozenset):
        tokens = []
        for member in obj:
            mcls = classify(member)
            if mcls == UNSUPPORTED_MUTABLE:
                raise UnrepresentableState(path, member, "unsupported set member")
            tokens.append((canonical_value(member, mcls, path), member))
        tokens.sort(key=lambda pair: repr(pair[0]))   # sorts TOKENS, not objects
        return [(("member", tok), m) for tok, m in tokens[:MAX_FANOUT]]
    if t is bytearray:
        return []
    return []


def take_witness(
    engine,
    exclusions: Optional[Dict[str, str]] = None,
    max_depth: int = MAX_DEPTH,
) -> StateSnapshot:
    exclusions = exclusions or {}
    for name, why in exclusions.items():
        if not why or len(why) < 20:
            raise ValueError(f"exclusion {name!r} needs a written justification")

    snap = StateSnapshot()
    queue: collections.deque = collections.deque()

    # The engine itself is a node, so a root-level attribute is a real EDGE.
    # Without this the identity graph is rooted but its roots are outside it:
    # `engine.b = engine.a` for a mutable would add no edge at all, and
    # root-level aliasing was only ever detected indirectly, through children.
    engine_wid = id(engine)
    snap.engine_id = engine_wid
    snap.nodes[engine_wid] = WitnessNode(
        witness_id=engine_wid, runtime_type=type(engine).__name__,
        classification=OWNED_OBJECT,
        canonical_value=("engine", type(engine).__name__),
        mutable=True, tracks_identity=True,
    )
    snap.canonical_paths[engine_wid] = "engine"

    for name, value in sorted(vars(engine).items()):
        if name in exclusions:
            snap.exclusions.append(
                (f"engine.{name}", type(value).__name__, exclusions[name])
            )
            continue
        queue.append((f"engine.{name}", value, 0, engine_wid, ("root", name)))

    visited: Set[int] = set()
    while queue:
        path, obj, depth, parent_wid, label = queue.popleft()
        wid = id(obj)

        snap.edges.append((parent_wid, label, wid))
        if parent_wid == snap.engine_id and isinstance(label, tuple) \
                and label[0] == "root":
            snap.roots[label[1]] = wid

        prev = snap.canonical_paths.get(wid)
        if prev is None or (len(path), path) < (len(prev), prev):
            snap.canonical_paths[wid] = path

        if wid in visited:
            continue                       # back-edge recorded above, not re-walked
        visited.add(wid)

        cls = classify(obj)
        if cls == UNSUPPORTED_MUTABLE:
            raise UnrepresentableState(path, obj)

        snap.nodes[wid] = WitnessNode(
            witness_id=wid, runtime_type=type(obj).__name__, classification=cls,
            canonical_value=canonical_value(obj, cls, path),
            mutable=is_mutable(cls), tracks_identity=tracks_identity(cls),
        )

        if not descends(cls):
            continue
        if depth >= max_depth:
            snap.truncations.append(path)
            continue
        for clabel, cvalue in _children(obj, cls, path):
            queue.append((f"{path}|{clabel!r}", cvalue, depth + 1, wid, clabel))

    return snap


def compare(before: StateSnapshot, after: StateSnapshot) -> StateComparison:
    bv, av = before.value_signature(), after.value_signature()
    ba, aa = before.alias_signature(), after.alias_signature()
    return StateComparison(
        persistent_value_state_equal=bv == av,
        persistent_alias_topology_equal=ba == aa,
        persistent_root_identity_equal=before.root_identity() == after.root_identity(),
        value_diff_paths=sorted(p for p in set(bv) | set(av)
                                if bv.get(p) != av.get(p))[:40],
        alias_only_before=sorted(ba - aa, key=repr)[:40],
        alias_only_after=sorted(aa - ba, key=repr)[:40],
    )
