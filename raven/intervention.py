"""
raven.intervention — causal probes over the retrieval graph.

Runners built ON TOP of AdaptiveMemoryEngine.intervene(). Nothing here touches
the engine's internals: the primitive is read-only with respect to the field,
and these are just search strategies over which cells to silence.

Scope of the claim (docs/INTERVENTION_DESIGN.md §1): everything measured here is
**retrieval causal influence** — the effect of a silencing on RAVEN's own
deterministic recall. It says nothing about a downstream agent's answer, which
depends on a model this module never runs.

Two things make the search worth more than one-cell-at-a-time probing:

  suppress(A)      → no change
  suppress(B)      → no change
  suppress(A,B)    → X disappears        ⇒ redundant paths to X

  suppress(A)      → X disappears
  suppress(B)      → X disappears        ⇒ two necessary dependencies

and the invariant fuzzer, which points the same machinery at the rescue rule.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from raven.memory_engine import (
    AdaptiveMemoryEngine,
    ExclusionReason,
    InterventionResult,
    InterventionSpec,
    MemoryState,
)


@dataclass
class Ablation:
    """One probe: the silenced population and what it did to retrieval."""
    targets: Tuple[str, ...]
    result: InterventionResult

    @property
    def changed(self) -> bool:
        return self.result.changed

    def dropped(self, memory_id: str) -> bool:
        """True when `memory_id` was retrieved at baseline and is gone under
        the intervention. Membership is judged on the FULL scored set, so
        falling out of the top-k does not count as being dropped."""
        b = {r.memory.memory_id for r in self.result.baseline_scored}
        p = {r.memory.memory_id for r in self.result.perturbed_scored}
        return memory_id in b and memory_id not in p

    def rank_shift(self, memory_id: str) -> Optional[int]:
        """Signed rank change over the full scored sets (positive = demoted).
        None when the memory is absent from either side."""
        b = {r.memory.memory_id: i for i, r in enumerate(self.result.baseline_scored)}
        p = {r.memory.memory_id: i for i, r in enumerate(self.result.perturbed_scored)}
        if memory_id not in b or memory_id not in p:
            return None
        return p[memory_id] - b[memory_id]

    def moved(self, memory_id: str, min_rank_shift: int = 1) -> bool:
        """Disappearance OR displacement by at least `min_rank_shift` places.

        Disappearance alone is a coarse criterion in this engine: the k-NN
        graph is symmetrized in _rebuild_kdtree(), so every active cell keeps K
        edges and a memory is rarely cut off entirely. Real influence usually
        shows up as demotion, not deletion — measured here, not assumed.
        """
        if self.dropped(memory_id):
            return True
        shift = self.rank_shift(memory_id)
        return shift is not None and abs(shift) >= min_rank_shift


def _probe(engine, query_embedding, targets, **kw) -> Ablation:
    spec = InterventionSpec.suppress(tuple(targets))
    return Ablation(targets=tuple(targets),
                    result=engine.intervene(query_embedding, spec, **kw))


# ----------------------------------------------------------
# Ablation strategies
# ----------------------------------------------------------

def single_cell_ablation(
    engine: AdaptiveMemoryEngine,
    query_embedding: np.ndarray,
    memory_ids: Sequence[str],
    **kw,
) -> List[Ablation]:
    """Silence each candidate on its own."""
    return [_probe(engine, query_embedding, (m,), **kw) for m in memory_ids]


def pairwise_ablation(
    engine: AdaptiveMemoryEngine,
    query_embedding: np.ndarray,
    memory_ids: Sequence[str],
    **kw,
) -> List[Ablation]:
    """Silence every unordered pair. Quadratic by construction — pass a
    shortlist (e.g. the top-N activated), not the whole field."""
    return [
        _probe(engine, query_embedding, pair, **kw)
        for pair in itertools.combinations(memory_ids, 2)
    ]


def random_subset_ablation(
    engine: AdaptiveMemoryEngine,
    query_embedding: np.ndarray,
    memory_ids: Sequence[str],
    size: int,
    trials: int = 20,
    seed: int = 42,
    **kw,
) -> List[Ablation]:
    """Sample `trials` distinct subsets of a fixed size. Seeded, so a run is
    reproducible and a surprising subset can be replayed exactly."""
    pool = list(memory_ids)
    if size > len(pool):
        raise ValueError(f"subset size {size} exceeds pool of {len(pool)}")
    rng = random.Random(seed)
    seen, out = set(), []
    for _ in range(trials):
        subset = tuple(sorted(rng.sample(pool, size)))
        if subset in seen:
            continue
        seen.add(subset)
        out.append(_probe(engine, query_embedding, subset, **kw))
    return out


# ----------------------------------------------------------
# Taxonomy
# ----------------------------------------------------------

REDUNDANT_PATHS      = "REDUNDANT_PATHS"
SINGLE_NECESSARY     = "SINGLE_NECESSARY"
MULTIPLE_NECESSARY   = "MULTIPLE_NECESSARY"
NO_DEPENDENCE        = "NO_DEPENDENCE"


def classify_dependence(
    watched: str,
    singles: Sequence[Ablation],
    combos: Sequence[Ablation] = (),
    min_rank_shift: Optional[int] = None,
) -> Dict:
    """
    Read a dependence structure out of single and combined ablations.

    REDUNDANT_PATHS is the finding single-cell probing cannot produce: no
    individual silencing moves `watched`, but silencing a set does.

    `min_rank_shift=None` counts only outright disappearance. Pass an integer
    to count demotion too — on a densely connected field that is usually the
    criterion that carries the signal (see Ablation.moved).
    """
    def hit(a: Ablation) -> bool:
        return a.dropped(watched) if min_rank_shift is None \
            else a.moved(watched, min_rank_shift)

    killers = [a.targets for a in singles if hit(a)]
    combo_killers = [a.targets for a in combos if hit(a)]

    if not killers and combo_killers:
        verdict = REDUNDANT_PATHS
    elif len(killers) == 1:
        verdict = SINGLE_NECESSARY
    elif len(killers) > 1:
        verdict = MULTIPLE_NECESSARY
    else:
        verdict = NO_DEPENDENCE

    return {
        "watched": watched,
        "verdict": verdict,
        "single_killers": [list(t) for t in killers],
        "combo_killers": [list(t) for t in combo_killers],
    }


# ----------------------------------------------------------
# Invariant fuzzing — the rescue rule
# ----------------------------------------------------------

def fuzz_rescue_rule(
    engine: AdaptiveMemoryEngine,
    query_embedding: np.ndarray,
    subset_size: int = 2,
    trials: int = 25,
    seed: int = 42,
    **kw,
) -> Dict:
    """
    Fault injection against:

        a validated truth cannot be silenced by an unverified claim

    The naive oracle ("a REINFORCED memory disappeared") is WRONG, and would
    report false violations. A suppression may legitimately remove a REINFORCED
    memory — by silencing the only path that reached it. The rescue rule
    promises protection against *inhibition by an unvalidated claim*, not
    topological immortality.

    So the falsifiable form is narrower: silencing only NON-REINFORCED cells,
    never the target's own cell, must never cause a REINFORCED memory to leave
    the result set **via the inhibition path**. Any other exit is legal and is
    recorded separately rather than counted as a violation.

    Returns the violations plus the legal exits, so a run that finds nothing
    still shows what it actually exercised.
    """
    reinforced = [
        m.memory_id
        for m in engine.list_memories(limit=100000)
        if m.state == MemoryState.REINFORCED
    ]
    pool = [
        m.memory_id
        for m in engine.list_memories(limit=100000)
        if m.state == MemoryState.NEUTRAL
    ]
    if not reinforced or len(pool) < subset_size:
        return {
            "trials": 0,
            "violations": [],
            "legal_exits": [],
            "note": "insufficient field: need ≥1 REINFORCED memory and "
                    f"≥{subset_size} NEUTRAL cells to suppress",
        }

    probes = random_subset_ablation(
        engine, query_embedding, pool, size=subset_size,
        trials=trials, seed=seed, **kw,
    )

    violations: List[Dict] = []
    legal_exits: List[Dict] = []
    for ab in probes:
        for mem_id in reinforced:
            if not ab.dropped(mem_id):
                continue
            reason = engine.absence_reason(ab.result.perturbed_exclusions, mem_id)
            record = {
                "suppressed": list(ab.targets),
                "reinforced_memory": mem_id,
                "reason": reason,
            }
            if reason == ExclusionReason.INHIBITED:
                violations.append(record)
            else:
                # UNREACHABLE / STATE_FILTER / LAYER_FILTER — real mechanisms,
                # none of which the rescue rule ever claimed to cover.
                legal_exits.append(record)

    return {
        "trials": len(probes),
        "violations": violations,
        "legal_exits": legal_exits,
        "reinforced_watched": reinforced,
    }
