#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Red-team follow-up (finding RT-1): pin the discriminating power of
fuzz_rescue_rule()'s violation oracle.

Under real engine semantics the violation branch is unreachable — the rescue
loop removes REINFORCED cells from inhibited_cells before any INHIBITED
exclusion is written (see the vacuity note in raven/intervention.py). That
makes a bare "violations == []" result unfalsifiable: it would stay green even
if the wiring were dead.

So this test runs the fuzzer against a STUBBED engine that counterfactually
produces the one exit the oracle counts as a violation, and asserts the
fuzzer records it. The tripwire is proven live; the vacuity of the real
engine is proven separately (and is a property of the engine, not of this
fuzzer).

Run: pytest tests/test_fuzz_oracle.py -q
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raven.intervention import fuzz_rescue_rule
from raven.memory_engine import ExclusionReason, MemoryState


def _mem(mid, state):
    return types.SimpleNamespace(memory_id=mid, state=state)


class _StubResult:
    """Minimal stand-in for InterventionResult (only what Ablation reads)."""

    def __init__(self, perturbed_exclusions):
        self.baseline_scored = [types.SimpleNamespace(memory=_mem("R_truth", None))]
        self.perturbed_scored = []  # the watched memory dropped under treatment
        self.perturbed_exclusions = perturbed_exclusions


class _StubEngine:
    """Counterfactual engine: every probe drops the REINFORCED memory with
    the exit reason it was configured to produce."""

    def __init__(self, reason):
        self._reason = reason
        self._reinforced = _mem("R_truth", MemoryState.REINFORCED)
        self._neutrals = [_mem(f"N{i}", MemoryState.NEUTRAL) for i in range(4)]

    def list_memories(self, limit=100000):
        return [self._reinforced] + self._neutrals

    def intervene(self, query_embedding, spec, **kw):
        exclusions = {"R_truth": self._reason} if self._reason else {}
        # raw result only — random_subset_ablation wraps it in Ablation itself
        return _StubResult(exclusions)

    def absence_reason(self, exclusions, memory_id):
        return exclusions.get(memory_id, ExclusionReason.UNREACHABLE)


def test_oracle_records_violation_when_inhibited_exit_occurs():
    """Counterfactual: if a REINFORCED memory ever exited via INHIBITED, the
    fuzzer MUST report a violation. Proves the tripwire is wired."""
    out = fuzz_rescue_rule(_StubEngine(ExclusionReason.INHIBITED),
                           np.zeros(4), subset_size=2, trials=5, seed=1)
    assert out["trials"] > 0
    assert len(out["violations"]) > 0, "oracle failed to record a violation it must count"
    assert all(v["reason"] == ExclusionReason.INHIBITED for v in out["violations"])
    assert out["legal_exits"] == []


def test_oracle_counts_other_exits_as_legal():
    """The same machinery, with any other exit, must land in legal_exits —
    the narrow rescue-rule promise, not topological immortality."""
    out = fuzz_rescue_rule(_StubEngine(ExclusionReason.UNREACHABLE),
                           np.zeros(4), subset_size=2, trials=5, seed=1)
    assert out["violations"] == []
    assert len(out["legal_exits"]) > 0
    assert all(e["reason"] == ExclusionReason.UNREACHABLE for e in out["legal_exits"])
