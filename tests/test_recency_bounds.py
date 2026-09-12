#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — the recency term is bounded, including under clock skew.

Regression tests for a defect found by the intervention work's adversarial pass
(docs/INTERVENTION_DESIGN.md §13, finding B). The defect predates that work:
reproduced identically on e539df8 (= main).

    recency_bonus = RECENCY_WEIGHT · exp(−ln2 · age / RECENCY_HALFLIFE)

With `age < 0` — a last_activation in the future — the exponential GROWS
instead of decaying, without bound. The failure is not the eventual
OverflowError at roughly +2.8 years; it is everything before it, where the
engine keeps returning well-formed, serialisable, confidently wrong rankings.

These tests state the PROPERTY, deliberately not the numbers the buggy code
happens to produce:

    age_effective = max(0, now − last_activation)
    0 ≤ recency_bonus ≤ RECENCY_WEIGHT           for every possible timestamp
    a future timestamp scores exactly like `now`, never better

"Activity in the future" cannot mean "more recent than now": the most recent a
memory can be is now, and the term's ceiling is its value at age = 0.

Run: pytest tests/test_recency_bounds.py -q
"""

import hashlib
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raven.memory_engine import (
    AdaptiveMemoryEngine,
    RECENCY_HALFLIFE,
    RECENCY_WEIGHT,
)
from raven.portability import export_field, import_field

DAY = 86400.0
SKEWS = [
    ("+1 second", 1.0),
    ("+1 hour", 3600.0),
    ("+1 day", DAY),
    ("+30 days", 30 * DAY),
    ("+1 year", 365 * DAY),
    ("+3 years", 3 * 365 * DAY),      # past the OverflowError threshold
]


def make_emb(text: str, dim: int = 384) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(dim).astype(np.float32)
    e /= np.linalg.norm(e) + 1e-10
    return e


def near(base: str, k: int, jitter: float = 0.2) -> np.ndarray:
    e = make_emb(base) + jitter * make_emb(f"{base}::{k}")
    return (e / np.linalg.norm(e)).astype(np.float32)


def build(tmp_path, name="rec.db", n=6) -> AdaptiveMemoryEngine:
    eng = AdaptiveMemoryEngine(db_path=tmp_path / name)
    for i in range(n):
        eng.store(f"document number {i} with body text", near("cluster", i))
    eng.recall(near("cluster", 0))          # populate last_activation
    return eng


def set_all_activations(engine, value: float):
    with sqlite3.connect(engine._db.db_path) as conn:
        conn.execute("UPDATE memories SET last_activation = ?", (value,))
        conn.commit()


def reopen(engine) -> AdaptiveMemoryEngine:
    return AdaptiveMemoryEngine(db_path=engine._db.db_path)


# ============================================================
# The ceiling
# ============================================================

@pytest.mark.parametrize("label,skew", SKEWS)
def test_future_activation_never_exceeds_the_recency_ceiling(tmp_path, label, skew):
    """0 ≤ recency_bonus ≤ RECENCY_WEIGHT, for any timestamp whatsoever."""
    eng = build(tmp_path, f"ceil_{int(skew)}.db")
    set_all_activations(eng, time.time() + skew)
    eng = reopen(eng)

    results, _ = eng.recall(near("cluster", 0), top_k=10)
    assert results
    for r in results:
        assert 0.0 <= r.recency_bonus <= RECENCY_WEIGHT, (
            f"{label}: recency_bonus={r.recency_bonus!r} escaped "
            f"[0, {RECENCY_WEIGHT}] — a future timestamp cannot be more "
            f"recent than now"
        )


@pytest.mark.parametrize("label,skew", SKEWS)
def test_future_activation_scores_exactly_like_now(tmp_path, label, skew):
    """Every future timestamp collapses to the same effective age of zero, so
    +1 second and +3 years must be indistinguishable from `now`."""
    ref = build(tmp_path, "ref.db")
    now = time.time()
    set_all_activations(ref, now)
    ref = reopen(ref)
    baseline = {r.memory.content: r.recency_bonus
                for r in ref.recall(near("cluster", 0), top_k=10)[0]}

    eng = build(tmp_path, f"fut_{int(skew)}.db")
    set_all_activations(eng, now + skew)
    eng = reopen(eng)
    observed = {r.memory.content: r.recency_bonus
                for r in eng.recall(near("cluster", 0), top_k=10)[0]}

    assert observed.keys() == baseline.keys()
    for k in baseline:
        assert observed[k] == pytest.approx(baseline[k], abs=1e-6), (
            f"{label}: {k!r} scored differently from an activation at now"
        )


def test_extreme_future_activation_does_not_raise(tmp_path):
    """Past ~2.8 years of skew the exponential overflowed outright. Crashing is
    not the fix; it is simply the most visible symptom."""
    eng = build(tmp_path, "extreme.db")
    set_all_activations(eng, time.time() + 50 * 365 * DAY)
    eng = reopen(eng)
    results, _ = eng.recall(near("cluster", 0), top_k=10)
    assert results
    assert all(np.isfinite(r.final_score) for r in results)


def test_future_timestamp_confers_no_ranking_advantage(tmp_path):
    """The ordering claim, not just the term: a memory stamped in the future
    must not outrank one activated at `now`."""
    eng = build(tmp_path, "rank.db", n=8)
    mems = sorted(eng.list_memories(limit=100), key=lambda m: m.cell_id)
    future_ids = {m.memory_id for m in mems[::2]}

    t0 = time.time()
    with sqlite3.connect(eng._db.db_path) as conn:
        for m in mems:
            stamp = t0 + 30 * DAY if m.memory_id in future_ids else t0
            conn.execute("UPDATE memories SET last_activation=? WHERE memory_id=?",
                         (stamp, m.memory_id))
        conn.commit()
    eng = reopen(eng)

    results, _ = eng.recall(near("cluster", 0), top_k=20)
    elapsed = time.time() - t0

    by_id = {r.memory.memory_id: r for r in results}
    fut = [by_id[i].recency_bonus for i in future_ids if i in by_id]
    pres = [r.recency_bonus for r in results if r.memory.memory_id not in future_ids]
    assert fut and pres

    # The clamp makes a future stamp mean exactly "now". A memory stamped at t0
    # is then older than "now" by however long this test took to reach the
    # scorer, so a residual gap is expected — but ONLY as much as that elapsed
    # wall-clock explains. Anything beyond it is the future timestamp buying
    # rank, which is the defect. (The bug produced a gap of order 1e7.)
    max_explainable = RECENCY_WEIGHT * (
        1.0 - math.exp(-math.log(2) * elapsed / RECENCY_HALFLIFE)
    )
    gap = max(fut) - max(pres)
    assert gap <= max_explainable + 1e-12, (
        f"future timestamp gained {gap:.3e} of recency bonus; elapsed "
        f"wall-clock ({elapsed:.3f}s) explains at most {max_explainable:.3e}"
    )
    assert max(fut) <= RECENCY_WEIGHT


# ============================================================
# The public path that makes this a product defect
# ============================================================

def test_imported_field_with_future_activation_stays_bounded(tmp_path):
    """import_field() copies last_activation verbatim — which is correct for
    portability. Interpreting it safely is the scorer's job, not the
    importer's, so the field must import faithfully AND score sanely."""
    src = build(tmp_path, "src.db")
    skewed = time.time() + 90 * DAY
    set_all_activations(src, skewed)

    out = tmp_path / "field.jsonl"
    export_field(src._db.db_path, out)
    dst_path = tmp_path / "imported.db"
    report = import_field(out, dst_path)
    assert report["chain_intact"] and report["hash_integrity"]

    dst = AdaptiveMemoryEngine(db_path=dst_path)

    # Fidelity: the importer preserved the timestamp as recorded.
    assert all(abs(m.last_activation - skewed) < 1.0
               for m in dst.list_memories(limit=100))

    # Safety: the scorer refuses to turn it into unbounded score.
    results, _ = dst.recall(near("cluster", 0), top_k=10)
    assert results
    for r in results:
        assert 0.0 <= r.recency_bonus <= RECENCY_WEIGHT
        assert r.final_score <= 10.0, (
            f"imported field produced final_score={r.final_score!r} — a clock "
            f"ahead on the exporting machine silently corrupted the ranking"
        )
