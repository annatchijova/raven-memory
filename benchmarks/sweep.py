#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — Hyperparameter sensitivity sweep (plan 4.3).

Varies the scoring coefficients that were, until now, chosen by intuition
(HOP_LAMBDA, RECENCY_WEIGHT, RESONANT_BOOST) and measures their effect with
the quality harness. The point is not to auto-tune — it is to know which
knobs matter and document why each default survives (or should change).

Run:  python benchmarks/sweep.py [--seed 42]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import raven.memory_engine as me  # noqa: E402
from quality import run  # noqa: E402  (benchmarks/ on path via __file__)

DEFAULTS = {
    "HOP_LAMBDA": me.HOP_LAMBDA,
    "RECENCY_WEIGHT": me.RECENCY_WEIGHT,
    "RESONANT_BOOST": me.RESONANT_BOOST,
}

GRID = {
    "HOP_LAMBDA": [0.0, 0.15, 0.5, 1.0],
    "RECENCY_WEIGHT": [0.0, 0.05, 0.3],
    "RESONANT_BOOST": [0.0, 0.5, 1.5],
}


def one_at_a_time(seed: int):
    """Vary each knob alone, others at default — sensitivity, not search."""
    rows = []
    for name, values in GRID.items():
        for value in values:
            for k, v in DEFAULTS.items():
                setattr(me, k, v)
            setattr(me, name, value)
            m = run(seed=seed)
            rows.append({
                "param": name,
                "value": value,
                "is_default": value == DEFAULTS[name],
                "recall5_field": m["recall5_field"],
                "val_first_field": m["val_first_field"],
                "n_pairs": m["n_pairs"],
            })
    for k, v in DEFAULTS.items():   # restore
        setattr(me, k, v)
    return rows


def main():
    parser = argparse.ArgumentParser(description="raven-memory hyperparameter sweep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = one_at_a_time(args.seed)
    print(f"\n🦅 sensitivity sweep (one-at-a-time, seed={args.seed}) — "
          f"defaults marked with *\n")
    print(f"{'param':<16} {'value':>7}  {'recall@5':>9}  {'validated-first':>15}")
    print("-" * 52)
    for r in rows:
        mark = "*" if r["is_default"] else " "
        print(f"{r['param']:<16} {r['value']:>6}{mark}  "
              f"{r['recall5_field']:>9.3f}  "
              f"{r['val_first_field']:>12}/{r['n_pairs']}")
    print()


if __name__ == "__main__":
    main()
