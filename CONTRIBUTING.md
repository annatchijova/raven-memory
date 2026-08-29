# Contributing to raven-memory

## Setup

```bash
git clone https://github.com/annatchijova/raven-memory.git
cd raven-memory
pip install -r requirements-lock.txt   # reproducible light profile (no torch)
pip install -e . --no-deps
```

Optional extras: `pip install -e ".[local-embeddings]"` (real offline
embeddings, multi-GB), `".[demo]"` (Gradio), `".[mcp]"`, `".[ann]"` (hnswlib).
Everything — tests included — runs without them on deterministic dummy
embeddings.

## Tests

```bash
pytest -q                        # full suite (collected via pyproject.toml)
python tests/demo_stress_test.py # adversarial stress script (not pytest-collected)
```

CI runs both on Python 3.11/3.12 plus a Docker build on every push. A PR with
a red suite does not merge.

## Benchmarks — measure before you optimize

Any performance-motivated change must cite before/after numbers from the same
machine:

```bash
python benchmarks/perf.py --sizes 1000,5000    # latency / throughput
python benchmarks/quality.py                   # field vs. plain top-k
python benchmarks/sweep.py                     # scoring-coefficient sensitivity
```

Results live in `benchmarks/RESULTS.md` and `benchmarks/QUALITY.md` — update
them honestly, including what did *not* improve.

## Conventions that are load-bearing

Read `docs/IMPROVEMENT_PLAN.md` → "Qué NO cambiar" before proposing changes
to any of these:

- SQLite + WAL as the only storage; the audit hash chain must remain
  recomputable from stored columns alone.
- Forgetting is exclusion, never deletion.
- The rescue rule (a validated truth cannot be silenced by an unverified
  claim) and metadata-based contradiction detection (no NLI model).
- Degradation is always loud: a fallback that serves lower-quality results
  must say so in logs AND in responses/metrics.
- Schema changes go through `PRAGMA user_version` migrations in
  `MemoryStore._init_db` (numbered, idempotent), never ad-hoc ALTERs.
- Scoring coefficients are named constants in `raven/memory_engine.py`;
  new ones should come with a `benchmarks/sweep.py` sensitivity entry.

## Regenerating the lockfile

```bash
python -m venv /tmp/lockenv
/tmp/lockenv/bin/pip install numpy scipy scikit-learn fastapi \
    "uvicorn[standard]" pydantic requests pytest httpx
/tmp/lockenv/bin/pip freeze   # → paste under the header of requirements-lock.txt
```
