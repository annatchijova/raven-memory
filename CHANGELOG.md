# Changelog

## Unreleased

### Added — causal intervention probes (docs/INTERVENTION_DESIGN.md)
- **`AdaptiveMemoryEngine.intervene(query, InterventionSpec)`** — measures the
  *retrieval causal influence* of silencing a set of cells: runs the scoring
  core twice (baseline and `do(suppress)`) against one field snapshot, under one
  lock, sharing one timestamp, and returns a decomposed delta. The claim is
  scoped to RAVEN's retrieval, never to a downstream agent's answer.
- **`_recall_core()`** — the scoring pipeline extracted as a pure function.
  `recall()` is now that core plus its effects (STDP, activation timestamps,
  alerts, audit). A probe runs the same core and discards the effects, so
  "the probe does not mutate the field" is structural rather than a convention.
- **Per-memory exclusion provenance** (`ExclusionReason`, `absence_reason()`) —
  direct suppression, inhibition, unreachability and state/layer filtering are
  now distinguishable. The engine can answer why something did *not* surface,
  not only why something did.
- **`raven/intervention.py`** — single, pairwise and seeded-random subset
  ablation, a dependence taxonomy, and targeted fuzzing of the rescue rule
  ("a validated truth cannot be silenced by an unverified claim") whose oracle
  separates inhibition from legitimate topological exits.
- **Schema v4** — `audit_log.intervention`. A probe seals its *resolved* target
  population (memory_id → cell_id) and both branches' outcome into the hash
  chain under the operation `recall_intervention`.

### Changed
- Recall results are now ordered by `(-final_score, memory_id)`. Ties
  previously fell back to insertion order, which is unstable on the >999-cell
  chunked load path — a rank delta could have been a sorting artifact.

### Notes — red-team pass (findings RT-1 / RT-2)
- **RT-1 (documented, tested):** `fuzz_rescue_rule()`'s violation branch is
  *unreachable* under current engine semantics — the rescue loop removes
  REINFORCED cells from `inhibited_cells` before any INHIBITED exclusion is
  written. The docstring now says so; the branch is kept as a tripwire, with
  its discriminating power pinned by a counterfactual stubbed-engine test
  (`tests/test_fuzz_oracle.py`).
- **RT-2 (reviewed, kept as design):** an empty `targets` tuple is the null
  probe and stays allowed — the witness purity tests rely on it. Its cost (a
  sealed no-op audit row) is documented at the validation site.
- Unknown intervention modes, stages or fields raise `InterventionError`. A
  silently ignored intervention would yield delta = 0, which is
  indistinguishable from the genuine finding "no causal influence".

### Fixed
- **Unbounded recency bonus under clock skew.** `recency_bonus` is
  `RECENCY_WEIGHT · exp(−ln2 · age / 24h)`; with `last_activation` in the
  future the age went negative and the exponential grew without bound instead
  of decaying (+30 days scored ~5.4e7, ~+2.8 years raised `OverflowError`).
  Age is now clamped at zero: activity in the future cannot mean more recent
  than now, so the term's ceiling is its value at age 0. Reachable without
  touching the database — `import_field()` preserves `last_activation`
  verbatim, so a field exported from a machine whose clock ran ahead silently
  produced corrupted ranking with no degraded flag. Predates the intervention
  work (reproduced identically on 1.2.0); found by its adversarial pass.
  Clock skew is made harmless, not yet diagnosed — see
  docs/INTERVENTION_DESIGN.md §13.

### Compatibility
- Audit rows without an intervention omit the key from the hashed payload
  entirely (never `null`), so every pre-v4 chain keeps verifying.
- `raven/portability.py` exports/imports the new column; a round-tripped field
  still verifies.

## 1.2.0 — 2026-08-29

Full execution of [the improvement plan](docs/IMPROVEMENT_PLAN.md) (Fases 0–4).

### Fixed (critical)
- **Spectral module never loaded** in any real entry point (package-relative
  import bug) — resonance/coherence were silently dead in production. Now
  live end-to-end; absence logs a WARNING.
- **API event loop blocking**: embedding + Qwen calls (30 s timeout) ran
  inline in `async` endpoints, freezing the whole server. Offloaded to the
  threadpool; the engine gained an internal RLock.
- **Stale `/recall` cache**: never invalidated on mutations, unbounded.
  Now generation-keyed, TTL-bounded, LRU-capped; degraded results uncached.
- **Cross-user state leak**: all HTTP clients shared one conversation and
  one STDP turn history. Now isolated per `session_id`.
- **Unauthenticated reads**: with a token set, `/memories`, `/stats`,
  `/audit`, `/alerts` were open. Now token-gated; constant-time compares;
  `X-Forwarded-For` honoured only behind a declared proxy.
- **MCP env-var mismatch**: server read `QWEN_API_KEY` while all docs say
  `DASHSCOPE_API_KEY` → silent dummy embeddings. Both accepted, canonical
  name first.
- **Stylometric forensics**: single-sample author profile replaced by a
  rolling per-(author, language) mean with a minimum-sample gate;
  auto-quarantine (FORGOTTEN) is now opt-in (`RAVEN_STYLO_ENFORCE=1`),
  default is alert-only. Alert dedup per memory.
- Stale pre-reorganization imports in the test suite and stress test.

### Added
- `pyproject.toml` packaging with extras and console scripts (`raven-api`,
  `raven-mcp`, `raven-consolidate`, `raven-export`, `raven-import`).
- Versioned SQLite schema (`PRAGMA user_version`, migrations v1→v3);
  newer-schema DBs are refused loudly.
- GitHub Actions CI (Python 3.11/3.12, lockfile install, full suite,
  stress test, Docker build) + `requirements-lock.txt`.
- **Hot consolidation**: `engine.consolidate()`, `POST /consolidate`, MCP
  `raven_consolidate` — no restart needed.
- `GET /metrics` (Prometheus text format, zero new dependencies).
- Optional **hnsw ANN backend** (`RAVEN_ANN_BACKEND=hnsw`,
  `raven-memory[ann]`): first recall at 5k memories 8.5 s → 0.9 s.
- **Field export/import** (`raven-export` / `raven-import`, JSONL) with
  honest audit-chain verification on import.
- MCP: `raven_verify_chain`, topic filter on `raven_recall`.
- **Benchmarks**: `benchmarks/perf.py` (latency/throughput),
  `benchmarks/quality.py` (field vs. plain top-k with ground truth),
  `benchmarks/sweep.py` (hyperparameter sensitivity) + results docs.

### Changed
- Audit entries store the query embedding's SHA-256 instead of the raw
  vector: 11.2 KB → 2.8 KB per recall; legacy rows still verify.
- Recall hot path: BFS-recorded hop distances, in-memory link index,
  batched post-recall writes — p50 at 5k memories 128 ms → 24 ms.
- `demo_killer.py` → `demo/gradio_demo.py`.
- Scoring coefficients `RESONANT_BOOST` and `SYNAPTIC_SCORE_WEIGHT` are now
  named constants (as the README already promised).

## 1.1.0

Security & hardening release — 55-finding audit resolved. See
[docs/FIXES_v1.1.md](docs/FIXES_v1.1.md).

## 1.0.0

Initial hackathon release (Qwen Cloud Hackathon, Track 1: MemoryAgent).
