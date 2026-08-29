# Changelog

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
