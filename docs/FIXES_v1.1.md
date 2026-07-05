# raven-memory v1.1 — Hardening Pass

Maps every applied fix to its source: the 55-bug internal audit (P0/P1/P2)
and the external review of `spectral.py` (2026-06). Suites: **17/17 engine,
11/11 spectral, 7/7 API smoke, E2E consolidation verified.**

## spectral.py → v2.5 (schema unchanged, ENGINE_VERSION stays pca_v2.4)

| Source | Fix |
|---|---|
| Review P1-A | Legacy fields loaded with reconstructed (zero) mean now set `requires_rebuild=True`; `is_stale()` forces rebuild, `summary()` warns. A PCA over the wrong centre no longer degrades silently. |
| Review P1-B | `build_from_memories()` validates per-embedding shape + finiteness before `vstack` — clear `TensorValidationError` instead of an opaque numpy broadcast error. |
| Review P1-C | `project()` returns a zero vector on non-finite projections — NaN never reaches resonance scores. |
| Review (observability) | Cross-process determinism test reports **PASS / WARN / FAIL** as distinct states; the suite counts WARNs separately. A failed child can no longer masquerade as green. |
| Audit P1 #8 | `k >= 1` invariant explicit in mode truncation. |
| Audit P1 #9 | Cross-process child also verifies the **JSON round-trip** (the persistence path production actually uses). |
| Audit P2 #12 | `build_and_persist_spectral_field()` catches store errors — a persistence failure no longer destroys a valid in-memory field. |
| New tests | `test_build_dimension_validation`, `test_projection_finite_guard`, extended backward-compat assertions. |

## memory_engine.py

| Source | Fix |
|---|---|
| Audit P0 #7 | **Real tamper-evidence.** `compute_audit_hash()` hashes the full retrieved payload including each memory's `content_hash`, with a single captured timestamp. Every input is a stored column → `verify_audit_chain()` recomputes each hash from the DB row alone. Editing audited content now breaks verification (test 16). |
| Audit P0 #1 | SQLite **WAL mode** + `synchronous=NORMAL` at init; every connection gets `busy_timeout=5000`. Concurrent API server + consolidator no longer race into corruption. |
| Audit P0 #4 | STDP pruning uses `<= STDP_PRUNE_EPS` (1e-9) instead of `== 0.0` — dead synaptic links are actually pruned (progressive leak closed). |
| Audit P0 #5 | `final_score` clamped to `>= 0` (anti-correlated embeddings produced negative scores). |
| Audit P0 #6 | `.tolist()` guard — `query_embedding` accepted as ndarray or list. |
| Audit P0 (BLAS) | `spectral` imported **before** numpy/scipy — BLAS env vars land in time; late-import RuntimeWarning eliminated for all entry points. |
| Audit P1 #8 | `_linked_pairs` cache (hydrated from DB on load) deduplicates auto-INHIBITORY links — O(n²) redundant writes per hot topic eliminated. |
| Audit P1 #9 | Stylometry split into ES/EN function-word sets with dominant-language detection. The forensic check **skips cross-language comparisons**: a bilingual author is not a tamperer (test 17). Same-language style change still detected (stress test, mismatch 0.836). |
| Audit P1 #11 | `MAX_HOP_SEARCH` named constant. |
| Audit P1 #12 | `retention_ratio` added to stats — exposes the denominator MSS silently drops (1 REINFORCED + 99 FORGOTTEN no longer reads as a perfectly stable system). |
| Audit P1 #5 (api) | `get_audit_trail` orders by `id DESC` (true chain sequence), not timestamp. |
| Robustness | `rebuild_spectral_field()` catches `TensorValidationError` from corrupt rows. |

## sleep_consolidator.py

| Source | Fix |
|---|---|
| New P0 (found this pass) | **Atomic consolidation.** Insert of merged node + delete of sources + cell-link cascade now run in ONE `BEGIN IMMEDIATE` transaction (`apply_consolidation`). A crash mid-merge previously left node AND sources alive. Non-atomic legacy helpers removed. |
| Audit P0 #3 | `_verify_chain_tail()` checks linkage of the last 25 entries before appending — chaining onto a broken chain is loudly reported, never laundered. |
| Audit P0 #7 (shared) | Consolidation entries use the same canonical `compute_audit_hash` — verifiable by `verify_audit_chain()` alongside recall entries (E2E verified). |
| Audit P1 #5 | Empty extractive summary falls back to the longest source content. |
| Audit P0 #1 (shared) | All connections via `_connect()` with busy timeout. |

## api_server.py

| Source | Fix |
|---|---|
| Audit P0 (CORS) | Origins from `RAVEN_ALLOWED_ORIGINS` (default: localhost only). Wildcard requires explicit opt-in and is logged loudly. Methods/headers narrowed. |
| Audit P0 (WS auth) | `RAVEN_API_TOKEN` enables auth: Bearer/X-API-Key on mutating endpoints, `?token=` on `/ws` (close 4401 on mismatch). Open mode preserved for the one-command local demo, with a startup warning. |
| Audit P0 #3 | Sliding-window rate limiter (`RAVEN_RATE_LIMIT`, default 30/min per IP) on `/recall` and `POST /memories` — denial-of-wallet on the Qwen API closed. |
| Audit P0 #7 | `/audit` performs full cryptographic verification: linkage **and** per-row hash recomputation (`hash_integrity` + `issues` in the response). |
| Audit P1 #4 | Embedding dimension validated post-embed → clean 502 instead of a deep engine ValueError. |
| Observability | `/health` exposes `embedding_provider` (incl. `degraded`) and the active security configuration. |
| Race | WS register/unregister under the same lock as `_broadcast`. |

## qwen_client.py

| Source | Fix |
|---|---|
| Audit P0 (qwen #3) | **Dummy fallback is no longer silent.** Provider tracks its active tier, counts degradations, logs a loud throttled `SEMANTIC QUALITY DEGRADED` warning, and exposes `provider_status()` — surfaced in `/health`, `/memories`, and every `process_message` result. |
| Audit P0 (qwen #2) | Conversation history sanitized: only `user`/`assistant` dicts with string content reach the LLM. `role="system"` smuggled through stored state is dropped (prompt-injection vector closed). |
| Audit P1 #4 | `engine.recall()` failures degrade to an answer without memory + `recall_error` field — never a raw 500. |
| Audit P1 #5 | Context budget: 400 chars/memory, 6000 chars/block, explicit truncation marker. |
| Audit P2 #15 | `_conversation_history` capped at 20 entries. |
| Reliability | Qwen embedding API: 3 attempts with exponential backoff before declaring the tier dead. |

## run_all.py / demos / requirements

| Source | Fix |
|---|---|
| Audit P1 (run_all) | Batch phases run with a 900 s hard timeout. |
| Audit P1 (demo_killer #1) | `share=True` removed — public gradio.live tunnelling requires `RAVEN_GRADIO_SHARE=1`. |
| Audit P1 (demo_killer #2) | MSS history mutated through `_record_mss()` under a lock; render uses a consistent snapshot. |
| Audit P2 (stress) | Temporary DB (+ WAL/SHM sidecars) cleaned up at the end; chain status now reports real verification, not entry count. |
| Stress test | Tamper simulation rewritten as same-language style change (the previous cross-language version exercised the bilingual false positive, not genuine detection). |
| Audit P2 #18 | `torch` documented as transitive (sentence-transformers) — removable together. |

## Architectural note — single vs. separate spectral fields

The review raised whether episodic and semantic memories should share one
PCA field or get separate ones. Decision for v1.x: **single field, kept
deliberately.** Sleep consolidation continuously migrates episodic clusters
into semantic nodes; a separate episodic field would lose exactly the
cross-layer resonance that makes consolidation meaningful — the consolidated
semantic structure *should* shape how new episodic queries resonate. The
cavity is one. If semantic dominance becomes measurable as the corpus grows,
stratified fields are v3 scope (alongside Procrustes alignment and energy
weighting, already deferred).

---

## v1.1 — Second hardening pass (external review triage)

Three review documents were triaged against the real source. Many findings
referenced code that doesn't exist in this tree (imagined `_ensure_kdtree`
recomputation, O(n²) KDTree query, missing rollback) — **false positives,
not applied**. The Chinese-translation review was also a false positive: the
`zh` dictionary was already complete (verified programmatically, 0 missing
keys). The genuine findings, verified and fixed:

| Source | Fix | Test |
|---|---|---|
| #24 scalability (P0) | `get_stats()` deserialized the whole table on every recall/broadcast/dashboard tick. New `count_stats()` aggregates in SQL. | smoke |
| #2/#5 | `reinforce()` never restored a FORGOTTEN cell to the KDTree — reinforced memory stayed invisible. Now re-registers + marks dirty. | test 18 |
| #4 | `_points` was a list growing monotonically with dead-cell holes. Migrated to a sparse dict + monotonic id allocator. | test 19 |
| #15 | Consolidation clustering built an n×n cosine matrix + O(n³) agglomerative. Added greedy O(n·k) fallback above 2000 memories. | test 20 |
| (latent) | `_load_from_db` didn't reset in-memory state on reload — a post-consolidation reload referenced deleted cells and crashed the KDTree rebuild. Exposed by test 19; now resets all rebuildable structures. | test 19 |
| #6 | `_cosine_sim` clamped to [-1, 1] so float rounding can't push a score above its state-boost ceiling. | — |
| #19 | `_broadcast` sends with a per-client 2 s timeout — one slow WS client can't stall the HTTP request. | — |
| #20 | WebSocket auth close code 4401 → standard 1008 (Policy Violation). | — |
| #21 | `/audit?limit=` clamped to ≤1000 (one-request DoS). | — |
| #22 | `export_graph(max_nodes=1000)` caps nodes + edges, returns `truncated`/`total_nodes`. | — |

### Website honesty (the most damaging finding if left)

- The page claimed "runs entirely in your browser / faithful miniature of the
  real engine." Rewritten (EN + ZH) to "interactive simulation — same rules as
  the Python engine," with an explicit pointer to the real backend.
- The demo audit chain now hashes each retrieved memory's **content hash**
  inside the payload, mirroring `compute_audit_hash`. Tampering edits the
  memory content; recomputation then fails — verified headless.
- Audit verdict re-translates on language switch.
- `api_server` now serves `site/index.html` at `/` and the JSON root at
  `/api`, so the page and the real backend ship as one deployable unit.

### Deployment

- `Dockerfile` + `DEPLOY.md`: one-command judge run on deterministic local
  embeddings (no key, no model download), plus production paths for
  sentence-transformers and the Qwen API, and the security env vars.

**Suites:** 20/20 engine · 11/11 spectral · stress + API smoke green.

---

## v1.1 — Third pass: real Qwen Cloud embeddings wiring

Triggered by switching the live test/recording session to real Qwen embeddings
(text-embedding-v3, 512-dim — 384 is not a supported Matryoshka cut for that
model; supported values are the 64–2048 step series). Found while wiring it:

| Bug | Real impact | Fix |
|---|---|---|
| Fallback chain was never 3-tier | `embed()` only tried the Qwen API when `use_local_embeddings=False` from the start. If local was *enabled* but unavailable (package missing, or — newly — a dimension mismatch), it skipped straight to dummy even with a valid API key configured. | API tier now tried whenever local didn't produce a result, not only when local was disabled by config. |
| Dummy fallback ignored `embedding_dim` | `_dummy_embed(t)` always used its hardcoded default (384), regardless of what dimension the engine actually expected. | Both call sites now pass `dim=self.config.embedding_dim`. |
| Local tier could silently mismatch | `all-MiniLM-L6-v2` always outputs 384-dim vectors. If `embedding_dim` was set to anything else (e.g. 512 for real Qwen), local would still "succeed" and hand back 384-dim vectors. | `_init_local` now checks the native dim against config and refuses to activate local on mismatch — falls through instead of silently misreporting `dim={config.embedding_dim}` in its own log line. |
| Qwen embedding API never requested a dimension | `_embed_api`'s request body had no `dimensions` field, so text-embedding-v3/v4 returned their native default (1024) regardless of `config.embedding_dim` — guaranteed mismatch the moment a non-default dimension was configured. | `dimensions: config.embedding_dim` now sent for any `v3`/`v4` model. |
| No validation at the source | A dimension mismatch from the API surfaced only as a generic 502 three layers away (api_server.py), with no indication of *why*. | `_embed_api` now validates response shape immediately and raises a message naming the model, the dimension it returned, and the dimension expected. |
| Engine and orchestrator dimension could silently diverge | `api_server.py` built `AdaptiveMemoryEngine()` (always 384, the module default) and `QwenConfig()` (independently defaulted to 384) — they only agreed by coincidence, not by construction. | Both now read from one set of env vars (`RAVEN_USE_LOCAL_EMBEDDINGS`, `RAVEN_EMBEDDING_DIM`, `RAVEN_EMBEDDING_MODEL`, `RAVEN_LLM_MODEL`) and are constructed from the same resolved values — structurally can't drift apart. |

Verified with `unittest.mock` (no real network — the sandbox has no egress to
`aliyuncs.com`): real-Qwen-512 path requests the right dimension and returns
correctly-shaped vectors; local-enabled-but-incompatible correctly falls through
to the API tier instead of dummy; an API dimension mismatch correctly falls
through to a dummy vector of the *right* shape with full degradation telemetry.
Default mode (local, 384) unchanged — 20/20 engine, 11/11 spectral still green.
