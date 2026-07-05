# Deploying raven-memory

Three ways to run it, fastest first. All three serve the bilingual landing
page at `/` and the JSON API at `/recall`, `/memories`, `/audit`, `/health`,
plus interactive docs at `/docs`.

## What's real vs. simulated (read this first)

- The **Python engine** (`memory_engine.py`, `spectral.py`,
  `sleep_consolidator.py`) is the real system: ternary states, STDP, Voronoi
  k-NN graph, SVD spectral field, SHA-256 audit chain, atomic consolidation.
- The **landing page** (`site/index.html`) is an *interactive simulation* of
  the engine's rules, for intuition. It does not call the backend. The page
  says so. The real engine is what the API and the demo video exercise.

## 1. Docker (one command, reproducible)

```bash
docker build -t raven-memory .
docker run -p 8000:8000 raven-memory
```

Open http://localhost:8000 — the landing page. API docs at
http://localhost:8000/docs. This uses deterministic local embeddings, so it
needs no API key and no model download; recall geometry is exercised end to
end, and `/health` reports `embedding_provider.degraded: true` honestly.

## 2. Local Python (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install fastapi "uvicorn[standard]" pydantic numpy scipy scikit-learn requests
uvicorn api_server:app --port 8000
```

Run the test suites to verify the install:

```bash
python test_suite.py        # 20/20 engine integration tests
python spectral.py          # 11/11 spectral tests (incl. cross-process determinism)
python demo_stress_test.py  # end-to-end scenario + audit-chain verification
```

## 3. Real Qwen embeddings (production-quality recall)

Dummy embeddings are deterministic but semantically empty. For real recall,
pick one tier:

**Local sentence-transformers** (offline, ~80 MB model download):

```bash
pip install sentence-transformers torch
# QwenConfig(use_local_embeddings=True) — the default path
```

**Qwen Cloud API** (best quality):

```bash
export DASHSCOPE_API_KEY="your-key"
# QwenConfig(use_local_embeddings=False) → text-embedding-v3, with retries
```

The provider falls back local → Qwen API → deterministic dummy, and the dummy
tier is never silent: it logs `SEMANTIC QUALITY DEGRADED` and reports
`degraded: true` in `/health` and every response. **For a live demo, prefer
local sentence-transformers** so an external API outage can't derail you.

## Security (before exposing to a network)

Defaults are demo-safe (localhost only, open auth). To harden:

```bash
export RAVEN_API_TOKEN="a-long-secret"                 # auth on writes + WS
export RAVEN_ALLOWED_ORIGINS="https://your.domain"     # CORS allow-list
export RAVEN_RATE_LIMIT="30"                            # requests/min per IP
```

Mutating endpoints then require `Authorization: Bearer <token>` (or
`X-API-Key`); the WebSocket needs `?token=<token>`. Without a token the server
logs a loud OPEN-mode warning at startup.

Note: the rate limiter is in-process. Run a single worker (`--workers 1`, the
default) or put a shared limiter (e.g. a reverse proxy) in front for multi-worker.

## Sleep consolidation

Consolidation is an offline pass — run it on a schedule (cron, 3 a.m.) or
on demand:

```bash
python sleep_consolidator.py --db raven_memory.db --threshold 0.83
```

It clusters redundant episodic memories, fuses each cluster into one semantic
node in a single atomic transaction, logs the operation into the same audit
chain, and the engine rebuilds the spectral field over the new space on its
next start.
