#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — FastAPI REST Server
Full CRUD for memories, recall, stats, audit trail, forensic alerts.

Run:
    python api_server.py
    → http://localhost:8000/docs  (Swagger UI)
    → http://localhost:8000/redoc (ReDoc)

Authors: Anna Tchijova + Claude (VIGÍA AI Collective)
License: Apache 2.0
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from raven.memory_engine import AdaptiveMemoryEngine, MemoryState, LinkType, verify_audit_chain
from raven.qwen_client import MemoryAgentOrchestrator, QwenConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("raven.api")

# ============================================================
# SECURITY CONFIG — environment-driven
# ============================================================
# P0: allow_origins=["*"] meant any web page could read/write memories and
# watch live WebSocket events (CSRF + exfiltration). Origins now come from
# the environment; the wildcard requires an explicit opt-in.
#
#   RAVEN_ALLOWED_ORIGINS="https://demo.example.com,http://localhost:7860"
#   RAVEN_ALLOWED_ORIGINS="*"           # explicit opt-in, logged loudly
#   RAVEN_API_TOKEN="<secret>"          # enables auth on mutating endpoints + WS
#   RAVEN_RATE_LIMIT="30"               # requests/min per client on /recall & POST /memories

_DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:7860,http://localhost:8000"
_raw_origins = os.environ.get("RAVEN_ALLOWED_ORIGINS", _DEFAULT_ORIGINS).strip()
if _raw_origins == "*":
    ALLOWED_ORIGINS = ["*"]
    logger.warning("CORS: wildcard origins explicitly enabled via RAVEN_ALLOWED_ORIGINS=* "
                   "— do NOT run like this outside an isolated demo")
else:
    ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

API_TOKEN = os.environ.get("RAVEN_API_TOKEN", "").strip()
if not API_TOKEN:
    logger.warning("RAVEN_API_TOKEN not set — API runs in OPEN mode (local demo only). "
                   "Set a token before exposing this server to a network.")

RATE_LIMIT_PER_MIN = int(os.environ.get("RAVEN_RATE_LIMIT", "30"))


def require_token(request: Request):
    """
    P0: auth dependency for mutating endpoints. Accepts
    'Authorization: Bearer <token>' or 'X-API-Key: <token>'.
    No-op in open mode (token unset) so the local demo stays one-command.
    """
    if not API_TOKEN:
        return
    auth = request.headers.get("authorization", "")
    provided = auth[7:] if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if provided != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


class SlidingWindowLimiter:
    """
    P0: denial-of-wallet guard — every /recall can fan out to the Qwen API.
    In-memory sliding window per client IP; no external dependencies.
    Single-process only by design (matches the SQLite deployment model).
    """

    def __init__(self, max_per_min: int):
        self.max = max_per_min
        self.window = 60.0
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, client_id: str):
        now = time.monotonic()
        async with self._lock:
            q = self._hits[client_id]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.max:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit: {self.max} requests/min. Retry shortly.",
                )
            q.append(now)


rate_limiter = SlidingWindowLimiter(RATE_LIMIT_PER_MIN)


async def rate_limited(request: Request):
    # Respect X-Forwarded-For when behind a reverse proxy (nginx, Cloudflare).
    # Fall back to the TCP connection address only if the header is absent.
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client = forwarded_for.split(",")[0].strip()
    else:
        client = request.client.host if request.client else "unknown"
    await rate_limiter.check(client)

# ============================================================
# GLOBALS — initialised at startup
# ============================================================

engine: Optional[AdaptiveMemoryEngine] = None
orchestrator: Optional[MemoryAgentOrchestrator] = None
ws_clients: List[WebSocket] = []
_ws_lock = asyncio.Lock()  # P3-1: guard concurrent connect/disconnect

# Embedding configuration — read once, fed to BOTH the engine and the
# orchestrator's QwenConfig. Previously each constructed its own default
# (engine: memory_engine.EMBEDDING_DIM=384, orchestrator: QwenConfig.
# embedding_dim=384) — they happened to agree by coincidence, not by
# construction. Switching to a real Qwen embedding deployment (text-
# embedding-v3/v4, which don't support 384 — supported values are 64-2048
# in Matryoshka steps; 512 is the closest fit) meant changing one and
# forgetting the other, which fails as an opaque dimension-mismatch 502
# three layers away from the actual misconfiguration.
RAVEN_USE_LOCAL_EMBEDDINGS = os.environ.get("RAVEN_USE_LOCAL_EMBEDDINGS", "1") == "1"
RAVEN_EMBEDDING_DIM = int(os.environ.get(
    "RAVEN_EMBEDDING_DIM", "384" if RAVEN_USE_LOCAL_EMBEDDINGS else "512"
))
RAVEN_EMBEDDING_MODEL = os.environ.get("RAVEN_EMBEDDING_MODEL", "text-embedding-v3")
RAVEN_LLM_MODEL = os.environ.get("RAVEN_LLM_MODEL", "qwen-max")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, orchestrator
    logger.info(
        f"Starting raven-memory engine… "
        f"(embeddings: {'local' if RAVEN_USE_LOCAL_EMBEDDINGS else 'qwen_api'}, "
        f"dim={RAVEN_EMBEDDING_DIM})"
    )
    engine = AdaptiveMemoryEngine(embedding_dim=RAVEN_EMBEDDING_DIM)
    orchestrator = MemoryAgentOrchestrator(engine, QwenConfig(
        use_local_embeddings=RAVEN_USE_LOCAL_EMBEDDINGS,
        embedding_dim=RAVEN_EMBEDDING_DIM,
        embedding_model=RAVEN_EMBEDDING_MODEL,
        model=RAVEN_LLM_MODEL,
    ))
    logger.info(f"Engine ready. Stats: {engine.get_stats()}")
    yield
    logger.info("Shutting down raven-memory engine")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="raven-memory API",
    version="1.1.0",
    description=(
        "Adaptive Memory Substrate — stateful, graph-based memory retrieval "
        "with ternary states, neighbourhood activation, and STDP dynamics."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # P0: env-driven, no implicit wildcard
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class StoreRequest(BaseModel):
    content: str = Field(..., description="Text content to memorise")
    layer: str = Field("semantic", description="Memory layer: semantic | episodic | procedural")
    state: str = Field("NEUTRAL", description="Initial state: REINFORCED | NEUTRAL | FORGOTTEN")
    session_id: str = Field("default")
    author_id: str = Field("user")
    metadata: Dict[str, Any] = Field(default_factory=dict)


import hashlib
_recall_cache: Dict[str, Dict] = {}
_recall_cache_lock = asyncio.Lock()

def _cache_key(query: str, top_k: int, hops: int, layer_filter: Optional[str]) -> str:
    normalized = query.strip().lower()
    raw = f"{normalized}|{top_k}|{hops}|{layer_filter or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


class RecallRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(5, ge=1, le=20)
    hops: int = Field(2, ge=0, le=4)
    layer_filter: Optional[str] = None
    store_interaction: bool = Field(False, description="Store query as episodic memory")
    no_cache: bool = Field(False, description="Bypass cache, force a live Qwen call")


class CellLinkRequest(BaseModel):
    from_cell_id: int
    to_cell_id: int
    link_type: str = Field("NEUTRAL", description="RESONANT | NEUTRAL | INHIBITORY")


class StateUpdate(BaseModel):
    memory_id: str


# ============================================================
# HELPERS
# ============================================================

def _get_engine() -> AdaptiveMemoryEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return engine


def _get_orch() -> MemoryAgentOrchestrator:
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")
    return orchestrator


async def _broadcast(event: str, data: Dict):
    """Push a JSON event to all connected WebSocket clients."""
    if not ws_clients:
        return
    message = json.dumps({"event": event, "data": data, "ts": time.time()})
    # Snapshot the client list under lock, then release before doing I/O.
    async with _ws_lock:
        snapshot = list(ws_clients)

    dead = []
    for ws in snapshot:
        try:
            await asyncio.wait_for(ws.send_text(message), timeout=2.0)
        except Exception:
            dead.append(ws)

    if dead:
        async with _ws_lock:
            for ws in dead:
                if ws in ws_clients:
                    ws_clients.remove(ws)


# ============================================================
# HEALTH
# ============================================================

from fastapi.responses import FileResponse, JSONResponse

_SITE_DIR = Path(__file__).parent / "site"
_SITE_INDEX = _SITE_DIR / "index.html"
_SITE_DEMO = _SITE_DIR / "demo.html"


@app.get("/", include_in_schema=False)
async def root():
    # Serve the landing page when it ships alongside the server, so the demo
    # site and the real backend are one deployable unit (the page's API
    # examples can call this same origin). Falls back to JSON if absent.
    if _SITE_INDEX.exists():
        return FileResponse(str(_SITE_INDEX))
    return JSONResponse({"name": "raven-memory", "version": "1.1.0", "status": "ok"})


@app.get("/demo", include_in_schema=False)
async def demo():
    """Interactive demo — public, no API key needed."""
    if _SITE_DEMO.exists():
        return FileResponse(str(_SITE_DEMO))
    raise HTTPException(404, "Demo page not found")


@app.get("/api", tags=["health"])
async def api_root():
    return {"name": "raven-memory", "version": "1.1.0", "status": "ok"}


@app.get("/health", tags=["health"])
async def health():
    eng = _get_engine()
    orch = _get_orch()
    return {
        "status": "ok",
        "stats": eng.get_stats(),
        # P0: degradation is observable, not silent. "degraded": true means
        # recall is running over dummy embeddings — semantically meaningless.
        "embedding_provider": orch.embedder.provider_status(),
        "security": {
            "auth_enabled": bool(API_TOKEN),
            "cors_origins": ALLOWED_ORIGINS,
            "rate_limit_per_min": RATE_LIMIT_PER_MIN,
        },
    }


# ============================================================
# MEMORIES — CRUD
# ============================================================

@app.post("/memories", tags=["memories"], summary="Store a new memory",
          dependencies=[Depends(require_token), Depends(rate_limited)])
async def store_memory(req: StoreRequest):
    eng = _get_engine()
    orch = _get_orch()

    try:
        state = MemoryState[req.state.upper()]
    except KeyError:
        raise HTTPException(400, f"Invalid state '{req.state}'. Use REINFORCED | NEUTRAL | FORGOTTEN")

    emb = orch.embedder.embed([req.content])[0]
    # P1: validate the provider's output BEFORE it reaches the engine —
    # a misconfigured API model (e.g. 1024-dim) would otherwise surface
    # as an opaque engine ValueError deep in the call stack.
    if emb.shape != (eng.embedding_dim,):
        raise HTTPException(
            502,
            f"Embedding provider returned dim {emb.shape}, engine expects "
            f"({eng.embedding_dim},). Check embedding model configuration.",
        )

    entry = eng.store(
        content=req.content,
        embedding=emb,
        layer=req.layer,
        state=state,
        session_id=req.session_id,
        author_id=req.author_id,
        metadata=req.metadata,
    )

    result = {
        "memory_id": entry.memory_id,
        "cell_id": entry.cell_id,
        "state": entry.state.name,
        "layer": entry.layer,
        "created_at": entry.created_at,
        # P0: degradation visible at the point of storage
        "embedding_provider": orch.embedder.provider_status()["active"],
    }
    await _broadcast("memory_stored", result)
    return result


@app.get("/memories", tags=["memories"], summary="List memories with optional filters")
async def list_memories(
    layer: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    eng = _get_engine()
    mems = eng.list_memories(layer=layer, state=state, limit=limit, offset=offset)
    return {
        "count": len(mems),
        "offset": offset,
        "memories": [
            {
                "memory_id": m.memory_id,
                "cell_id": m.cell_id,
                "content_preview": m.content[:100],
                "state": m.state.name,
                "layer": m.layer,
                "author_id": m.author_id,
                "recall_count": m.recall_count,
                "last_activation": m.last_activation,
                "created_at": m.created_at,
                "metadata": m.metadata,
            }
            for m in mems
        ],
    }


@app.get("/memories/{memory_id}", tags=["memories"], summary="Get a single memory by ID")
async def get_memory(memory_id: str):
    eng = _get_engine()
    m = eng._db.load_memory(memory_id)
    if not m:
        raise HTTPException(404, f"Memory '{memory_id}' not found")
    d = m.to_dict()
    d.pop("embedding", None)  # don't send raw floats over API
    return d


@app.post("/memories/{memory_id}/reinforce", tags=["memories"], summary="Reinforce → state = REINFORCED",
          dependencies=[Depends(require_token)])
async def reinforce_memory(memory_id: str):
    eng = _get_engine()
    try:
        entry = eng.reinforce(memory_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    stats = eng.get_stats()
    result = {"memory_id": memory_id, "new_state": entry.state.name, "mss": stats["memory_stability_score"]}
    await _broadcast("memory_reinforced", result)
    return result


@app.post("/memories/{memory_id}/forget", tags=["memories"], summary="Forget → state = FORGOTTEN",
          dependencies=[Depends(require_token)])
async def forget_memory(memory_id: str):
    eng = _get_engine()
    try:
        entry = eng.forget(memory_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    stats = eng.get_stats()
    result = {"memory_id": memory_id, "new_state": entry.state.name, "mss": stats["memory_stability_score"]}
    await _broadcast("memory_forgotten", result)
    return result


# ============================================================
# RECALL
# ============================================================

@app.post("/recall", tags=["recall"], summary="Recall memories matching a query",
          dependencies=[Depends(require_token), Depends(rate_limited)])
async def recall(req: RecallRequest):
    key = _cache_key(req.query, req.top_k, req.hops, req.layer_filter)

    if not req.no_cache:
        async with _recall_cache_lock:
            cached = _recall_cache.get(key)
        if cached is not None:
            cached_copy = dict(cached)
            cached_copy["_served_from_cache"] = True
            return cached_copy

    orch = _get_orch()
    result = orch.process_message(
        user_text=req.query,
        store_as_memory=req.store_interaction,
        top_k=req.top_k,
        hops=req.hops,
        layer_filter=req.layer_filter,
    )
    result["_served_from_cache"] = False

    async with _recall_cache_lock:
        _recall_cache[key] = dict(result)

    await _broadcast("recall_executed", {"query": req.query, "hits": len(result["recalled_memories"])})
    return result


# ============================================================
# CELL LINKS
# ============================================================

@app.post("/cell-links", tags=["graph"], summary="Create a ternary cell link manually",
          dependencies=[Depends(require_token)])
async def create_cell_link(req: CellLinkRequest):
    eng = _get_engine()
    try:
        lt = LinkType[req.link_type.upper()]
    except KeyError:
        raise HTTPException(400, f"Invalid link_type '{req.link_type}'. Use RESONANT | NEUTRAL | INHIBITORY")
    eng.create_cell_link(req.from_cell_id, req.to_cell_id, lt)
    return {"from": req.from_cell_id, "to": req.to_cell_id, "type": lt.name}


@app.get("/graph", tags=["graph"], summary="Export full memory graph (nodes + edges)",
         dependencies=[Depends(require_token)])
async def export_graph():
    eng = _get_engine()
    return eng.export_graph()


# ============================================================
# STATS / AUDIT / ALERTS
# ============================================================

@app.get("/stats", tags=["analytics"])
async def get_stats():
    return _get_engine().get_stats()


@app.get("/audit", tags=["analytics"], summary="Audit trail (hash-chain, fully verified)")
async def get_audit(limit: int = 50):
    eng = _get_engine()
    # Bug #21: clamp the limit. ?limit=1000000 would load a million audit rows
    # (each carrying a memories_retrieved payload) into memory — a one-request DoS.
    limit = max(1, min(limit, 1000))
    entries = eng.get_audit_trail(limit=limit)
    # P0: full cryptographic verification — linkage (prev_hash continuity)
    # AND per-row hash recomputation from stored columns. The old check only
    # verified linkage, so an attacker could edit a row's payload and the
    # chain still read "intact". Tamper-evidence is now real: every entry's
    # hash covers query, cells, and the retrieved memories' content_hash.
    report = verify_audit_chain(entries)
    return {
        "chain_intact": report["chain_intact"],
        "hash_integrity": report["hash_integrity"],
        "issues": report["issues"],
        "count": len(entries),
        "entries": entries,
    }


@app.get("/alerts", tags=["analytics"], summary="Forensic tamper-detection alerts")
async def get_alerts(limit: int = 30):
    eng = _get_engine()
    alerts = eng.get_alerts(limit=limit)
    return {
        "count": len(alerts),
        "alerts": [
            {
                "alert_id": a.alert_id,
                "timestamp": a.timestamp,
                "memory_id": a.memory_id,
                "expected_author": a.expected_author,
                "mismatch_score": a.mismatch_score,
                "action_taken": a.action_taken,
            }
            for a in alerts
        ],
    }


# ============================================================
# WEBSOCKET — Real-time updates
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # P0: unauthenticated /ws let any page watch every memory event live.
    # Browsers can't set headers on WebSocket upgrade, so the token travels
    # as a query parameter: ws://host:8000/ws?token=<RAVEN_API_TOKEN>.
    # Open mode (no token configured) keeps the local demo frictionless.
    if API_TOKEN:
        provided = ws.query_params.get("token", "")
        if provided != API_TOKEN:
            # Bug #20: 1008 (Policy Violation) is the registered close code for
            # an auth/policy failure; 4xxx codes are non-standard.
            await ws.close(code=1008, reason="Invalid or missing token")
            logger.warning("WebSocket connection rejected: bad token")
            return

    await ws.accept()
    async with _ws_lock:                     # P3-1: registry mutations under lock
        ws_clients.append(ws)
    logger.info(f"WebSocket client connected ({len(ws_clients)} total)")
    try:
        # Send current stats on connection
        try:
            stats = _get_engine().get_stats()
        except Exception as e:
            await ws.send_text(json.dumps({"event": "error", "data": str(e), "ts": time.time()}))
            return
        await ws.send_text(json.dumps({"event": "connected", "data": stats, "ts": time.time()}))
        while True:
            # Keep alive — client can send pings
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text(json.dumps({"event": "pong", "ts": time.time()}))
            elif msg == "stats":
                await ws.send_text(json.dumps({
                    "event": "stats",
                    "data": _get_engine().get_stats(),
                    "ts": time.time(),
                }))
    except WebSocketDisconnect:
        pass
    finally:
        # P0: removal under the same lock as _broadcast — a disconnect racing
        # a broadcast could corrupt the registry or double-remove.
        async with _ws_lock:
            if ws in ws_clients:
                ws_clients.remove(ws)
        logger.info(f"WebSocket client disconnected ({len(ws_clients)} remaining)")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    _port = int(os.environ.get("RAVEN_API_PORT", "8000"))
    print("\n🦅 raven-memory API Server")
    print(f"   Swagger UI: http://localhost:{_port}/docs")
    print(f"   ReDoc:      http://localhost:{_port}/redoc")
    print(f"   WebSocket:  ws://localhost:{_port}/ws")
    print()
    uvicorn.run("api_server:app", host="0.0.0.0", port=_port, reload=False, log_level="info")
