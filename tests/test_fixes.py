#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — Regression tests for the Fase 0 fixes
(docs/IMPROVEMENT_PLAN.md items 0.1–0.7).

Run: pytest tests/test_fixes.py -q
"""

import asyncio
import hashlib
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raven.memory_engine import (
    AdaptiveMemoryEngine,
    AuthorStyleProfile,
    _SPECTRAL_AVAILABLE,
)
from raven.qwen_client import (
    MemoryAgentOrchestrator,
    QwenConfig,
    resolve_dashscope_api_key,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_emb(text: str, dim: int = 384) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(dim).astype(np.float32)
    e /= np.linalg.norm(e) + 1e-10
    return e


@pytest.fixture
def engine(tmp_path):
    return AdaptiveMemoryEngine(db_path=tmp_path / "raven_test.db")


# ============================================================
# 0.1 — spectral must actually load
# ============================================================

def test_spectral_module_is_importable():
    """The bare `from spectral import` left the whole module dead (plan 0.1)."""
    assert _SPECTRAL_AVAILABLE, (
        "spectral must import via the raven package — if this fails, "
        "resonance/coherence are silently disabled again"
    )


def test_recall_reports_nonzero_resonance(engine):
    rng = np.random.default_rng(7)
    for i in range(6):
        e = rng.standard_normal(384).astype(np.float32)
        e /= np.linalg.norm(e)
        engine.store(f"memoria spectral {i}", e)
    assert engine.rebuild_spectral_field() is True
    q = rng.standard_normal(384).astype(np.float32)
    q /= np.linalg.norm(q)
    results, _ = engine.recall(q, top_k=3)
    assert results, "recall must return candidates"
    assert any(r.resonance_score != 0.0 for r in results), (
        "spectral field is built — resonance must be real metadata, not 0.0"
    )


def test_consolidator_runs_as_module_and_script():
    """Both invocation styles must resolve the raven package (plan 0.1)."""
    for cmd in (
        [sys.executable, "-m", "raven.sleep_consolidator", "--help"],
        [sys.executable, str(REPO_ROOT / "raven" / "sleep_consolidator.py"), "--help"],
    ):
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, timeout=60)
        assert proc.returncode == 0, f"{cmd}: {proc.stderr.decode()[:500]}"


# ============================================================
# 0.4 — per-session conversation isolation
# ============================================================

def _offline_orchestrator(tmp_path) -> MemoryAgentOrchestrator:
    eng = AdaptiveMemoryEngine(db_path=tmp_path / "raven_orch.db")
    cfg = QwenConfig(api_key="", use_local_embeddings=False)
    return MemoryAgentOrchestrator(eng, cfg)


def test_sessions_do_not_share_conversation_history(tmp_path):
    orch = _offline_orchestrator(tmp_path)
    orch.process_message("dato secreto de ana", session_id="ana", store_as_memory=False)
    orch.process_message("consulta de bruno", session_id="bruno", store_as_memory=False)

    ana = orch._session("ana").conversation_history
    bruno = orch._session("bruno").conversation_history
    assert all("bruno" not in m["content"] for m in ana)
    assert all("secreto" not in m["content"] for m in bruno)


def test_reset_conversation_clears_one_session(tmp_path):
    orch = _offline_orchestrator(tmp_path)
    orch.process_message("hola", session_id="a", store_as_memory=False)
    orch.process_message("hola", session_id="b", store_as_memory=False)
    orch.reset_conversation("a")
    assert "a" not in orch._sessions
    assert "b" in orch._sessions


def test_store_uses_the_caller_session_id(tmp_path):
    orch = _offline_orchestrator(tmp_path)
    orch.process_message("recuerda esto", session_id="mi-sesion", store_as_memory=True)
    mems = orch.engine.list_memories(limit=10)
    assert any(m.session_id == "mi-sesion" for m in mems)


# ============================================================
# 0.6 — API key resolution (DASHSCOPE_API_KEY canonical)
# ============================================================

def test_api_key_resolution(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    assert resolve_dashscope_api_key() == ""

    monkeypatch.setenv("QWEN_API_KEY", "legacy")
    assert resolve_dashscope_api_key() == "legacy"

    monkeypatch.setenv("DASHSCOPE_API_KEY", "canonical")
    assert resolve_dashscope_api_key() == "canonical", "canonical name must win"


# ============================================================
# 0.7 — rolling stylometric profile, alert-only by default
# ============================================================

LONG_ES = [
    "El sistema guarda los recuerdos en celdas espaciales y cada celda "
    "tiene un estado ternario que modula toda la recuperación semántica.",
    "La memoria adaptativa conecta los conceptos mediante enlaces y los "
    "estados ternarios deciden qué recuerdos aparecen en cada consulta.",
    "Cada consulta activa una vecindad de celdas y el motor propaga la "
    "señal mediante saltos que van decayendo con la distancia recorrida.",
    "Los enlaces inhibitorios silencian las contradicciones mientras que "
    "los resonantes amplifican los recuerdos relacionados entre ellos.",
]

TAMPERED = (
    "BUY NOW!!! cheap pills... click; here; NOW; limited!! offer!!! "
    "amazing; deal; wow!!! free; money; guaranteed!! results!!! today!!"
)


def test_no_false_positive_on_varied_same_author_texts(engine):
    base = make_emb("estilo_base")
    for i, text in enumerate(LONG_ES):
        engine.store(text, make_emb(f"estilo_{i}"), author_id="anna")
    results, audit = engine.recall(base, top_k=10, hops=3)
    assert audit.filtered_by_estilometria == 0
    assert engine.get_alerts() == [], "varied but genuine texts must not alert"


def test_min_samples_gate_before_any_verdict(engine):
    """One sample is an anecdote: no profile of size 1 may produce alerts."""
    engine.store(LONG_ES[0], make_emb("solo_1"), author_id="nueva")
    engine.store(TAMPERED, make_emb("solo_2"), author_id="nueva")
    engine.recall(make_emb("solo_1"), top_k=10, hops=3)
    # With < STYLO_MIN_SAMPLES per (author, language) profile there is no
    # baseline to compare against — TAMPERED is English, the ES profile has
    # one sample: neither side reaches the threshold count.
    assert engine.get_alerts() == []


def test_mismatch_alerts_but_does_not_forget_by_default(engine):
    for i, text in enumerate(LONG_ES):
        engine.store(text, make_emb(f"perfil_{i}"), author_id="anna")
    # Same author_id and same detected language (all-Spanish function words),
    # but a disjoint function-word set, telegraphic sentences and aggressive
    # punctuation — a radical break from the profile's flowing prose.
    impostor_text = (
        "sí!!! también;; ya;; ni!!! sin;; sobre!!! entre;; hasta!!! desde;; "
        "todo!!! sí!!! también;; ya;; ni!!! sin;; sobre!!! entre;; hasta!!! "
        "desde;; todo!!!"
    )
    m_bad = engine.store(impostor_text, make_emb("impostor"), author_id="anna")
    engine.recall(make_emb("impostor"), top_k=10, hops=3)

    alerts = engine.get_alerts()
    assert alerts, "a radical style break within the same language must alert"
    assert alerts[0].action_taken == "ALERT_ONLY"
    reloaded = engine._db.load_memory(m_bad.memory_id)
    assert reloaded.state.name != "FORGOTTEN", (
        "without RAVEN_STYLO_ENFORCE=1, recall() must never destroy state"
    )


def test_profile_mean_is_average_of_samples():
    from raven.memory_engine import StylometricExtractor
    ext = StylometricExtractor()
    profile = AuthorStyleProfile()
    for text in LONG_ES:
        profile.add(ext.extract(text, "anna"))
    mean = profile.mean_fingerprint()
    assert profile.count == len(LONG_ES)
    assert mean.language == "es"
    assert 0 < mean.avg_sentence_length < 40


# ============================================================
# 1.4 — versioned schema migrations
# ============================================================

def test_schema_versioning_migrates_legacy_db(tmp_path):
    """A pre-versioning DB (user_version=0, no recall_count) upgrades in place."""
    import sqlite3
    from raven.memory_engine import MemoryStore, SCHEMA_VERSION

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE memories (
            memory_id TEXT PRIMARY KEY, layer TEXT NOT NULL,
            content TEXT NOT NULL, content_hash TEXT NOT NULL,
            embedding BLOB NOT NULL, state TEXT NOT NULL,
            cell_id INTEGER NOT NULL, created_at REAL NOT NULL,
            session_id TEXT NOT NULL, author_id TEXT NOT NULL,
            metadata TEXT, synaptic_links TEXT,
            last_activation REAL DEFAULT 0.0, fingerprint TEXT
        )
    """)
    conn.commit()
    conn.close()

    MemoryStore(db)   # opening runs the migrations

    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)")]
    assert "recall_count" in cols, "v2 migration must add the column"
    indexes = [r[1] for r in conn.execute("PRAGMA index_list(memories)")]
    assert "uq_mem_cell_id" in indexes, "v2 migration must enforce UNIQUE(cell_id)"
    conn.close()


def test_newer_schema_refuses_to_open(tmp_path):
    """Old code must fail loudly on a DB from a newer version, not run over it."""
    import sqlite3
    from raven.memory_engine import MemoryStore

    db = tmp_path / "future.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="upgrade raven-memory"):
        MemoryStore(db)


# ============================================================
# API server — 0.2 event loop, 0.3 cache, 0.5 auth
# ============================================================

def _fresh_api(tmp_path, token: str = ""):
    """(Re)import api_server with a controlled environment."""
    os.environ["RAVEN_DB_PATH"] = str(tmp_path / "raven_api.db")
    os.environ["RAVEN_API_TOKEN"] = token
    os.environ.pop("DASHSCOPE_API_KEY", None)
    os.environ.pop("QWEN_API_KEY", None)
    import api_server
    return importlib.reload(api_server)


def _fake_result():
    return {
        "qwen_response": {"role": "assistant", "content": "ok"},
        "recalled_memories": [],
        "audit_log": None,
        "stats": {},
        "turn_memory_ids": [],
        "session_id": "default",
        "embedding_provider": {"degraded": False, "active": "test"},
        "recall_error": None,
        "latency_ms": 1,
    }


def _lifespan_client(api):
    import httpx
    transport = httpx.ASGITransport(app=api.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_recall_cache_hits_and_invalidates(tmp_path):
    api = _fresh_api(tmp_path)

    calls = {"n": 0}

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            def fake_process(**kwargs):
                calls["n"] += 1
                return _fake_result()
            api.orchestrator.process_message = fake_process

            async with _lifespan_client(api) as client:
                r1 = await client.post("/recall", json={"query": "hola"})
                r2 = await client.post("/recall", json={"query": "hola"})
                assert r1.json()["_served_from_cache"] is False
                assert r2.json()["_served_from_cache"] is True
                assert calls["n"] == 1, "second identical query must hit the cache"

                # Any field mutation must invalidate every cached recall.
                await client.post("/cell-links", json={
                    "from_cell_id": 1, "to_cell_id": 2, "link_type": "NEUTRAL",
                })
                r3 = await client.post("/recall", json={"query": "hola"})
                assert r3.json()["_served_from_cache"] is False
                assert calls["n"] == 2, "mutation must bust the cache"

                # Different sessions must never share cache entries.
                await client.post("/recall", json={"query": "hola", "session_id": "otra"})
                assert calls["n"] == 3

    asyncio.run(scenario())


def test_degraded_results_are_not_cached(tmp_path):
    api = _fresh_api(tmp_path)

    calls = {"n": 0}

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            def fake_process(**kwargs):
                calls["n"] += 1
                result = _fake_result()
                result["embedding_provider"]["degraded"] = True
                return result
            api.orchestrator.process_message = fake_process

            async with _lifespan_client(api) as client:
                await client.post("/recall", json={"query": "hola"})
                await client.post("/recall", json={"query": "hola"})
                assert calls["n"] == 2, "dummy-embedding noise must never be cached"

    asyncio.run(scenario())


def test_slow_recall_does_not_block_event_loop(tmp_path):
    api = _fresh_api(tmp_path)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            def slow_process(**kwargs):
                time.sleep(1.0)   # simulates the Qwen API round-trip
                return _fake_result()
            api.orchestrator.process_message = slow_process

            async with _lifespan_client(api) as client:
                recall_task = asyncio.create_task(
                    client.post("/recall", json={"query": "lenta", "no_cache": True})
                )
                await asyncio.sleep(0.1)   # let the recall start blocking
                t0 = time.monotonic()
                health = await client.get("/health")
                elapsed = time.monotonic() - t0
                await recall_task
                assert health.status_code == 200
                assert elapsed < 0.5, (
                    f"/health took {elapsed:.2f}s while /recall was in flight — "
                    "the blocking work is not offloaded from the event loop"
                )

    asyncio.run(scenario())


def test_reads_require_token_when_auth_enabled(tmp_path):
    api = _fresh_api(tmp_path, token="secreto123")

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            async with _lifespan_client(api) as client:
                for path in ("/memories", "/stats", "/audit", "/alerts", "/graph"):
                    resp = await client.get(path)
                    assert resp.status_code == 401, f"{path} must require the token"
                    resp = await client.get(
                        path, headers={"Authorization": "Bearer secreto123"})
                    assert resp.status_code == 200, f"{path} must accept the token"

                # /health stays public but must not leak configuration.
                health = (await client.get("/health")).json()
                assert health["status"] == "ok"
                assert "cors_origins" not in health.get("security", {})
                assert "stats" not in health

    asyncio.run(scenario())
    # Restore open mode for any test that imports api_server afterwards.
    _fresh_api(tmp_path, token="")
