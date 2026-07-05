#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — Qwen Client & Agent Orchestrator
Dual embedding provider (local sentence-transformers or Qwen API)
+ LLM orchestration through Qwen Cloud.

Set DASHSCOPE_API_KEY to enable Qwen Cloud.
Without it the system runs fully offline with deterministic embeddings.

Authors: Anna Tchijova + Claude (VIGÍA AI Collective)
License: Apache 2.0
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("raven.client")

# ============================================================
# CONFIG
# ============================================================

@dataclass
class QwenConfig:
    api_key: str = field(
        default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", "")
    )
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "QWEN_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    )
    model: str = "qwen-max"
    embedding_model: str = "text-embedding-v3"
    use_local_embeddings: bool = True   # prefer local (offline, fast)
    embedding_dim: int = 384
    max_tokens: int = 1024
    temperature: float = 0.7
    request_timeout: int = 30


# ============================================================
# EMBEDDING PROVIDER
# ============================================================

class EmbeddingProvider:
    """
    Embedding provider with three-tier fallback:
      1. Local sentence-transformers (fast, offline, preferred)
      2. Qwen text-embedding-v3 via API (high quality, requires key)
      3. Deterministic SHA-256-seeded dummy (always works, for testing)

    P0: the fallback to dummy is no longer silent. Dummy vectors are random
    projections — semantically meaningless — so every recall over them is
    noise dressed up as memory. The provider tracks its active tier, counts
    degradations, and exposes provider_status() so the API and the
    orchestrator can surface the degradation instead of hiding it.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    LOCAL_MODEL_DIM = 384    # native output dim of MODEL_NAME — fixed, not configurable
    API_RETRIES = 2          # total attempts = 1 + API_RETRIES
    API_BACKOFF_BASE = 0.5   # seconds, exponential

    def __init__(self, config: QwenConfig):
        self.config = config
        self._local: Any = None
        self._use_local = config.use_local_embeddings
        # P0: degradation telemetry
        self.active_provider: str = "uninitialised"
        self.dummy_fallbacks: int = 0
        self._dummy_warned: bool = False
        if self._use_local:
            self._init_local()

    def _init_local(self):
        # The local model has a fixed native dimension. If the engine was
        # configured for a different embedding_dim (e.g. 512, to match a real
        # Qwen text-embedding-v3/v4 deployment), silently returning 384-dim
        # local vectors would be the exact same class of bug we closed in the
        # dummy tier: a fallback that quietly serves the wrong shape. Refuse
        # to activate local in that case and fall through to the next tier.
        if self.config.embedding_dim != self.LOCAL_MODEL_DIM:
            logger.warning(
                f"Local model {self.MODEL_NAME} outputs {self.LOCAL_MODEL_DIM}-dim "
                f"vectors but engine is configured for {self.config.embedding_dim}-dim "
                f"embeddings — skipping local tier (would silently mismatch). "
                f"Falling through to Qwen API / dummy."
            )
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._local = SentenceTransformer(self.MODEL_NAME)
            logger.info(f"Local embeddings: {self.MODEL_NAME} (dim={self.LOCAL_MODEL_DIM})")
        except ImportError:
            logger.warning("sentence-transformers not installed → using deterministic dummy embeddings")
        except Exception as e:
            logger.warning(f"Failed to load local model ({e}) → using deterministic dummy embeddings")

    def provider_status(self) -> Dict[str, Any]:
        """Degradation telemetry for /health and orchestrator output."""
        return {
            "preferred": "local" if self._use_local else "qwen_api",
            "active": self.active_provider,
            "dummy_fallbacks": self.dummy_fallbacks,
            "degraded": self.active_provider == "dummy",
        }

    def _alert_dummy(self, count: int):
        """P0: loud, throttled alert — semantic quality is compromised."""
        self.active_provider = "dummy"
        self.dummy_fallbacks += count
        if not self._dummy_warned:
            self._dummy_warned = True
            logger.warning(
                "SEMANTIC QUALITY DEGRADED: falling back to deterministic dummy "
                "embeddings. Vectors are random projections — recall results are "
                "NOT semantically meaningful. Install sentence-transformers or "
                "set DASHSCOPE_API_KEY. (further fallbacks counted silently; "
                "see provider_status())"
            )

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of texts. Returns list of float32 unit-norm vectors."""
        if not texts:
            return []

        if self._use_local and self._local is not None:
            return self._embed_local(texts)

        # P0: try the API tier whenever local didn't produce a result — not
        # only when use_local_embeddings was False from the start. Previously
        # this was gated on `not self._use_local`, so if local was *enabled*
        # but unavailable (package missing, or a configured embedding_dim
        # that doesn't match the local model's fixed native dimension), the
        # chain skipped straight to dummy even with a valid API key sitting
        # right there. "Three-tier fallback" was only ever two reachable tiers.
        if self.config.api_key:
            embs = self._embed_api(texts)
            if embs:
                self.active_provider = "qwen_api"
                return embs

        self._alert_dummy(len(texts))
        return [self._dummy_embed(t, dim=self.config.embedding_dim) for t in texts]

    def _embed_local(self, texts: List[str]) -> List[np.ndarray]:
        try:
            vecs = self._local.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.active_provider = "local"
            return [v.astype(np.float32) for v in vecs]
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            self._alert_dummy(len(texts))
            return [self._dummy_embed(t, dim=self.config.embedding_dim) for t in texts]

    def _embed_api(self, texts: List[str]) -> List[np.ndarray]:
        """P0: retry with exponential backoff before declaring the tier dead."""
        import requests  # type: ignore
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.config.embedding_model,
            "input": texts,
            "encoding_format": "float",
        }
        # text-embedding-v3/v4 are Matryoshka models: they support a
        # `dimensions` parameter and otherwise return their native default
        # (1024), which silently mismatches whatever embedding_dim the engine
        # was built with. Request the engine's dimension explicitly so the
        # two can never drift apart. Harmless to omit for models that ignore
        # the field, but only sent for models known to support it.
        if any(v in self.config.embedding_model for v in ("v3", "v4")):
            body["dimensions"] = self.config.embedding_dim
        last_err: Optional[Exception] = None
        for attempt in range(1 + self.API_RETRIES):
            try:
                resp = requests.post(
                    f"{self.config.base_url}/embeddings",
                    headers=headers,
                    json=body,
                    timeout=self.config.request_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                items = sorted(data["data"], key=lambda x: x["index"])
                vecs = [np.array(item["embedding"], dtype=np.float32) for item in items]
                # Catch a dimension mismatch at the source, with a message
                # that names the actual cause, rather than letting a
                # wrong-shaped vector propagate to where it's harder to debug
                # (a generic 502 three layers up in api_server.py).
                bad = [v.shape[0] for v in vecs if v.shape != (self.config.embedding_dim,)]
                if bad:
                    raise ValueError(
                        f"{self.config.embedding_model} returned {bad[0]}-dim vectors, "
                        f"engine expects {self.config.embedding_dim}-dim. The model may "
                        f"not support the requested `dimensions` value — check the "
                        f"model's supported dimension list."
                    )
                return vecs
            except Exception as e:
                last_err = e
                if attempt < self.API_RETRIES:
                    delay = self.API_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        f"Qwen embedding API attempt {attempt + 1} failed ({e}); "
                        f"retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
        logger.error(f"Qwen embedding API failed after {1 + self.API_RETRIES} attempts: {last_err}")
        return []

    @staticmethod
    def _dummy_embed(text: str, dim: int = 384) -> np.ndarray:
        """
        Deterministic, reproducible embedding from SHA-256.
        Same text always produces the same vector across sessions
        (unlike Python's built-in hash() which is randomised by PYTHONHASHSEED).
        """
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal(dim).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-10
        return emb


# ============================================================
# QWEN LLM CLIENT
# ============================================================

class QwenLLMClient:
    """Thin wrapper around Qwen Cloud Chat Completions API."""

    SYSTEM_PROMPT = """You are a memory-augmented AI assistant backed by raven-memory —
a stateful, graph-based memory field with ternary states and neighbourhood activation.

When answering:
- Prioritise [REINFORCED] memories (user-validated, high confidence).
- If memories conflict, acknowledge both perspectives explicitly.
- Keep responses concise and grounded in the provided memory context.
- If context is absent or sparse, say so rather than hallucinating.

Memory state legend:
  [REINFORCED] ×1.5 — user-validated truth
  [NEUTRAL]    ×1.0 — standard memory
"""

    def __init__(self, config: QwenConfig):
        self.config = config

    def complete(
        self,
        user_text: str,
        context_block: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """Run a single LLM turn with memory context injected into the prompt."""
        # P0: sanitize history — an entry smuggled in with role="system"
        # (or any non-dialogue role) would override or compete with the
        # system prompt: classic prompt injection through stored state.
        # Only user/assistant turns with string content survive.
        messages: List[Dict] = [
            {"role": m["role"], "content": m["content"]}
            for m in (conversation_history or [])
            if isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
        ]
        content = f"{context_block}\n\nUser: {user_text}" if context_block else user_text
        messages.append({"role": "user", "content": content})

        if not self.config.api_key:
            return self._offline_response(user_text, context_block)

        try:
            import requests  # type: ignore
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.config.model,
                "messages": [{"role": "system", "content": self.SYSTEM_PROMPT}] + messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            resp = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Qwen LLM API error: {e}")
            return self._offline_response(user_text, context_block)

    @staticmethod
    def _offline_response(user_text: str, context: str) -> str:
        preview = context[:300].replace("\n", " ") if context else "no memories"
        return (
            f"[OFFLINE MODE — set DASHSCOPE_API_KEY for Qwen responses]\n\n"
            f"Query: {user_text}\n\n"
            f"Memory context available:\n{preview}…"
        )


# ============================================================
# ORCHESTRATOR
# ============================================================

class MemoryAgentOrchestrator:
    """
    Main entry point for agent interactions.

    Coordinates:
      embedding → recall → LLM → (optionally store) → return
    """

    def __init__(self, engine: Any, config: QwenConfig):
        self.engine = engine
        self.config = config
        self.embedder = EmbeddingProvider(config)
        self.llm = QwenLLMClient(config)
        # Rolling window of memory IDs activated in recent turns (for STDP)
        self._turn_history: List[str] = []
        self._conversation_history: List[Dict] = []

    def process_message(
        self,
        user_text: str,
        store_as_memory: bool = True,
        top_k: int = 5,
        hops: int = 2,
        layer_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: embed → recall → LLM → optionally store → return structured result."""
        t0 = time.time()

        # 1. Embed query
        query_emb = self.embedder.embed([user_text])[0]

        # 2. Recall from memory field
        # P1: an engine failure (locked DB, corrupt row) must degrade to an
        # answer without memory, not take the whole request down with a 500.
        try:
            recall_results, audit = self.engine.recall(
                query_embedding=query_emb,
                query_text=user_text,
                top_k=top_k,
                hops=hops,
                layer_filter=layer_filter,
                current_turn_memories=self._turn_history[-20:] if self._turn_history else None,
            )
            recall_error = None
        except Exception as e:
            logger.error(f"engine.recall() failed: {e}")
            recall_results, audit = [], None
            recall_error = str(e)

        # 3. Update turn history for STDP
        current_ids = [r.memory.memory_id for r in recall_results]
        self._turn_history.extend(current_ids)
        self._turn_history = self._turn_history[-40:]  # cap at 40

        # 4. Build context block for LLM
        context_block = self._build_context(recall_results)

        # 5. LLM completion
        llm_response = self.llm.complete(
            user_text=user_text,
            context_block=context_block,
            conversation_history=self._conversation_history[-6:],  # last 3 turns
        )

        # 6. Update conversation history
        self._conversation_history.append({"role": "user", "content": user_text})
        self._conversation_history.append({"role": "assistant", "content": llm_response})
        # P2: unbounded growth — only a window is ever sent to the LLM,
        # but the list itself leaked memory over long sessions.
        self._conversation_history = self._conversation_history[-20:]

        # 7. Optionally store the interaction
        if store_as_memory:
            self.engine.store(
                content=user_text,
                embedding=query_emb,
                layer="episodic",
                session_id="agent",
                metadata={"type": "user_query", "response_preview": llm_response[:80]},
            )

        # 8. Format output
        recalled_formatted = [
            {
                "memory_id": r.memory.memory_id,
                "content_preview": r.memory.content[:120],
                "state": r.memory.state.name,
                "state_boost": r.state_boost,
                "base_score": round(r.base_score, 4),
                "hop_decay": round(r.hop_decay, 4),
                "synaptic_boost": round(r.synaptic_boost, 4),
                "recency_bonus": round(r.recency_bonus, 4),
                "final_score": round(r.final_score, 4),
                "hop_distance": r.hop_distance,
                "source": r.source,
                "cell_id": r.cell_id,
                "layer": r.memory.layer,
            }
            for r in recall_results
        ]

        return {
            "qwen_response": {"role": "assistant", "content": llm_response},
            "recalled_memories": recalled_formatted,
            "audit_log": audit.to_dict() if audit is not None else None,
            "stats": self.engine.get_stats(),
            "turn_memory_ids": current_ids,
            # P0: surface degradation — the caller must be able to see that
            # results came from dummy embeddings or that recall failed.
            "embedding_provider": self.embedder.provider_status(),
            "recall_error": recall_error,
            "latency_ms": round((time.time() - t0) * 1000),
        }

    def reinforce_memory(self, memory_id: str) -> Dict:
        entry = self.engine.reinforce(memory_id)
        return {
            "memory_id": memory_id,
            "new_state": entry.state.name,
            "stats": self.engine.get_stats(),
        }

    def forget_memory(self, memory_id: str) -> Dict:
        entry = self.engine.forget(memory_id)
        return {
            "memory_id": memory_id,
            "new_state": entry.state.name,
            "stats": self.engine.get_stats(),
        }

    def reset_conversation(self):
        """Clear turn history and conversation context."""
        self._turn_history.clear()
        self._conversation_history.clear()

    # P1: context budget — unbounded blocks waste tokens and can push the
    # actual user query out of the model's effective window.
    MAX_MEMORY_CHARS = 400
    MAX_CONTEXT_CHARS = 6000

    @staticmethod
    def _build_context(results: List[Any]) -> str:
        if not results:
            return ""
        lines = ["=== RAVEN-MEMORY CONTEXT ==="]
        budget = MemoryAgentOrchestrator.MAX_CONTEXT_CHARS
        used = len(lines[0])
        truncated = False
        for i, r in enumerate(results, 1):
            content = r.memory.content
            if len(content) > MemoryAgentOrchestrator.MAX_MEMORY_CHARS:
                content = content[:MemoryAgentOrchestrator.MAX_MEMORY_CHARS] + "…"
            score_info = (
                f"score={r.final_score:.3f} "
                f"[sim={r.base_score:.3f} × state={r.state_boost} × hop_decay={r.hop_decay:.3f}"
                f" + syn={r.synaptic_boost:.3f}]"
            )
            block = (
                f"[{i}] [{r.memory.state.name}] {content}\n"
                f"     {score_info} | hop={r.hop_distance} | src={r.source}"
            )
            if used + len(block) > budget:
                truncated = True
                break
            lines.append(block)
            used += len(block)
        if truncated:
            lines.append("(context truncated — more memories matched than fit the budget)")
        lines.append("=== END CONTEXT ===")
        return "\n".join(lines)


if __name__ == "__main__":
    print("raven-memory — Qwen client OK")
