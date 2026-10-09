"""RAVEN-MEMORY — curated read-only MCP bridge.

mcp_server.py exposes 12 tools: store, recall, reinforce, forget,
create_link, stats, get_memory, audit_trail, export_graph, verify_chain,
consolidate, info. This file exposes only the six that never write to the
database:

- raven_stats, raven_get_memory, raven_audit_trail, raven_export_graph,
  raven_verify_chain, raven_info.

Deliberately excluded, not just unlisted:

- raven_store, raven_reinforce, raven_forget, raven_create_link,
  raven_consolidate — obvious writes.
- raven_recall — looks read-only but is not: memory_engine.py's
  recall() calls self._db.store_audit(...) and
  self._db.update_activations(...) on every call, and (only when
  RAVEN_STYLO_ENFORCE=1) can even flip a memory's state to FORGOTTEN. A
  read tool with a write side effect does not belong in a bridge whose
  whole point is "cannot change state."

Because none of the six kept tools embeds text, this file skips
constructing the Qwen/sentence-transformers embedding provider entirely
(raven_store/raven_recall are the only callers of it in the original
server) — one fewer heavy dependency path (torch, sentence-transformers)
on the import graph for a bridge that never needs it.

Implementations are copied verbatim from mcp_server.py, not
reimplemented.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raven.memory_engine import AdaptiveMemoryEngine, verify_audit_chain  # noqa: E402

log = logging.getLogger("raven.mcp.readonly")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), stream=sys.stderr)

mcp = FastMCP("raven-memory-readonly")

_DB_PATH = Path(os.environ.get("RAVEN_DB_PATH", "raven_memory.db"))
_engine = AdaptiveMemoryEngine(db_path=_DB_PATH)

_MAX_ID = 128


def _sanitize_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    value = value.strip()
    if len(value) > _MAX_ID:
        raise ValueError(f"{name} too long (max {_MAX_ID} chars).")
    return value


@mcp.tool()
def raven_stats() -> dict:
    """Engine statistics: memory/cell counts, state distribution, MSS. Read-only."""
    return _engine.get_stats()


@mcp.tool()
def raven_get_memory(memory_id: str) -> dict:
    """Fetch a single memory by ID (embedding stripped from response). Read-only."""
    memory_id = _sanitize_id(memory_id, "memory_id")
    entry = _engine._db.load_memory(memory_id)
    if entry is None:
        return {"error": f"Memory {memory_id} not found."}
    d = entry.to_dict()
    d.pop("embedding", None)
    return d


@mcp.tool()
def raven_audit_trail(limit: int = 20) -> dict:
    """Most recent entries from the tamper-evident audit chain. Read-only."""
    limit = max(1, min(int(limit), 100))
    entries = _engine._db.get_audit_trail(limit=limit)
    verification = verify_audit_chain(entries) if entries else {
        "chain_intact": True, "hash_integrity": True, "issues": [],
    }
    return {
        "entries": len(entries),
        "chain_intact": verification.get("chain_intact", True),
        "hash_integrity": verification.get("hash_integrity", True),
        "issues": verification.get("issues", []),
        "recent": [
            {
                "id": e.get("id"),
                "timestamp": e.get("timestamp"),
                "operation": e.get("operation"),
                "query_text": (e.get("query_text") or "")[:100],
                "returned_to_agent": e.get("returned_to_agent"),
                "audit_hash": e.get("audit_hash", "")[:16] + "...",
            }
            for e in entries[:10]
        ],
    }


@mcp.tool()
def raven_export_graph(max_nodes: int = 200) -> dict:
    """Export the memory graph (nodes + edges) as structured JSON. Read-only."""
    max_nodes = max(1, min(int(max_nodes), 500))
    return _engine.export_graph(max_nodes=max_nodes)


@mcp.tool()
def raven_verify_chain(limit: int = 1000) -> dict:
    """Full cryptographic verification of the audit hash chain. Read-only."""
    limit = max(1, min(int(limit), 10_000))
    entries = _engine.get_audit_trail(limit=limit)
    if not entries:
        return {"entries_verified": 0, "chain_intact": True,
                "hash_integrity": True, "issues": []}
    report = verify_audit_chain(entries)
    return {"entries_verified": len(entries), **report}


@mcp.tool()
def raven_info() -> dict:
    """Architecture, scoring formula, and field dynamics. Read-only."""
    return {
        "name": "raven-memory",
        "version": "1.1",
        "description": (
            "Adaptive semantic memory engine with resonant field dynamics. "
            "Replaces flat vector search with KDTree spatial indexing, BFS hop "
            "expansion, STDP synaptic learning, and ternary state modulation."
        ),
        "scoring_formula": (
            "final_score = cosine_sim * state_boost * hop_decay + "
            "synaptic_boost + recency_bonus"
        ),
        "state_boosts": {"REINFORCED": 1.5, "NEUTRAL": 1.0, "FORGOTTEN": 0.0},
        "link_types": {
            "RESONANT": "Memories reinforce each other in BFS expansion",
            "INHIBITORY": "Memories suppress each other (contradiction)",
            "NEUTRAL": "No scoring effect",
        },
        "note": "read-only bridge: raven_recall is intentionally not exposed "
                "here (it writes audit/activation rows on every call).",
    }


if __name__ == "__main__":
    log.info("RAVEN-MEMORY read-only MCP server starting — db=%s", _DB_PATH)
    mcp.run()
