"""
RAVEN-MEMORY MCP Server
========================
Exposes the raven-memory adaptive memory engine as an MCP server, so any
MCP-capable agent (Claude, Claude Code, custom agents) can store, recall,
reinforce, forget, and audit episodic memories through the resonant field model.

Design notes
------------
- FastMCP over stdio, same transport model as CRONOS and CORVUS MCP servers.
- Embeddings are generated server-side via the three-tier fallback:
  local sentence-transformers > Qwen Cloud API > deterministic dummy.
  The MCP client passes text; the server handles vectorization.
- The server shares the SQLite database (WAL mode) with the FastAPI server
  and Gradio demo, so memories stored via any interface are visible everywhere.
- Input sanitization pattern adapted from CORVUS/VIGIA: hard caps on text
  size and list lengths to prevent OOM/ReDoS via MCP.

Run
---
    python mcp_server.py            # stdio transport

Register (Claude Code settings.json or claude_desktop_config.json)
------------------------------------------------------------------
    {
      "mcpServers": {
        "raven-memory": {
          "command": "python",
          "args": ["/home/labestiadevigia/raven-memory/mcp_server.py"],
          "env": {
            "RAVEN_DB_PATH": "/home/labestiadevigia/raven-memory/raven_memory.db"
          }
        }
      }
    }
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Ensure the package resolves regardless of the invoking CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raven.memory_engine import (
    AdaptiveMemoryEngine,
    MemoryState,
    LinkType,
    verify_audit_chain,
)
from raven.qwen_client import EmbeddingProvider, QwenConfig

log = logging.getLogger("raven.mcp")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    stream=sys.stderr,  # stdout belongs to the MCP protocol
)

mcp = FastMCP("raven-memory")

# -- Config ------------------------------------------------------------------

_DB_PATH = Path(os.environ.get("RAVEN_DB_PATH", "raven_memory.db"))
_engine = AdaptiveMemoryEngine(db_path=_DB_PATH)

# Embedding provider — three-tier fallback (local > API > dummy)
_qwen_config = QwenConfig(
    api_key=os.environ.get("QWEN_API_KEY", ""),
    use_local_embeddings=True,
    embedding_dim=_engine.embedding_dim,
)
_embedder = EmbeddingProvider(_qwen_config)

# -- Input limits (VIGIA/CORVUS pattern) ------------------------------------

_MAX_TEXT = 50_000
_MAX_ID = 128
_MAX_RESULTS = 50


def _trunc(text: str, limit: int = _MAX_TEXT) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text if len(text) <= limit else text[:limit - 1] + "..."


def _sanitize_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    value = value.strip()
    if len(value) > _MAX_ID:
        raise ValueError(f"{name} too long (max {_MAX_ID} chars).")
    return value


# -- MCP tools ---------------------------------------------------------------

@mcp.tool()
def raven_store(
    content: str,
    layer: str = "semantic",
    author_id: str = "mcp_agent",
    session_id: str = "mcp",
    topic: str = "",
    claim: str = "",
) -> dict:
    """
    Store a memory in the raven-memory field. The server generates the
    embedding automatically from the content text (384-dim MiniLM or
    Qwen API, with deterministic fallback).

    Args:
        content: The text content of the memory to store.
        layer: Memory layer/category (e.g. "semantic", "episodic", "procedural").
        author_id: Identity of the author storing this memory.
        session_id: Session identifier for grouping related operations.
        topic: Optional topic tag for automatic contradiction detection.
        claim: Optional claim text — memories with same topic but different
            claims get INHIBITORY links (epistemic self-organization).

    Returns:
        memory_id, cell_id, content_hash, layer, state, and embedding provider used.
    """
    content = _trunc(content)
    author_id = _sanitize_id(author_id, "author_id")

    if not content.strip():
        return {"error": "content must be a non-empty string."}

    # Generate embedding via the three-tier fallback
    embeddings = _embedder.embed([content])
    if not embeddings:
        return {"error": "Embedding generation failed across all providers."}

    metadata = {}
    if topic:
        metadata["topic"] = _trunc(topic, 256)
    if claim:
        metadata["claim"] = _trunc(claim, 512)

    entry = _engine.store(
        content=content,
        embedding=embeddings[0],
        layer=layer,
        session_id=session_id,
        author_id=author_id,
        metadata=metadata,
    )

    return {
        "memory_id": entry.memory_id,
        "cell_id": entry.cell_id,
        "content_hash": entry.content_hash,
        "layer": entry.layer,
        "state": entry.state.name,
        "provider": _embedder.active_provider,
    }


@mcp.tool()
def raven_recall(
    query: str,
    top_k: int = 5,
    hops: int = 2,
    layer_filter: str = "",
) -> dict:
    """
    Recall memories from the adaptive field using semantic similarity,
    BFS hop expansion, STDP synaptic weights, and ternary state boosts.

    The field dynamics mean:
    - REINFORCED memories score 1.5x (validated truths surface first)
    - FORGOTTEN memories are invisible (state=0.0, excluded from KDTree)
    - INHIBITORY links suppress contradicted claims during hop expansion
    - Co-activated pairs strengthen over time (STDP potentiation)

    Args:
        query: Natural language query text to search for.
        top_k: Maximum number of results to return (1-50).
        hops: BFS expansion depth from the seed cell (0=exact match only, 2=default).
        layer_filter: Optional — only return memories from this layer.

    Returns:
        List of recalled memories with composite scores, hop distances,
        source labels, and the audit hash for this operation.
    """
    query = _trunc(query)
    if not query.strip():
        return {"error": "query must be a non-empty string."}

    top_k = max(1, min(int(top_k), _MAX_RESULTS))
    hops = max(0, min(int(hops), 5))

    embeddings = _embedder.embed([query])
    if not embeddings:
        return {"error": "Query embedding generation failed."}

    results, audit = _engine.recall(
        query_embedding=embeddings[0],
        query_text=query,
        top_k=top_k,
        hops=hops,
        layer_filter=layer_filter or None,
    )

    return {
        "count": len(results),
        "results": [
            {
                "memory_id": r.memory.memory_id,
                "content": r.memory.content,
                "layer": r.memory.layer,
                "state": r.memory.state.name,
                "author_id": r.memory.author_id,
                "final_score": round(r.final_score, 6),
                "base_score": round(r.base_score, 6),
                "state_boost": round(r.state_boost, 2),
                "hop_decay": round(r.hop_decay, 4),
                "hop_distance": r.hop_distance,
                "source": r.source,
                "cell_id": r.cell_id,
                "resonance_score": round(r.resonance_score, 4),
                "coherence_score": round(r.coherence_score, 4),
            }
            for r in results
        ],
        "audit_hash": audit.audit_hash,
        "provider": _embedder.active_provider,
    }


@mcp.tool()
def raven_reinforce(memory_id: str) -> dict:
    """
    Reinforce a memory — sets its state to REINFORCED (1.5x boost in recall).

    Reinforcement is the epistemic act of validating a memory as true/useful.
    In the field dynamics, reinforced memories:
    - Score 50% higher in recall
    - Are immune to orphan sweeps
    - Trigger INHIBITORY suppression of contradicting claims (Collapse Around Truth)

    Args:
        memory_id: The ID of the memory to reinforce.

    Returns:
        Updated memory state and confidence level.
    """
    memory_id = _sanitize_id(memory_id, "memory_id")
    try:
        entry = _engine.reinforce(memory_id)
    except KeyError as exc:
        return {"error": str(exc)}

    return {
        "memory_id": entry.memory_id,
        "state": entry.state.name,
        "cell_id": entry.cell_id,
        "content": entry.content[:200],
    }


@mcp.tool()
def raven_forget(memory_id: str) -> dict:
    """
    Forget a memory — sets its state to FORGOTTEN (invisible to recall).

    Forgotten memories are NOT deleted (evidence preservation). They are
    excluded from the KDTree index and score 0.0, making them invisible
    to recall. They can be re-reinforced later if evidence changes.

    Args:
        memory_id: The ID of the memory to forget.

    Returns:
        Updated memory state.
    """
    memory_id = _sanitize_id(memory_id, "memory_id")
    try:
        entry = _engine.forget(memory_id)
    except KeyError as exc:
        return {"error": str(exc)}

    return {
        "memory_id": entry.memory_id,
        "state": entry.state.name,
        "cell_id": entry.cell_id,
        "content": entry.content[:200],
    }


@mcp.tool()
def raven_create_link(
    from_cell_id: int,
    to_cell_id: int,
    link_type: str = "RESONANT",
) -> dict:
    """
    Create a ternary cell link between two memory cells.

    Link types:
    - RESONANT: memories reinforce each other during BFS hop expansion
    - INHIBITORY: memories suppress each other (contradiction signal)
    - NEUTRAL: no effect on scoring

    Automatic INHIBITORY links are created when memories share the same
    topic but have different claims. This tool allows manual override or
    explicit RESONANT linking of related memories.

    Args:
        from_cell_id: Source cell ID.
        to_cell_id: Target cell ID.
        link_type: "RESONANT", "INHIBITORY", or "NEUTRAL".

    Returns:
        Confirmation of the link created.
    """
    try:
        lt = LinkType[link_type.upper()]
    except KeyError:
        return {"error": f"Invalid link_type: {link_type}. Use RESONANT, INHIBITORY, or NEUTRAL."}

    _engine.create_cell_link(from_cell_id, to_cell_id, lt)
    return {
        "from_cell_id": from_cell_id,
        "to_cell_id": to_cell_id,
        "link_type": lt.name,
        "created": True,
    }


@mcp.tool()
def raven_stats() -> dict:
    """
    Return engine statistics: total memories, Voronoi cells, state distribution,
    layer distribution, Memory Stability Score (MSS), retention ratio, and
    cell link counts.

    MSS = weighted_reinforced / (weighted_reinforced + weighted_neutral).
    A high MSS means the field has converged on validated truths.
    """
    return _engine.get_stats()


@mcp.tool()
def raven_get_memory(memory_id: str) -> dict:
    """
    Fetch a single memory by ID. Returns full metadata including content,
    state, layer, author, cell_id, and creation time.

    Args:
        memory_id: The ID of the memory to retrieve.
    """
    memory_id = _sanitize_id(memory_id, "memory_id")
    entry = _engine._db.load_memory(memory_id)
    if entry is None:
        return {"error": f"Memory {memory_id} not found."}
    d = entry.to_dict()
    # Remove the full embedding from the response (too large for MCP)
    d.pop("embedding", None)
    return d


@mcp.tool()
def raven_audit_trail(limit: int = 20) -> dict:
    """
    Return the most recent entries from the tamper-evident audit chain.

    Every recall operation appends a hash-linked entry recording: what was
    queried, which cells activated, which memories were returned, and the
    cryptographic hash linking to the previous entry. If any entry was
    modified after the fact, the chain breaks.

    Args:
        limit: Number of recent entries to return (max 100).

    Returns:
        Recent audit entries and chain verification status.
    """
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
    """
    Export the memory graph for visualization or analysis. Returns nodes
    (memories) and edges (cell links) as structured JSON. Capped to prevent
    oversized responses.

    Args:
        max_nodes: Maximum number of nodes to include (most-recalled first).

    Returns:
        Nodes with metadata, edges with link types, and truncation flag.
    """
    max_nodes = max(1, min(int(max_nodes), 500))
    graph = _engine.export_graph(max_nodes=max_nodes)
    return graph


@mcp.tool()
def raven_info() -> dict:
    """
    Describe raven-memory's architecture, scoring formula, and field dynamics.
    Call this first to understand how the memory system works and how to
    interpret recall scores.
    """
    stats = _engine.get_stats()
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
        "state_boosts": {
            "REINFORCED": 1.5,
            "NEUTRAL": 1.0,
            "FORGOTTEN": 0.0,
        },
        "link_types": {
            "RESONANT": "Memories reinforce each other in BFS expansion",
            "INHIBITORY": "Memories suppress each other (contradiction)",
            "NEUTRAL": "No scoring effect",
        },
        "key_behaviors": [
            "Collapse Around Truth: reinforcing one claim silences contradictions via INHIBITORY links",
            "STDP potentiation: co-activated memories strengthen their synaptic connection",
            "Hop decay: distant memories score exponentially less (lambda=0.15)",
            "Recency bonus: recently accessed memories get a small additive boost",
            "Stylometric fingerprinting: detects authorship anomalies",
        ],
        "embedding_provider": _embedder.active_provider,
        "embedding_dim": _engine.embedding_dim,
        "current_stats": stats,
    }


# -- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    log.info("raven-memory MCP server starting — db=%s, provider=%s",
             _DB_PATH, _embedder.active_provider)
    mcp.run()
