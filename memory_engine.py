#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY v1.0
Adaptive Memory Substrate for Agentic Systems
Track 1: MemoryAgent — Qwen Cloud Hackathon

Architecture:
- KDTree + k-NN graph for neighbourhood activation
- Ternary states: REINFORCED (×1.5), NEUTRAL (×1.0), FORGOTTEN (×0.0)
- Hop decay: score *= exp(-λ * hop_distance)
- STDP: LTP (potentiation) + LTD (depression) synaptic dynamics
- Ternary cell links: RESONANT / NEUTRAL / INHIBITORY
- Recency scoring with configurable half-life
- Stylometric fingerprinting for author verification
- Audit hash-chain for tamper-proof traceability
- SQLite persistence with lazy KDTree rebuild

Authors: Anna Tchijova + Claude (VIGÍA AI Collective)
License: Apache 2.0
"""

import json
import sqlite3
import hashlib
import time
import math
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("raven.engine")

# P0: spectral must be imported BEFORE numpy — it sets the single-thread
# BLAS env vars that underpin the determinism guarantee, and they only take
# effect if no BLAS backend has been initialised yet. Importing it first
# from the engine (the usual entry point) silences the late-import warning
# for every downstream module (api_server, consolidator, demos).
# Spectral module is optional — engine degrades gracefully without it.
try:
    from spectral import SpectralField, SpectralStore
    _SPECTRAL_AVAILABLE = True
except ImportError:
    _SPECTRAL_AVAILABLE = False

import numpy as np
from scipy.spatial import KDTree

if not _SPECTRAL_AVAILABLE:
    logger.debug("spectral module not found — resonance/coherence metadata disabled")


# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("raven_memory.db")
EMBEDDING_DIM = 384
K_NEIGHBORS = 6
HOP_LAMBDA = 0.15
STDP_MAX_WEIGHT = 2.0
STDP_MIN_WEIGHT = 0.0
STDP_POTENTIATION = 0.10   # LTP: co-activated pairs strengthen
STDP_DEPRESSION = 0.02     # LTD: previously active but absent weaken
# P0: prune threshold for dead synaptic links. Float accumulation
# (0.10 up / 0.02 down) can leave residues ~1e-17 that `== 0.0` never
# matches — links would accumulate forever (progressive memory leak).
STDP_PRUNE_EPS = 1e-9
# P1: BFS search ceiling for hop-distance queries. Named (not magic):
# with k=6 neighbours the graph diameter is far below this in practice;
# the ceiling only bounds worst-case pathological geometries.
MAX_HOP_SEARCH = 10
ESTILOMETRIA_THRESHOLD = 0.5
RECENCY_HALFLIFE = 86400.0  # seconds — 24 h half-life for recency bonus
RECENCY_WEIGHT = 0.05       # small additive bonus for recently-accessed memories


# ============================================================
# ENUMS
# ============================================================

class MemoryState(Enum):
    REINFORCED = 1.5
    NEUTRAL = 1.0
    FORGOTTEN = 0.0


class LinkType(Enum):
    RESONANT = 1.0
    NEUTRAL = 0.0
    INHIBITORY = -1.0


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class StylometricFingerprint:
    functional_words: Dict[str, float]
    avg_sentence_length: float
    punctuation_profile: Dict[str, float]
    fingerprint_hash: str
    # P1: dominant language of the sample ("es" | "en" | "und").
    # Default keeps backward compatibility with fingerprints persisted
    # before this field existed.
    language: str = "und"


@dataclass
class MemoryEntry:
    memory_id: str
    layer: str
    content: str
    content_hash: str
    embedding: np.ndarray
    state: MemoryState
    cell_id: int
    created_at: float
    session_id: str
    author_id: str
    metadata: Dict
    synaptic_links: Dict[str, float] = field(default_factory=dict)
    last_activation: float = 0.0
    recall_count: int = 0
    fingerprint: Optional[StylometricFingerprint] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["embedding"] = self.embedding.tolist()
        d["state"] = self.state.name
        d["fingerprint"] = asdict(self.fingerprint) if self.fingerprint else None
        return d


@dataclass
class RecallResult:
    memory: MemoryEntry
    base_score: float
    state_boost: float
    hop_decay: float
    synaptic_boost: float
    recency_bonus: float
    final_score: float
    hop_distance: int
    cell_id: int
    source: str                  # "similarity" | "synaptic" | "resonant"
    resonance_score: float = 0.0 # spectral eigen-mode resonance (epistemic, not ranking)
    coherence_score: float = 1.0 # RESONANT/(RESONANT+INHIBITORY) links (epistemic flag)


@dataclass
class AuditLog:
    timestamp: float
    operation: str
    query_text: Optional[str]
    query_embedding: Optional[List[float]]
    cells_activated: List[int]
    memories_retrieved: List[Dict]
    total_candidates: int
    filtered_by_state: int
    filtered_by_estilometria: int
    filtered_by_inhibitory: int
    synaptic_activated: int
    returned_to_agent: int
    audit_hash: str
    prev_hash: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ForensicAlert:
    alert_id: str
    timestamp: float
    memory_id: str
    detected_author: str
    expected_author: str
    mismatch_score: float
    action_taken: str


# ============================================================
# AUDIT HASH — canonical, verifiable, content-aware
# ============================================================

def compute_audit_hash(
    timestamp: float,
    operation: str,
    query_text: Optional[str],
    cells_activated: List[int],
    memories_retrieved,
    prev_hash: str,
    query_embedding=None,
) -> str:
    """
    Canonical audit hash for the tamper-evident chain.

    P0 fix — two structural flaws in the previous scheme:
      1. Only memory IDs were hashed: content could be modified post-audit
         and the chain stayed "valid". memories_retrieved now carries each
         memory's content_hash, and the FULL retrieved payload is hashed.
      2. The hashed timestamp was a separate time.time() call from the one
         stored in the row, so the chain could never be recomputed from
         persisted data. Every input here is a stored column — any verifier
         can recompute the hash from the DB row alone.

    Verification contract:
      audit_hash == sha256(canonical_json(payload) + prev_hash)
      payload    == {ts, op, query, cells, results} exactly as persisted.
    """
    payload = json.dumps(
        {
            "ts": round(float(timestamp), 6),
            "op": operation,
            "query": query_text,
            "cells": cells_activated,
            "results": memories_retrieved,
            "qemb_sha256": hashlib.sha256(
                str(query_embedding).encode("utf-8") if query_embedding is not None else b""
            ).hexdigest(),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256((payload + prev_hash).encode("utf-8")).hexdigest()


def verify_audit_chain(entries_desc: List[Dict]) -> Dict:
    """
    Verify both properties of the audit chain over rows ordered by id DESC:
      linkage   — entries[i].prev_hash == entries[i+1].audit_hash
      integrity — each audit_hash recomputes from its own stored columns
                  (detects post-hoc edits to query, cells, or retrieved
                  content metadata, including content_hash).

    Returns {"chain_intact", "hash_integrity", "issues": [...]}.
    Pre-fix rows (hashed under the legacy ID-only scheme) fail integrity
    recomputation — reported as issues, not as exceptions.
    """
    issues: List[Dict] = []
    chain_ok = True
    integrity_ok = True

    for i in range(len(entries_desc) - 1):
        if entries_desc[i]["prev_hash"] != entries_desc[i + 1]["audit_hash"]:
            chain_ok = False
            issues.append({
                "type": "linkage_broken",
                "at_id": entries_desc[i].get("id"),
            })

    for e in entries_desc:
        try:
            cells = json.loads(e["cells_activated"]) if isinstance(
                e["cells_activated"], str) else e["cells_activated"]
            results = json.loads(e["memories_retrieved"]) if isinstance(
                e["memories_retrieved"], str) else e["memories_retrieved"]
            qemb_stored = e.get("query_embedding")
            qemb = json.loads(qemb_stored) if qemb_stored else None
            recomputed = compute_audit_hash(
                e["timestamp"], e["operation"], e["query_text"],
                cells, results, e["prev_hash"], qemb,
            )
            if recomputed != e["audit_hash"]:
                integrity_ok = False
                issues.append({
                    "type": "hash_mismatch",
                    "at_id": e.get("id"),
                    "note": "stored columns do not reproduce audit_hash "
                            "(tampered row, or legacy pre-content-hash entry)",
                })
        except Exception as exc:
            integrity_ok = False
            issues.append({"type": "verify_error", "at_id": e.get("id"), "error": str(exc)})

    return {"chain_intact": chain_ok, "hash_integrity": integrity_ok, "issues": issues}


# ============================================================
# STYLOMETRIC EXTRACTOR
# ============================================================

class StylometricExtractor:
    """Extracts stylometric fingerprint from text for authorship verification."""

    # P1: language-separated sets. Mixing both in one bag made a bilingual
    # author look like two different people (forensic false positives) —
    # function-word frequency profiles are only comparable WITHIN a language.
    FUNCTIONAL_WORDS_ES = {
        "el", "la", "de", "que", "y", "en", "un", "es", "se", "por", "con",
        "para", "los", "las", "del", "al", "lo", "le", "me", "te", "su", "mi",
        "nos", "les", "pero", "como", "más", "sin", "sobre", "entre", "hasta",
        "desde", "todo", "también", "ya", "sí", "ni",
    }
    FUNCTIONAL_WORDS_EN = {
        "the", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "can", "so", "very", "just",
        "now", "then", "here", "there", "when", "where", "all", "each", "both",
        "some", "only", "same", "than", "also", "after", "even", "new", "any",
    }
    # Ambiguous tokens valid in both languages — counted for frequency but
    # not for language detection.
    FUNCTIONAL_WORDS_SHARED = {"a", "no", "o"}
    FUNCTIONAL_WORDS = FUNCTIONAL_WORDS_ES | FUNCTIONAL_WORDS_EN | FUNCTIONAL_WORDS_SHARED

    def extract(self, text: str, author_id: str) -> StylometricFingerprint:
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        words = text.lower().split()
        total_words = max(len(words), 1)

        func_counts: Dict[str, int] = {}
        es_hits = en_hits = 0
        for w in words:
            clean = w.strip(".,;:!?\"'")
            if clean in self.FUNCTIONAL_WORDS:
                func_counts[clean] = func_counts.get(clean, 0) + 1
                if clean in self.FUNCTIONAL_WORDS_ES:
                    es_hits += 1
                elif clean in self.FUNCTIONAL_WORDS_EN:
                    en_hits += 1
        func_freq = {w: c / total_words for w, c in func_counts.items()}

        # P1: dominant-language detection (cheap, no external deps).
        if es_hits > en_hits:
            language = "es"
        elif en_hits > es_hits:
            language = "en"
        else:
            language = "und"

        avg_len = float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0

        punct = {",": 0, ";": 0, ":": 0, ".": 0, "!": 0, "?": 0}
        for c in text:
            if c in punct:
                punct[c] += 1
        total_chars = max(len(text), 1)
        punct_profile = {k: v / total_chars for k, v in punct.items()}

        payload = json.dumps({"func": func_freq, "avg": avg_len, "punct": punct_profile, "lang": language}, sort_keys=True)
        fp_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]

        return StylometricFingerprint(
            functional_words=func_freq,
            avg_sentence_length=avg_len,
            punctuation_profile=punct_profile,
            fingerprint_hash=fp_hash,
            language=language,
        )

    def compare(self, fp1: StylometricFingerprint, fp2: StylometricFingerprint) -> float:
        """Returns stylometric distance: 0.0 (identical) → 1.0 (completely different)."""
        all_words = set(fp1.functional_words) | set(fp2.functional_words)
        if all_words:
            v1 = [fp1.functional_words.get(w, 0) for w in all_words]
            v2 = [fp2.functional_words.get(w, 0) for w in all_words]
            func_dist = 1.0 - self._cosine_sim(v1, v2)
        else:
            func_dist = 0.0

        max_len = max(fp1.avg_sentence_length, fp2.avg_sentence_length, 1.0)
        len_diff = abs(fp1.avg_sentence_length - fp2.avg_sentence_length) / max_len

        keys = list(fp1.punctuation_profile.keys())
        p1 = [fp1.punctuation_profile.get(k, 0) for k in keys]
        p2 = [fp2.punctuation_profile.get(k, 0) for k in keys]
        punct_dist = 1.0 - self._cosine_sim(p1, p2) if p1 else 0.0

        return 0.5 * func_dist + 0.3 * len_diff + 0.2 * punct_dist

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ============================================================
# SQLITE STORE
# ============================================================

class MemoryStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._prev_audit_hash = "0" * 64
        self._init_db()
        self._load_prev_hash()

    def _connect(self) -> sqlite3.Connection:
        """
        P0: every connection gets a busy timeout so concurrent writers
        (api_server + sleep_consolidator) wait instead of throwing
        'database is locked'. WAL journal mode (set once in _init_db,
        persistent in the DB file) lets readers proceed during writes.
        """
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            # P0: WAL is a persistent DB property — concurrent reader/writer
            # safety between the API server and the sleep consolidator.
            # synchronous=NORMAL is the recommended pairing for WAL
            # (durable at checkpoint, ~an order of magnitude faster).
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    layer TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    state TEXT NOT NULL,
                    cell_id INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    metadata TEXT,
                    synaptic_links TEXT,
                    last_activation REAL DEFAULT 0.0,
                    recall_count INTEGER DEFAULT 0,
                    fingerprint TEXT
                )
            """)
            # Add recall_count column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN recall_count INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cell_links (
                    from_cell_id INTEGER,
                    to_cell_id INTEGER,
                    link_type TEXT,
                    created_at REAL,
                    auto_generated INTEGER,
                    PRIMARY KEY (from_cell_id, to_cell_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    operation TEXT NOT NULL,
                    query_text TEXT,
                    query_embedding BLOB,
                    cells_activated TEXT,
                    memories_retrieved TEXT,
                    total_candidates INTEGER,
                    filtered_by_state INTEGER,
                    filtered_by_estilometria INTEGER,
                    filtered_by_inhibitory INTEGER,
                    synaptic_activated INTEGER,
                    returned_to_agent INTEGER,
                    audit_hash TEXT,
                    prev_hash TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forensic_alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    memory_id TEXT NOT NULL,
                    detected_author TEXT,
                    expected_author TEXT,
                    mismatch_score REAL,
                    action_taken TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_cell   ON memories(cell_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_layer  ON memories(layer)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_author ON memories(author_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_state  ON memories(state)")
            conn.commit()

    def _load_prev_hash(self):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT audit_hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                self._prev_audit_hash = row[0]

    # ---- MEMORIES ----

    def store_memory(self, entry: MemoryEntry):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memories
                (memory_id, layer, content, content_hash, embedding, state, cell_id,
                 created_at, session_id, author_id, metadata, synaptic_links,
                 last_activation, recall_count, fingerprint)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry.memory_id, entry.layer, entry.content, entry.content_hash,
                    entry.embedding.astype(np.float32).tobytes(),
                    entry.state.name, entry.cell_id, entry.created_at,
                    entry.session_id, entry.author_id,
                    json.dumps(entry.metadata),
                    json.dumps(entry.synaptic_links),
                    entry.last_activation, entry.recall_count,
                    json.dumps(asdict(entry.fingerprint)) if entry.fingerprint else None,
                ),
            )
            conn.commit()

    def load_memories(
        self,
        cell_ids: Optional[List[int]] = None,
        author_id: Optional[str] = None,
        layer: Optional[str] = None,
        state: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[MemoryEntry]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conditions = []
            params: List = []

            if cell_ids is not None:
                if len(cell_ids) == 0:
                    return []
                if len(cell_ids) <= 999:
                    placeholders = ",".join("?" * len(cell_ids))
                    conditions.append(f"cell_id IN ({placeholders})")
                    params.extend(cell_ids)
                else:
                    # P1-4: SQLite max 999 bound params — chunk the query
                    all_rows = []
                    for i in range(0, len(cell_ids), 999):
                        chunk = cell_ids[i:i+999]
                        ph = ",".join("?" * len(chunk))
                        rows = conn.execute(
                            f"SELECT * FROM memories WHERE cell_id IN ({ph})", chunk
                        ).fetchall()
                        all_rows.extend(rows)
                    entries = [self._row_to_entry(r) for r in all_rows]
                    if author_id:
                        entries = [e for e in entries if e.author_id == author_id]
                    if layer:
                        entries = [e for e in entries if e.layer == layer]
                    if state:
                        entries = [e for e in entries if e.state.name == state]
                    if offset:
                        entries = entries[offset:]
                    if limit:
                        entries = entries[:limit]
                    return entries
            if author_id:
                conditions.append("author_id = ?")
                params.append(author_id)
            if layer:
                conditions.append("layer = ?")
                params.append(layer)
            if state:
                conditions.append("state = ?")
                params.append(state)

            sql = "SELECT * FROM memories"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY cell_id ASC"
            if limit:
                sql += f" LIMIT {limit} OFFSET {offset}"

            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def load_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            return self._row_to_entry(row) if row else None

    def count_memories(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def count_stats(self) -> Dict:
        """
        Aggregate counts computed in SQL — no row deserialization.

        get_stats() previously called load_memories(), which materialised every
        row (embedding bytes, fingerprint JSON, the lot) just to tally states.
        Since get_stats() runs on every recall, every WebSocket broadcast and
        every dashboard refresh, that turned a counter into a full-table scan
        that scaled with corpus size. This stays O(1) in Python memory.
        """
        with self._connect() as conn:
            state_dist = {"REINFORCED": 0, "NEUTRAL": 0, "FORGOTTEN": 0}
            for name, c in conn.execute(
                "SELECT state, COUNT(*) FROM memories GROUP BY state"
            ).fetchall():
                state_dist[name] = c

            layer_dist: Dict[str, int] = {}
            for layer, c in conn.execute(
                "SELECT layer, COUNT(*) FROM memories GROUP BY layer"
            ).fetchall():
                layer_dist[layer] = c

            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            total_recalls = conn.execute(
                "SELECT COALESCE(SUM(recall_count), 0) FROM memories"
            ).fetchone()[0]
            authors = conn.execute(
                "SELECT COUNT(DISTINCT author_id) FROM memories"
            ).fetchone()[0]

            link_types = {"RESONANT": 0, "NEUTRAL": 0, "INHIBITORY": 0}
            for lt, c in conn.execute(
                "SELECT link_type, COUNT(*) FROM cell_links GROUP BY link_type"
            ).fetchall():
                link_types[lt] = c

        return {
            "state_distribution": state_dist,
            "layer_distribution": layer_dist,
            "total_memories": total,
            "total_recalls": int(total_recalls),
            "authors": authors,
            "cell_links": link_types,
        }

    def update_state(self, memory_id: str, new_state: MemoryState):
        with self._connect() as conn:
            conn.execute("UPDATE memories SET state=? WHERE memory_id=?",
                         (new_state.name, memory_id))
            conn.commit()

    def update_synaptic_links(self, memory_id: str, links: Dict[str, float]):
        with self._connect() as conn:
            conn.execute("UPDATE memories SET synaptic_links=? WHERE memory_id=?",
                         (json.dumps(links), memory_id))
            conn.commit()

    def update_activation(self, memory_id: str, timestamp: float):
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET last_activation=?, recall_count=recall_count+1 WHERE memory_id=?",
                (timestamp, memory_id),
            )
            conn.commit()

    # ---- CELL LINKS ----

    def store_cell_link(self, from_id: int, to_id: int, link_type: LinkType, auto: bool = True):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cell_links
                (from_cell_id, to_cell_id, link_type, created_at, auto_generated)
                VALUES (?,?,?,?,?)""",
                (from_id, to_id, link_type.name, time.time(), int(auto)),
            )
            conn.commit()

    def load_cell_links(self, cell_id: int) -> List[Tuple[int, LinkType, bool]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT to_cell_id, link_type, auto_generated FROM cell_links WHERE from_cell_id=?",
                (cell_id,),
            ).fetchall()
            return [(r[0], LinkType[r[1]], bool(r[2])) for r in rows]

    def load_all_cell_links(self) -> List[Tuple[int, int, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT from_cell_id, to_cell_id, link_type FROM cell_links"
            ).fetchall()
            return [(r[0], r[1], r[2]) for r in rows]

    def load_all_cell_links_indexed(self) -> Dict[int, List[Tuple[int, "LinkType"]]]:
        """
        Load all cell links in one query, indexed by from_cell_id.
        Used in recall() to avoid N queries inside the BFS loop.
        Returns {from_cell_id: [(to_cell_id, LinkType), ...]}
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT from_cell_id, to_cell_id, link_type FROM cell_links"
            ).fetchall()
        index: Dict[int, List[Tuple[int, LinkType]]] = {}
        for from_id, to_id, lt_str in rows:
            index.setdefault(from_id, []).append((to_id, LinkType[lt_str]))
        return index

    def load_memories_by_ids(self, memory_ids: List[str]) -> List["MemoryEntry"]:
        """
        Batch load memories by ID in a single query.
        Used in recall() synaptic pull and _update_stdp() to avoid N+1 queries.
        """
        if not memory_ids:
            return []
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if len(memory_ids) <= 999:
                placeholders = ",".join("?" * len(memory_ids))
                rows = conn.execute(
                    f"SELECT * FROM memories WHERE memory_id IN ({placeholders})",
                    memory_ids,
                ).fetchall()
            else:
                rows = []
                for i in range(0, len(memory_ids), 999):
                    chunk = memory_ids[i:i + 999]
                    ph = ",".join("?" * len(chunk))
                    rows.extend(
                        conn.execute(
                            f"SELECT * FROM memories WHERE memory_id IN ({ph})", chunk
                        ).fetchall()
                    )
        return [self._row_to_entry(r) for r in rows]

    # ---- AUDIT ----

    def store_audit(self, audit: AuditLog):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO audit_log
                (timestamp, operation, query_text, query_embedding, cells_activated,
                 memories_retrieved, total_candidates, filtered_by_state,
                 filtered_by_estilometria, filtered_by_inhibitory, synaptic_activated,
                 returned_to_agent, audit_hash, prev_hash)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    audit.timestamp, audit.operation, audit.query_text,
                    json.dumps(audit.query_embedding) if audit.query_embedding else None,
                    json.dumps(audit.cells_activated),
                    json.dumps(audit.memories_retrieved),
                    audit.total_candidates, audit.filtered_by_state,
                    audit.filtered_by_estilometria, audit.filtered_by_inhibitory,
                    audit.synaptic_activated, audit.returned_to_agent,
                    audit.audit_hash, audit.prev_hash,
                ),
            )
            conn.commit()
        self._prev_audit_hash = audit.audit_hash

    def get_audit_trail(self, limit: int = 100) -> List[Dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            # P1: id ordering (not timestamp) — two entries can share a
            # timestamp under load; AUTOINCREMENT id is the true sequence
            # the hash chain was built over.
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- FORENSIC ALERTS ----

    def store_alert(self, alert: ForensicAlert):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO forensic_alerts
                (alert_id, timestamp, memory_id, detected_author, expected_author,
                 mismatch_score, action_taken)
                VALUES (?,?,?,?,?,?,?)""",
                (alert.alert_id, alert.timestamp, alert.memory_id,
                 alert.detected_author, alert.expected_author,
                 alert.mismatch_score, alert.action_taken),
            )
            conn.commit()

    def get_alerts(self, limit: int = 50) -> List[ForensicAlert]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM forensic_alerts ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [ForensicAlert(**dict(r)) for r in rows]

    def get_prev_audit_hash(self) -> str:
        # Always read from DB — the consolidator may have written since our
        # last recall, and a cached value would break the audit chain.
        self._load_prev_hash()
        return self._prev_audit_hash

    # ---- INTERNAL ----

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        fingerprint = None
        fp_raw = row["fingerprint"] if "fingerprint" in row.keys() else None
        if fp_raw:
            try:
                fingerprint = StylometricFingerprint(**json.loads(fp_raw))
            except Exception as _fp_exc:
                logger.error(
                    f"Corrupted stylometric fingerprint in row — forensic verification disabled for this memory. "
                    f"Error: {_fp_exc}"
                )

        recall_count = 0
        try:
            recall_count = int(row["recall_count"] or 0)
        except Exception:
            pass

        return MemoryEntry(
            memory_id=row["memory_id"],
            layer=row["layer"],
            content=row["content"],
            content_hash=row["content_hash"],
            embedding=np.frombuffer(row["embedding"], dtype=np.float32).copy(),
            state=MemoryState[row["state"]],
            cell_id=row["cell_id"],
            created_at=row["created_at"],
            session_id=row["session_id"],
            author_id=row["author_id"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            synaptic_links=json.loads(row["synaptic_links"]) if row["synaptic_links"] else {},
            last_activation=float(row["last_activation"] or 0.0),
            recall_count=recall_count,
            fingerprint=fingerprint,
        )


# ============================================================
# ENGINE
# ============================================================

class AdaptiveMemoryEngine:
    """
    Raven-Memory core engine.

    Replaces flat vector search with a dynamical memory field:
      - Cells ≈ Voronoi regions in embedding space
      - Recall = neighbourhood activation + BFS hop expansion
      - Scoring = cosine × state_boost × hop_decay + bonuses
      - STDP strengthens co-activated cell pairs (LTP + LTD)
      - Ternary cell links gate propagation (RESONANT/INHIBITORY)
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, k_neighbors: int = K_NEIGHBORS,
                 db_path: Path = DB_PATH):
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors
        self._db = MemoryStore(db_path)
        self.kdtree: Optional[KDTree] = None
        self._points: Dict[int, np.ndarray] = {}     # cell_id → embedding (sparse)
        self._next_cell_id: int = 0                   # monotonic allocator
        self._cell_to_memid: Dict[int, str] = {}      # cell_id → memory_id (last stored per cell)
        self.cell_neighbors: Dict[int, Set[int]] = {}
        self._kdtree_dirty: bool = False
        # Live cell set — excludes FORGOTTEN/deleted cells (P0-1 fix).
        self._active_cells: Set[int] = set()
        # Maps KDTree array index → cell_id (rebuilt with KDTree, P0-1 fix).
        self._kdtree_idx_to_cell: List[int] = []

        # Inverted index: topic → [(memory_id, cell_id, claim)]
        self._topic_index: Dict[str, List[Tuple[str, int, str]]] = {}

        # P1: pairs already linked by _auto_link_contradictions. Without this,
        # every store on a hot topic re-issued O(n) redundant INSERT OR REPLACE
        # writes per existing contradiction (O(n²) total per topic).
        self._linked_pairs: Set[Tuple[int, int, str]] = set()
        # RESONANT neighbours for BFS hop-distance — keyed by from_cell_id.
        self._resonant_neighbors: Dict[int, List[Tuple[int, LinkType]]] = {}

        self.stylometric = StylometricExtractor()
        self._author_fingerprints: Dict[str, StylometricFingerprint] = {}

        # Spectral field — optional, loaded from DB if available.
        # Call rebuild_spectral_field() after bulk stores or consolidation.
        self._spectral: Optional["SpectralField"] = None  # type: ignore[name-defined]
        if _SPECTRAL_AVAILABLE:
            try:
                loaded = SpectralStore(db_path).load()
                if loaded and loaded.is_built:
                    self._spectral = loaded
                    logger.info(f"Spectral field loaded: {loaded.summary()}")
            except Exception as _e:
                logger.debug(f"Could not load spectral field from DB: {_e}")

        # Rebuild in-memory state from persisted DB
        self._load_from_db()

    # ----------------------------------------------------------
    # STARTUP — rebuild in-memory structures from DB
    # ----------------------------------------------------------

    def _load_from_db(self):
        """Reconstruct KDTree, topic index, and author fingerprints from persisted data."""
        # Reset every rebuildable structure first. _load_from_db can be called
        # on an existing instance (e.g. after the consolidator deletes rows);
        # carrying over stale _active_cells / _cell_to_memid would reference
        # cell_ids that no longer exist in _points and crash the KDTree rebuild.
        self._points = {}
        self._cell_to_memid = {}
        self._active_cells = set()
        self._topic_index = {}
        self._linked_pairs = set()
        self._resonant_neighbors = {}
        self._next_cell_id = 0

        mems = self._db.load_memories()
        if not mems:
            self.kdtree = None
            self.cell_neighbors = {}
            self._kdtree_idx_to_cell = []
            return

        # Points are stored sparsely, keyed by cell_id. cell_id only ever grows
        # (consolidation deletes rows without reusing ids), so a list would
        # accumulate zero-vectors for every dead cell — unbounded memory for a
        # field that is mostly holes after many consolidation cycles.
        max_cell = max(m.cell_id for m in mems)
        self._next_cell_id = max_cell + 1

        for m in mems:
            self._points[m.cell_id] = m.embedding.copy()
            self._cell_to_memid[m.cell_id] = m.memory_id
            if m.state.name != "FORGOTTEN":
                self._active_cells.add(m.cell_id)   # only live cells enter KDTree
            self._index_topic(m)
            if m.fingerprint and m.author_id not in self._author_fingerprints:
                self._author_fingerprints[m.author_id] = m.fingerprint

        self._rebuild_kdtree()

        # P1: hydrate the linked-pairs cache so dedup survives restarts.
        # Also populate _resonant_neighbors for BFS hop-distance via RESONANT links.
        for f, t, lt in self._db.load_all_cell_links():
            lt_str = lt.name if hasattr(lt, 'name') else str(lt)
            self._linked_pairs.add((f, t, lt_str))
            if lt_str == "RESONANT":
                lt_enum = LinkType[lt_str]
                self._resonant_neighbors.setdefault(f, []).append((t, lt_enum))
                self._resonant_neighbors.setdefault(t, []).append((f, lt_enum))

        logger.info(f"Loaded {len(mems)} memories, {len(self._points)} live cells from DB")

    # ----------------------------------------------------------
    # KDTREE
    # ----------------------------------------------------------

    def _ensure_kdtree(self):
        if self._kdtree_dirty:
            self._rebuild_kdtree()
            self._kdtree_dirty = False

    def _rebuild_kdtree(self):
        """
        Build KDTree from ACTIVE cells only.
        Deleted/FORGOTTEN cells are excluded — no ghost vectors in the index.
        _kdtree_idx_to_cell maps array position → cell_id for query translation.
        """
        if not self._active_cells:
            self.kdtree = None
            self.cell_neighbors = {}
            self._kdtree_idx_to_cell = []
            return

        self._kdtree_idx_to_cell = sorted(self._active_cells)
        arr = np.array([self._points[cid] for cid in self._kdtree_idx_to_cell])
        self.kdtree = KDTree(arr)

        k = min(self.k_neighbors, len(self._kdtree_idx_to_cell) - 1)
        if k <= 0:
            self.cell_neighbors = {cid: set() for cid in self._kdtree_idx_to_cell}
            return

        _, indices = self.kdtree.query(arr, k=k + 1)
        self.cell_neighbors = {}
        for arr_i, neighbors in enumerate(indices):
            cell_id = self._kdtree_idx_to_cell[arr_i]
            real = {self._kdtree_idx_to_cell[int(n)] for n in neighbors if n != arr_i}
            self.cell_neighbors[cell_id] = real
            for n_cell in real:
                self.cell_neighbors.setdefault(n_cell, set()).add(cell_id)

    # ----------------------------------------------------------
    # TOPIC INDEX
    # ----------------------------------------------------------

    def _index_topic(self, entry: MemoryEntry):
        topic = entry.metadata.get("topic", "")
        if not topic:
            return
        claim = entry.metadata.get("claim", "")
        self._topic_index.setdefault(topic, []).append(
            (entry.memory_id, entry.cell_id, claim)
        )

    # ----------------------------------------------------------
    # STORE
    # ----------------------------------------------------------

    def store(
        self,
        content: str,
        embedding: np.ndarray,
        layer: str = "semantic",
        state: MemoryState = MemoryState.NEUTRAL,
        session_id: str = "default",
        author_id: str = "user",
        metadata: Optional[Dict] = None,
    ) -> MemoryEntry:
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {embedding.shape}")
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains NaN or Inf values — cannot store corrupted vector")

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        memory_id = f"mem_{content_hash[:16]}_{int(time.time() * 1000)}_{self._next_cell_id}"
        cell_id = self._next_cell_id
        self._next_cell_id += 1

        self._points[cell_id] = embedding.copy()
        self._cell_to_memid[cell_id] = memory_id
        self._active_cells.add(cell_id)          # register as live cell
        self._kdtree_dirty = True  # Lazy — don't rebuild on every store

        fingerprint = self.stylometric.extract(content, author_id)
        if author_id not in self._author_fingerprints:
            self._author_fingerprints[author_id] = fingerprint

        entry = MemoryEntry(
            memory_id=memory_id, layer=layer, content=content,
            content_hash=content_hash, embedding=embedding.copy(),
            state=state, cell_id=cell_id, created_at=time.time(),
            session_id=session_id, author_id=author_id,
            metadata=metadata or {}, fingerprint=fingerprint,
        )

        self._db.store_memory(entry)
        self._index_topic(entry)
        self._auto_link_contradictions(entry)

        return entry

    def _auto_link_contradictions(self, new_entry: MemoryEntry):
        """Create INHIBITORY links between memories with same topic but different claims."""
        topic = new_entry.metadata.get("topic", "")
        new_claim = new_entry.metadata.get("claim", "")
        if not topic or not new_claim:
            return

        for mem_id, cell_id, claim in self._topic_index.get(topic, []):
            if mem_id == new_entry.memory_id:
                continue
            if claim and claim != new_claim:
                if (new_entry.cell_id, cell_id, "INHIBITORY") in self._linked_pairs:
                    continue  # P1: already inhibitory-linked — skip redundant writes
                self._db.store_cell_link(new_entry.cell_id, cell_id, LinkType.INHIBITORY, auto=True)
                self._db.store_cell_link(cell_id, new_entry.cell_id, LinkType.INHIBITORY, auto=True)
                self._linked_pairs.add((new_entry.cell_id, cell_id, "INHIBITORY"))
                self._linked_pairs.add((cell_id, new_entry.cell_id, "INHIBITORY"))
                logger.info(
                    f"INHIBITORY link: cell {new_entry.cell_id} ({new_claim}) "
                    f"↔ cell {cell_id} ({claim}) [topic={topic}]"
                )

    # ----------------------------------------------------------
    # RECALL
    # ----------------------------------------------------------

    def recall(
        self,
        query_embedding: np.ndarray,
        query_text: Optional[str] = None,
        top_k: int = 5,
        hops: int = 2,
        layer_filter: Optional[str] = None,
        current_turn_memories: Optional[List[str]] = None,
    ) -> Tuple[List[RecallResult], AuditLog]:

        # P0-3: reject wrong dimensions early (store() does this too)
        if query_embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"recall() embedding shape {query_embedding.shape} "
                f"does not match engine dim ({self.embedding_dim},)"
            )
        self._ensure_kdtree()

        if self.kdtree is None or not self._active_cells:
            audit = self._build_audit(query_text, query_embedding, set(), [], 0, 0, 0, 0, 0)
            self._db.store_audit(audit)
            return [], audit

        # ---- Find seed cell (nearest neighbour) ----
        # KDTree is built over active cells only; idx is an array position.
        _, idx = self.kdtree.query(query_embedding.reshape(1, -1))
        query_cell = self._kdtree_idx_to_cell[int(idx[0])]  # array pos → cell_id

        # ---- Batch load all cell links before BFS ----
        # Avoids one SQL query per cell per hop (was ~43 queries for hops=2, k=6).
        # The dict is read-only within this recall call; no invalidation needed.
        all_cell_links: Dict[int, List[Tuple[int, LinkType]]] = (
            self._db.load_all_cell_links_indexed()
        )

        # ---- BFS hop expansion with ternary links ----
        activated_cells: Set[int] = set()
        inhibited_cells: Set[int] = set()
        resonant_boosts: Dict[int, float] = {}
        frontier = {query_cell}

        for _ in range(hops + 1):
            new_frontier: Set[int] = set()
            for cell in frontier:
                if cell in inhibited_cells:
                    continue
                activated_cells.add(cell)

                for neighbor in self.cell_neighbors.get(cell, set()):
                    new_frontier.add(neighbor)

                for target_id, link_type in all_cell_links.get(cell, []):
                    if link_type == LinkType.INHIBITORY:
                        inhibited_cells.add(target_id)
                    elif link_type == LinkType.RESONANT:
                        new_frontier.add(target_id)
                        resonant_boosts[target_id] = resonant_boosts.get(target_id, 0) + 0.5
                    else:
                        new_frontier.add(target_id)

            frontier = new_frontier - activated_cells - inhibited_cells

        # ---- Rescue REINFORCED cells that got inhibited during BFS ----
        # A validated truth cannot be silenced by an unverified claim:
        # if any inhibited cell holds a REINFORCED memory, restore it.
        if inhibited_cells:
            candidate_mems = self._db.load_memories(cell_ids=list(inhibited_cells))
            for _m in candidate_mems:
                if _m.state == MemoryState.REINFORCED:
                    inhibited_cells.discard(_m.cell_id)
                    activated_cells.add(_m.cell_id)

        # ---- Load and score candidates ----
        memories = self._db.load_memories(cell_ids=list(activated_cells))
        if layer_filter:
            memories = [m for m in memories if m.layer == layer_filter]

        total_cands = len(memories)
        f_state = f_estilo = f_inhib = 0
        results: List[RecallResult] = []
        now = time.time()

        for mem in memories:
            if mem.state == MemoryState.FORGOTTEN:
                f_state += 1
                continue

            if mem.cell_id in inhibited_cells and mem.cell_id != query_cell:
                f_inhib += 1
                continue

            # Stylometric forensic check (only meaningful for texts of ≥15 words)
            if (mem.fingerprint and mem.author_id in self._author_fingerprints
                    and len(mem.content.split()) >= 15):
                historical = self._author_fingerprints[mem.author_id]
                # P1: a bilingual author is not a tamperer. Function-word
                # profiles are only comparable within one language — skip
                # the forensic comparison on a confirmed language switch.
                lang_a = getattr(mem.fingerprint, "language", "und")
                lang_b = getattr(historical, "language", "und")
                if lang_a != lang_b and "und" not in (lang_a, lang_b):
                    logger.info(
                        f"Stylometric check skipped for {mem.memory_id[:16]}: "
                        f"language switch {lang_b}→{lang_a} (not evidence of tampering)"
                    )
                else:
                    dist = self.stylometric.compare(mem.fingerprint, historical)
                    if dist > ESTILOMETRIA_THRESHOLD:
                        self._db.update_state(mem.memory_id, MemoryState.FORGOTTEN)
                        self._db.store_alert(ForensicAlert(
                            alert_id=f"alert_{int(time.time()*1000)}_{mem.memory_id[:8]}",
                            timestamp=now,
                            memory_id=mem.memory_id,
                            detected_author="UNKNOWN_TAMPERER",
                            expected_author=mem.author_id,
                            mismatch_score=dist,
                            action_taken="DEGRADED_TO_FORGOTTEN",
                        ))
                        self._active_cells.discard(mem.cell_id)
                        self._kdtree_dirty = True
                        f_estilo += 1
                        logger.warning(f"Forensic: tampered memory {mem.memory_id[:16]} → FORGOTTEN (dist={dist:.3f})")
                        continue

            # Score components
            sim = float(self._cosine_sim(query_embedding, mem.embedding))
            state_boost = mem.state.value
            hop_dist = self._hop_distance(query_cell, mem.cell_id)
            hop_decay = math.exp(-HOP_LAMBDA * hop_dist) if hop_dist >= 0 else 1.0
            resonant_boost = resonant_boosts.get(mem.cell_id, 0.0)

            synaptic_boost = 0.0
            if current_turn_memories:
                for act_id in current_turn_memories:
                    if act_id in mem.synaptic_links:
                        synaptic_boost += mem.synaptic_links[act_id]

            recency_bonus = 0.0
            if mem.last_activation > 0:
                age = now - mem.last_activation
                recency_bonus = RECENCY_WEIGHT * math.exp(-math.log(2) * age / RECENCY_HALFLIFE)

            # P1-1: resonant contribution scaled by similarity.
            # Flat additive would let an irrelevant-but-linked memory
            # outrank a semantically relevant one.
            resonant_contribution = resonant_boost * min(sim, 1.0)
            final_score = (
                sim * state_boost * hop_decay
                + resonant_contribution
                + synaptic_boost * 0.3
                + recency_bonus
            )
            # P0: anti-correlated embeddings (sim < 0) could push the final
            # score negative. Negative magnitudes carry no ranking information
            # the agent should act on — clamp to the floor of irrelevance.
            final_score = max(0.0, final_score)

            # ---- Spectral resonance + coherence (epistemic metadata) ----
            # These do NOT modify final_score; they are reported as audit metadata
            # so the agent can inspect field-level alignment and graph integrity.
            spectral_res = 0.0
            coherence = 1.0
            if self._spectral is not None and self._spectral.is_built:
                try:
                    spectral_res = self._spectral.resonance(query_embedding, mem.embedding)
                    # Reuse the already-loaded cell_links dict — no extra DB call.
                    mem_links = all_cell_links.get(mem.cell_id, [])
                    r_links = sum(1 for _, lt in mem_links if lt == LinkType.RESONANT)
                    i_links = sum(1 for _, lt in mem_links if lt == LinkType.INHIBITORY)
                    coherence = SpectralField.coherence(r_links, i_links)
                except Exception:
                    pass  # spectral is optional — never crash recall()

            results.append(RecallResult(
                memory=mem, base_score=sim, state_boost=state_boost,
                hop_decay=hop_decay, synaptic_boost=synaptic_boost,
                recency_bonus=recency_bonus, final_score=final_score,
                hop_distance=hop_dist, cell_id=mem.cell_id, source="similarity",
                resonance_score=round(spectral_res, 4),
                coherence_score=round(coherence, 4),
            ))

        # ---- Synaptic pull (STDP-driven cross-turn association) ----
        # Batch-load to avoid N+1 queries (was N + N*M individual DB calls).
        synaptic_count = 0
        if current_turn_memories:
            existing_ids = {r.memory.memory_id for r in results}

            # Batch 1: load all act_mems in one query
            act_mems_batch = {
                m.memory_id: m
                for m in self._db.load_memories_by_ids(current_turn_memories)
            }

            # Collect all linked_ids that clear the weight filter,
            # keeping the highest weight when multiple sources point to the same target.
            link_candidates: Dict[str, float] = {}
            for act_id in current_turn_memories:
                act_mem = act_mems_batch.get(act_id)
                if not act_mem:
                    continue
                for linked_id, weight in act_mem.synaptic_links.items():
                    if weight >= 0.5 and linked_id not in existing_ids:
                        link_candidates[linked_id] = max(
                            link_candidates.get(linked_id, 0.0), weight
                        )

            # Batch 2: load all candidate linked memories in one query
            if link_candidates:
                linked_batch = {
                    m.memory_id: m
                    for m in self._db.load_memories_by_ids(list(link_candidates.keys()))
                }
                for linked_id, weight in link_candidates.items():
                    linked = linked_batch.get(linked_id)
                    if not linked or linked.state == MemoryState.FORGOTTEN:
                        continue
                    if linked.cell_id in inhibited_cells:
                        continue
                    results.append(RecallResult(
                        memory=linked, base_score=0.0, state_boost=linked.state.value,
                        hop_decay=1.0, synaptic_boost=weight, recency_bonus=0.0,
                        final_score=weight * 0.3, hop_distance=-1,
                        cell_id=linked.cell_id, source="synaptic",
                    ))
                    existing_ids.add(linked_id)
                    synaptic_count += 1

        results.sort(key=lambda r: r.final_score, reverse=True)
        top_results = results[:top_k]

        # ---- Update STDP weights ----
        if current_turn_memories:
            self._update_stdp(current_turn_memories, [r.memory.memory_id for r in top_results])

        # ---- Update activation timestamps ----
        for r in top_results:
            self._db.update_activation(r.memory.memory_id, now)

        audit = self._build_audit(
            query_text, query_embedding, activated_cells, top_results,
            total_cands, f_state, f_estilo, f_inhib, synaptic_count,
        )
        self._db.store_audit(audit)

        return top_results, audit

    # ----------------------------------------------------------
    # STDP — Long-Term Potentiation + Depression
    # ----------------------------------------------------------

    def _update_stdp(self, previous: List[str], current: List[str]):
        curr_set = set(current)
        # Batch load all previous memories in one query (avoids N individual DB calls).
        prev_mems = {
            m.memory_id: m
            for m in self._db.load_memories_by_ids(previous)
        }
        for prev_id in previous:
            mem = prev_mems.get(prev_id)
            if not mem:
                continue
            links = dict(mem.synaptic_links)

            for curr_id in curr_set:
                if curr_id == prev_id:
                    continue
                # LTP: co-activated → strengthen
                links[curr_id] = min(STDP_MAX_WEIGHT, links.get(curr_id, 0.0) + STDP_POTENTIATION)

            # LTD: had a link but target NOT in current turn → slight weakening
            for eid in list(links.keys()):
                if eid not in curr_set and eid != prev_id:
                    links[eid] = max(STDP_MIN_WEIGHT, links[eid] - STDP_DEPRESSION)
                    # P0: float accumulation (0.10/0.02 steps) leaves residues
                    # ~1e-17 that `== 0.0` never matches — dead links would
                    # pile up forever. EPS threshold actually prunes them.
                    if links[eid] <= STDP_PRUNE_EPS:
                        del links[eid]

            self._db.update_synaptic_links(prev_id, links)

    # ----------------------------------------------------------
    # STATE MANAGEMENT
    # ----------------------------------------------------------

    def reinforce(self, memory_id: str) -> MemoryEntry:
        self._db.update_state(memory_id, MemoryState.REINFORCED)
        m = self._db.load_memory(memory_id)
        if not m:
            raise KeyError(f"Memory {memory_id} not found")
        # A memory may have been FORGOTTEN (removed from the KDTree) before being
        # reinforced — the API allows that transition. Re-register it as a live
        # cell and rebuild the index, otherwise a reinforced memory would stay
        # invisible to recall forever.
        if m.cell_id not in self._active_cells:
            self._points[m.cell_id] = m.embedding.copy()
            self._active_cells.add(m.cell_id)
            self._cell_to_memid[m.cell_id] = m.memory_id
            self._kdtree_dirty = True
        logger.info(f"Reinforced: {memory_id}")
        return m

    def forget(self, memory_id: str) -> MemoryEntry:
        self._db.update_state(memory_id, MemoryState.FORGOTTEN)
        m = self._db.load_memory(memory_id)
        if not m:
            raise KeyError(f"Memory {memory_id} not found")
        # P0-1e: remove from active cells so KDTree excludes this ghost
        self._active_cells.discard(m.cell_id)
        self._kdtree_dirty = True
        # Remove from topic index to prevent ghost contradictions
        for topic, entries in list(self._topic_index.items()):
            self._topic_index[topic] = [e for e in entries if e[1] != m.cell_id]
            if not self._topic_index[topic]:
                del self._topic_index[topic]
        logger.info(f"Forgotten: {memory_id}")
        return m

    def create_cell_link(self, from_cell_id: int, to_cell_id: int, link_type: LinkType):
        """Manually create a ternary cell link."""
        self._db.store_cell_link(from_cell_id, to_cell_id, link_type, auto=False)
        self._linked_pairs.add((from_cell_id, to_cell_id, link_type.name))
        if link_type == LinkType.RESONANT:
            self._resonant_neighbors.setdefault(from_cell_id, []).append((to_cell_id, link_type))
            self._resonant_neighbors.setdefault(to_cell_id, []).append((from_cell_id, link_type))

    # ----------------------------------------------------------
    # STATS & EXPORT
    # ----------------------------------------------------------

    def get_stats(self) -> Dict:
        # Bug #24: aggregate in SQL instead of materialising every row.
        # This runs on every recall / broadcast / dashboard tick, so it must
        # not scale with corpus size.
        agg = self._db.count_stats()
        state_dist = agg["state_distribution"]
        layer_dist = agg["layer_distribution"]

        reinforced_w = state_dist["REINFORCED"] * 1.5
        neutral_w = state_dist["NEUTRAL"] * 1.0
        total_w = reinforced_w + neutral_w
        mss = round(reinforced_w / total_w, 4) if total_w > 0 else 0.0

        # P1: MSS is computed over LIVE memories only — with 1 REINFORCED and
        # 99 FORGOTTEN it reads 1.0, which is technically true but misleading.
        # retention_ratio exposes the denominator the MSS silently drops.
        total_all = agg["total_memories"]
        live = state_dist["REINFORCED"] + state_dist["NEUTRAL"]
        retention = round(live / total_all, 4) if total_all > 0 else 0.0

        avg_neighbors = float(np.mean([len(n) for n in self.cell_neighbors.values()])) \
            if self.cell_neighbors else 0.0

        return {
            "total_memories": agg["total_memories"],
            "voronoi_cells": len(self._active_cells),
            "state_distribution": state_dist,
            "layer_distribution": layer_dist,
            "memory_stability_score": mss,
            "retention_ratio": retention,
            "avg_neighbors": round(avg_neighbors, 2),
            "total_recalls": agg["total_recalls"],
            "cell_links": agg["cell_links"],
            "authors": agg["authors"],
        }

    def export_graph(self, max_nodes: int = 1000) -> Dict:
        """
        Export the memory graph for external visualisation.

        Bug #22: capped. An uncapped export over 50k memories returns 50k nodes
        and potentially far more edges — tens of MB of JSON that times out the
        request. We return the most-recalled `max_nodes` cells and only the
        edges between them, plus a `truncated` flag so the client knows.
        """
        mems = self._db.load_memories()
        total = len(mems)
        truncated = total > max_nodes
        if truncated:
            mems = sorted(mems, key=lambda m: m.recall_count, reverse=True)[:max_nodes]

        included = {m.cell_id for m in mems}
        all_links = self._db.load_all_cell_links()

        nodes = [
            {
                "id": m.cell_id,
                "memory_id": m.memory_id,
                "label": m.content[:40] + "…",
                "state": m.state.name,
                "layer": m.layer,
                "recall_count": m.recall_count,
            }
            for m in mems
        ]
        edges = [
            {"from": f, "to": t, "type": lt}
            for f, t, lt in all_links
            if f in included and t in included
        ]
        return {"nodes": nodes, "edges": edges, "truncated": truncated, "total_nodes": total}

    def list_memories(
        self,
        layer: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MemoryEntry]:
        return self._db.load_memories(layer=layer, state=state, limit=limit, offset=offset)

    def get_audit_trail(self, limit: int = 100) -> List[Dict]:
        return self._db.get_audit_trail(limit)

    def get_alerts(self, limit: int = 50) -> List[ForensicAlert]:
        return self._db.get_alerts(limit)

    # ----------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------

    def _build_audit(
        self, query_text, query_embedding, cells, results,
        total_cand, f_state, f_est, f_inhib, synaptic_act,
    ) -> AuditLog:
        prev = self._db.get_prev_audit_hash()

        # P0: capture ONE timestamp — it goes both into the hash payload and
        # the stored row, so the chain is recomputable from persisted data.
        ts = time.time()
        cells_sorted = sorted(cells)

        # P0: content_hash travels inside the hashed payload. Modifying a
        # memory's text after it was audited now breaks hash recomputation.
        mem_dicts = [
            {
                "memory_id": r.memory.memory_id,
                "content_hash": r.memory.content_hash,
                "content_preview": r.memory.content[:100],
                "state": r.memory.state.name,
                "base_score": round(r.base_score, 4),
                "state_boost": r.state_boost,
                "hop_decay": round(r.hop_decay, 4),
                "synaptic_boost": round(r.synaptic_boost, 4),
                "recency_bonus": round(r.recency_bonus, 4),
                "final_score": round(r.final_score, 4),
                "hop_distance": r.hop_distance,
                "source": r.source,
                "resonance_score": r.resonance_score,
                "coherence_score": r.coherence_score,
            }
            for r in results
        ]

        # P0: .tolist() guard — query_embedding may arrive as ndarray or list.
        qe = None
        if query_embedding is not None:
            qe = (query_embedding.tolist()
                  if isinstance(query_embedding, np.ndarray)
                  else list(query_embedding))

        audit_hash = compute_audit_hash(
            ts, "recall", query_text, cells_sorted, mem_dicts, prev, qe,
        )

        return AuditLog(
            timestamp=ts, operation="recall", query_text=query_text,
            query_embedding=qe,
            cells_activated=cells_sorted,
            memories_retrieved=mem_dicts,
            total_candidates=total_cand,
            filtered_by_state=f_state,
            filtered_by_estilometria=f_est,
            filtered_by_inhibitory=f_inhib,
            synaptic_activated=synaptic_act,
            returned_to_agent=len(results),
            audit_hash=audit_hash,
            prev_hash=prev,
        )

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        # P0-4: check norms independently; < (not <=) prevents the edge case
        # where norm == 1e-10 passes the guard but div still produces NaN
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        sim = float(np.dot(a, b)) / (norm_a * norm_b)
        # Clamp to the mathematical range. Floating-point rounding can yield
        # 1.0000000000002, which would otherwise leak into final_score and let
        # a recall score drift above its state-boost ceiling.
        return max(-1.0, min(1.0, sim))

    def _hop_distance(self, from_cell: int, to_cell: int) -> int:
        if from_cell == to_cell:
            return 0
        visited = {from_cell}
        frontier = {from_cell}
        hops = 0
        while frontier and hops < MAX_HOP_SEARCH:
            hops += 1
            new_frontier: Set[int] = set()
            for cell in frontier:
                # k-NN neighbors
                for n in self.cell_neighbors.get(cell, set()):
                    if n == to_cell:
                        return hops
                    if n not in visited:
                        visited.add(n)
                        new_frontier.add(n)
                # RESONANT cell_links — extend BFS through explicit links
                for linked_cell, link_type in self._resonant_neighbors.get(cell, []):
                    if linked_cell == to_cell:
                        return hops
                    if linked_cell not in visited:
                        visited.add(linked_cell)
                        new_frontier.add(linked_cell)
            frontier = new_frontier
        return -1  # unreachable


    def rebuild_spectral_field(self) -> bool:
        """
        Rebuild the spectral field from current active memories and persist to DB.

        Call after bulk store operations or sleep consolidation.
        No-op (returns False) if the spectral module is not installed or fewer
        than 2 active memories exist.

        The field is stored in-process in self._spectral and persisted to the
        same SQLite DB as the rest of raven-memory. On the next engine restart
        it will be loaded automatically.
        """
        if not _SPECTRAL_AVAILABLE:
            logger.warning("spectral module not available — install spectral.py to enable")
            return False
        mems = self._db.load_memories()
        active_embs = [m.embedding for m in mems if m.state != MemoryState.FORGOTTEN]
        if len(active_embs) < 2:
            logger.info("Spectral rebuild skipped: fewer than 2 active memories")
            return False
        field = SpectralField(self.embedding_dim)
        try:
            ok = field.build_from_memories(active_embs)
        except Exception as _e:
            # spectral v2.5 raises TensorValidationError on corrupt embeddings;
            # a corrupt row must not crash the engine — log and skip rebuild.
            logger.warning(f"Spectral rebuild aborted: {_e}")
            return False
        if ok:
            self._spectral = field
            try:
                SpectralStore(self._db.db_path).save(field)
            except Exception as _e:
                logger.warning(f"Could not persist spectral field: {_e}")
            logger.info(f"Spectral field rebuilt: {field.summary()}")
        return ok


if __name__ == "__main__":
    print("raven-memory v1.0 — Engine OK")
