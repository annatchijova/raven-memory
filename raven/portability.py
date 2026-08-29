#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — Field export/import (plan 3.2).

Portable JSONL snapshot of an entire memory field: memories (embeddings as
base64 float32), cell links, the full audit chain, and forensic alerts.

Honesty contract:
  - A FULL export → import preserves the audit chain verbatim, so
    verify_audit_chain() reports on the imported copy exactly what it
    reported on the source (including pre-existing breaks — importing
    never launders a broken chain).
  - Import refuses a non-empty target database: merging two fields would
    interleave two hash chains and make both unverifiable.

CLI:
    raven-export --db raven_memory.db --out field.jsonl
    raven-import --in field.jsonl --db new_field.db

Authors: Anna Tchijova + Claude (VIGÍA AI Collective)
License: Apache 2.0
"""

import argparse
import base64
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raven.memory_engine import MemoryStore, verify_audit_chain

FORMAT_VERSION = 1

_MEMORY_COLS = (
    "memory_id", "layer", "content", "content_hash", "embedding", "state",
    "cell_id", "created_at", "session_id", "author_id", "metadata",
    "synaptic_links", "last_activation", "recall_count", "fingerprint",
)
_LINK_COLS = ("from_cell_id", "to_cell_id", "link_type", "created_at", "auto_generated")
_AUDIT_COLS = (
    "id", "timestamp", "operation", "query_text", "query_embedding",
    "cells_activated", "memories_retrieved", "total_candidates",
    "filtered_by_state", "filtered_by_estilometria", "filtered_by_inhibitory",
    "synaptic_activated", "returned_to_agent", "audit_hash", "prev_hash",
    "qemb_sha256",
)
_ALERT_COLS = (
    "alert_id", "timestamp", "memory_id", "detected_author",
    "expected_author", "mismatch_score", "action_taken",
)


def export_field(db_path: Path, out_path: Path) -> Dict:
    db_path, out_path = Path(db_path), Path(out_path)
    if not db_path.exists():
        return {"error": f"Database not found: {db_path}"}

    counts = {"memories": 0, "cell_links": 0, "audit_log": 0, "forensic_alerts": 0}
    with sqlite3.connect(db_path) as conn, open(out_path, "w", encoding="utf-8") as out:
        conn.row_factory = sqlite3.Row
        schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
        out.write(json.dumps({
            "kind": "header",
            "format_version": FORMAT_VERSION,
            "schema_version": schema_version,
            "source": db_path.name,
        }) + "\n")

        for row in conn.execute(
                f"SELECT {','.join(_MEMORY_COLS)} FROM memories ORDER BY cell_id"):
            rec = dict(row)
            rec["embedding"] = base64.b64encode(rec["embedding"]).decode("ascii")
            out.write(json.dumps({"kind": "memory", **rec}, ensure_ascii=False) + "\n")
            counts["memories"] += 1

        for row in conn.execute(
                f"SELECT {','.join(_LINK_COLS)} FROM cell_links "
                "ORDER BY from_cell_id, to_cell_id"):
            out.write(json.dumps({"kind": "cell_link", **dict(row)}) + "\n")
            counts["cell_links"] += 1

        # Audit rows in id order — the id sequence IS the chain order.
        for row in conn.execute(
                f"SELECT {','.join(_AUDIT_COLS)} FROM audit_log ORDER BY id"):
            out.write(json.dumps({"kind": "audit", **dict(row)}, ensure_ascii=False) + "\n")
            counts["audit_log"] += 1

        for row in conn.execute(
                f"SELECT {','.join(_ALERT_COLS)} FROM forensic_alerts ORDER BY timestamp"):
            out.write(json.dumps({"kind": "alert", **dict(row)}, ensure_ascii=False) + "\n")
            counts["forensic_alerts"] += 1

    return {"out": str(out_path), **counts}


def import_field(in_path: Path, db_path: Path) -> Dict:
    in_path, db_path = Path(in_path), Path(db_path)
    if not in_path.exists():
        return {"error": f"Export file not found: {in_path}"}

    # Refuse a non-empty target: two interleaved hash chains verify as broken
    # and there is no honest way to merge them.
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            try:
                existing = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            except sqlite3.OperationalError:
                existing = 0
        if existing:
            return {"error": f"Target DB {db_path} already holds {existing} memories — "
                             "import requires an empty or absent target."}

    store = MemoryStore(db_path)   # creates schema at current version
    counts = {"memories": 0, "cell_links": 0, "audit_log": 0, "forensic_alerts": 0}
    header = None

    with sqlite3.connect(db_path) as conn, open(in_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            kind = rec.pop("kind")
            if kind == "header":
                header = rec
                if rec.get("format_version", 0) > FORMAT_VERSION:
                    return {"error": f"Export format v{rec['format_version']} is newer "
                                     f"than this code supports (v{FORMAT_VERSION})."}
                continue
            if kind == "memory":
                rec["embedding"] = base64.b64decode(rec["embedding"])
                conn.execute(
                    f"INSERT INTO memories ({','.join(_MEMORY_COLS)}) "
                    f"VALUES ({','.join('?' * len(_MEMORY_COLS))})",
                    [rec[c] for c in _MEMORY_COLS])
                counts["memories"] += 1
            elif kind == "cell_link":
                conn.execute(
                    f"INSERT INTO cell_links ({','.join(_LINK_COLS)}) "
                    f"VALUES ({','.join('?' * len(_LINK_COLS))})",
                    [rec[c] for c in _LINK_COLS])
                counts["cell_links"] += 1
            elif kind == "audit":
                # Explicit ids preserve the exact chain sequence.
                conn.execute(
                    f"INSERT INTO audit_log ({','.join(_AUDIT_COLS)}) "
                    f"VALUES ({','.join('?' * len(_AUDIT_COLS))})",
                    [rec.get(c) for c in _AUDIT_COLS])
                counts["audit_log"] += 1
            elif kind == "alert":
                conn.execute(
                    f"INSERT INTO forensic_alerts ({','.join(_ALERT_COLS)}) "
                    f"VALUES ({','.join('?' * len(_ALERT_COLS))})",
                    [rec.get(c) for c in _ALERT_COLS])
                counts["forensic_alerts"] += 1
        conn.commit()

    # Verify the imported chain and REPORT — never assert success blindly.
    entries = store.get_audit_trail(limit=100000)
    report = verify_audit_chain(entries) if entries else {
        "chain_intact": True, "hash_integrity": True, "issues": []}

    return {
        "db": str(db_path),
        "header": header,
        **counts,
        "chain_intact": report["chain_intact"],
        "hash_integrity": report["hash_integrity"],
        "issues": report["issues"],
    }


def main_export():
    p = argparse.ArgumentParser(description="Export a raven-memory field to JSONL")
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = export_field(Path(args.db), Path(args.out))
    print(json.dumps(result, indent=2))
    sys.exit(1 if "error" in result else 0)


def main_import():
    p = argparse.ArgumentParser(description="Import a raven-memory field from JSONL")
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--db", required=True)
    args = p.parse_args()
    result = import_field(Path(args.infile), Path(args.db))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(1 if "error" in result else 0)


if __name__ == "__main__":
    main_export()
