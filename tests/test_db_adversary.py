#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — adversary for a future DBWitness, written BEFORE the witness.

The point of the order: designing the canonicaliser first is how you fall in love
with the hash. These are concrete pairs of databases that a plausible "semantic
snapshot" calls EQUAL and that RAVEN distinguishes behaviourally. Each one is a
requirement DBWitness has to meet, expressed as a failing oracle for the naive
snapshot rather than as prose.

The promotion property they serve:

    DBWitness(A) == DBWitness(B)
      + same non-DB semantic state
      + same operation inputs
    ⇒ same DB-dependent observable behaviour

That cannot be proven by tests. Trying to falsify it is what forces the missing
causal state into the open, and it already has: three times here.

The dangerous direction is asymmetric. `DBWitness !=` with identical behaviour is
a conservative over-approximation. `DBWitness ==` with different behaviour is the
failure that makes a purity gate a lie.

Nothing here implements DBWitness.

Run: pytest tests/test_db_adversary.py -q
"""

import hashlib
import json
import shutil
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raven.memory_engine import AdaptiveMemoryEngine

TEXTS = [
    "el motor es determinista de punta a punta y la cadena de auditoria lo demuestra",
    "cada recuerdo vive en exactamente un estado de recuperacion dentro del campo",
    "la propagacion por saltos amplifica ramas resonantes y silencia contradictorias",
    "un hecho validado nunca puede ser silenciado por una afirmacion sin verificar",
    "las memorias episodicas se consolidan por agrupamiento coseno mientras duerme",
    "el puntaje final combina similitud estado decaimiento por salto y recencia",
    "la huella estilometrica compara palabras funcionales longitud y puntuacion",
    "los enlaces ternarios conectan celdas resonante neutra o inhibitoria siempre",
    "la plasticidad dependiente del tiempo de disparo refuerza pares coactivados",
    "el indice espacial se reconstruye de forma perezosa cuando el campo cambia",
    "olvidar es exclusion del campo activo y nunca destruccion del registro",
    "el campo espectral expone resonancia y coherencia como metadatos aparte",
    "las intervenciones causales miden influencia sobre la recuperacion no el agente",
    "la degradacion honesta grita en los registros en vez de matematica silenciosa",
]


def emb(text: str, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
    e = rng.standard_normal(dim).astype(np.float32)
    return e / np.linalg.norm(e)


def build(path: Path) -> AdaptiveMemoryEngine:
    eng = AdaptiveMemoryEngine(db_path=path)
    for t in TEXTS:
        eng.store(t, emb(t), author_id="anna")
    return eng


def clone(src: Path, dst: Path):
    """WAL matters: the main file is not the whole state until a checkpoint.
    A DBWitness that reads the file without checkpointing would observe a
    different database than the engine does."""
    with sqlite3.connect(src) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copy(src, dst)


def naive_snapshot(db: Path) -> str:
    """The canonicaliser one is tempted to write: keyed by memory_id, cell_id
    normalised away as an "internal id", rows as sets, audit ignored."""
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        mems = sorted(
            (r["memory_id"], r["content_hash"], r["state"], r["layer"],
             r["author_id"], round(r["last_activation"], 6),
             r["synaptic_links"], r["metadata"])
            for r in conn.execute("SELECT * FROM memories"))
        links = sorted(set(tuple(r) for r in conn.execute(
            "SELECT from_cell_id, to_cell_id, link_type FROM cell_links")))
    return hashlib.sha256(
        json.dumps([mems, links], sort_keys=True, default=str).encode()
    ).hexdigest()


def naive_active_only(db: Path) -> str:
    """An even more common shortcut: "only what is retrievable matters"."""
    with sqlite3.connect(db) as conn:
        rows = sorted(conn.execute(
            "SELECT memory_id, content_hash, state FROM memories "
            "WHERE state != 'FORGOTTEN'"))
    return hashlib.sha256(str(rows).encode()).hexdigest()


def permute_cell_ids(db: Path):
    with sqlite3.connect(db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.execute("UPDATE memories SET cell_id = cell_id + 100000")
        for old in range(n):
            conn.execute("UPDATE memories SET cell_id=? WHERE cell_id=?",
                         (n - 1 - old, old + 100000))
        conn.commit()


# ============================================================
# Adversary 1 — cell_id is NOT an irrelevant internal id
# ============================================================

def test_ADVERSARY_permuted_cell_ids_look_equal_but_change_forensics(tmp_path):
    """`cell_id` is the most tempting field to normalise away. It is causal.

    `load_memories()` orders by cell_id, and `_load_from_db` feeds every
    fingerprint into a rolling 10-sample author profile in that order. With more
    memories than the window, permuting cell_ids changes WHICH samples survive,
    hence the mean fingerprint, hence every future stylometric distance — the
    quantity a forensic alert is thresholded on.
    """
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    build(a)
    clone(a, b)
    permute_cell_ids(b)

    assert naive_snapshot(a) == naive_snapshot(b), \
        "the adversary is only interesting if the naive snapshot calls them equal"

    ea, eb = AdaptiveMemoryEngine(db_path=a), AdaptiveMemoryEngine(db_path=b)
    key = ("anna", "es")
    assert ea._author_profiles[key].count == 10, "window must be saturated"

    pa = asdict(ea._author_profiles[key].mean_fingerprint())
    pb = asdict(eb._author_profiles[key].mean_fingerprint())
    assert pa != pb, "permutation no longer moves the profile"
    assert pa["avg_sentence_length"] != pb["avg_sentence_length"]

    # And the behavioural quantity, not just internal state: the forensic
    # distance for one identical probe text.
    probe = ("texto sospechoso con un estilo bastante distinto porque usa "
             "muchisimas palabras raras sin funcionales frecuentes")
    def distance(eng):
        fp = eng.stylometric.extract(probe, "anna")
        return eng.stylometric.compare(
            fp, eng._author_profiles[("anna", fp.language)].mean_fingerprint())

    da, db = distance(ea), distance(eb)
    assert da != db, "the forensic distance is identical — adversary neutralised"
    # Measured 0.760 vs 0.682 against ESTILOMETRIA_THRESHOLD = 0.5. Both fire
    # here; a probe whose distance falls between the two values receives
    # OPPOSITE verdicts, which follows from the threshold being a constant.
    # Stated as the arithmetic consequence it is, not as a measured flip.
    assert abs(da - db) > 1e-6


# ============================================================
# Adversary 2 — the audit chain head is DB state
# ============================================================

def test_ADVERSARY_audit_head_looks_equal_but_changes_the_next_seal(tmp_path):
    """A snapshot of memories and links ignores `audit_log`, so two databases
    differing only in their chain head compare equal — and the very next recall
    seals a different hash. A probe able to move the head unobserved is the
    precise shape of a false "read-only"."""
    c, d = tmp_path / "c.db", tmp_path / "d.db"
    ec = build(c)
    ec.recall(emb(TEXTS[1]))
    clone(c, d)
    with sqlite3.connect(d) as conn:
        conn.execute("DELETE FROM audit_log WHERE id=(SELECT MAX(id) FROM audit_log)")
        conn.commit()

    assert naive_snapshot(c) == naive_snapshot(d)

    e1, e2 = AdaptiveMemoryEngine(db_path=c), AdaptiveMemoryEngine(db_path=d)
    _, a1 = e1.recall(emb(TEXTS[2]))
    _, a2 = e2.recall(emb(TEXTS[2]))
    assert a1.prev_hash != a2.prev_hash
    assert a1.audit_hash != a2.audit_hash


# ============================================================
# Adversary 3 — FORGOTTEN is not deletion
# ============================================================

def test_ADVERSARY_forgotten_versus_deleted_looks_equal_but_is_not(tmp_path):
    """RAVEN's own stance is that forgetting is exclusion, not destruction. A
    snapshot restricted to retrievable memories erases exactly that
    distinction."""
    e_path, f_path = tmp_path / "e.db", tmp_path / "f.db"
    ee = build(e_path)
    victim = ee.list_memories(limit=100)[3]
    ee.forget(victim.memory_id)
    clone(e_path, f_path)
    with sqlite3.connect(f_path) as conn:
        conn.execute("DELETE FROM memories WHERE memory_id=?", (victim.memory_id,))
        conn.commit()

    assert naive_active_only(e_path) == naive_active_only(f_path)

    e3, e4 = AdaptiveMemoryEngine(db_path=e_path), AdaptiveMemoryEngine(db_path=f_path)
    assert len(e3.list_memories(limit=100)) == len(e4.list_memories(limit=100)) + 1
    assert e3.reinforce(victim.memory_id).state.name == "REINFORCED"
    with pytest.raises(KeyError):
        e4.reinforce(victim.memory_id)


# ============================================================
# Establishing set-versus-multiset by CONSTRAINT, not by convenience
# ============================================================

def test_link_identity_is_the_endpoint_pair_established_by_constraint(tmp_path):
    """Do not decide a table is a set because sorting and hashing rows is
    convenient. `cell_links` has PRIMARY KEY (from_cell_id, to_cell_id), so the
    endpoint pair IS the identity and `link_type` is one of its VALUES: two
    links between the same pair with different polarity cannot coexist. That is
    a fact about the schema, and it is what licenses set semantics here."""
    db = tmp_path / "g.db"
    build(db)
    with sqlite3.connect(db) as conn:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='cell_links'").fetchone()[0]
        assert "PRIMARY KEY (from_cell_id, to_cell_id)" in ddl.replace("\n", " ").replace("  ", " ")

        conn.execute("INSERT OR REPLACE INTO cell_links VALUES (1,2,'RESONANT',0,0)")
        conn.execute("INSERT OR REPLACE INTO cell_links VALUES (1,2,'INHIBITORY',0,0)")
        conn.commit()
        rows = list(conn.execute(
            "SELECT link_type FROM cell_links WHERE from_cell_id=1 AND to_cell_id=2"))
    assert len(rows) == 1 and rows[0][0] == "INHIBITORY", (
        "polarity is an attribute of the pair, not part of its identity — a "
        "snapshot keyed on (from, to, type) would model a relation the schema "
        "cannot represent"
    )


def test_wal_means_the_file_is_not_the_whole_state(tmp_path):
    """A DBWitness that reads the database file without checkpointing can
    observe a different database than the engine does. Found while building
    these adversaries: a plain file copy produced an empty schema."""
    db, copy_path = tmp_path / "h.db", tmp_path / "h_copy.db"
    build(db)
    shutil.copy(db, copy_path)          # no checkpoint
    with sqlite3.connect(copy_path) as conn:
        uncheckpointed = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='memories'").fetchone()[0]

    checkpointed_path = tmp_path / "h_ck.db"
    clone(db, checkpointed_path)
    with sqlite3.connect(checkpointed_path) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    assert rows == len(TEXTS)
    assert uncheckpointed == 0, (
        "WAL no longer hides the data — re-derive whether DBWitness still needs "
        "to checkpoint before reading"
    )
