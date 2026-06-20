#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADAPTIVE MEMORY FIELD — Stress Test & Collapse Visualization
Demo script para hackathon Qwen Cloud.

Ejecutar: python demo_stress_test.py

Muestra:
1. Test adversarial semantico real (paraphrases, negaciones, contradicciones)
2. STDP observable (que memoria activo cual, peso antes/despues)
3. "Collapse visualization": antes/despues de reinforcement

Autor: Anna Tchijova (VIGIA AI Collective)
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, ".")

from memory_engine import AdaptiveMemoryEngine, MemoryState
from qwen_client import MemoryAgentOrchestrator, QwenConfig


def print_header(title):
    print("\n" + "=" * 70)
    print("  " + title)
    print("=" * 70)


def print_subheader(title):
    print("\n--- " + title + " ---")


# ============================================================
# SETUP
# ============================================================

print_header("ADAPTIVE MEMORY FIELD v1.0 — STRESS TEST & COLLAPSE DEMO")

# Clean DB
db_path = Path("adaptive_memory_field.db")
if db_path.exists():
    db_path.unlink()

engine = AdaptiveMemoryEngine()
orch = MemoryAgentOrchestrator(engine, QwenConfig(use_local_embeddings=True))

print("\n[SETUP] Engine initialized with clean database")


# ============================================================
# 1. TEST ADVERSARIAL SEMANTICO REAL
# ============================================================

print_header("TEST 1: ADVERSARIAL SEMANTIC TEST (Real Paraphrases & Contradictions)")

# Insert memories with real semantic variations
adversarial_memories = [
    {
        "content": "VIGIA es completamente determinista y no utiliza machine learning en ninguna de sus capas.",
        "layer": "semantic", "state": MemoryState.NEUTRAL,
        "metadata": {"topic": "vigia_nature", "claim": "deterministic", "language": "es"},
    },
    {
        "content": "VIGIA no es determinista; emplea tecnicas de aprendizaje automatico y embeddings vectoriales.",
        "layer": "semantic", "state": MemoryState.NEUTRAL,
        "metadata": {"topic": "vigia_nature", "claim": "ml_hybrid", "language": "es"},
    },
    {
        "content": "The VIGIA system is fully deterministic and avoids any machine learning components.",
        "layer": "semantic", "state": MemoryState.NEUTRAL,
        "metadata": {"topic": "vigia_nature", "claim": "deterministic", "language": "en"},
    },
    {
        "content": "VIGIA no usa ML, es puro analisis semiologico forense con aritmetica racional.",
        "layer": "semantic", "state": MemoryState.NEUTRAL,
        "metadata": {"topic": "vigia_nature", "claim": "deterministic", "language": "es"},
    },
    {
        "content": "SANS hackathon deadline is June 15, 2026. Rob T. Lee approved the dataset.",
        "layer": "episodic", "state": MemoryState.REINFORCED,
        "metadata": {"topic": "hackathon", "event": "sans_2026", "person": "Rob T. Lee"},
    },
]

print_subheader("Inserting adversarial memories...")
for mem in adversarial_memories:
    emb = orch.embedder.embed([mem["content"]])[0]
    entry = engine.store(
        content=mem["content"], embedding=emb, layer=mem["layer"],
        state=mem["state"], metadata=mem["metadata"],
    )
    print("  Stored: " + entry.memory_id[:20] + "... | " + mem["content"][:60] + "...")

# Test queries with paraphrases
queries = [
    "Es VIGIA determinista?",
    "Does VIGIA use machine learning?",
    "VIGIA usa ML o no?",
    "Is VIGIA deterministic or probabilistic?",
    "Que es VIGIA tecnicamente?",
]

print_subheader("Querying with paraphrases...")
for q in queries:
    emb = orch.embedder.embed([q])[0]
    results, audit = engine.recall(emb, query_text=q, top_k=5, hops=2)

    print("\n  Query: "" + q + """)
    print("  Retrieved " + str(len(results)) + " memories:")
    for r in results:
        claim = r.memory.metadata.get("claim", "unknown")
        lang = r.memory.metadata.get("language", "?")
        print("    * [" + r.memory.state.name + "] Score=" + str(round(r.final_score, 3)) + 
              " | Claim=" + claim + " | Lang=" + lang + " | " + r.memory.content[:50] + "...")


# ============================================================
# 2. STDP OBSERVABLE
# ============================================================

print_header("TEST 2: STDP OBSERVABLE (Synaptic Weights Before/After)")

# Create two memories that should form synaptic link
mem_a_content = "Rob T. Lee approved VIGIA dataset for SANS competition"
mem_b_content = "SANS hackathon deadline is June 15, 2026 at 14:00 PDT"

emb_a = orch.embedder.embed([mem_a_content])[0]
emb_b = orch.embedder.embed([mem_b_content])[0]

mem_a = engine.store(mem_a_content, emb_a, layer="episodic", metadata={"topic": "hackathon"})
mem_b = engine.store(mem_b_content, emb_b, layer="episodic", metadata={"topic": "hackathon"})

print_subheader("Initial synaptic weights")
print("  Memory A: " + mem_a.memory_id[:20] + "...")
print("  Memory B: " + mem_b.memory_id[:20] + "...")
print("  Synaptic link A->B: " + str(mem_a.synaptic_links.get(mem_b.memory_id, "NOT EXISTS")))
print("  Synaptic link B->A: " + str(mem_b.synaptic_links.get(mem_a.memory_id, "NOT EXISTS")))

# First turn: query about Rob -> should retrieve both
print_subheader("Turn 1: Query 'What did Rob say about VIGIA?'")
q1 = "What did Rob say about VIGIA?"
emb_q1 = orch.embedder.embed([q1])[0]
results1, _ = engine.recall(emb_q1, query_text=q1, top_k=5, hops=2, current_turn_memories=None)
turn1_ids = [r.memory.memory_id for r in results1]

print("  Retrieved: " + str([r.memory.content[:40] + "..." for r in results1]))

# Second turn: query about SANS -> should retrieve both via STDP
print_subheader("Turn 2: Query 'When is the SANS deadline?'")
q2 = "When is the SANS deadline?"
emb_q2 = orch.embedder.embed([q2])[0]
results2, _ = engine.recall(emb_q2, query_text=q2, top_k=5, hops=2, current_turn_memories=turn1_ids)

synaptic_results = [r for r in results2 if r.source == "synaptic"]
similarity_results = [r for r in results2 if r.source == "similarity"]

print("  Retrieved via similarity: " + str([r.memory.content[:40] + "..." for r in similarity_results]))
print("  Retrieved via synaptic:   " + str([r.memory.content[:40] + "..." for r in synaptic_results]))

# Check synaptic weights after co-activation
mem_a_reloaded = engine._db.load_memory(mem_a.memory_id)
mem_b_reloaded = engine._db.load_memory(mem_b.memory_id)

print_subheader("Synaptic weights AFTER co-activation")
print("  A->B: " + str(mem_a_reloaded.synaptic_links.get(mem_b.memory_id, "NOT EXISTS")))
print("  B->A: " + str(mem_b_reloaded.synaptic_links.get(mem_a.memory_id, "NOT EXISTS")))

if mem_a.memory_id in mem_b_reloaded.synaptic_links:
    print("  STDP link formed! Weight: " + str(round(mem_b_reloaded.synaptic_links[mem_a.memory_id], 2)))
else:
    print("  No STDP link formed (may need more co-activations)")


# ============================================================
# 3. COLLAPSE VISUALIZATION
# ============================================================

print_header("TEST 3: COLLAPSE VISUALIZATION (Reinforcement Dynamics)")

# Insert two contradictory memories
mem_x_content = "VIGIA es 100% determinista, sin floats, sin probabilidades"
mem_y_content = "VIGIA usa embeddings y cosine similarity, es un sistema ML hibrido"

emb_x = orch.embedder.embed([mem_x_content])[0]
emb_y = orch.embedder.embed([mem_y_content])[0]

mem_x = engine.store(mem_x_content, emb_x, layer="semantic",
                     metadata={"topic": "vigia_nature", "claim": "deterministic"})
mem_y = engine.store(mem_y_content, emb_y, layer="semantic",
                     metadata={"topic": "vigia_nature", "claim": "ml_hybrid"})

print_subheader("BEFORE reinforcement")
stats_before = engine.get_stats()
print("  Memory Stability Score (MSS): " + str(round(stats_before["memory_stability_score"], 3)))
print("  State distribution: " + str(stats_before["state_distribution"]))

# Query before reinforcement
q = "Es VIGIA un sistema de machine learning?"
emb_q = orch.embedder.embed([q])[0]
results_before, _ = engine.recall(emb_q, query_text=q, top_k=5, hops=2)

print("\n  Query: '" + q + "'")
print("  Results BEFORE reinforcement:")
for r in results_before:
    claim = r.memory.metadata.get("claim", "unknown")
    print("    * [" + r.memory.state.name + "] Score=" + str(round(r.final_score, 3)) + 
          " | Claim=" + claim + " | " + r.memory.content[:50] + "...")

# Reinforce mem_x (deterministic)
print_subheader("Action: REINFORCE mem_x (deterministic claim)")
engine.reinforce(mem_x.memory_id)

print_subheader("AFTER reinforcement")
stats_after = engine.get_stats()
print("  Memory Stability Score (MSS): " + str(round(stats_after["memory_stability_score"], 3)))
print("  State distribution: " + str(stats_after["state_distribution"]))

# Query after reinforcement
results_after, _ = engine.recall(emb_q, query_text=q, top_k=5, hops=2)

print("\n  Query: '" + q + "'")
print("  Results AFTER reinforcement:")
for r in results_after:
    claim = r.memory.metadata.get("claim", "unknown")
    print("    * [" + r.memory.state.name + "] Score=" + str(round(r.final_score, 3)) + 
          " | Claim=" + claim + " | " + r.memory.content[:50] + "...")

# Show the collapse
print_subheader("COLLAPSE ANALYSIS")
if results_before and results_after:
    before_scores = {}
    for r in results_before:
        claim = r.memory.metadata.get("claim", "?")
        before_scores[claim] = r.final_score

    after_scores = {}
    for r in results_after:
        claim = r.memory.metadata.get("claim", "?")
        after_scores[claim] = r.final_score

    all_claims = set(before_scores.keys()) | set(after_scores.keys())
    for claim in all_claims:
        b = before_scores.get(claim, 0)
        a = after_scores.get(claim, 0)
        delta = a - b
        if delta > 0:
            arrow = "UP"
        elif delta < 0:
            arrow = "DOWN"
        else:
            arrow = "SAME"
        print("  " + claim + ": " + str(round(b, 3)) + " -> " + str(round(a, 3)) + 
              " (" + arrow + " " + str(round(delta, 3)) + ")")


# ============================================================
# 4. ESTILOMETRIA FORENSE
# ============================================================

print_header("TEST 4: FORENSIC STYLOMETRY (Tampering Detection)")

# Insert memory with Anna style
anna_text = "El sistema VIGIA es completamente determinista. No usa floats ni probabilidades. Es puro analisis semiologico forense."
emb_anna = orch.embedder.embed([anna_text])[0]
mem_anna = engine.store(anna_text, emb_anna, layer="semantic", author_id="anna",
                        metadata={"topic": "vigia_nature", "claim": "deterministic"})

print_subheader("Memory stored with author_id=anna")
print("  Content: " + anna_text[:60] + "...")
if mem_anna.fingerprint:
    print("  Fingerprint hash: " + mem_anna.fingerprint.fingerprint_hash)

# Simulate tampering: same topic, SAME language, radically different style.
# (v1.1: a language switch alone is no longer treated as tampering — that was
# a bilingual-author false positive. Stylometry must detect a style change
# WITHIN the language: different function-word profile, sentence rhythm,
# punctuation density.)
hacker_text = ("Bueno o sea aca lo que pasa, y mira que esto ya lo dije antes no?, es que todo el tema ese que vos decis, "
               "ese de los sistemas y todo lo demas, va asi: cada cosa que entra, sale, y listo, no hay mas vuelta que darle, "
               "porque al final del dia, y esto es lo que importa, todo se reduce a eso, a que funcione, nada mas!!!")
emb_hacker = orch.embedder.embed([hacker_text])[0]

print_subheader("Simulating tampering: same topic, different author style")
print("  Content: " + hacker_text[:60] + "...")

# Manually insert with wrong author_id (simulating DB tampering)
mem_hacker = engine.store(hacker_text, emb_hacker, layer="semantic", author_id="anna",
                           metadata={"topic": "vigia_nature", "claim": "deterministic"})

# Query to trigger estilometria check
q = "Que es VIGIA?"
emb_q = orch.embedder.embed([q])[0]
results, audit = engine.recall(emb_q, query_text=q, top_k=10, hops=2)

print_subheader("Recall results + forensic check")
print("  Total candidates: " + str(audit.total_candidates))
print("  Filtered by estilometria: " + str(audit.filtered_by_estilometria))

# Check alerts
alerts = engine.get_alerts(limit=5)
if alerts:
    print("\n  FORENSIC ALERTS DETECTED:")
    for alert in alerts:
        print("    * Alert " + alert.alert_id[:20] + "...")
        print("      Memory: " + alert.memory_id[:20] + "...")
        print("      Expected author: " + alert.expected_author)
        print("      Mismatch score: " + str(round(alert.mismatch_score, 3)))
        print("      Action: " + alert.action_taken)
else:
    print("\n  No forensic alerts (hacker style was similar enough or not detected)")


# ============================================================
# 5. AUDIT HASH CHAIN
# ============================================================

print_header("TEST 5: AUDIT HASH CHAIN (Tamper-Proof Trace)")

audits = engine.get_audit_trail(limit=10)
print("Total audit entries: " + str(len(audits)))

if len(audits) >= 2:
    print_subheader("Last 3 audit entries (hash chain)")
    for i, audit in enumerate(audits[:3]):
        print("  [" + str(i+1) + "] " + audit['operation'] + 
              " | hash=" + audit['audit_hash'][:16] + "... | prev=" + audit['prev_hash'][:16] + "...")

    # Verify chain continuity
    print_subheader("Verifying chain continuity...")
    broken = False
    for i in range(len(audits) - 1):
        if audits[i]['prev_hash'] != audits[i+1]['audit_hash']:
            print("  Chain BROKEN between audit " + str(i+1) + " and " + str(i+2))
            broken = True
            break
    if not broken:
        print("  Hash chain intact (" + str(len(audits)) + " entries)")
else:
    print("  Not enough audits to verify chain")


# ============================================================
# SUMMARY
# ============================================================

print_header("SUMMARY")

final_stats = engine.get_stats()
print("\nFinal System State:")
print("   * Total memories: " + str(final_stats["total_memories"]))
print("   * Memory Stability Score: " + str(round(final_stats["memory_stability_score"], 3)))
print("   * State distribution: " + str(final_stats["state_distribution"]))
print("   * Layer distribution: " + str(final_stats["layer_distribution"]))
print("   * Average neighbors per cell: " + str(round(final_stats["avg_neighbors"], 1)))

print("\nAudit Trail:")
print("   * Total audit entries: " + str(len(audits)))
if len(audits) >= 2:
    from memory_engine import verify_audit_chain
    report = verify_audit_chain(audits)
    status = "INTACT (linkage + hash recomputation)" if (
        report["chain_intact"] and report["hash_integrity"]) else f"BROKEN: {report['issues'][:2]}"
    print("   * Hash chain: " + status)
else:
    print("   * Hash chain: N/A")

print("\nForensic Alerts:")
print("   * Total alerts: " + str(len(engine.get_alerts(limit=10))))

print("\nAll stress tests completed successfully.")

print("\n" + "=" * 70)
print("  Adaptive Memory Field v1.0 — Ready for hackathon demo")
print("=" * 70)

# P2: clean up the temporary stress DB (and its WAL/SHM sidecars) so
# repeated runs start from a known state and don't leave artifacts behind.
for suffix in ("", "-wal", "-shm"):
    p = Path(str(db_path) + suffix)
    if p.exists():
        p.unlink()
print("Temporary stress DB cleaned up.")
