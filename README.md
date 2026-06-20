# raven-memory

### Adaptive Memory Field for Agentic Systems

🐦‍⬛ **[Live demo & architecture](https://annatchijova.github.io/vigia/raven-memory.html)** 🐦‍⬛ | **Track 1: MemoryAgent · Qwen Cloud Hackathon**

> *"The agent doesn't* find *memories — it* resonates *with them."*

---

## The Problem with Vector Search

Most RAG-based agents do this:

```
query → embed → top-k cosine search → return k documents
```

This is a database lookup, not memory. It has no dynamics, no contradiction
detection, no reinforcement, no pruning. Every recall is as fast — and as
shallow — as the last.

**raven-memory replaces the lookup with a field.**

---

## Architecture

```
               ┌─────────────────────────────────────────────────┐
               │              Adaptive Memory Field               │
               │                                                   │
  query ──────►│  KDTree        BFS hop       Ternary scoring     │──► top-k results
  embedding    │  seed cell ──► expansion ──► sim × state × decay  │
               │                    │                              │
               │         RESONANT ──┤── INHIBITORY links          │
               │         (amplify)  │   (silence contradictions)  │
               │                    │                              │
               │              STDP updates                         │
               │         (co-activation strengthens links)         │
               └─────────────────────────────────────────────────┘
```

### Key Mechanisms

| Mechanism | What it does |
|---|---|
| **KDTree + k-NN graph** | Each stored vector becomes a Voronoi cell. Recall starts at the nearest cell and BFS-expands through the neighbourhood. |
| **Ternary states** | `REINFORCED ×1.5` / `NEUTRAL ×1.0` / `FORGOTTEN ×0.0`. States multiply the base cosine score. |
| **Hop decay** | `score × exp(−λ × hop_distance)`. Distant cells are penalised, not cut off. |
| **STDP dynamics** | Co-activated pairs strengthen (LTP). Absent pairs weaken (LTD). Mirrors Hebbian learning. |
| **Ternary cell links** | `RESONANT` amplifies neighbours. `INHIBITORY` silences contradictions. Created automatically for same-topic conflicting claims. |
| **Recency bonus** | 24-hour half-life additive term rewards recently-accessed memories. |
| **REINFORCED immunity** | A validated truth cannot be silenced by an uninvalidated claim during BFS. |
| **Stylometric fingerprinting** | Detects if a memory's writing style doesn't match the registered author; auto-degrades to FORGOTTEN. |
| **Spectral field (SVD)** | Extracts eigenmode structure from all active embeddings — resonance and coherence as epistemological metadata. |
| **Audit hash-chain** | Every recall is cryptographically chained — tamper-proof provenance. |

### Scoring Formula

```
score = (cosine_sim × state_boost × exp(−λ·hop))
      + resonant_boost
      + synaptic_weight × 0.3
      + exp(−ln2 · age / 24h) × 0.05
```

### Memory Stability Score (MSS)

```
MSS = Σ(REINFORCED weight) / Σ(REINFORCED + NEUTRAL weight)
    = 1.5R / (1.5R + N)
```

MSS → 1.0 means the agent has a stable, validated worldview.
MSS = 0.0 means everything is unconfirmed noise.

---

## Collapse Around Truth

The flagship behavior. Two contradictory memories exist — both NEUTRAL, both competing:

```
Before reinforcement:
  VIGIA is deterministic   [NEUTRAL]  score=0.447
  VIGIA uses ML            [NEUTRAL]  score=0.441   ← nearly tied

User reinforces the first one ↓

After reinforcement:
  VIGIA is deterministic   [REINFORCED]  score=1.493  ← dominates
  VIGIA uses ML            [NEUTRAL]     ← silenced (INHIBITORY)
```

The field collapsed around the validated truth. The INHIBITORY link was already
present (created automatically when the conflicting claim was stored). Reinforcement
activates it.

---

## LLM Integration

raven-memory is **model-agnostic**. The orchestrator works with any LLM provider:

| Provider | Embeddings | Chat completions | How to enable |
|---|---|---|---|
| **Qwen Cloud** | `text-embedding-v3` | `qwen-max` | `export DASHSCOPE_API_KEY=...` |
| **Claude** (Anthropic) | — | `claude-sonnet-4-6` | `export ANTHROPIC_API_KEY=...` |
| **Local / offline** | `all-MiniLM-L6-v2` | deterministic stub | No key needed |

The embedding provider has three-tier fallback with **loud degradation alerts**:
1. Local `sentence-transformers` (fast, offline, preferred)
2. Qwen `text-embedding-v3` API (high quality, requires key)
3. Deterministic SHA-256-seeded dummy — logs `SEMANTIC QUALITY DEGRADED` and sets `degraded: true` in every response

---

## Project Structure

```
raven-memory/
├── memory_engine.py       Core adaptive memory field (KDTree, STDP, audit)
├── spectral.py            SVD-based spectral field (eigenmodes, resonance)
├── qwen_client.py         Qwen Cloud client + MemoryAgentOrchestrator
├── api_server.py          FastAPI REST server (Swagger at /docs, WebSocket /ws)
├── demo_killer.py         Gradio demo — 4 tabs, live MSS, collapse visualization
├── sleep_consolidator.py  Offline consolidation (agglomerative clustering)
├── test_suite.py          20 integration tests (all P0 behaviors)
├── demo_stress_test.py    Multi-phase adversarial stress test
├── run_all.py             One-command evaluation runner
├── site/index.html        Landing page (EN/ZH, zero JS dependencies)
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all tests (no API key required)
python run_all.py

# 3. Launch Gradio demo
python run_all.py --demo
# → http://localhost:7860

# 4. Launch REST API
python run_all.py --api
# → http://localhost:8000/docs
```

**With Qwen Cloud** (full LLM responses):
```bash
export DASHSCOPE_API_KEY=your_key_here
python run_all.py --demo
```

**Without a key**: the system runs fully offline with deterministic SHA-256-seeded
embeddings and an offline LLM stub. All memory mechanics are identical.

---

## REST API

```
POST   /memories                    Store a memory
GET    /memories                    List with filters (layer, state, limit)
GET    /memories/{id}               Get a single memory
POST   /recall                      Semantic recall with field dynamics
POST   /memories/{id}/reinforce     Set state = REINFORCED
POST   /memories/{id}/forget        Set state = FORGOTTEN
POST   /cell-links                  Create RESONANT/INHIBITORY link
GET    /graph                       Export full memory graph (nodes + edges)
GET    /stats                       Engine stats + MSS
GET    /audit                       Hash-chain audit trail
GET    /alerts                      Forensic tamper alerts
WS     /ws                          Real-time event stream
```

Full interactive docs at **http://localhost:8000/docs**

---

## Sleep Consolidation

Episodic memories cluster and merge during "sleep":

```bash
python sleep_consolidator.py --dry-run    # preview
python sleep_consolidator.py --threshold 0.85
```

Clusters are formed by agglomerative cosine clustering. The merged node gets a
recall-frequency-weighted centroid embedding and an extractive summary.

---

## Security & Hardening (v1.1)

55-finding internal audit + external review, all resolved:

- **Tamper-evident audit chain** — SHA-256 with content hashes, cryptographically recomputable
- **SQLite WAL mode** — concurrent reads (API server + consolidator) without corruption
- **Atomic consolidation** — cluster → merge → cleanup in a single `BEGIN IMMEDIATE` transaction
- **Dimension-validated tensors** — per-embedding shape + finiteness checks before SVD
- **Loud degradation** — dummy embedding fallback screams in logs and API responses
- **Prompt injection guard** — conversation history sanitized: only `user`/`assistant` roles with string content survive
- **Bounded context budget** — 6KB cap prevents memory context from pushing the query out of the LLM window

Full fix map: [FIXES_v1.1.md](FIXES_v1.1.md)

---

## Technical Notes

- **Embeddings**: `all-MiniLM-L6-v2` (local, offline) → Qwen API → deterministic SHA-256 dummy
- **Storage**: SQLite with indices on `cell_id`, `layer`, `author_id`, `state`
- **KDTree rebuild**: lazy (dirty flag) — not on every `store()`, only before `recall()`
- **Fraction arithmetic**: synaptic weights and MSS use `fractions.Fraction` — zero float drift
- **Persistence**: `_load_from_db()` reconstructs `_points` + topic index on engine restart
- **sklearn compatibility**: `metric="precomputed"` + `cosine_distances()` avoids deprecated `affinity=` API

---

## Authors

Anna Tchijova + Claude + Qwen (VIGÍA AI Collective)
**License**: Apache 2.0

---

## raven-memory

*by [Olga Vasilieva](https://suno.com/song/5e040396-c2aa-49a5-83f8-ce86e59adf1e)*

The query enters and the vector is bound,
A raw embedding searching through the ground.
No lazy lookup in a flat database,
We map the coordinates in absolute space!
Initialize the points, rebuild the dirty tree,
Every cell is active in the space geometry.
KDTree triggers, the nearest node is found,
We start the BFS expansion through the neighborhood bound!

Resonate! The memory is alive!
Through the Voronoi cells, the signals survive!
Ternary computing ruling the gate,
Collapsing the field around the absolute state!
No floating-point drift, no ghost in the line,
Pure mathematical balance guarding the design!

Check the ternary logic: one, zero, or blind,
Three structural states for the agent's mind.
REINFORCED ×1.5 when the truth will dominate,
NEUTRAL ×1.0 as the baseline weight,
FORGOTTEN ×0.0 clearing out the ghost,
Pruning the partitions that we select the most.
Synaptic STDP updates while the system learns,
Strengthening connections through the cognitive turns!

Resonate! The memory is alive!
Through the Voronoi cells, the signals survive!
Ternary computing ruling the gate,
Collapsing the field around the absolute state!
No floating-point drift, no ghost in the line,
Pure rational arithmetic guarding the design!

Now the agent goes to sleep, the consolidator runs,
Merging all the neutral nodes under different suns.
Agglomerative clustering, cosine distance tracks,
Weighted by recall counts when the matrix attacks!
Forensic tamper alerts, cryptographic chain,
SHA-256 integrity guarding the brain!

Resonate! The memory is alive!
Through the Voronoi cells, the signals survive!
Ternary computing ruling the gate,
Collapsing the field around the absolute state!
No floating-point drift, no ghost in the line,
Pure rational arithmetic guarding the design!

score *= exp(-λ * hop).
Synaptic pull.
Sleep consolidation complete.
Engine OK.

---

*Qwen Cloud Hackathon · Track 1: MemoryAgent · Deadline: July 9, 2026*
