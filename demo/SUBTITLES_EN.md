# 🦅 raven-memory — English subtitles & on-screen explanations

Aligned 1:1 with `GUION_DEMO_ES.md`. Two formats included:
**(A)** subtitle blocks you can paste into your editor as captions, and
**(B)** short on-screen explainer captions (lower-thirds) per act.

Timecodes are indicative — adjust to your final cut.

---

## (A) SUBTITLE BLOCKS

### ACT 0 — Opening (0:00–0:40)

```
00:00 → Vector stores don't remember. They store.
00:05 → Query Pinecone today and tomorrow — identical results.
00:09 → Nothing changed. Nothing learned. Nothing forgot.
00:14 → This is raven-memory: an adaptive memory field.
00:18 → Every memory has a state. Every recall leaves a scar.
00:23 → And the system... dreams. Literally. Let me show you.
00:28 → First, proof this is real: live on Alibaba Cloud,
00:32 → real Qwen embeddings and LLM. No mocks.
00:36 → embedding_provider: qwen_api. degraded: false. We're live.
```

### ACT 1 — Seeding the field (0:40–1:30)

```
00:40 → We seed the field with ~40 real memories:
00:44 → architecture, cognitive science, and — key —
00:48 → deliberate contradictions.
00:51 → Real agents don't live in consistent worlds.
00:55 → They hear one thing on Monday, the opposite on Friday.
01:00 → Each memory enters with a layer, a state, and metadata.
01:05 → Right column: qwen_api on every single one.
01:09 → Real embeddings, not toy hashes.
01:15 → Forty-plus memories. Voronoi cells built.
01:20 → The field is alive. Now the fun part.
```

### ACT 2 — The conflict (1:30–2:45)

```
01:30 → A memory lives in one of three states.
01:34 → REINFORCED: validated truth — score boosted ×1.5.
01:39 → NEUTRAL: the baseline.
01:42 → FORGOTTEN: suppressed to ×0.5 — but not deleted.
01:46 → Forgetting is reversible, like in a brain.
01:50 → Ternary Łukasiewicz logic, applied to memory.
01:56 → I ask the field: does VIGÍA use machine learning?
02:02 → Look what it retrieves: one memory says
02:06 → "fully deterministic"... another says "hybrid ML system".
02:12 → Both. With their scores.
02:16 → A normal vector store dumps both on you — good luck.
02:21 → raven-memory lets you RESOLVE the conflict.
02:26 → Note those two memory IDs. Here comes the good part.
```

### ACT 3 — The collapse (2:45–4:00) ⭐

```
02:45 → This is what no RAG does:
02:48 → we tell the field which version is TRUE.
02:52 → Reinforce one. Forget the other.
02:55 → And the entire field reorganizes.
03:00 → Reinforce... and forget. Two calls.
03:05 → Now the exact same question again.
03:12 → ...and the false memory VANISHED from recall.
03:17 → The reinforced one dominates with a ×1.5 boost.
03:22 → Same query. Same field. Different answer —
03:27 → because the field LEARNED.
03:31 → Memory as a process, not memory as a file.
```

### ACT 4 — Synapses: STDP (4:00–4:50)

```
04:00 → Second mechanism stolen from neuroscience: STDP.
04:05 → "Cells that fire together, wire together."
04:09 → When two memories co-activate in the same recall,
04:13 → the synapse between them strengthens.
04:17 → Retrieval isn't free: retrieval MODIFIES the field.
04:24 → Three recalls on the same topic.
04:28 → The Voronoi and hop memories are now synaptically linked.
04:34 → Next time one fires, it drags the others along.
04:39 → The field specializes in what you actually use.
```

### ACT 5 — Sleep (4:50–5:50)

```
04:50 → And at night, the field dreams.
04:54 → An offline consolidator — run it like a 3 a.m. cron —
04:59 → clusters redundant episodic memories,
05:03 → merges each cluster into one semantic node
05:07 → with a recall-frequency-weighted centroid,
05:11 → and commits in a single atomic transaction.
05:15 → A crash can never duplicate the past.
05:20 → Dry-run first: preview the clusters, touch nothing.
05:26 → Then the real consolidation.
05:31 → The agent wakes with a cleaner field and a rebuilt
05:36 → SVD spectral space — sharper eigen-modes.
05:41 → It literally dreams and wakes up smarter.
```

### ACT 6 — Forensic chain (5:50–6:40)

```
05:50 → Last piece, and the most serious one:
05:54 → every operation is chained with SHA-256
05:58 → over its full payload — including the content hash
06:02 → of every retrieved memory.
06:06 → Hand-edit a memory, and the chain breaks:
06:10 → a forensic alert fires.
06:14 → Plus stylometry: insert a memory faking another author,
06:19 → and the stylistic profile gives you away.
06:25 → Every recall, reinforce, consolidation — hash-chained.
06:31 → Tamper-evident, end to end.
```

### ACT 7 — Closing (6:40–7:30)

```
06:40 → One line per rival:
06:43 → RAG searches global top-k — raven propagates a local wave.
06:49 → Pinecone is stateless — here every recall leaves a trace.
06:55 → LangChain memory is an append-only log — here memories evolve.
07:01 → MemGPT pages memory — here everything lives in one continuous field.
07:08 → raven-memory: ternary states, STDP synapses,
07:13 → sleep consolidation, forensic audit.
07:17 → Running in production on Qwen Cloud. Apache 2.0.
07:22 → Memory is not a file. It's a process. Thank you.
```

---

## (B) ON-SCREEN EXPLAINER CAPTIONS (lower-thirds)

One short technical caption per act, shown while the terminal runs:

| Act | Lower-third caption |
|---|---|
| 0 | `LIVE DEPLOY · Alibaba Cloud · Qwen embeddings + LLM · no mocks` |
| 1 | `Bulk seeding: ~40 memories · semantic + episodic layers · deliberate contradictions` |
| 2 | `Ternary states: REINFORCED ×1.5 · NEUTRAL ×1.0 · FORGOTTEN ×0.5 (reversible)` |
| 3 | `Conflict collapse: reinforce truth + forget falsehood → same query, new answer` |
| 4 | `STDP plasticity: co-activated memories wire together · recall mutates the field` |
| 5 | `Sleep consolidation: cluster → merge → atomic commit → SVD spectral rebuild` |
| 6 | `Forensic layer: SHA-256 hash chain + stylometric tamper detection` |
| 7 | `Not RAG · not a vector store · not a log · memory as a process` |

---

## (C) GLOSSARY CARD (optional end screen / description box)

- **Ternary states** — three-valued memory logic (Łukasiewicz): reinforced / neutral / forgotten.
- **Voronoi field** — each memory owns a geometric cell; adjacency defines activation hops.
- **STDP** — Spike-Timing-Dependent Plasticity; co-recalled memories strengthen their link.
- **Hop decay** — activation fades exponentially (e^(−0.15·hops)) with semantic distance.
- **Sleep consolidation** — offline clustering + merging of episodic memories into semantic nodes.
- **MSS** — Memory Stability Score; global health metric of the field.
- **Audit chain** — SHA-256 hash chain over every operation, tamper-evident by construction.
