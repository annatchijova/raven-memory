# 🦅 raven-memory — Demo video script (ENGLISH)

**Target length:** 6–8 minutes
**Structure:** 7 acts. Each act = [SLIDE] → [SCREEN/TERMINAL] → [NARRATION].
Slides go BETWEEN the runs, as a visual breather and explanation.

**Golden rule:** never explain while a command is running. Let the terminal
speak, and narrate over the slide or over the frozen result.

---

## ACT 0 — Opening (0:00 – 0:40)

**[SLIDE 1 — Title]**

> **NARRATION:**
> "Vector stores don't remember. They store.
> Query Pinecone today and query it tomorrow — identical results,
> because nothing changed. Nothing learned. Nothing forgot.
>
> This is raven-memory: an adaptive memory field where every memory
> has a state, every recall leaves a scar, and the system... dreams.
> Literally. Let me show you."

**[TERMINAL — Environment check]**

```bash
curl -s http://8.222.219.67:8012/health | python3 -m json.tool
```

> **NARRATION (over the frozen result):**
> "Before we start, one thing: this is not a mock. It's running in
> production on Alibaba Cloud, with real Qwen embeddings and a real
> Qwen LLM. Look: `embedding_provider.active: qwen_api`,
> `degraded: false`. Everything you're about to see is live."

✅ **Pre-recording checklist:** health OK, `qwen_api` active, stats at zero —
or with the field already loaded if you'd rather skip Act 1 in real time.

---

## ACT 1 — Seeding the field (0:40 – 1:30)

**[SLIDE 2 — "The problem"]**

> **NARRATION:**
> "We'll seed the field with about 40 real memories: architecture,
> cognitive science, and — this is key — **deliberate contradictions**.
> Because a real agent doesn't live in a consistent world. It lives
> in a world that tells it one thing on Monday and the opposite
> on Friday."

**[TERMINAL — Bulk load]** *(speed up ×4 in editing, or time-lapse)*

```bash
bash load_memories.sh
```

> **NARRATION (over the sped-up run):**
> "Each memory enters with a layer — semantic or episodic — a state,
> and metadata. Watch the right-hand column: `qwen_api` on every
> single one. Real embeddings, not toy hashes."

**[TERMINAL — Field status]**

```bash
curl -s http://8.222.219.67:8012/stats | python3 -m json.tool
```

> **NARRATION:**
> "Forty-plus memories, Voronoi cells built — the field is alive.
> Now the fun part."

---

## ACT 2 — The conflict (1:30 – 2:45)

**[SLIDE 3 — Ternary states]**

> **NARRATION:**
> "In raven-memory a memory lives in one of three states.
> REINFORCED: validated truth, score multiplied by 1.5.
> NEUTRAL: the baseline. FORGOTTEN: suppressed, punished down to 0.5 —
> but not deleted. Forgetting is reversible, like in a brain.
> It's Łukasiewicz ternary logic applied to memory."

**[TERMINAL — Recall with contradiction]**

```bash
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "Does VIGÍA use machine learning?", "top_k": 5, "hops": 2}' \
  | python3 -m json.tool
```

> **NARRATION (over the JSON, highlighting with the cursor):**
> "I ask the field: does VIGÍA use machine learning?
> And look what it retrieves: one memory saying VIGÍA is
> **fully deterministic**... and another saying it's a
> **hybrid ML system**. Both. With their scores.
>
> A normal vector store dumps both on you — good luck.
> raven-memory lets you **resolve** the conflict. Note those two
> memory IDs, because here comes the spectacular part."

📝 **Production note:** copy both `memory_id` values into a text file
BEFORE recording Act 3, so you can paste them fast.

---

## ACT 3 — The collapse (2:45 – 4:00) ⭐ CLIMAX

**[SLIDE 4 — "The collapse"]**

> **NARRATION:**
> "This is what no RAG does: we're going to tell the field which of
> the two versions is true. Reinforce one. Forget the other.
> And the entire field reorganizes."

**[TERMINAL — Reinforce + Forget]**

```bash
# Reinforce the truth
curl -s -X POST http://8.222.219.67:8012/memories/<DETERMINISTIC_ID>/reinforce \
  | python3 -m json.tool

# Suppress the false one
curl -s -X POST http://8.222.219.67:8012/memories/<ML_HYBRID_ID>/forget \
  | python3 -m json.tool
```

> **NARRATION:**
> "Reinforce... and forget. Two calls. Now the same question again.
> Exactly the same."

**[TERMINAL — Same recall, repeated]**

```bash
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "Does VIGÍA use machine learning?", "top_k": 5, "hops": 2}' \
  | python3 -m json.tool
```

> **NARRATION (dramatic pause over the result):**
> "...And the false memory **vanished from recall**. The reinforced
> one boosted its score ×1.5 and dominates the answer. Same query,
> same field, different answer — because the field **learned**.
>
> This is memory as a process, not memory as a file."

---

## ACT 4 — Synapses: STDP (4:00 – 4:50)

**[SLIDE 5 — STDP]**

> **NARRATION:**
> "Second mechanism stolen from neuroscience: STDP.
> *Cells that fire together, wire together.* When two memories
> co-activate in the same recall, the synapse between them
> strengthens. Retrieval is never free: retrieval **modifies
> the field**."

**[TERMINAL — Repeated recalls]**

```bash
# Run the same thematic query 3 times
for i in 1 2 3; do
  curl -s -X POST http://8.222.219.67:8012/recall \
    -H "Content-Type: application/json" \
    -d '{"query": "Voronoi geometry and hop-based activation", "top_k": 5, "hops": 3}' \
    > /dev/null
  echo "Recall $i done"
done

curl -s http://8.222.219.67:8012/stats | python3 -m json.tool
```

> **NARRATION:**
> "Three recalls on the same topic. Look at `total_recalls` and the
> neighborhoods: the Voronoi and hop memories are now synaptically
> linked. Next time one fires, it drags the others along.
> The field specializes in what you actually use."

---

## ACT 5 — Sleep (4:50 – 5:50)

**[SLIDE 6 — Sleep consolidation]**

> **NARRATION:**
> "And at night, the field dreams. An offline consolidator — run it
> like a 3 a.m. cron — clusters redundant episodic memories, merges
> each cluster into a single semantic node with a recall-frequency-
> weighted centroid, and commits everything in one atomic transaction.
> A crash can never duplicate the past."

**[TERMINAL — Dry run first, then real]**

```bash
python sleep_consolidator.py --dry-run
# (show the cluster preview)

python sleep_consolidator.py --threshold 0.85
```

> **NARRATION:**
> "Dry-run first: it shows me which clusters it found without touching
> anything. Then the real consolidation. The agent 'wakes up' with a
> cleaner field and a rebuilt SVD spectral space — sharper eigen-modes.
> It literally dreams and wakes up smarter."

---

## ACT 6 — The forensic chain (5:50 – 6:40)

**[SLIDE 7 — Audit chain]**

> **NARRATION:**
> "Last piece, and the most serious one: every operation is chained
> with SHA-256 over the full payload — including the content hash of
> every retrieved memory. If someone hand-edits a memory, the chain
> breaks and a forensic alert fires. On top of that: stylometry.
> Insert a memory pretending to be another author, and the stylistic
> profile gives you away."

**[TERMINAL — Audit trail]**

```bash
curl -s "http://8.222.219.67:8012/audit?limit=5" | python3 -m json.tool
curl -s http://8.222.219.67:8012/alerts | python3 -m json.tool
```

> **NARRATION:**
> "There it all is: every recall, every reinforce, every consolidation,
> each with its hash and the hash of the previous one.
> Tamper-evident, end to end."

---

## ACT 7 — Closing (6:40 – 7:30)

**[SLIDE 8 — Comparison]**

> **NARRATION:**
> "To wrap up, one line per rival:
> RAG searches the global top-k — raven propagates a local wave.
> Pinecone is stateless — here every recall leaves a trace.
> LangChain memory is an append-only log — here memories evolve.
> MemGPT pages memory — here everything lives in one continuous field."

**[SLIDE 9 — Closing]**

> **NARRATION:**
> "raven-memory. Ternary states, STDP synapses, a consolidating sleep
> cycle, forensic auditing. Running in production on Qwen Cloud.
> Apache 2.0 — the code is yours.
>
> Memory is not a file. It's a process. Thank you."

---

## Appendix — Plan B (if anything fails live)

| Failure | Plan B |
|---|---|
| `embedding_provider: dummy` | Cut. Check `DASHSCOPE_API_KEY`, `docker logs`. Do NOT record in dummy mode. |
| Recall latency > 10s | Record the command, cut, paste the result from a previous take. |
| Consolidator finds no clusters | Lower threshold to 0.80, or run 2–3 episodic recalls first. |
| Everything breaks | `python run_all.py --demo` locally with Gradio as the visual fallback. |

## Appendix — Recording tips

- Terminal: large font (18pt+), dark theme, clean window with no tabs.
- Keep commands in a `demo_commands.sh` file and paste them — don't type live.
- Record each act separately. Editing stitches it all together.
- Zoom / cursor-highlight the key JSON fields (scores, states, hashes).
- Act 3 is THE moment: give it 2 seconds of silence before "...vanished".
