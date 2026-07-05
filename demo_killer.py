#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — Hackathon Demo
"The memory that collapses around truth."

Tabs:
  🎬  Conflict Demo  — insert contradictory memories, query, reinforce, watch collapse
  🧠  Memory Field   — browse & manage all memories
  📊  Analytics      — live MSS chart + state distribution
  🔍  Forensic Trail — audit hash-chain + tamper alerts

Authors: Anna Tchijova + Claude (VIGÍA AI Collective)
License: Apache 2.0
"""

import json
import time
from pathlib import Path

import gradio as gr
import numpy as np

from raven.memory_engine import AdaptiveMemoryEngine, MemoryState, LinkType
from raven.qwen_client import MemoryAgentOrchestrator, QwenConfig

# ============================================================
# SETUP
# ============================================================

DB_PATH = Path("raven_demo.db")
if DB_PATH.exists():
    print(f"⚠️  WARNING: deleting demo DB ({DB_PATH}) — pass --preserve-db to keep it")
    DB_PATH.unlink()

engine = AdaptiveMemoryEngine(db_path=DB_PATH)
orch = MemoryAgentOrchestrator(engine, QwenConfig(use_local_embeddings=True))

print("🦅 raven-memory Demo — initialising…")

# Pre-loaded conflict scenario
_SCENARIO = [
    {
        "content": "VIGÍA es 100% determinista. No usa machine learning. Es puro análisis semiótico forense con aritmética racional.",
        "layer": "semantic", "state": MemoryState.NEUTRAL,
        "metadata": {"topic": "vigia_nature", "claim": "deterministic"},
    },
    {
        "content": "VIGÍA usa embeddings y similitud coseno. Es técnicamente un sistema de ML híbrido.",
        "layer": "semantic", "state": MemoryState.NEUTRAL,
        "metadata": {"topic": "vigia_nature", "claim": "ml_hybrid"},
    },
    {
        "content": "SANS hackathon deadline: June 15 2026. Rob T. Lee personally approved the VIGÍA dataset.",
        "layer": "episodic", "state": MemoryState.REINFORCED,
        "metadata": {"topic": "hackathon", "event": "sans_2026", "person": "Rob Lee"},
    },
    {
        "content": "raven-memory uses KDTree neighbourhoods for local activation — not global top-k search.",
        "layer": "semantic", "state": MemoryState.NEUTRAL,
        "metadata": {"topic": "raven_arch", "claim": "kdtree_local"},
    },
    {
        "content": "The Qwen Cloud hackathon track 1 (MemoryAgent) rewards novel memory architectures over fine-tuning.",
        "layer": "episodic", "state": MemoryState.NEUTRAL,
        "metadata": {"topic": "hackathon", "event": "qwen_cloud"},
    },
]

for mem in _SCENARIO:
    emb = orch.embedder.embed([mem["content"]])[0]
    engine.store(
        content=mem["content"], embedding=emb,
        layer=mem["layer"], state=mem["state"],
        metadata=mem["metadata"],
    )

print(f"✅  {len(_SCENARIO)} scenario memories loaded")
# P3-5: bounded history — prevents unbounded growth in long-running demos
# P1: Gradio serves handlers from a thread pool — concurrent users mutate
# this deque simultaneously. deque.append is atomic, but append+render as a
# sequence is not; the lock keeps each update-and-snapshot consistent.
import collections as _collections
import threading as _threading
_mss_lock = _threading.Lock()
_mss_history = _collections.deque(
    [engine.get_stats()["memory_stability_score"]], maxlen=200
)


def _record_mss(value: float) -> list:
    """Append under lock and return a consistent snapshot for rendering."""
    with _mss_lock:
        _mss_history.append(value)
        return list(_mss_history)

# ============================================================
# CSS — dark professional theme
# ============================================================

CSS = """
:root {
    --raven-dark: #0d1117;
    --raven-card: #161b22;
    --raven-border: #30363d;
    --raven-blue: #58a6ff;
    --raven-green: #3fb950;
    --raven-yellow: #d29922;
    --raven-red: #f85149;
    --raven-text: #c9d1d9;
}
body { background: var(--raven-dark); }
.score-bar { font-family: monospace; font-size: 13px; line-height: 1.8; }
.state-reinforced { color: #3fb950; font-weight: bold; }
.state-neutral    { color: #58a6ff; }
.state-forgotten  { color: #6e7681; }
.mss-number { font-size: 2.8em; font-weight: 900; color: #3fb950; font-family: monospace; }
.mss-label  { font-size: 0.85em; color: #8b949e; letter-spacing: 0.12em; text-transform: uppercase; }
.audit-hash { font-family: monospace; font-size: 11px; color: #8b949e; }
"""


# ============================================================
# SHARED HELPERS
# ============================================================

def _state_emoji(state: str) -> str:
    return {"REINFORCED": "🟢", "NEUTRAL": "🔵", "FORGOTTEN": "⚫"}.get(state, "❓")


def _source_emoji(src: str) -> str:
    return {"similarity": "🔍", "synaptic": "⚡", "resonant": "🔗"}.get(src, "❓")


def _render_memories_html(memories: list) -> str:
    if not memories:
        return "<p style='color:#8b949e;'>No memories retrieved.</p>"
    rows = []
    for i, m in enumerate(memories, 1):
        state = m["state"]
        cls = {"REINFORCED": "state-reinforced", "NEUTRAL": "state-neutral", "FORGOTTEN": "state-forgotten"}.get(state, "")
        bar_pct = min(int(m["final_score"] * 200), 100)
        bar_color = {"REINFORCED": "#3fb950", "NEUTRAL": "#58a6ff", "FORGOTTEN": "#6e7681"}.get(state, "#aaa")
        rows.append(f"""
        <div style='border:1px solid #30363d; border-radius:6px; padding:10px; margin-bottom:8px; background:#161b22;'>
          <div style='display:flex; justify-content:space-between; align-items:center;'>
            <span class='{cls}'>{_state_emoji(state)} [{state}] &nbsp; {_source_emoji(m.get("source",""))} {m.get("source","")}</span>
            <span style='font-family:monospace; font-size:13px; color:#e6edf3;'>
              score <strong>{m["final_score"]:.4f}</strong>
            </span>
          </div>
          <div style='background:#0d1117; border-radius:3px; height:6px; margin:6px 0;'>
            <div style='background:{bar_color}; height:6px; border-radius:3px; width:{bar_pct}%;'></div>
          </div>
          <div style='font-size:12px; color:#8b949e; font-family:monospace;'>
            sim={m["base_score"]:.3f} × state={m["state_boost"]} × hop_decay={m["hop_decay"]:.3f}
            + syn={m["synaptic_boost"]:.3f} + rec={m["recency_bonus"]:.4f}
            | hop={m["hop_distance"]} | cell={m["cell_id"]}
          </div>
          <div style='margin-top:6px; font-size:13px; color:#c9d1d9;'>{m["content_preview"][:110]}…</div>
          <div style='margin-top:4px; font-size:11px; color:#6e7681;'>{m["memory_id"]}</div>
        </div>
        """)
    return "".join(rows)


def _render_stats_html(stats: dict, mss_history: list) -> str:
    mss = stats.get("memory_stability_score", 0)
    sd = stats.get("state_distribution", {})
    ld = stats.get("layer_distribution", {})
    cl = stats.get("cell_links", {})

    # Mini ASCII sparkline for MSS
    spark = ""
    if len(mss_history) > 1:
        hi = max(mss_history) or 1
        lo = min(mss_history)
        chars = "▁▂▃▄▅▆▇█"
        for v in mss_history[-24:]:
            idx = int((v - lo) / max(hi - lo, 0.001) * 7)
            spark += chars[idx]

    return f"""
    <div style='display:flex; gap:24px; flex-wrap:wrap;'>
      <div style='text-align:center; padding:16px 24px; background:#161b22; border:1px solid #30363d; border-radius:8px;'>
        <div class='mss-number'>{mss:.3f}</div>
        <div class='mss-label'>Memory Stability Score</div>
        <div style='font-family:monospace; font-size:16px; color:#58a6ff; margin-top:4px; letter-spacing:2px;'>{spark}</div>
      </div>
      <div style='padding:16px 24px; background:#161b22; border:1px solid #30363d; border-radius:8px; min-width:140px;'>
        <div style='font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.1em;'>States</div>
        <div style='margin-top:8px;'>
          <div class='state-reinforced'>🟢 REINFORCED: {sd.get("REINFORCED", 0)}</div>
          <div class='state-neutral'>🔵 NEUTRAL: {sd.get("NEUTRAL", 0)}</div>
          <div class='state-forgotten'>⚫ FORGOTTEN: {sd.get("FORGOTTEN", 0)}</div>
        </div>
      </div>
      <div style='padding:16px 24px; background:#161b22; border:1px solid #30363d; border-radius:8px; min-width:140px;'>
        <div style='font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.1em;'>Graph</div>
        <div style='margin-top:8px; font-family:monospace; font-size:13px; color:#c9d1d9;'>
          <div>Cells: {stats.get("voronoi_cells", 0)}</div>
          <div>Avg neighbours: {stats.get("avg_neighbors", 0)}</div>
          <div>🔗 Resonant: {cl.get("RESONANT", 0)}</div>
          <div>🚫 Inhibitory: {cl.get("INHIBITORY", 0)}</div>
        </div>
      </div>
      <div style='padding:16px 24px; background:#161b22; border:1px solid #30363d; border-radius:8px; min-width:140px;'>
        <div style='font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.1em;'>Layers</div>
        <div style='margin-top:8px; font-family:monospace; font-size:13px; color:#c9d1d9;'>
          {''.join(f"<div>{k}: {v}</div>" for k, v in ld.items())}
          <div style='margin-top:4px;'>Total recalls: {stats.get("total_recalls", 0)}</div>
        </div>
      </div>
    </div>
    """


# ============================================================
# TAB 1 — CONFLICT DEMO
# ============================================================

def run_query(query: str, hops: int, top_k: int):
    if not query.strip():
        return "", "<p>Enter a query.</p>", "", ""

    result = orch.process_message(
        user_text=query, store_as_memory=True,
        top_k=int(top_k), hops=int(hops),
    )
    _hist = _record_mss(result["stats"]["memory_stability_score"])

    qwen_text = result["qwen_response"]["content"]
    mems_html = _render_memories_html(result["recalled_memories"])
    stats_html = _render_stats_html(result["stats"], _hist)
    audit_json = json.dumps(
        {k: v for k, v in result["audit_log"].items() if k != "query_embedding"},
        indent=2, ensure_ascii=False,
    )
    return qwen_text, mems_html, stats_html, audit_json


def do_reinforce(memory_id: str):
    mid = memory_id.strip()
    if not mid:
        return "⚠️ Paste a memory ID first.", "", ""
    try:
        res = orch.reinforce_memory(mid)
        _hist = _record_mss(res["stats"]["memory_stability_score"])
        mss = res["stats"]["memory_stability_score"]
        msg = f"✅ {mid[:24]}… → **REINFORCED**\n📊 New MSS: **{mss:.4f}**"
        return msg, _render_stats_html(res["stats"], _hist), _get_memory_list_html()
    except Exception as e:
        return f"❌ {e}", "", ""


def do_forget(memory_id: str):
    mid = memory_id.strip()
    if not mid:
        return "⚠️ Paste a memory ID first.", "", ""
    try:
        res = orch.forget_memory(mid)
        _hist = _record_mss(res["stats"]["memory_stability_score"])
        mss = res["stats"]["memory_stability_score"]
        msg = f"⚫ {mid[:24]}… → **FORGOTTEN**\n📊 New MSS: **{mss:.4f}**"
        return msg, _render_stats_html(res["stats"], _hist), _get_memory_list_html()
    except Exception as e:
        return f"❌ {e}", "", ""


# ============================================================
# TAB 2 — MEMORY FIELD
# ============================================================

def _get_memory_list_html() -> str:
    mems = engine.list_memories(limit=100)
    if not mems:
        return "<p style='color:#8b949e;'>No memories stored.</p>"
    rows = []
    for m in mems:
        se = _state_emoji(m.state.name)
        rows.append(f"""
        <div style='display:flex; gap:12px; align-items:center; padding:7px 10px;
                    border-bottom:1px solid #21262d; font-size:12px;'>
          <span style='min-width:20px;'>{se}</span>
          <span style='min-width:90px; font-family:monospace; color:#8b949e;'>{m.state.name}</span>
          <span style='min-width:70px; color:#8b949e;'>{m.layer}</span>
          <span style='flex:1; color:#c9d1d9;'>{m.content[:80]}…</span>
          <span style='font-family:monospace; font-size:11px; color:#6e7681; cursor:pointer;'
                title='{m.memory_id}'>{m.memory_id[:20]}…</span>
        </div>
        """)
    header = f"""
    <div style='background:#161b22; border:1px solid #30363d; border-radius:8px; overflow:hidden;'>
      <div style='padding:8px 10px; background:#0d1117; font-size:11px; color:#8b949e;
                  text-transform:uppercase; letter-spacing:0.1em; display:flex; gap:12px;'>
        <span style='min-width:20px;'>&nbsp;</span>
        <span style='min-width:90px;'>STATE</span>
        <span style='min-width:70px;'>LAYER</span>
        <span style='flex:1;'>CONTENT</span>
        <span>ID (click to copy)</span>
      </div>
      {''.join(rows)}
    </div>
    """
    return header


def search_memories(query: str):
    if not query.strip():
        return _get_memory_list_html()
    emb = orch.embedder.embed([query])[0]
    results, _ = engine.recall(emb, query_text=query, top_k=20, hops=3)
    return _render_memories_html(
        [{"memory_id": r.memory.memory_id, "content_preview": r.memory.content,
          "state": r.memory.state.name, "state_boost": r.state_boost,
          "base_score": r.base_score, "hop_decay": r.hop_decay,
          "synaptic_boost": r.synaptic_boost, "recency_bonus": r.recency_bonus,
          "final_score": r.final_score, "hop_distance": r.hop_distance,
          "source": r.source, "cell_id": r.cell_id}
         for r in results]
    )


def add_memory_manual(content: str, layer: str, state_str: str, metadata_str: str):
    if not content.strip():
        return "⚠️ Content cannot be empty.", _get_memory_list_html()
    try:
        metadata = json.loads(metadata_str) if metadata_str.strip() else {}
    except json.JSONDecodeError:
        return "⚠️ Metadata must be valid JSON.", _get_memory_list_html()
    try:
        state = MemoryState[state_str.upper()]
    except KeyError:
        return f"⚠️ Invalid state '{state_str}'.", _get_memory_list_html()

    emb = orch.embedder.embed([content])[0]
    entry = engine.store(content=content, embedding=emb, layer=layer, state=state, metadata=metadata)
    return f"✅ Stored: `{entry.memory_id}`", _get_memory_list_html()


# ============================================================
# TAB 3 — ANALYTICS
# ============================================================

def get_analytics():
    stats = engine.get_stats()
    _hist = _record_mss(stats["memory_stability_score"])
    return _render_stats_html(stats, _hist)


# ============================================================
# TAB 4 — FORENSIC TRAIL
# ============================================================

def get_audit():
    entries = engine.get_audit_trail(limit=20)
    if not entries:
        return "<p style='color:#8b949e;'>No audit entries yet.</p>"

    # Check chain integrity
    chain_ok = True
    if len(entries) >= 2:
        for i in range(len(entries) - 1):
            if entries[i]["prev_hash"] != entries[i + 1]["audit_hash"]:
                chain_ok = False
                break

    badge = (
        "<span style='background:#1a7f37; color:#fff; padding:2px 8px; border-radius:4px; font-size:12px;'>CHAIN INTACT ✓</span>"
        if chain_ok else
        "<span style='background:#b91c1c; color:#fff; padding:2px 8px; border-radius:4px; font-size:12px;'>CHAIN BROKEN ✗</span>"
    )
    rows = []
    for e in entries[:15]:
        rows.append(f"""
        <div style='padding:8px 12px; border-bottom:1px solid #21262d; font-size:12px;'>
          <div style='display:flex; justify-content:space-between;'>
            <span style='color:#8b949e;'>{time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))}</span>
            <span style='color:#58a6ff;'>{e["operation"]}</span>
            <span style='color:#e6edf3;'>→ {e["returned_to_agent"]} memories</span>
          </div>
          <div class='audit-hash' style='margin-top:3px;'>
            hash: {e["audit_hash"][:24]}… | prev: {e["prev_hash"][:24]}…
          </div>
          {f'<div style="color:#c9d1d9; margin-top:3px;">{e["query_text"][:80]}</div>' if e.get("query_text") else ''}
        </div>
        """)

    return f"""
    <div style='background:#161b22; border:1px solid #30363d; border-radius:8px; overflow:hidden;'>
      <div style='padding:10px 14px; background:#0d1117; display:flex; justify-content:space-between;'>
        <span style='font-size:13px; font-weight:bold; color:#c9d1d9;'>Audit Hash-Chain</span>
        {badge}
      </div>
      {''.join(rows)}
    </div>
    """


def get_alerts():
    alerts = engine.get_alerts(limit=20)
    if not alerts:
        return "<p style='color:#3fb950; font-size:13px;'>✅ No forensic alerts detected.</p>"
    rows = []
    for a in alerts:
        rows.append(f"""
        <div style='padding:10px; border:1px solid #f85149; border-radius:6px; margin-bottom:8px; background:#1a0d0d;'>
          <div style='color:#f85149; font-weight:bold; font-size:13px;'>🚨 TAMPER DETECTED</div>
          <div style='font-size:12px; font-family:monospace; color:#c9d1d9; margin-top:4px;'>
            Memory: {a.memory_id[:30]}…<br>
            Expected author: {a.expected_author}<br>
            Mismatch score: {a.mismatch_score:.3f} (threshold=0.5)<br>
            Action: {a.action_taken}
          </div>
        </div>
        """)
    return "".join(rows)


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="raven-memory — Adaptive Memory Field",
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif"],
    ),
    css=CSS,
) as demo:

    gr.HTML("""
    <div style='text-align:center; padding:24px 0 12px;'>
      <div style='font-size:2.2em; font-weight:900; letter-spacing:-0.02em;'>🦅 raven-memory</div>
      <div style='font-size:1.05em; color:#8b949e; margin-top:4px;'>
        Adaptive Memory Field &nbsp;·&nbsp; Track 1: MemoryAgent &nbsp;·&nbsp; Qwen Cloud Hackathon
      </div>
      <div style='margin-top:10px; font-size:0.9em; color:#58a6ff; font-style:italic;'>
        "The agent doesn't <em>find</em> memories — it <em>resonates</em> with them."
      </div>
    </div>
    """)

    with gr.Tabs():

        # ---- TAB 1 ----
        with gr.Tab("🎬 Conflict Demo"):
            gr.Markdown("""
            ### Demo: Memory Collapse Around Truth

            Two contradictory memories exist. Query → watch both compete.
            **Reinforce one** → scores split, system *collapses* toward the reinforced truth.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    query_in = gr.Textbox(
                        label="Query",
                        value="Es VIGÍA un sistema de machine learning?",
                        lines=2,
                    )
                    with gr.Row():
                        hops_sl = gr.Slider(0, 4, value=2, step=1, label="Hops")
                        topk_sl = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                    query_btn = gr.Button("🔍 Query Memory Field", variant="primary", size="lg")

                    gr.Markdown("### ⚡ Memory Actions")
                    mem_id_in = gr.Textbox(label="Memory ID (copy from results below)", lines=1)
                    with gr.Row():
                        reinforce_btn = gr.Button("🟢 Reinforce", variant="primary")
                        forget_btn    = gr.Button("⚫ Forget",    variant="secondary")
                    action_out = gr.Markdown()

                with gr.Column(scale=2):
                    qwen_out   = gr.Textbox(label="Agent Response", lines=5, interactive=False)
                    mems_out   = gr.HTML(label="Retrieved Memories")
                    stats_out  = gr.HTML(label="System Metrics")
                    audit_out  = gr.Code(label="Audit Entry", language="json", lines=8)

            query_btn.click(
                run_query,
                inputs=[query_in, hops_sl, topk_sl],
                outputs=[qwen_out, mems_out, stats_out, audit_out],
            )
            reinforce_btn.click(
                do_reinforce,
                inputs=[mem_id_in],
                outputs=[action_out, stats_out, gr.HTML()],  # stats + memory list
            )
            forget_btn.click(
                do_forget,
                inputs=[mem_id_in],
                outputs=[action_out, stats_out, gr.HTML()],
            )

        # ---- TAB 2 ----
        with gr.Tab("🧠 Memory Field"):
            gr.Markdown("### Browse, search, and manage memories")
            with gr.Row():
                search_in  = gr.Textbox(label="Search memories", placeholder="Semantic search…", scale=3)
                search_btn = gr.Button("🔍 Search", scale=1)
                refresh_btn = gr.Button("🔄 All", scale=1)

            mem_list_out = gr.HTML(value=_get_memory_list_html())

            gr.Markdown("### ➕ Add Memory Manually")
            with gr.Row():
                add_content = gr.Textbox(label="Content", lines=2, scale=3)
                add_layer   = gr.Dropdown(["semantic", "episodic", "procedural"], value="semantic", label="Layer", scale=1)
                add_state   = gr.Dropdown(["NEUTRAL", "REINFORCED", "FORGOTTEN"], value="NEUTRAL", label="State", scale=1)
            add_meta    = gr.Textbox(label="Metadata (JSON)", value="{}", lines=1)
            add_btn     = gr.Button("Add to Memory Field", variant="primary")
            add_out     = gr.Markdown()

            search_btn.click(search_memories, inputs=[search_in], outputs=[mem_list_out])
            refresh_btn.click(lambda: _get_memory_list_html(), outputs=[mem_list_out])
            add_btn.click(
                add_memory_manual,
                inputs=[add_content, add_layer, add_state, add_meta],
                outputs=[add_out, mem_list_out],
            )

        # ---- TAB 3 ----
        with gr.Tab("📊 Analytics"):
            gr.Markdown("### Live System Metrics")
            analytics_out = gr.HTML(value=get_analytics())
            refresh_analytics = gr.Button("🔄 Refresh")
            refresh_analytics.click(get_analytics, outputs=[analytics_out])

        # ---- TAB 4 ----
        with gr.Tab("🔍 Forensic Trail"):
            gr.Markdown("### Tamper-proof audit hash-chain + forensic alerts")
            with gr.Row():
                audit_out_tab  = gr.HTML(value=get_audit())
                alerts_out_tab = gr.HTML(value=get_alerts())
            refresh_forensic = gr.Button("🔄 Refresh Trail")
            refresh_forensic.click(
                lambda: (get_audit(), get_alerts()),
                outputs=[audit_out_tab, alerts_out_tab],
            )

    gr.HTML("""
    <div style='text-align:center; padding:16px 0 8px; font-size:12px; color:#6e7681;'>
      raven-memory v1.0 &nbsp;·&nbsp; Apache 2.0 &nbsp;·&nbsp;
      VIGÍA AI Collective &nbsp;·&nbsp; Qwen Cloud Hackathon 2025
    </div>
    """)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    stats = engine.get_stats()
    print(f"\n🦅 raven-memory Demo")
    print(f"   Memories loaded : {stats['total_memories']}")
    print(f"   Initial MSS     : {stats['memory_stability_score']:.3f}")
    print(f"   Cells           : {stats['voronoi_cells']}")
    print(f"   Inhibitory links: {stats['cell_links'].get('INHIBITORY', 0)}")
    print()
    # P1: share=True published the demo on a public gradio.live URL by
    # default — combined with an unauthenticated engine that means anyone
    # on the internet could mutate the memory field. Public tunnelling is
    # now an explicit opt-in: RAVEN_GRADIO_SHARE=1
    import os as _os
    _share = _os.environ.get("RAVEN_GRADIO_SHARE", "0") == "1"
    if _share:
        print("   ⚠️  RAVEN_GRADIO_SHARE=1 — demo will be PUBLICLY reachable via gradio.live")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=_share)
