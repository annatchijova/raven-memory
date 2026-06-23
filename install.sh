#!/usr/bin/env bash
# Copyright 2026 Anna Tchijova — VIGÍA AI Collective
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# raven-memory — Adaptive Memory Field for Agentic Systems
# Installation script
# Usage: bash install.sh

set -euo pipefail

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
ok()   { echo -e "${G}  ✓${N} $1"; }
warn() { echo -e "${Y}  ⚠${N} $1"; }
err()  { echo -e "${R}  ✗${N} $1"; exit 1; }

echo ""
echo "raven-memory — Adaptive Memory Field for Agentic Systems"
echo "Installation"
echo ""

# Python version check (3.10+)
command -v python3 >/dev/null 2>&1 || err "python3 not found"
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJ=$(echo "$PY_VER" | cut -d. -f1)
PY_MIN=$(echo "$PY_VER" | cut -d. -f2)
[[ "$PY_MAJ" -ge 3 && "$PY_MIN" -ge 10 ]] || err "Python 3.10+ required (found $PY_VER)"
ok "Python $PY_VER"

# Virtual environment
if [ ! -d .venv ]; then
    python3 -m venv .venv
    ok ".venv created"
else
    ok ".venv already exists"
fi

# shellcheck source=/dev/null
source .venv/bin/activate
ok "Virtual environment activated"

pip install --quiet --upgrade pip
echo "  Installing dependencies (torch + sentence-transformers may take a few minutes)..."
pip install --quiet -r requirements.txt
ok "Dependencies installed"

# API key hint
if [ -n "${DASHSCOPE_API_KEY:-}" ]; then
    ok "DASHSCOPE_API_KEY set — Qwen Cloud embeddings enabled"
else
    warn "DASHSCOPE_API_KEY not set — will use offline deterministic embeddings (all mechanics intact)"
    warn "  export DASHSCOPE_API_KEY=your_key_here   to enable Qwen Cloud"
fi

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    ok "ANTHROPIC_API_KEY set — Claude backend available"
fi

echo ""
echo "Installation complete."
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python run_all.py                   # run tests"
echo "  python run_all.py --demo            # Gradio demo on :7860"
echo "  python run_all.py --api             # REST API on :8000 (/docs)"
