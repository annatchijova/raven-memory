#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY v1.0 — One-Command Evaluation

Usage:
    python run_all.py           # tests + stress demo
    python run_all.py --api     # tests + stress + REST API
    python run_all.py --demo    # tests + stress + Gradio demo

Authors: Anna Tchijova + Claude (VIGÍA AI Collective)
License: Apache 2.0
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🦅  raven-memory v1.0 — One-Command Eval           ║
║   Adaptive Memory Field · Track 1: MemoryAgent              ║
║   Qwen Cloud Hackathon · Apache 2.0                         ║
╚══════════════════════════════════════════════════════════════╝
"""


def run(label: str, script: str, timeout: int = 900) -> bool:
    print(f"\n{'━'*66}")
    print(f"  PHASE: {label}")
    print(f"{'━'*66}")
    # P1: batch phases get a hard timeout — a hung test or stress run must
    # not block the evaluation pipeline forever. Interactive phases
    # (API/demo servers) intentionally run without one.
    env = os.environ.copy()
    root = str(Path(__file__).parent)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    try:
        result = subprocess.run([sys.executable, script], capture_output=False, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        print(f"\n⏱️  Phase '{label}' exceeded {timeout}s — aborted.")
        return False
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="raven-memory — One-Command Evaluation")
    parser.add_argument("--api",  action="store_true", help="Launch REST API after tests")
    parser.add_argument("--demo", action="store_true", help="Launch Gradio demo after tests")
    parser.add_argument("--skip-stress", action="store_true", help="Skip stress test")
    args = parser.parse_args()

    print(BANNER)

    # 1. Tests
    if not run("INTEGRITY TESTS", "tests/test_suite.py"):
        print("\n❌  Tests failed. Fix before proceeding.")
        sys.exit(1)

    # 2. Stress
    if not args.skip_stress:
        ok = run("STRESS TEST & COLLAPSE VISUALIZATION", "tests/demo_stress_test.py")
        if not ok:
            print("\n⚠️  Stress test had warnings — continuing.")

    # 3. Optional servers
    if args.api:
        api_port = os.environ.get("RAVEN_API_PORT", "8000")
        print(f"\n{'━'*66}")
        print("  PHASE: REST API SERVER")
        print(f"{'━'*66}")
        print(f"  Swagger UI : http://localhost:{api_port}/docs")
        print(f"  WebSocket  : ws://localhost:{api_port}/ws")
        print("  Press Ctrl+C to stop")
        print(f"{'━'*66}")
        subprocess.run([sys.executable, "api_server.py"])

    elif args.demo:
        print(f"\n{'━'*66}")
        print("  PHASE: GRADIO DEMO")
        print(f"{'━'*66}")
        print("  Local URL  : http://localhost:7860")
        print("  Press Ctrl+C to stop")
        print(f"{'━'*66}")
        subprocess.run([sys.executable, "demo_killer.py"])

    else:
        print(f"\n{'━'*66}")
        print("  ✅  EVALUATION COMPLETE")
        print(f"{'━'*66}")
        print()
        print("  Launch REST API:  python run_all.py --api")
        print("  Launch Demo:      python run_all.py --demo")
        print(f"{'━'*66}\n")


if __name__ == "__main__":
    main()
