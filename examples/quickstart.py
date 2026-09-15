#!/usr/bin/env python3
"""Quickstart: run the multi-step agent and print tool trace + answer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langgraph_eval_demo import run_agent


def main() -> None:
    queries = [
        "What is 15 * 7?",
        "What's the weather in Seattle?",
        "Search for capital of Japan and calculate 2 + 2",
        "Find the population of Tokyo then multiply that by 2",
    ]
    for q in queries:
        result = run_agent(q)
        print(f"Q: {q}")
        print(f"  tools:  {result['tool_trace']}")
        print(f"  answer: {result['final_answer']}")
        print()


if __name__ == "__main__":
    main()
