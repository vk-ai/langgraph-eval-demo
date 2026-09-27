#!/usr/bin/env python3
"""HITL interrupt-lite demo: pause before a flagged tool, then approve or reject.

Usage:
  python examples/interrupt_demo.py
  python examples/interrupt_demo.py --query "What's the weather in Seattle?" --approve
  python examples/interrupt_demo.py --query "What's the weather in Paris?" --reject
  python examples/interrupt_demo.py --approval-tools weather --approve

Honesty: stdlib LangGraph-*style* pause/resume — not real LangGraph interrupt(),
not ApprovalNode, not employer production approval infra.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langgraph_eval_demo import resume_agent, run_agent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HITL interrupt-lite demo (approve/reject)")
    p.add_argument(
        "--query",
        default="What's the weather in Seattle?",
        help="User query (default: Seattle weather)",
    )
    p.add_argument(
        "--approval-tools",
        nargs="+",
        default=["weather"],
        help="Tool names that require human approval before execute",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--approve", action="store_true", help="Approve pending tool and resume")
    g.add_argument("--reject", action="store_true", help="Reject pending tool and abort")
    p.add_argument("--json", action="store_true", help="Print full snapshots as JSON")
    args = p.parse_args(argv)

    paused = run_agent(args.query, approval_tools=set(args.approval_tools))
    if args.json:
        print(json.dumps({"paused": paused}, indent=2))
    else:
        print(f"Q: {args.query}")
        print(f"  interrupted: {paused['interrupted']}")
        print(f"  pending:     {paused.get('pending_decision')}")
        print(f"  tools so far:{paused['tool_trace']}")

    if not paused["interrupted"]:
        print("  (no interrupt — query did not hit an approval-gated tool)")
        print(f"  answer: {paused['final_answer']}")
        return 0

    if not args.approve and not args.reject:
        print("  (paused — re-run with --approve or --reject to resume)")
        return 0

    decision = "approve" if args.approve else "reject"
    done = resume_agent(paused, decision)
    if args.json:
        print(json.dumps({"resumed": done}, indent=2))
    else:
        print(f"  decision:    {decision}")
        print(f"  tools:       {done['tool_trace']}")
        print(f"  answer:      {done['final_answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
