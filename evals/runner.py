"""Golden-task eval runner.

Asserts expected tool sequence and final answer. Designed so pytest fails on
regression when agent behavior drifts.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

# Allow running as `python evals/runner.py` from repo root
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langgraph_eval_demo.agent import run_agent  # noqa: E402

AnswerMatch = Literal["exact", "contains"]


@dataclass(frozen=True)
class GoldenTask:
    id: str
    query: str
    expected_tools: list[str]
    expected_answer: str
    answer_match: AnswerMatch = "exact"


@dataclass
class TaskResult:
    task: GoldenTask
    actual_tools: list[str]
    actual_answer: str | None
    tools_ok: bool
    answer_ok: bool

    @property
    def passed(self) -> bool:
        return self.tools_ok and self.answer_ok


def load_golden_tasks(path: Path | None = None) -> list[GoldenTask]:
    path = path or Path(__file__).with_name("golden_tasks.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenTask(**item) for item in data]


def _match_answer(actual: str | None, expected: str, mode: AnswerMatch) -> bool:
    if actual is None:
        return False
    if mode == "exact":
        return actual.strip() == expected.strip()
    if mode == "contains":
        return expected.lower() in actual.lower()
    raise ValueError(f"Unknown answer_match: {mode}")


def evaluate_task(
    task: GoldenTask,
    agent_fn: Callable[[str], dict[str, Any]] | None = None,
) -> TaskResult:
    agent_fn = agent_fn or run_agent
    result = agent_fn(task.query)
    actual_tools = list(result.get("tool_trace") or [])
    actual_answer = result.get("final_answer")
    tools_ok = actual_tools == list(task.expected_tools)
    answer_ok = _match_answer(actual_answer, task.expected_answer, task.answer_match)
    return TaskResult(
        task=task,
        actual_tools=actual_tools,
        actual_answer=actual_answer,
        tools_ok=tools_ok,
        answer_ok=answer_ok,
    )


def run_eval(
    tasks: list[GoldenTask] | None = None,
    agent_fn: Callable[[str], dict[str, Any]] | None = None,
) -> list[TaskResult]:
    tasks = tasks or load_golden_tasks()
    return [evaluate_task(t, agent_fn=agent_fn) for t in tasks]


def report(results: list[TaskResult]) -> str:
    lines = ["# Golden eval report", ""]
    passed = sum(1 for r in results if r.passed)
    lines.append(f"**Score:** {passed}/{len(results)}")
    lines.append("")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"## [{status}] {r.task.id}")
        lines.append(f"- query: {r.task.query!r}")
        lines.append(f"- tools expected: {r.task.expected_tools}")
        lines.append(f"- tools actual:   {r.actual_tools} ({'ok' if r.tools_ok else 'MISMATCH'})")
        lines.append(f"- answer expected ({r.task.answer_match}): {r.task.expected_answer!r}")
        lines.append(f"- answer actual: {r.actual_answer!r} ({'ok' if r.answer_ok else 'MISMATCH'})")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    results = run_eval()
    print(report(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
