"""Golden-task eval runner.

Asserts expected tool sequence, final answer, and optional tool-call budgets.
Designed so pytest fails on regression when agent behavior drifts (including
runaway tool loops).
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
    # Budget gates (community: catch runaway tool loops in CI)
    max_tool_calls: int | None = None
    max_graph_steps: int | None = None  # optional related budget (graph invoke steps)


@dataclass
class TaskResult:
    task: GoldenTask
    actual_tools: list[str]
    actual_answer: str | None
    tools_ok: bool
    answer_ok: bool
    budget_ok: bool = True
    budget_detail: str | None = None

    @property
    def passed(self) -> bool:
        return self.tools_ok and self.answer_ok and self.budget_ok


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


def _check_budget(task: GoldenTask, actual_tools: list[str], result: dict[str, Any]) -> tuple[bool, str | None]:
    if task.max_tool_calls is not None and len(actual_tools) > task.max_tool_calls:
        return (
            False,
            f"tool_calls={len(actual_tools)} exceeds max_tool_calls={task.max_tool_calls}",
        )
    if task.max_graph_steps is not None:
        steps = result.get("graph_steps")
        if steps is not None and int(steps) > task.max_graph_steps:
            return (
                False,
                f"graph_steps={steps} exceeds max_graph_steps={task.max_graph_steps}",
            )
    return True, None


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
    budget_ok, budget_detail = _check_budget(task, actual_tools, result)
    return TaskResult(
        task=task,
        actual_tools=actual_tools,
        actual_answer=actual_answer,
        tools_ok=tools_ok,
        answer_ok=answer_ok,
        budget_ok=budget_ok,
        budget_detail=budget_detail,
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
        if r.task.max_tool_calls is not None:
            lines.append(
                f"- max_tool_calls: {r.task.max_tool_calls} "
                f"(actual={len(r.actual_tools)}; {'ok' if r.budget_ok else 'OVER BUDGET'})"
            )
        if r.budget_detail and not r.budget_ok:
            lines.append(f"- budget: {r.budget_detail}")
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
