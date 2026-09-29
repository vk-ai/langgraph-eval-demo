"""Golden-task eval runner.

Asserts expected tool sequence (with a configurable ``trajectory_match`` mode
and optional ``must_precede`` precedence rules), final answer, optional
tool-call budgets, optional ``expected_tool_args`` digests, and frozen
tool-result fixtures.
Designed so pytest fails on regression when agent behavior drifts (including
runaway tool loops and arg/environment drift).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

# Allow running as `python evals/runner.py` from repo root
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langgraph_eval_demo.agent import run_agent  # noqa: E402
from langgraph_eval_demo.fixtures import (  # noqa: E402
    load_catalog,
    verify_against_fixture,
)

AnswerMatch = Literal["exact", "contains"]
# Deterministic trajectory match modes (same *idea* as agentevals /
# LangSmith trajectory evals; stdlib re-implementation, not that library).
TrajectoryMatch = Literal["strict", "unordered", "subset", "superset"]
TRAJECTORY_MODES: tuple[str, ...] = ("strict", "unordered", "subset", "superset")


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
    # Optional SHA-256 digests of resolved tool args (canonical sorted JSON)
    expected_tool_args: list[str] | None = None
    # When True (default), compare tool outputs to frozen fixtures when present
    check_tool_fixtures: bool = True
    # How to compare the actual tool trace to ``expected_tools``
    trajectory_match: TrajectoryMatch = "strict"
    # Precedence (partial-order) rules: [["search", "calculator"]] means
    # "if calculator is called, search must have been called before it".
    must_precede: list[list[str]] | None = None


@dataclass
class TaskResult:
    task: GoldenTask
    actual_tools: list[str]
    actual_answer: str | None
    tools_ok: bool
    answer_ok: bool
    budget_ok: bool = True
    budget_detail: str | None = None
    args_ok: bool = True
    args_detail: str | None = None
    fixtures_ok: bool = True
    fixtures_detail: str | None = None
    actual_arg_digests: list[str] = field(default_factory=list)
    precedence_ok: bool = True
    precedence_detail: str | None = None
    # Mode-aware partial credit in [0, 1] (see trajectory_score). Informational only.
    trajectory_score: float = 1.0

    @property
    def passed(self) -> bool:
        return (
            self.tools_ok
            and self.precedence_ok
            and self.answer_ok
            and self.budget_ok
            and self.args_ok
            and self.fixtures_ok
        )


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


def match_trajectory(
    actual: list[str], expected: list[str], mode: TrajectoryMatch = "strict"
) -> bool:
    """Compare tool traces. Multiset semantics (duplicate calls count).

    - ``strict``:    same tools, same order, same counts
    - ``unordered``: same tools and counts, any order
    - ``subset``:    agent only called tools from ``expected`` (no extras)
    - ``superset``:  agent called at least every tool in ``expected`` (extras ok)
    """
    if mode == "strict":
        return list(actual) == list(expected)
    a, e = Counter(actual), Counter(expected)
    if mode == "unordered":
        return a == e
    if mode == "subset":
        return all(a[k] <= e[k] for k in a)
    if mode == "superset":
        return all(a[k] >= e[k] for k in e)
    raise ValueError(f"Unknown trajectory_match: {mode} (expected one of {TRAJECTORY_MODES})")


def check_precedence(
    actual: list[str], rules: list[list[str]] | None
) -> tuple[bool, str | None]:
    """Check ``[before, after]`` rules against the actual tool trace.

    A rule ``[A, B]`` holds when every call to B has at least one call to A
    earlier in the trace (i.e. first A index < first B index). If B is never
    called the rule is vacuously satisfied — use ``trajectory_match`` to
    require that B is called at all.
    """
    if not rules:
        return True, None
    violations: list[str] = []
    for rule in rules:
        if len(rule) != 2:
            raise ValueError(f"must_precede rule must be [before, after], got {rule!r}")
        before, after = rule
        if after not in actual:
            continue
        first_after = actual.index(after)
        if before not in actual[:first_after]:
            violations.append(
                f"{before!r} must precede {after!r} (first {after!r} at index {first_after}, "
                f"no earlier {before!r})"
            )
    if violations:
        return False, "; ".join(violations)
    return True, None


def trajectory_score(
    actual: list[str], expected: list[str], mode: TrajectoryMatch = "strict"
) -> float:
    """Deterministic partial credit in [0, 1] (informational; never gates CI).

    - ``strict``: matched prefix length / len(expected) — order-aware, so
      ``[search, weather]`` vs ``[search, calculator]`` scores 0.5.
    - ``unordered`` / ``superset``: multiset overlap / len(expected) — how
      much of the reference trajectory was covered, in any order.
    - ``subset``: multiset overlap / len(actual) — share of the agent's calls
      that were allowed (1.0 when the agent called nothing).

    An empty expected trajectory scores 1.0 only if the agent called no tools
    (except in ``subset`` / ``superset``, where it is trivially satisfied).
    """
    overlap = sum((Counter(actual) & Counter(expected)).values())
    if mode == "subset":
        return 1.0 if not actual else round(overlap / len(actual), 4)
    if not expected:
        return 1.0 if (not actual or mode == "superset") else 0.0
    if mode == "strict":
        matched = 0
        for a, e in zip(actual, expected):
            if a != e:
                break
            matched += 1
        return round(matched / len(expected), 4)
    if mode in ("unordered", "superset"):
        return round(overlap / len(expected), 4)
    raise ValueError(f"Unknown trajectory_match: {mode} (expected one of {TRAJECTORY_MODES})")


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


def _check_arg_digests(
    task: GoldenTask, actual_digests: list[str]
) -> tuple[bool, str | None]:
    if task.expected_tool_args is None:
        return True, None
    expected = list(task.expected_tool_args)
    if actual_digests == expected:
        return True, None
    # Non-strict trajectory modes allow reordering, so pin args as a multiset.
    if task.trajectory_match != "strict" and Counter(actual_digests) == Counter(expected):
        return True, None
    return (
        False,
        f"tool_arg digests mismatch: expected={expected} actual={actual_digests}",
    )


def _check_fixtures(result: dict[str, Any], task: GoldenTask) -> tuple[bool, str | None]:
    if not task.check_tool_fixtures:
        return True, None
    catalog = load_catalog()
    details: list[str] = []
    for tr in result.get("tool_results") or []:
        name = tr.get("name") or ""
        args = tr.get("args") or {}
        output = tr.get("output") or ""
        ok, detail = verify_against_fixture(name, args, output, catalog=catalog)
        if not ok and detail:
            details.append(detail)
    if details:
        return False, "; ".join(details)
    return True, None


def evaluate_task(
    task: GoldenTask,
    agent_fn: Callable[[str], dict[str, Any]] | None = None,
) -> TaskResult:
    agent_fn = agent_fn or run_agent
    result = agent_fn(task.query)
    actual_tools = list(result.get("tool_trace") or [])
    actual_answer = result.get("final_answer")
    actual_digests = list(result.get("tool_arg_digests") or [])
    tools_ok = match_trajectory(actual_tools, list(task.expected_tools), task.trajectory_match)
    precedence_ok, precedence_detail = check_precedence(actual_tools, task.must_precede)
    answer_ok = _match_answer(actual_answer, task.expected_answer, task.answer_match)
    budget_ok, budget_detail = _check_budget(task, actual_tools, result)
    args_ok, args_detail = _check_arg_digests(task, actual_digests)
    fixtures_ok, fixtures_detail = _check_fixtures(result, task)
    return TaskResult(
        task=task,
        actual_tools=actual_tools,
        actual_answer=actual_answer,
        tools_ok=tools_ok,
        answer_ok=answer_ok,
        budget_ok=budget_ok,
        budget_detail=budget_detail,
        args_ok=args_ok,
        args_detail=args_detail,
        fixtures_ok=fixtures_ok,
        fixtures_detail=fixtures_detail,
        actual_arg_digests=actual_digests,
        precedence_ok=precedence_ok,
        precedence_detail=precedence_detail,
        trajectory_score=trajectory_score(
            actual_tools, list(task.expected_tools), task.trajectory_match
        ),
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
    if results:
        mean_traj = sum(r.trajectory_score for r in results) / len(results)
        lines.append(f"**Mean trajectory_score:** {mean_traj:.2f}")
    lines.append("")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"## [{status}] {r.task.id}")
        lines.append(f"- query: {r.task.query!r}")
        lines.append(f"- tools expected: {r.task.expected_tools}")
        lines.append(f"- tools actual:   {r.actual_tools} ({'ok' if r.tools_ok else 'MISMATCH'})")
        lines.append(
            f"- trajectory_match: {r.task.trajectory_match} "
            f"(trajectory_score={r.trajectory_score:.2f})"
        )
        if r.task.must_precede:
            lines.append(
                f"- must_precede: {r.task.must_precede} "
                f"({'ok' if r.precedence_ok else 'VIOLATED'})"
            )
        if r.precedence_detail and not r.precedence_ok:
            lines.append(f"- precedence: {r.precedence_detail}")
        if r.task.max_tool_calls is not None:
            lines.append(
                f"- max_tool_calls: {r.task.max_tool_calls} "
                f"(actual={len(r.actual_tools)}; {'ok' if r.budget_ok else 'OVER BUDGET'})"
            )
        if r.budget_detail and not r.budget_ok:
            lines.append(f"- budget: {r.budget_detail}")
        if r.task.expected_tool_args is not None:
            lines.append(
                f"- tool_arg digests: {'ok' if r.args_ok else 'MISMATCH'} "
                f"(expected={r.task.expected_tool_args} actual={r.actual_arg_digests})"
            )
        if r.args_detail and not r.args_ok:
            lines.append(f"- args: {r.args_detail}")
        if not r.fixtures_ok:
            lines.append(f"- fixtures: {r.fixtures_detail}")
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
