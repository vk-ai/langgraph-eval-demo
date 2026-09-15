"""Golden eval must pass; regression (wrong tools/answer) must fail assertions."""

from __future__ import annotations

import copy

import pytest

from evals.runner import GoldenTask, evaluate_task, load_golden_tasks, run_eval


def test_all_golden_tasks_pass():
    results = run_eval()
    failures = [r for r in results if not r.passed]
    assert not failures, "\n".join(
        f"{r.task.id}: tools={r.actual_tools} answer={r.actual_answer!r}" for r in failures
    )


def test_golden_file_nonempty():
    tasks = load_golden_tasks()
    assert len(tasks) >= 5


def test_regression_wrong_tools_fails():
    """Simulated regression: agent skips calculator → eval detects mismatch."""
    task = GoldenTask(
        id="regr_tools",
        query="What is 15 * 7?",
        expected_tools=["calculator"],
        expected_answer="105",
        answer_match="exact",
    )

    def broken_agent(query: str):
        return {"tool_trace": ["search"], "final_answer": "105"}

    result = evaluate_task(task, agent_fn=broken_agent)
    assert result.passed is False
    assert result.tools_ok is False


def test_regression_wrong_answer_fails():
    """Simulated regression: wrong final answer → eval detects mismatch."""
    task = GoldenTask(
        id="regr_answer",
        query="What is 15 * 7?",
        expected_tools=["calculator"],
        expected_answer="105",
        answer_match="exact",
    )

    def broken_agent(query: str):
        return {"tool_trace": ["calculator"], "final_answer": "106"}

    result = evaluate_task(task, agent_fn=broken_agent)
    assert result.passed is False
    assert result.answer_ok is False


def test_pytest_asserts_on_single_golden_regression():
    """End-to-end: mutating expected answer makes the golden check fail."""
    tasks = load_golden_tasks()
    task = copy.deepcopy(tasks[0])
    # Force an impossible expectation to prove the harness fails closed
    broken = GoldenTask(
        id=task.id,
        query=task.query,
        expected_tools=task.expected_tools,
        expected_answer="___NOT_THE_REAL_ANSWER___",
        answer_match="exact",
    )
    result = evaluate_task(broken)
    with pytest.raises(AssertionError):
        assert result.passed, "golden regression should fail"
