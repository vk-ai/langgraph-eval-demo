"""Trajectory match modes, must_precede rules, and partial trajectory_score."""

from __future__ import annotations

import pytest

from evals.runner import (
    GoldenTask,
    check_precedence,
    evaluate_task,
    load_golden_tasks,
    match_trajectory,
    trajectory_score,
)


# --- match_trajectory: one row per mode -------------------------------------

@pytest.mark.parametrize(
    "actual, expected, mode, ok",
    [
        # strict: same order, same counts
        (["search", "calculator"], ["search", "calculator"], "strict", True),
        (["calculator", "search"], ["search", "calculator"], "strict", False),
        # unordered: same multiset, any order
        (["calculator", "search"], ["search", "calculator"], "unordered", True),
        (["search", "search", "calculator"], ["search", "calculator"], "unordered", False),
        (["search"], ["search", "calculator"], "unordered", False),
        # subset: only tools from the reference (no extras)
        (["search"], ["search", "calculator"], "subset", True),
        ([], ["search", "calculator"], "subset", True),
        (["search", "weather"], ["search", "calculator"], "subset", False),
        (["search", "search"], ["search", "calculator"], "subset", False),
        # superset: at least the reference tools (extras ok)
        (["search", "calculator", "weather"], ["calculator"], "superset", True),
        (["weather"], ["calculator"], "superset", False),
        (["calculator"], ["calculator", "calculator"], "superset", False),
    ],
)
def test_match_trajectory_modes(actual, expected, mode, ok):
    assert match_trajectory(actual, expected, mode) is ok


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        match_trajectory(["search"], ["search"], "fuzzy")  # type: ignore[arg-type]


# --- must_precede ------------------------------------------------------------

def test_precedence_satisfied():
    ok, detail = check_precedence(["search", "calculator"], [["search", "calculator"]])
    assert ok and detail is None


def test_precedence_violated():
    ok, detail = check_precedence(["calculator", "search"], [["search", "calculator"]])
    assert ok is False
    assert "'search' must precede 'calculator'" in (detail or "")


def test_precedence_vacuous_when_after_tool_not_called():
    ok, _ = check_precedence(["search"], [["search", "calculator"]])
    assert ok is True


def test_precedence_multiple_rules_reports_each_violation():
    ok, detail = check_precedence(
        ["calculator", "weather", "search"],
        [["search", "calculator"], ["search", "weather"]],
    )
    assert ok is False
    assert detail is not None and detail.count("must precede") == 2


def test_precedence_rule_shape_validated():
    with pytest.raises(ValueError):
        check_precedence(["search"], [["search"]])


# --- trajectory_score --------------------------------------------------------

@pytest.mark.parametrize(
    "actual, expected, mode, score",
    [
        # strict → matched prefix / len(expected)
        (["search", "calculator"], ["search", "calculator"], "strict", 1.0),
        (["search", "weather"], ["search", "calculator"], "strict", 0.5),
        (["calculator", "search"], ["search", "calculator"], "strict", 0.0),
        (["search"], ["search", "calculator", "weather"], "strict", 0.3333),
        (["search", "calculator", "weather"], ["search", "calculator"], "strict", 1.0),
        ([], [], "strict", 1.0),
        (["search"], [], "strict", 0.0),
        # unordered / superset → coverage of the reference, any order
        (["calculator", "search"], ["search", "calculator"], "unordered", 1.0),
        (["weather", "search"], ["search", "calculator"], "unordered", 0.5),
        (["search", "calculator", "weather"], ["calculator"], "superset", 1.0),
        (["weather"], ["calculator"], "superset", 0.0),
        # subset → share of the agent's calls that were allowed
        (["search"], ["search", "calculator"], "subset", 1.0),
        (["search", "weather"], ["search", "calculator"], "subset", 0.5),
        ([], ["search"], "subset", 1.0),
    ],
)
def test_trajectory_score_partial_credit(actual, expected, mode, score):
    assert trajectory_score(actual, expected, mode) == score


# --- evaluate_task integration ----------------------------------------------

def _fake_agent(trace: list[str], answer: str = "ok"):
    def agent(query: str):
        return {"tool_trace": list(trace), "final_answer": answer}

    return agent


def _task(**kw) -> GoldenTask:
    base = dict(
        id="t",
        query="q",
        expected_tools=["search", "calculator"],
        expected_answer="ok",
        answer_match="exact",
        max_tool_calls=3,
        check_tool_fixtures=False,
    )
    base.update(kw)
    return GoldenTask(**base)


def test_unordered_passes_where_strict_fails():
    agent = _fake_agent(["calculator", "search"])
    assert evaluate_task(_task(trajectory_match="strict"), agent_fn=agent).passed is False
    assert evaluate_task(_task(trajectory_match="unordered"), agent_fn=agent).passed is True


def test_unordered_plus_precedence_catches_causal_violation():
    """The 'between strict and unordered' case: order is free except search→calculator."""
    task = _task(trajectory_match="unordered", must_precede=[["search", "calculator"]])
    good = evaluate_task(task, agent_fn=_fake_agent(["search", "calculator"]))
    bad = evaluate_task(task, agent_fn=_fake_agent(["calculator", "search"]))
    assert good.passed is True
    assert bad.tools_ok is True  # multiset matches...
    assert bad.precedence_ok is False  # ...but the causal rule is broken
    assert bad.passed is False
    # Partial score reflects the multiset match; precedence is reported separately.
    assert bad.trajectory_score == 1.0
    assert "must precede" in (bad.precedence_detail or "")


def test_superset_allows_extra_tools_but_not_missing():
    task = _task(expected_tools=["calculator"], trajectory_match="superset")
    assert evaluate_task(task, agent_fn=_fake_agent(["search", "calculator"])).passed is True
    assert evaluate_task(task, agent_fn=_fake_agent(["search"])).passed is False


def test_subset_rejects_unexpected_tool():
    task = _task(trajectory_match="subset")
    assert evaluate_task(task, agent_fn=_fake_agent(["search"])).passed is True
    assert evaluate_task(task, agent_fn=_fake_agent(["search", "weather"])).passed is False


def test_arg_digests_order_insensitive_only_in_non_strict_modes():
    def agent(query: str):
        return {
            "tool_trace": ["calculator", "search"],
            "final_answer": "ok",
            "tool_arg_digests": ["b", "a"],
        }

    strict = _task(trajectory_match="strict", expected_tools=["calculator", "search"],
                   expected_tool_args=["a", "b"])
    unordered = _task(trajectory_match="unordered", expected_tool_args=["a", "b"])
    assert evaluate_task(strict, agent_fn=agent).args_ok is False
    assert evaluate_task(unordered, agent_fn=agent).args_ok is True


def test_default_mode_is_strict_for_existing_tasks():
    tasks = {t.id: t for t in load_golden_tasks()}
    assert tasks["calc_simple"].trajectory_match == "strict"
    assert tasks["calc_simple"].must_precede is None


def test_golden_file_covers_every_mode_and_a_precedence_rule():
    tasks = load_golden_tasks()
    modes = {t.trajectory_match for t in tasks}
    assert modes == {"strict", "unordered", "subset", "superset"}
    assert any(t.must_precede for t in tasks)


def test_golden_any_order_task_would_fail_strict():
    """Guard: the 'any order' golden task is genuinely order-varying."""
    tasks = {t.id: t for t in load_golden_tasks()}
    t = tasks["weather_and_search_any_order"]
    r = evaluate_task(t)
    assert r.passed is True
    assert match_trajectory(r.actual_tools, t.expected_tools, "strict") is False


def test_report_includes_mode_and_score():
    from evals.runner import report, run_eval

    text = report(run_eval())
    assert "trajectory_match: unordered" in text
    assert "must_precede" in text
    assert "Mean trajectory_score" in text
