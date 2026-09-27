"""HITL interrupt-lite: pause before flagged tool, approve / reject / fail-closed."""

from __future__ import annotations

import pytest

from langgraph_eval_demo.agent import resume_agent, run_agent


def test_interrupt_pauses_before_weather():
    snap = run_agent(
        "What's the weather in Seattle?",
        approval_tools={"weather"},
    )
    assert snap["interrupted"] is True
    assert snap["tool_trace"] == []
    assert snap["final_answer"] is None
    pd = snap["pending_decision"]
    assert pd is not None
    assert pd["tool_name"] == "weather"
    assert pd["status"] == "pending"
    assert pd["args_digest"]
    assert pd["args"]["city"] == "Seattle"


def test_approve_resumes_and_runs_tool():
    paused = run_agent(
        "What's the weather in Seattle?",
        approval_tools={"weather"},
    )
    done = resume_agent(paused, "approve")
    assert done["interrupted"] is False
    assert done["tool_trace"] == ["weather"]
    assert done["final_answer"] is not None
    assert "Seattle" in done["final_answer"]
    assert done["pending_decision"] is None


def test_reject_aborts_without_tool():
    paused = run_agent(
        "What's the weather in Paris?",
        approval_tools={"weather"},
    )
    done = resume_agent(paused, "reject")
    assert done["interrupted"] is False
    assert done["tool_trace"] == []
    assert "Rejected" in (done["final_answer"] or "")
    assert "weather" in (done["final_answer"] or "")


def test_resume_fail_closed_on_digest_mismatch():
    paused = run_agent(
        "What's the weather in Tokyo?",
        approval_tools={"weather"},
    )
    done = resume_agent(paused, "approve", expected_digest="deadbeef")
    assert "failed closed" in (done["final_answer"] or "").lower()
    assert done["tool_trace"] == []


def test_no_approval_tools_runs_normally():
    # Default path unchanged (no HITL)
    result = run_agent("What's the weather in Seattle?")
    assert result["interrupted"] is False
    assert result["tool_trace"] == ["weather"]
    assert "Seattle" in result["final_answer"]


def test_interrupt_mid_plan_then_approve():
    """Search runs freely; weather is gated mid-plan."""
    paused = run_agent(
        "Search for capital of Japan and what's the weather in Tokyo?",
        approval_tools={"weather"},
    )
    # Planner order: weather first (weather keyword), then search — or search then weather.
    # Either way, if weather is first we interrupt with empty trace; if search first,
    # search may already be on the trace.
    assert paused["interrupted"] is True
    assert paused["pending_decision"]["tool_name"] == "weather"
    done = resume_agent(paused, "approve")
    assert done["interrupted"] is False
    assert "weather" in done["tool_trace"]
    assert done["final_answer"]


def test_resume_requires_interrupted_snapshot():
    snap = run_agent("What is 2 + 2?")
    with pytest.raises(ValueError, match="not interrupted"):
        resume_agent(snap, "approve")


def test_interrupt_snapshot_is_labeled_langgraph_style():
    snap = run_agent("What's the weather in Seattle?", approval_tools={"weather"})
    assert snap["interrupted"] is True
    assert snap["backend_label"] == "[langgraph-style]"
    assert snap["graph_mode"] == "langgraph-style"
    resumed = resume_agent(snap, "approve")
    assert resumed["backend_label"] == "[langgraph-style]"
    assert resumed["interrupted"] is False


def test_interrupt_with_use_real_env_stays_on_style_path(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_EVAL_USE_REAL", "true")
    snap = run_agent("What's the weather in Seattle?", approval_tools={"weather"})
    assert snap["interrupted"] is True
    assert snap["backend_label"] == "[langgraph-style]"
