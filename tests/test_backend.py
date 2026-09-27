"""Backend selection / honesty labels — all offline, no langgraph required."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from langgraph_eval_demo.agent import run_agent
from langgraph_eval_demo.backend import (
    backend_label,
    fallback_reason,
    graph_mode,
    langgraph_importable,
    wants_real_langgraph,
)


def test_default_run_agent_tagged_style(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_EVAL_USE_REAL", raising=False)
    result = run_agent("What is 15 * 7?")
    assert result["tool_trace"] == ["calculator"]
    assert result["final_answer"] == "105"
    assert result["graph_mode"] == "langgraph-style"
    assert result["backend_label"] == "[langgraph-style]"
    assert any("langgraph-style" in m for m in result["messages"])


def test_flag_false_never_requires_langgraph(monkeypatch):
    """With flag unset/false, selection stays style even if langgraph were present."""
    monkeypatch.delenv("LANGGRAPH_EVAL_USE_REAL", raising=False)
    assert wants_real_langgraph() is False
    assert graph_mode() == "langgraph-style"
    assert backend_label() == "[langgraph-style]"
    assert fallback_reason() is None

    monkeypatch.setenv("LANGGRAPH_EVAL_USE_REAL", "false")
    assert wants_real_langgraph() is False
    assert graph_mode() == "langgraph-style"


def test_flag_true_missing_langgraph_falls_back(monkeypatch):
    """Opt-in without the package → style snapshot + clear fallback reason."""
    monkeypatch.setenv("LANGGRAPH_EVAL_USE_REAL", "true")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langgraph" or name.startswith("langgraph."):
            raise ImportError("langgraph deliberately unavailable in offline test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for key in list(sys.modules):
        if key == "langgraph" or key.startswith("langgraph."):
            sys.modules.pop(key, None)

    assert langgraph_importable() is False
    assert graph_mode() == "langgraph-style"
    reason = fallback_reason()
    assert reason is not None
    assert "langgraph" in reason.lower()

    result = run_agent("What is 15 * 7?")
    assert result["graph_mode"] == "langgraph-style"
    assert result["backend_label"] == "[langgraph-style]"
    assert result["final_answer"] == "105"
    assert any("LANGGRAPH_EVAL_USE_REAL" in m for m in result["messages"])


def test_selection_helpers_when_importable(monkeypatch):
    """Unit-test real-path selection without requiring the dep: stub importable()."""
    monkeypatch.setenv("LANGGRAPH_EVAL_USE_REAL", "true")
    monkeypatch.setattr(
        "langgraph_eval_demo.backend.langgraph_importable",
        lambda: True,
    )
    assert graph_mode() == "langgraph"
    assert backend_label() == "[langgraph]"
    assert fallback_reason() is None


def test_real_path_invoke_failure_falls_back(monkeypatch):
    """If real backend raises, run_agent falls back to style without crashing."""
    monkeypatch.setenv("LANGGRAPH_EVAL_USE_REAL", "true")
    monkeypatch.setattr(
        "langgraph_eval_demo.backend.langgraph_importable",
        lambda: True,
    )

    fake = types.ModuleType("langgraph_eval_demo.langgraph_backend")

    def boom(_query: str):
        raise RuntimeError("simulated real-graph failure")

    fake.run_with_real_langgraph = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langgraph_eval_demo.langgraph_backend", fake)

    result = run_agent("What is 15 * 7?")
    assert result["graph_mode"] == "langgraph-style"
    assert result["backend_label"] == "[langgraph-style]"
    assert result["final_answer"] == "105"
    assert any("falling back" in m.lower() for m in result["messages"])


def test_real_langgraph_path_when_installed(monkeypatch):
    """Exercise real path only when the optional extra is present in the env."""
    if not langgraph_importable():
        pytest.skip("optional langgraph not installed (CI stays offline)")
    monkeypatch.setenv("LANGGRAPH_EVAL_USE_REAL", "true")
    result = run_agent("What is 15 * 7?")
    assert result["tool_trace"] == ["calculator"]
    assert result["final_answer"] == "105"
    assert result["graph_mode"] == "langgraph"
    assert result["backend_label"] == "[langgraph]"
