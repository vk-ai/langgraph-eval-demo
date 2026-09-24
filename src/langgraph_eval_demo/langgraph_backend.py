"""Optional real LangGraph backend (opt-in via LANGGRAPH_EVAL_USE_REAL).

Builds the same plan → tools* → respond graph using ``langgraph.graph.StateGraph``.
Reuses ``plan_node`` / ``execute_tool_node`` / ``respond_node`` and routers from
``agent.py``. Not imported on the default offline path.
"""

from __future__ import annotations

from typing import Any, TypedDict

from .agent import (
    execute_tool_node,
    plan_node,
    respond_node,
    route_after_plan,
    route_after_tool,
)
from .state import AgentState


class LGState(TypedDict):
    """LangGraph channel state: wraps our dataclass so nodes stay unchanged."""

    agent: AgentState


def _wrap_node(fn):  # type: ignore[no-untyped-def]
    def wrapped(state: LGState) -> LGState:
        updated = fn(state["agent"])
        return {"agent": updated}

    return wrapped


def _wrap_router(fn):  # type: ignore[no-untyped-def]
    def wrapped(state: LGState) -> str:
        return fn(state["agent"])

    return wrapped


def build_real_langgraph():
    """Compile a real LangGraph StateGraph mirroring the stdlib stand-in."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(LGState)
    g.add_node("plan", _wrap_node(plan_node))
    g.add_node("tools", _wrap_node(execute_tool_node))
    g.add_node("respond", _wrap_node(respond_node))
    g.add_edge(START, "plan")
    g.add_conditional_edges(
        "plan",
        _wrap_router(route_after_plan),
        {"tools": "tools", "respond": "respond"},
    )
    g.add_conditional_edges(
        "tools",
        _wrap_router(route_after_tool),
        {"tools": "tools", "respond": "respond"},
    )
    g.add_edge("respond", END)
    return g.compile()


def run_with_real_langgraph(query: str) -> AgentState:
    """Invoke the real LangGraph graph; returns final ``AgentState``."""
    compiled = build_real_langgraph()
    initial: LGState = {"agent": AgentState(query=query)}
    final = compiled.invoke(initial)
    agent = final["agent"]
    if isinstance(agent, AgentState):
        return agent
    # Defensive: some versions may return dict-shaped channel values
    if isinstance(agent, dict):
        return AgentState(**agent)
    raise TypeError(f"unexpected agent state type: {type(agent)!r}")
