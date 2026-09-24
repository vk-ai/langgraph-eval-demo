"""Multi-step tool-calling agent built on the thin StateGraph.

Uses a deterministic planner (no LLM / no network) so CI runs fully offline.
The planner emits an ordered plan of tool calls from the user query, then the
graph walks: plan → (execute_tool)* → respond.
"""

from __future__ import annotations

import re
from typing import Any

from .graph import END, StateGraph
from .state import AgentState, PendingDecision, ToolCall, ToolResult
from .fixtures import tool_arg_digest
from .tools import call_tool


def _extract_city(text: str) -> str | None:
    m = re.search(
        r"\b(?:in|for|at)\s+([A-Za-z][A-Za-z\s]+?)(?:\?|$|,|\.|$)",
        text,
        re.I,
    )
    if m:
        return m.group(1).strip().rstrip("?.!,")
    for city in ("seattle", "paris", "tokyo", "new york"):
        if city in text.lower():
            return city.title() if city != "new york" else "New York"
    return None


def _extract_math(text: str) -> str | None:
    # Prefer explicit "calculate ..." / "what is N op N"
    m = re.search(
        r"(?:calculate|compute|what(?:'s| is))\s+([\d\.\s\+\-\*\/×÷\^\(\)]+)",
        text,
        re.I,
    )
    if m:
        return m.group(1).strip().rstrip("?.!")
    m = re.search(r"([\d\.]+\s*[\+\-\*\/×÷\^]\s*[\d\.]+(?:\s*[\+\-\*\/×÷\^]\s*[\d\.]+)*)", text)
    if m:
        return m.group(1).strip()
    # "multiply X by Y" / "X times Y"
    m = re.search(r"(?:multiply|times)\s+(\d+(?:\.\d+)?)\s+(?:by|and|times)?\s*(\d+(?:\.\d+)?)", text, re.I)
    if m:
        return f"{m.group(1)} * {m.group(2)}"
    m = re.search(r"(\d+(?:\.\d+)?)\s+(?:times|multiplied by)\s+(\d+(?:\.\d+)?)", text, re.I)
    if m:
        return f"{m.group(1)} * {m.group(2)}"
    m = re.search(r"(?:add|plus)\s+(\d+(?:\.\d+)?)\s+(?:and|to|plus)?\s*(\d+(?:\.\d+)?)", text, re.I)
    if m:
        return f"{m.group(1)} + {m.group(2)}"
    return None


def _needs_search(text: str) -> str | None:
    lower = text.lower()
    patterns = [
        (r"capital of \w+", lambda m: m.group(0)),
        (r"population of \w+", lambda m: m.group(0)),
        (r"search(?:\s+for)?\s+(.+?)(?:\s+and\s+|\s+then\s+|$)", lambda m: m.group(1).strip()),
        (r"what is (langgraph|python programming)", lambda m: m.group(1)),
        (r"find (?:the )?(population of \w+|capital of \w+)", lambda m: m.group(1)),
    ]
    for pat, extract in patterns:
        m = re.search(pat, lower)
        if m:
            return extract(m)
    # bare knowledge lookups
    for key in (
        "capital of france",
        "capital of japan",
        "population of tokyo",
        "population of paris",
        "langgraph",
        "python programming",
    ):
        if key in lower:
            return key
    return None


def build_plan(query: str) -> list[ToolCall]:
    """Deterministic multi-step planner: may emit 0..N tool calls in order."""
    plan: list[ToolCall] = []
    q = query.strip()
    lower = q.lower()

    # Weather first if asked (often standalone)
    if any(w in lower for w in ("weather", "temperature", "forecast")):
        city = _extract_city(q) or "Seattle"
        plan.append(ToolCall("weather", {"city": city}))

    # Search lookups
    search_q = _needs_search(q)
    if search_q:
        plan.append(ToolCall("search", {"query": search_q}))

    # Calculator — including "multiply that / the population by N" after search
    math_expr = _extract_math(q)
    if math_expr:
        plan.append(ToolCall("calculator", {"expression": math_expr}))
    else:
        # "multiply/double/triple [the population|that|it] by N"
        m = re.search(
            r"(?:multiply|double|triple)\s+(?:the\s+)?(?:population|that|it|result)?\s*"
            r"(?:by\s+)?(\d+(?:\.\d+)?)?",
            lower,
        )
        if m and any(t.name == "search" for t in plan):
            # Pull a number out of upcoming search result at execute time via placeholder
            factor = m.group(1)
            if "double" in lower:
                factor = "2"
            elif "triple" in lower:
                factor = "3"
            if factor:
                plan.append(
                    ToolCall(
                        "calculator",
                        {"expression": f"__FROM_SEARCH__ * {factor}"},
                    )
                )

    return plan


def _number_from_text(text: str) -> str | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*million", text, re.I)
    if m:
        # keep as millions for simple demos, or expand
        val = float(m.group(1)) * 1_000_000
        return str(int(val)) if val == int(val) else str(val)
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    return None


def plan_node(state: AgentState) -> AgentState:
    """Entry node: build the tool plan once."""
    if not state.pending_tools and not state.done and state.final_answer is None:
        state.pending_tools = build_plan(state.query)
        state.messages.append(f"plan: {[t.name for t in state.pending_tools]}")
        if not state.pending_tools:
            # No tools needed — answer directly for trivial queries
            state.final_answer = (
                "I can help with search, calculations, and weather. "
                "Try asking about a capital city, math, or the weather."
            )
            state.done = True
    return state


def route_after_plan(state: AgentState) -> str:
    if state.done or state.final_answer is not None:
        return "respond"
    if state.pending_tools:
        return "tools"
    return "respond"


def _resolve_tool_args(state: AgentState, call: ToolCall) -> dict[str, Any]:
    """Resolve placeholders (e.g. __FROM_SEARCH__) before digest / approval / execute."""
    args = dict(call.args)
    if call.name == "calculator" and "__FROM_SEARCH__" in str(args.get("expression", "")):
        state.derived_from_search = True
        prior = next(
            (r.output for r in reversed(state.tool_results) if r.name == "search"),
            None,
        )
        num = _number_from_text(prior or "")
        if num:
            args["expression"] = args["expression"].replace("__FROM_SEARCH__", num)
        else:
            args["expression"] = "0"
    return args


def execute_tool_node(state: AgentState) -> AgentState:
    """Pop and run the next pending tool call (or pause for HITL interrupt-lite)."""
    if not state.pending_tools:
        return state

    # Already interrupted — stay paused until resume()
    if state.interrupted and state.pending_decision is not None:
        return state

    call = state.pending_tools[0]
    args = _resolve_tool_args(state, call)
    digest = tool_arg_digest(args)

    # HITL interrupt-lite: gate flagged tools behind approve/reject
    if call.name in state.approval_tools:
        pd = state.pending_decision
        if pd is None or pd.status == "pending":
            # First hit — pause before side effect; keep call on pending_tools
            state.pending_decision = PendingDecision(
                tool_name=call.name,
                args=dict(args),
                args_digest=digest,
                status="pending",
            )
            # Store resolved args back so resume executes the same digest
            call.args = dict(args)
            state.interrupted = True
            state.messages.append(
                f"interrupt: awaiting approval for {call.name} digest={digest}"
            )
            return state
        if pd.status == "rejected":
            state.pending_tools.pop(0)
            state.pending_decision = None
            state.interrupted = False
            state.final_answer = (
                f"Rejected tool call `{call.name}` "
                f"(digest={digest}); aborting remaining plan."
            )
            state.done = True
            state.messages.append(f"reject:{call.name}")
            return state
        if pd.status == "approved":
            # Fail closed: digest must still match the paused decision
            if pd.args_digest != digest or pd.tool_name != call.name:
                state.interrupted = True
                state.final_answer = (
                    "Resume failed closed: pending decision digest/tool mismatch."
                )
                state.done = True
                state.messages.append("resume_mismatch")
                return state
            # Fall through to execute; clear pending gate
            state.pending_decision = None
            state.interrupted = False

    # Execute
    state.pending_tools.pop(0)
    output = call_tool(call.name, args)
    state.tool_trace.append(call.name)
    state.tool_arg_digests.append(digest)
    state.tool_results.append(
        ToolResult(name=call.name, output=output, args=dict(args), args_digest=digest)
    )
    state.messages.append(f"tool:{call.name}({args}) -> {output}")
    return state


def route_after_tool(state: AgentState) -> str:
    if state.interrupted:
        return "end"
    if state.done or state.final_answer is not None:
        return "respond"
    if state.pending_tools:
        return "tools"
    return "respond"


def respond_node(state: AgentState) -> AgentState:
    """Compose a final answer from tool results (or prior direct answer)."""
    if state.final_answer is not None:
        state.done = True
        return state

    if not state.tool_results:
        state.final_answer = "No tools were needed; no answer produced."
        state.done = True
        return state

    # Prefer last calculator result if present, else last tool output, with context
    calc = next((r for r in reversed(state.tool_results) if r.name == "calculator"), None)
    search = next((r for r in reversed(state.tool_results) if r.name == "search"), None)
    weather_r = next((r for r in reversed(state.tool_results) if r.name == "weather"), None)

    parts: list[str] = []
    if weather_r:
        parts.append(weather_r.output)
    if search:
        parts.append(search.output)
    if calc:
        parts.append(calc.output)

    # Multi-step synthesis: search then derived calculator (population * N, etc.)
    if search and calc and state.derived_from_search:
        state.final_answer = f"{search.output} Multiplied result: {calc.output}."
    elif search and calc:
        state.final_answer = " ".join(parts)
    elif len(parts) == 1:
        state.final_answer = parts[0]
    else:
        state.final_answer = " ".join(parts)

    state.done = True
    return state


def build_agent_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("tools", execute_tool_node)
    g.add_node("respond", respond_node)
    g.set_entry_point("plan")
    g.add_conditional_edges(
        "plan",
        route_after_plan,
        {"tools": "tools", "respond": "respond"},
    )
    g.add_conditional_edges(
        "tools",
        route_after_tool,
        {"tools": "tools", "respond": "respond", "end": END},
    )
    g.add_edge("respond", END)
    return g


def run_agent(
    query: str,
    *,
    approval_tools: frozenset[str] | set[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Run the agent graph and return a serializable result snapshot.

    If ``approval_tools`` is set (e.g. ``{"weather"}``), the graph pauses
    *before* executing those tools and returns ``interrupted=True`` with a
    ``pending_decision``. Call :func:`resume_agent` to approve or reject.
    """
    graph = build_agent_graph().compile()
    state = AgentState(
        query=query,
        approval_tools=frozenset(approval_tools or ()),
    )
    final = graph.invoke(state)
    return final.snapshot()


def _state_from_snapshot(snap: dict[str, Any]) -> AgentState:
    """Rehydrate AgentState from a snapshot (interrupt-lite resume path)."""
    pending_tools = [
        ToolCall(name=t["name"], args=dict(t.get("args") or {}))
        for t in snap.get("pending_tools") or []
    ]
    tool_results = [
        ToolResult(
            name=r["name"],
            output=r["output"],
            args=dict(r.get("args") or {}),
            args_digest=str(r.get("args_digest") or ""),
        )
        for r in snap.get("tool_results") or []
    ]
    pd_raw = snap.get("pending_decision")
    pending_decision = PendingDecision.from_dict(pd_raw) if pd_raw else None
    return AgentState(
        query=str(snap.get("query") or ""),
        messages=list(snap.get("messages") or []),
        pending_tools=pending_tools,
        tool_trace=list(snap.get("tool_trace") or []),
        tool_results=tool_results,
        tool_arg_digests=list(snap.get("tool_arg_digests") or []),
        final_answer=snap.get("final_answer"),
        done=bool(snap.get("done")),
        derived_from_search=bool(snap.get("derived_from_search")),
        approval_tools=frozenset(snap.get("approval_tools") or ()),
        pending_decision=pending_decision,
        interrupted=bool(snap.get("interrupted")),
    )


def resume_agent(
    snapshot: dict[str, Any],
    decision: str,
    *,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    """Resume a paused (interrupted) agent with ``approve`` or ``reject``.

    Fail-closed: if ``expected_digest`` is given (or the snapshot has a pending
    digest) and the rehydrated pending tool digest no longer matches, abort.
    """
    decision = decision.strip().lower()
    if decision not in {"approve", "approved", "reject", "rejected"}:
        raise ValueError("decision must be 'approve' or 'reject'")

    state = _state_from_snapshot(snapshot)
    if not state.interrupted or state.pending_decision is None:
        raise ValueError("snapshot is not interrupted / has no pending_decision")

    if expected_digest is not None and state.pending_decision.args_digest != expected_digest:
        state.final_answer = "Resume failed closed: expected_digest mismatch."
        state.done = True
        state.interrupted = False
        return state.snapshot()

    if decision in {"approve", "approved"}:
        state.pending_decision.status = "approved"
        state.interrupted = False  # allow execute_tool_node to proceed
    else:
        state.pending_decision.status = "rejected"
        state.interrupted = False

    # Continue from tools node (pending call still at front of pending_tools)
    graph = build_agent_graph().compile()
    # Jump by invoking from a tiny wrapper: set entry via compiling full graph
    # but seed state so plan is skipped — use tools as synthetic entry by
    # calling execute path through a one-shot graph starting at tools.
    from .graph import StateGraph as _SG

    g = _SG(AgentState)
    g.add_node("tools", execute_tool_node)
    g.add_node("respond", respond_node)
    g.set_entry_point("tools")
    g.add_conditional_edges(
        "tools",
        route_after_tool,
        {"tools": "tools", "respond": "respond", "end": END},
    )
    g.add_edge("respond", END)
    final = g.compile().invoke(state)
    return final.snapshot()
