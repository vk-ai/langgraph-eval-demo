"""Multi-step tool-calling agent built on the thin StateGraph.

Uses a deterministic planner (no LLM / no network) so CI runs fully offline.
The planner emits an ordered plan of tool calls from the user query, then the
graph walks: plan → (execute_tool)* → respond.
"""

from __future__ import annotations

import re
from typing import Any

from .graph import END, StateGraph
from .state import AgentState, ToolCall, ToolResult
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


def execute_tool_node(state: AgentState) -> AgentState:
    """Pop and run the next pending tool call."""
    if not state.pending_tools:
        return state
    call = state.pending_tools.pop(0)
    args = dict(call.args)

    # Resolve placeholder that depends on prior search output
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

    output = call_tool(call.name, args)
    state.tool_trace.append(call.name)
    state.tool_results.append(ToolResult(name=call.name, output=output))
    state.messages.append(f"tool:{call.name}({args}) -> {output}")
    return state


def route_after_tool(state: AgentState) -> str:
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
        {"tools": "tools", "respond": "respond"},
    )
    g.add_edge("respond", END)
    return g


def run_agent(query: str) -> dict[str, Any]:
    """Run the agent graph and return a serializable result snapshot."""
    graph = build_agent_graph().compile()
    state = AgentState(query=query)
    final = graph.invoke(state)
    return final.snapshot()
