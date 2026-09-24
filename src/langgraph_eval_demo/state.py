"""Agent state passed between graph nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    name: str
    output: str
    args: dict[str, Any] = field(default_factory=dict)
    args_digest: str = ""


@dataclass
class AgentState:
    """Mutable state flowing through the StateGraph."""

    query: str
    messages: list[str] = field(default_factory=list)
    pending_tools: list[ToolCall] = field(default_factory=list)
    tool_trace: list[str] = field(default_factory=list)  # ordered tool names used
    tool_results: list[ToolResult] = field(default_factory=list)
    tool_arg_digests: list[str] = field(default_factory=list)
    final_answer: str | None = None
    done: bool = False
    derived_from_search: bool = False

    def snapshot(self, *, graph_mode: str = "langgraph-style") -> dict[str, Any]:
        label = "[langgraph]" if graph_mode == "langgraph" else "[langgraph-style]"
        return {
            "query": self.query,
            "tool_trace": list(self.tool_trace),
            "tool_arg_digests": list(self.tool_arg_digests),
            "final_answer": self.final_answer,
            "tool_results": [
                {
                    "name": r.name,
                    "output": r.output,
                    "args": dict(r.args),
                    "args_digest": r.args_digest,
                }
                for r in self.tool_results
            ],
            "graph_mode": graph_mode,
            "backend_label": label,
            "messages": list(self.messages),
        }
