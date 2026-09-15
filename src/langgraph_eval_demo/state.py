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


@dataclass
class AgentState:
    """Mutable state flowing through the StateGraph."""

    query: str
    messages: list[str] = field(default_factory=list)
    pending_tools: list[ToolCall] = field(default_factory=list)
    tool_trace: list[str] = field(default_factory=list)  # ordered tool names used
    tool_results: list[ToolResult] = field(default_factory=list)
    final_answer: str | None = None
    done: bool = False
    derived_from_search: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "tool_trace": list(self.tool_trace),
            "final_answer": self.final_answer,
            "tool_results": [
                {"name": r.name, "output": r.output} for r in self.tool_results
            ],
        }
