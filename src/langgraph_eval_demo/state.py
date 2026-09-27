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
class PendingDecision:
    """HITL interrupt-lite: a side-effecting tool waiting for approve/reject.

    Honesty: stdlib teaching contract — not LangGraph ``interrupt()``, not
    ApprovalNode, not a durable checkpointer.
    """

    tool_name: str
    args: dict[str, Any]
    args_digest: str
    status: str = "pending"  # pending | approved | rejected

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args": dict(self.args),
            "args_digest": self.args_digest,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingDecision:
        return cls(
            tool_name=str(data["tool_name"]),
            args=dict(data.get("args") or {}),
            args_digest=str(data.get("args_digest") or ""),
            status=str(data.get("status") or "pending"),
        )


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
    # HITL interrupt-lite
    approval_tools: frozenset[str] = field(default_factory=frozenset)
    pending_decision: PendingDecision | None = None
    interrupted: bool = False

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
            "interrupted": self.interrupted,
            "pending_decision": (
                self.pending_decision.to_dict() if self.pending_decision else None
            ),
            "pending_tools": [
                {"name": t.name, "args": dict(t.args)} for t in self.pending_tools
            ],
            "approval_tools": sorted(self.approval_tools),
            "done": self.done,
            "derived_from_search": self.derived_from_search,
            "graph_mode": graph_mode,
            "backend_label": label,
            "messages": list(self.messages),
        }
