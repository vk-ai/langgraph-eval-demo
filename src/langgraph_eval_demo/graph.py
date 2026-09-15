"""Thin LangGraph-style state graph in pure Python (zero third-party deps).

Mirrors the core mental model: nodes, edges, conditional edges, compile → invoke.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable

# Sentinel for terminal edge targets
END = "__end__"

NodeFn = Callable[[Any], Any]
RouterFn = Callable[[Any], Hashable]


class CompiledGraph:
    def __init__(
        self,
        nodes: dict[str, NodeFn],
        edges: dict[str, str],
        conditional: dict[str, tuple[RouterFn, dict[Hashable, str]]],
        entry: str,
        max_steps: int = 32,
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional
        self._entry = entry
        self._max_steps = max_steps

    def invoke(self, state: Any) -> Any:
        current = self._entry
        steps = 0
        while current != END:
            if steps >= self._max_steps:
                raise RuntimeError(f"Graph exceeded max_steps={self._max_steps}")
            if current not in self._nodes:
                raise KeyError(f"Unknown node: {current}")
            state = self._nodes[current](state)
            steps += 1
            if current in self._conditional:
                router, mapping = self._conditional[current]
                key = router(state)
                if key not in mapping:
                    raise KeyError(f"Router returned {key!r} not in {list(mapping)}")
                current = mapping[key]
            elif current in self._edges:
                current = self._edges[current]
            else:
                raise RuntimeError(f"No outgoing edge from node {current!r}")
        return state


class StateGraph:
    """Minimal stand-in for langgraph.graph.StateGraph."""

    def __init__(self, state_schema: type | None = None) -> None:
        self.state_schema = state_schema
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._conditional: dict[str, tuple[RouterFn, dict[Hashable, str]]] = {}
        self._entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> StateGraph:
        if name in self._nodes or name == END:
            raise ValueError(f"Duplicate or reserved node name: {name}")
        self._nodes[name] = fn
        return self

    def add_edge(self, source: str, target: str) -> StateGraph:
        self._edges[source] = target
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: RouterFn,
        path_map: dict[Hashable, str],
    ) -> StateGraph:
        self._conditional[source] = (router, path_map)
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        self._entry = name
        return self

    def compile(self, max_steps: int = 32) -> CompiledGraph:
        if not self._entry:
            raise ValueError("Entry point not set")
        if self._entry not in self._nodes:
            raise ValueError(f"Entry point {self._entry!r} is not a registered node")
        return CompiledGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditional=dict(self._conditional),
            entry=self._entry,
            max_steps=max_steps,
        )
