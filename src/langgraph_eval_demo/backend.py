"""Backend selection: default LangGraph-*style* stdlib graph vs optional real LangGraph.

Default path stays offline with zero third-party deps. Real LangGraph only runs when
``LANGGRAPH_EVAL_USE_REAL`` is truthy *and* the optional ``langgraph`` package is
importable. Missing install or invoke failures fall back to the style graph.
"""

from __future__ import annotations

import os
from typing import Literal

GraphMode = Literal["langgraph-style", "langgraph"]

_TRUTHY = {"1", "true", "yes", "on"}


def wants_real_langgraph() -> bool:
    """Return True when ``LANGGRAPH_EVAL_USE_REAL`` opts into the real package."""
    raw = os.environ.get("LANGGRAPH_EVAL_USE_REAL", "")
    return raw.strip().lower() in _TRUTHY


def langgraph_importable() -> bool:
    """True if real LangGraph graph APIs are importable (not an empty namespace)."""
    try:
        from langgraph.graph import END, START, StateGraph  # noqa: F401
    except ImportError:
        return False
    return True


def graph_mode(*, force_style: bool = False) -> GraphMode:
    """Return ``langgraph`` only when opted in and importable; else ``langgraph-style``."""
    if force_style:
        return "langgraph-style"
    if wants_real_langgraph() and langgraph_importable():
        return "langgraph"
    return "langgraph-style"


def backend_label(mode: GraphMode | None = None) -> str:
    """Bracket tag for snapshots / logs: ``[langgraph-style]`` or ``[langgraph]``."""
    m = mode if mode is not None else graph_mode()
    if m == "langgraph":
        return "[langgraph]"
    return "[langgraph-style]"


def fallback_reason() -> str | None:
    """Human-readable reason when real path was requested but cannot run."""
    if not wants_real_langgraph():
        return None
    if langgraph_importable():
        return None
    return (
        "LANGGRAPH_EVAL_USE_REAL=true but langgraph is not installed "
        "(pip install '.[langgraph]'). Falling back to [langgraph-style]."
    )
