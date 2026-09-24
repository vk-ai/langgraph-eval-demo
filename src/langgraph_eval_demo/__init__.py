"""langgraph-eval-demo: tiny multi-step tool-calling agent + golden eval harness.

OSS / learning demo only — not employer production software.
Default graph is a stdlib LangGraph-*style* stand-in ([langgraph-style]);
optional real LangGraph behind LANGGRAPH_EVAL_USE_REAL + pip install '.[langgraph]'.
"""

from .agent import run_agent
from .backend import backend_label, graph_mode
from .graph import StateGraph, END
from .state import AgentState
from .tools import TOOLS

__version__ = "0.1.0"
__all__ = [
    "run_agent",
    "backend_label",
    "graph_mode",
    "StateGraph",
    "END",
    "AgentState",
    "TOOLS",
    "__version__",
]
