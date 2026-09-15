"""Mock tools for offline demos (no network)."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Callable


# Tiny knowledge base for the mock search tool
_SEARCH_KB: dict[str, str] = {
    "capital of france": "Paris is the capital of France.",
    "capital of japan": "Tokyo is the capital of Japan.",
    "population of tokyo": "Tokyo has approximately 14 million people.",
    "population of paris": "Paris has approximately 2.1 million people.",
    "python programming": "Python is a high-level programming language created by Guido van Rossum.",
    "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
}

_WEATHER: dict[str, str] = {
    "seattle": "Seattle: 58°F, cloudy, light rain.",
    "paris": "Paris: 64°F, partly cloudy.",
    "tokyo": "Tokyo: 72°F, clear skies.",
    "new york": "New York: 68°F, humid with afternoon showers.",
}

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_expr(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    expr = expression.strip()
    # Normalize common words / symbols
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_expr(tree.body)
        if result == int(result):
            return str(int(result))
        return str(round(result, 6))
    except Exception as exc:  # noqa: BLE001 — mock tool surfaces errors as text
        return f"calculator error: {exc}"


def search(query: str) -> str:
    """Mock keyword search over a tiny local KB."""
    q = query.lower().strip()
    for key, value in _SEARCH_KB.items():
        if key in q or all(tok in q for tok in key.split()):
            return value
    # Fuzzy: any overlapping token hit
    tokens = set(re.findall(r"[a-z0-9]+", q))
    best: tuple[int, str] | None = None
    for key, value in _SEARCH_KB.items():
        overlap = len(tokens & set(key.split()))
        if overlap and (best is None or overlap > best[0]):
            best = (overlap, value)
    if best:
        return best[1]
    return f"No results found for: {query}"


def weather(city: str) -> str:
    """Mock weather lookup."""
    key = city.lower().strip()
    if key in _WEATHER:
        return _WEATHER[key]
    for name, report in _WEATHER.items():
        if name in key or key in name:
            return report
    return f"Weather unavailable for: {city}"


TOOLS: dict[str, Callable[..., str]] = {
    "search": search,
    "calculator": calculator,
    "weather": weather,
}


def call_tool(name: str, args: dict[str, Any]) -> str:
    if name not in TOOLS:
        return f"unknown tool: {name}"
    fn = TOOLS[name]
    return fn(**args)
