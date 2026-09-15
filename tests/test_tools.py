"""Unit tests for mock tools."""

from langgraph_eval_demo.tools import calculator, search, weather, call_tool


def test_calculator_basic():
    assert calculator("15 * 7") == "105"
    assert calculator("2 + 2") == "4"
    assert calculator("10 / 4") == "2.5"


def test_search_capital():
    assert "Paris" in search("capital of France")


def test_weather_seattle():
    assert "Seattle" in weather("Seattle")


def test_call_tool_dispatch():
    assert call_tool("calculator", {"expression": "3 * 3"}) == "9"
    assert "unknown tool" in call_tool("nope", {})
