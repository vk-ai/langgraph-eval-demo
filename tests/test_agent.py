"""Agent behavior tests (multi-step tool calling)."""

from langgraph_eval_demo.agent import run_agent


def test_calculator_path():
    result = run_agent("What is 15 * 7?")
    assert result["tool_trace"] == ["calculator"]
    assert result["final_answer"] == "105"


def test_weather_path():
    result = run_agent("What's the weather in Seattle?")
    assert result["tool_trace"] == ["weather"]
    assert "Seattle" in result["final_answer"]


def test_multi_step_search_and_calc():
    result = run_agent("Search for capital of Japan and calculate 2 + 2")
    assert result["tool_trace"] == ["search", "calculator"]
    assert "Tokyo" in result["final_answer"]
    assert "4" in result["final_answer"]


def test_population_then_multiply():
    result = run_agent("Find the population of Tokyo then multiply that by 2")
    assert result["tool_trace"] == ["search", "calculator"]
    assert "28000000" in result["final_answer"]
