"""Unit tests for tool-result fixtures / arg digests."""

from __future__ import annotations

from langgraph_eval_demo.fixtures import load_catalog, lookup_fixture, tool_arg_digest


def test_catalog_nonempty():
    catalog = load_catalog()
    tools = catalog.get("tools") or {}
    assert "calculator" in tools
    assert "search" in tools
    assert "weather" in tools
    assert len(tools["calculator"]) >= 1


def test_lookup_known_calculator():
    args = {"expression": "15 * 7"}
    out = lookup_fixture("calculator", args)
    assert out == "105"
    assert tool_arg_digest(args) == (
        "468ccb08daabbea0923e14a2cdbb6d6efc27976b748a8923ce844d458a2e97f6"
    )
