"""StateGraph unit tests."""

from langgraph_eval_demo.graph import END, StateGraph


def test_linear_graph():
    g = StateGraph()
    g.add_node("a", lambda s: s + ["a"])
    g.add_node("b", lambda s: s + ["b"])
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    assert g.compile().invoke([]) == ["a", "b"]


def test_conditional_loop():
    def tick(s):
        s["n"] += 1
        return s

    def router(s):
        return "again" if s["n"] < 3 else "done"

    g = StateGraph()
    g.add_node("tick", tick)
    g.add_node("finish", lambda s: {**s, "ok": True})
    g.set_entry_point("tick")
    g.add_conditional_edges("tick", router, {"again": "tick", "done": "finish"})
    g.add_edge("finish", END)
    out = g.compile().invoke({"n": 0})
    assert out["n"] == 3
    assert out["ok"] is True
