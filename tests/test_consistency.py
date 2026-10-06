"""pass^k consistency report + seeded flaky-tool mode."""

from __future__ import annotations

import pytest

from evals.consistency import (
    TaskConsistency,
    consistency_report,
    pass_at_k,
    pass_hat_k,
    run_consistency,
)
from evals.runner import GoldenTask, load_golden_tasks, main
from langgraph_eval_demo.agent import run_agent
from langgraph_eval_demo.flaky import (
    FlakyToolSpec,
    active_injector,
    flaky_tools,
    load_flaky_config,
    parse_flaky_config,
)
from langgraph_eval_demo.tools import call_tool

FLAKY = {"weather": {"fail_every": 3, "seed": 7}, "search": {"fail_rate": 0.15, "seed": 7}}


# ---- metric helpers -------------------------------------------------------


def test_pass_at_k_and_pass_hat_k_edges():
    assert pass_hat_k(5, 5, 5) == 1.0
    assert pass_hat_k(5, 4, 5) == 0.0  # one failure in 5 → not reliable at k=5
    assert pass_at_k(5, 1, 5) == 1.0  # one success → capable at k=5
    assert pass_at_k(5, 0, 5) == 0.0
    # k=1 reduces both to the mean success rate
    assert pass_at_k(4, 3, 1) == pytest.approx(0.75)
    assert pass_hat_k(4, 3, 1) == pytest.approx(0.75)


def test_unbiased_estimators_n_greater_than_k():
    # n=4 runs, c=3 passes, k=2: pass^2 = C(3,2)/C(4,2) = 3/6
    assert pass_hat_k(4, 3, 2) == pytest.approx(0.5)
    # pass@2 = 1 - C(1,2)/C(4,2) = 1
    assert pass_at_k(4, 3, 2) == 1.0
    assert pass_at_k(4, 1, 2) == pytest.approx(1 - 3 / 6)


@pytest.mark.parametrize("n,c,k", [(3, 4, 1), (3, 1, 0), (3, 1, 4), (0, 0, 1)])
def test_estimators_reject_bad_args(n, c, k):
    with pytest.raises(ValueError):
        pass_hat_k(n, c, k)


def test_task_consistency_gap():
    t = TaskConsistency("x", (True, False, True, True, True), k=5)
    assert t.pass_at_1 == 1.0
    assert t.mean_at_k == pytest.approx(0.8)
    assert t.pass_at_k == 1.0
    assert t.pass_hat_k == 0.0
    assert t.gap == pytest.approx(0.8)


# ---- flaky-tool mode --------------------------------------------------------


def test_flaky_spec_validation():
    with pytest.raises(ValueError):
        FlakyToolSpec()  # neither
    with pytest.raises(ValueError):
        FlakyToolSpec(fail_every=3, fail_rate=0.1)  # both
    with pytest.raises(ValueError):
        FlakyToolSpec(fail_every=0)
    with pytest.raises(ValueError):
        FlakyToolSpec(fail_rate=1.5)


def test_fail_every_fails_exactly_one_per_window_and_is_seeded():
    spec = FlakyToolSpec(fail_every=3, seed=7)
    decisions = [spec.should_fail("weather", i) for i in range(1, 31)]
    for w in range(10):
        assert sum(decisions[w * 3 : w * 3 + 3]) == 1
    # bit-identical for the same seed; different seed → different schedule
    again = [FlakyToolSpec(fail_every=3, seed=7).should_fail("weather", i) for i in range(1, 31)]
    other = [FlakyToolSpec(fail_every=3, seed=8).should_fail("weather", i) for i in range(1, 31)]
    assert decisions == again
    assert decisions != other


def test_fail_rate_extremes():
    assert not any(FlakyToolSpec(fail_rate=0.0).should_fail("t", i) for i in range(1, 50))
    assert all(FlakyToolSpec(fail_rate=1.0).should_fail("t", i) for i in range(1, 50))


def test_call_tool_injects_error_text_only_inside_context():
    assert active_injector() is None
    with flaky_tools({"weather": {"fail_rate": 1.0, "seed": 1}}) as inj:
        out = call_tool("weather", {"city": "Seattle"})
        assert out.startswith("weather error: transient failure (injected)")
        # other tools untouched
        assert call_tool("calculator", {"expression": "2 + 2"}) == "4"
        assert inj.failures == [("weather", 1)]
    assert active_injector() is None
    assert call_tool("weather", {"city": "Seattle"}) == "Seattle: 58°F, cloudy, light rain."


def test_flaky_failure_reaches_agent_answer():
    with flaky_tools({"weather": {"fail_rate": 1.0}}):
        snap = run_agent("What's the weather in Seattle?")
    assert snap["tool_trace"] == ["weather"]
    assert "weather error" in snap["final_answer"]


def test_empty_config_is_noop():
    with flaky_tools(None) as inj:
        assert inj is None
        assert active_injector() is None


def test_parse_and_load_config(tmp_path):
    specs = parse_flaky_config({"_comment": "x", "weather": {"fail_every": 3, "seed": 7}})
    assert list(specs) == ["weather"]
    assert specs["weather"] == FlakyToolSpec(fail_every=3, seed=7)
    path = tmp_path / "f.json"
    path.write_text('{"flaky_tools": {"search": {"fail_rate": 0.2, "seed": 3}}}')
    assert load_flaky_config(path)["search"].fail_rate == 0.2


def test_shipped_flaky_config_loads():
    from pathlib import Path

    specs = load_flaky_config(Path(__file__).resolve().parents[1] / "evals" / "flaky_tools.json")
    assert set(specs) == {"weather", "search"}


# ---- consistency runs -------------------------------------------------------


def test_deterministic_agent_has_zero_gap():
    rep = run_consistency(3)
    assert rep.pass_at_1 == rep.mean_at_k == rep.pass_hat_k == 1.0
    assert rep.gap == 0.0
    assert rep.injected_failures == 0
    assert len(rep.tasks) == len(load_golden_tasks())


def test_flaky_mode_shows_consistency_gap_and_is_reproducible():
    a = run_consistency(5, flaky=FLAKY)
    b = run_consistency(5, flaky=FLAKY)
    assert [t.outcomes for t in a.tasks] == [t.outcomes for t in b.tasks]
    assert a.injected_failures > 0
    assert a.pass_hat_k < a.mean_at_k <= 1.0
    assert a.gap > 0
    # pass@k >= mean@k >= pass^k always
    assert a.pass_at_k >= a.mean_at_k >= a.pass_hat_k
    # tasks that never touch a flaky tool stay perfectly consistent
    by_id = {t.task_id: t for t in a.tasks}
    assert by_id["calc_simple"].pass_hat_k == 1.0


def test_run_major_interleaving_spreads_failures():
    """fail_every must not lock onto a single task across runs."""
    rep = run_consistency(6, flaky={"weather": {"fail_every": 3, "seed": 7}})
    weather_tasks = [t for t in rep.tasks if "weather" in t.task_id]
    failed_somewhere = [t for t in weather_tasks if t.c < t.n]
    assert len(failed_somewhere) >= 2


def test_custom_agent_and_k_smaller_than_n():
    calls = {"n": 0}
    task = GoldenTask(id="t", query="q", expected_tools=[], expected_answer="ok")

    def alternating(query: str):
        calls["n"] += 1
        return {"tool_trace": [], "final_answer": "ok" if calls["n"] % 2 else "nope"}

    rep = run_consistency(4, k=2, tasks=[task], agent_fn=alternating)
    t = rep.tasks[0]
    assert t.outcomes == (True, False, True, False)
    assert t.pass_hat_k == pytest.approx(1 / 6)  # C(2,2)/C(4,2)
    with pytest.raises(ValueError):
        run_consistency(2, k=3, tasks=[task], agent_fn=alternating)
    with pytest.raises(ValueError):
        run_consistency(0, tasks=[task], agent_fn=alternating)


def test_markdown_report_mentions_metrics():
    md = consistency_report(run_consistency(2, flaky=FLAKY))
    for needle in ("pass^k", "mean@k", "pass@1", "gap", "flaky tools: weather"):
        assert needle in md


def test_cli_floor_gates_on_pass_hat_k(capsys, tmp_path):
    assert main(["--repeat", "3", "--min-pass-hat-k", "1.0"]) == 0
    cfg = tmp_path / "flaky.json"
    cfg.write_text('{"flaky_tools": {"weather": {"fail_every": 3, "seed": 7}}}')
    assert main(["--repeat", "5", "--flaky", str(cfg), "--min-pass-hat-k", "0.99"]) == 1
    assert "below floor" in capsys.readouterr().out


def test_cli_default_path_unchanged(capsys):
    assert main([]) == 0
    assert "# Golden eval report" in capsys.readouterr().out
