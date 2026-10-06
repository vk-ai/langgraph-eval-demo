"""pass^k consistency report over k repeated runs of the golden tasks.

"Your agent aced the task. Will it do it again?" — pass@1 says the agent *can*
solve a task; pass^k says it solves it *every time* over k tries. For each
task we run the golden eval ``k`` times and report:

- ``pass@1``   — first run passed (what a single CI run shows)
- ``mean@k``   — fraction of the k runs that passed (average success)
- ``pass@k``   — at least one of the k runs passed (capability)
- ``pass^k``   — all k runs passed (reliability)
- ``gap``      — ``mean@k − pass^k`` (the consistency gap)

Aggregates are means over tasks. The default agent is deterministic, so every
metric is 1.0; enable the seeded flaky-tool mode (``--flaky``) to see a gap.

``pass_at_k`` / ``pass_hat_k`` also implement the unbiased estimators from
``n >= k`` trials with ``c`` successes (Chen et al. 2021 for pass@k; the
tau-bench style ``C(c, k) / C(n, k)`` for pass^k), so ``--repeat 8 --k 4``
works too.

OSS/learning only — stdlib re-implementation of the metric idea, not a library.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Callable, Mapping

from evals.runner import GoldenTask, evaluate_task, load_golden_tasks
from langgraph_eval_demo.flaky import FlakyToolSpec, flaky_tools


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased P(at least one of k sampled runs passes) from n runs, c passes."""
    _check_nck(n, c, k)
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased P(all k sampled runs pass) from n runs, c passes (pass^k)."""
    _check_nck(n, c, k)
    return comb(c, k) / comb(n, k)


def _check_nck(n: int, c: int, k: int) -> None:
    if not (0 <= c <= n) or not (1 <= k <= n):
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n (got n={n} c={c} k={k})")


@dataclass(frozen=True)
class TaskConsistency:
    task_id: str
    outcomes: tuple[bool, ...]
    k: int

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def c(self) -> int:
        return sum(self.outcomes)

    @property
    def pass_at_1(self) -> float:
        return 1.0 if self.outcomes and self.outcomes[0] else 0.0

    @property
    def mean_at_k(self) -> float:
        return self.c / self.n if self.n else 0.0

    @property
    def pass_at_k(self) -> float:
        return pass_at_k(self.n, self.c, self.k)

    @property
    def pass_hat_k(self) -> float:
        return pass_hat_k(self.n, self.c, self.k)

    @property
    def gap(self) -> float:
        return self.mean_at_k - self.pass_hat_k


@dataclass(frozen=True)
class ConsistencyReport:
    tasks: list[TaskConsistency]
    n: int
    k: int
    flaky: Mapping[str, FlakyToolSpec] | None = None
    injected_failures: int = 0

    def _mean(self, attr: str) -> float:
        if not self.tasks:
            return 0.0
        return sum(getattr(t, attr) for t in self.tasks) / len(self.tasks)

    @property
    def pass_at_1(self) -> float:
        return self._mean("pass_at_1")

    @property
    def mean_at_k(self) -> float:
        return self._mean("mean_at_k")

    @property
    def pass_at_k(self) -> float:
        return self._mean("pass_at_k")

    @property
    def pass_hat_k(self) -> float:
        return self._mean("pass_hat_k")

    @property
    def gap(self) -> float:
        return self.mean_at_k - self.pass_hat_k

    def summary(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "k": self.k,
            "tasks": len(self.tasks),
            "pass@1": round(self.pass_at_1, 4),
            "mean@k": round(self.mean_at_k, 4),
            "pass@k": round(self.pass_at_k, 4),
            "pass^k": round(self.pass_hat_k, 4),
            "gap": round(self.gap, 4),
            "injected_failures": self.injected_failures,
        }


def run_consistency(
    n: int,
    *,
    k: int | None = None,
    tasks: list[GoldenTask] | None = None,
    flaky: Mapping[str, Any] | None = None,
    agent_fn: Callable[[str], dict[str, Any]] | None = None,
) -> ConsistencyReport:
    """Run every golden task ``n`` times and compute pass^k (``k`` defaults to n).

    Runs are interleaved run-major (all tasks for run 1, then run 2, ...) inside
    one flaky session, so a seeded ``fail_every`` schedule hits different tasks
    on different runs — like a real flaky upstream.
    """
    if n < 1:
        raise ValueError("n (repeat) must be >= 1")
    k = n if k is None else k
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, n] (got k={k}, n={n})")
    tasks = tasks or load_golden_tasks()
    outcomes: dict[str, list[bool]] = {t.id: [] for t in tasks}
    with flaky_tools(flaky) as injector:
        for _ in range(n):
            for t in tasks:
                outcomes[t.id].append(evaluate_task(t, agent_fn=agent_fn).passed)
        specs = dict(injector.specs) if injector else None
        injected = len(injector.failures) if injector else 0
    return ConsistencyReport(
        tasks=[TaskConsistency(t.id, tuple(outcomes[t.id]), k) for t in tasks],
        n=n,
        k=k,
        flaky=specs,
        injected_failures=injected,
    )


def consistency_report(rep: ConsistencyReport) -> str:
    """Markdown table for the consistency run."""
    s = rep.summary()
    lines = ["# Consistency report (pass^k)", ""]
    lines.append(f"- runs per task (n): {rep.n}; k: {rep.k}; tasks: {len(rep.tasks)}")
    if rep.flaky:
        desc = ", ".join(
            f"{tool}(fail_every={sp.fail_every}, seed={sp.seed})"
            if sp.fail_every is not None
            else f"{tool}(fail_rate={sp.fail_rate}, seed={sp.seed})"
            for tool, sp in rep.flaky.items()
        )
        lines.append(f"- flaky tools: {desc}; injected failures: {rep.injected_failures}")
    else:
        lines.append("- flaky tools: off (deterministic agent → expect gap 0.00)")
    lines.append("")
    lines.append(
        f"**pass@1** {s['pass@1']:.2f} · **mean@k** {s['mean@k']:.2f} · "
        f"**pass@k** {s['pass@k']:.2f} · **pass^k** {s['pass^k']:.2f} · "
        f"**gap** {s['gap']:.2f}"
    )
    lines.append("")
    lines.append("| task | runs | pass@1 | mean@k | pass@k | pass^k | gap |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in rep.tasks:
        runs = "".join("✓" if o else "✗" for o in t.outcomes)
        lines.append(
            f"| {t.task_id} | {runs} | {t.pass_at_1:.2f} | {t.mean_at_k:.2f} | "
            f"{t.pass_at_k:.2f} | {t.pass_hat_k:.2f} | {t.gap:.2f} |"
        )
    return "\n".join(lines)
