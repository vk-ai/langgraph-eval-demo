"""Seeded flaky-tool mode (deterministic fault injection for consistency evals).

The planner and mock tools are fully deterministic, so repeating a golden task
k times always gives the same answer. That makes ``pass^k`` trivially equal to
``pass@1`` — which is honest, but teaches nothing. This module lets a learner
inject *seeded, reproducible* transient tool failures so repeated runs show the
variance real agents have (a single 429 / timeout on run 3 of 5).

Config shape (also accepted from JSON, see ``evals/flaky_tools.json``)::

    {"weather": {"fail_every": 3, "seed": 7}}

- ``fail_every: N`` — exactly one of every N consecutive calls to that tool
  fails. Calls are counted across the whole repeated session (not reset per
  run), and the seed picks which slot in each window fails (re-drawn per
  window, so the failure does not lock onto one task).
- ``fail_rate: p`` — alternatively, each call fails with probability ``p``
  drawn from ``random.Random(f"{seed}:{tool}:{call_no}")`` (still bit-identical
  for a given seed).
- ``error`` — optional error text (default ``"transient failure (injected)"``).

A failed call returns ``"<tool> error: ..."`` text, the same convention the
mock tools already use for real errors, so the agent / eval see it naturally.

OSS/learning only — not a chaos-engineering framework.
"""

from __future__ import annotations

import contextvars
import json
import random
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

DEFAULT_ERROR = "transient failure (injected)"


@dataclass(frozen=True)
class FlakyToolSpec:
    """Seeded failure schedule for one tool."""

    fail_every: int | None = None
    fail_rate: float | None = None
    seed: int = 0
    error: str = DEFAULT_ERROR

    def __post_init__(self) -> None:
        if (self.fail_every is None) == (self.fail_rate is None):
            raise ValueError("FlakyToolSpec needs exactly one of fail_every / fail_rate")
        if self.fail_every is not None and int(self.fail_every) < 1:
            raise ValueError("fail_every must be >= 1")
        if self.fail_rate is not None and not 0.0 <= float(self.fail_rate) <= 1.0:
            raise ValueError("fail_rate must be in [0, 1]")

    def should_fail(self, tool: str, call_no: int) -> bool:
        """Deterministic decision for the ``call_no``-th (1-based) call to ``tool``."""
        if self.fail_every is not None:
            n = int(self.fail_every)
            # Seed picks which slot (0..n-1) of each n-call window fails. The
            # slot is re-drawn per window so the schedule does not alias with a
            # fixed task order (otherwise the same task would fail every run).
            window, slot = divmod(call_no - 1, n)
            return slot == random.Random(f"{self.seed}:{tool}:{window}").randrange(n)
        rng = random.Random(f"{self.seed}:{tool}:{call_no}")
        return rng.random() < float(self.fail_rate or 0.0)


@dataclass
class FlakyToolInjector:
    """Stateful injector: counts calls per tool and decides failures."""

    specs: dict[str, FlakyToolSpec]
    calls: Counter = field(default_factory=Counter)
    failures: list[tuple[str, int]] = field(default_factory=list)

    def maybe_fail(self, tool: str) -> str | None:
        """Return error text if this call should fail, else ``None``."""
        spec = self.specs.get(tool)
        if spec is None:
            return None
        self.calls[tool] += 1
        call_no = self.calls[tool]
        if spec.should_fail(tool, call_no):
            self.failures.append((tool, call_no))
            return f"{tool} error: {spec.error} (call #{call_no})"
        return None


_ACTIVE: contextvars.ContextVar[FlakyToolInjector | None] = contextvars.ContextVar(
    "langgraph_eval_demo_flaky", default=None
)


def parse_flaky_config(raw: Mapping[str, Any] | None) -> dict[str, FlakyToolSpec]:
    """Turn ``{"weather": {"fail_every": 3, "seed": 7}}`` into specs."""
    specs: dict[str, FlakyToolSpec] = {}
    for tool, cfg in (raw or {}).items():
        if str(tool).startswith("_"):
            continue  # allow "_comment" keys in JSON files
        cfg = dict(cfg or {})
        specs[str(tool)] = FlakyToolSpec(
            fail_every=cfg.get("fail_every"),
            fail_rate=cfg.get("fail_rate"),
            seed=int(cfg.get("seed", 0)),
            error=str(cfg.get("error", DEFAULT_ERROR)),
        )
    return specs


def load_flaky_config(path: str | Path) -> dict[str, FlakyToolSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_flaky_config(data.get("flaky_tools", data))


def active_injector() -> FlakyToolInjector | None:
    return _ACTIVE.get()


@contextmanager
def flaky_tools(
    config: Mapping[str, Any] | dict[str, FlakyToolSpec] | None,
) -> Iterator[FlakyToolInjector | None]:
    """Activate seeded tool failures for the duration of the block.

    ``config`` may be raw JSON-style dicts or parsed :class:`FlakyToolSpec`s.
    ``None`` / empty is a no-op (yields ``None``).
    """
    if not config:
        yield None
        return
    specs = {
        k: v if isinstance(v, FlakyToolSpec) else parse_flaky_config({k: v})[k]
        for k, v in config.items()
        if not str(k).startswith("_")
    }
    injector = FlakyToolInjector(specs=specs)
    token = _ACTIVE.set(injector)
    try:
        yield injector
    finally:
        _ACTIVE.reset(token)
