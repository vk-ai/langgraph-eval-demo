# langgraph-eval-demo

> **OSS / learning demo only.** This is a personal open-source teaching project by [vk-ai](https://github.com/vk-ai). It is **not** employer production software, is **not** affiliated with any employer, and must not be described as production agent infrastructure.

Tiny **multi-step tool-calling agent** built on a thin LangGraph-style state graph (pure Python, zero runtime deps), plus a **golden-task eval harness** and **pytest** suite that fails on regression.

## Why

LangGraph’s mental model (nodes → edges → conditional routing → compiled graph) is powerful — but for learning and CI you often want:

- **No API keys / no network** in the critical path
- **Deterministic** tool traces you can golden-test
- A harness that **breaks the build** when tool order or answers drift

This repo is that slice.

## What’s inside

| Piece | Role |
|---|---|
| `src/langgraph_eval_demo/graph.py` | Thin `StateGraph` / `END` / `compile().invoke()` (stdlib only) |
| `src/langgraph_eval_demo/agent.py` | Plan → tools\* → respond agent |
| `src/langgraph_eval_demo/tools.py` | Mock `search`, `calculator`, `weather` |
| `evals/golden_tasks.json` | Frozen tasks: expected tool sequence + answer |
| `evals/runner.py` | Eval runner + markdown report |
| `tests/` | Unit + golden tests (pytest fails on regression) |
| `.github/workflows/ci.yml` | CI on Python 3.11 / 3.12 |

## Quickstart

```bash
git clone https://github.com/vk-ai/langgraph-eval-demo.git
cd langgraph-eval-demo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python examples/quickstart.py
python evals/runner.py
```

Example agent run:

```text
Q: Find the population of Tokyo then multiply that by 2
  tools:  ['search', 'calculator']
  answer: Tokyo has approximately 14 million people. Multiplied result: 28000000.
```

## Graph shape

```text
          ┌────────┐
   start →│  plan  │─── no tools ──────────────┐
          └───┬────┘                           │
              │ has tools                      ▼
              ▼                           ┌─────────┐
          ┌───────┐  more tools           │ respond │ → END
          │ tools │──────────────┐        └─────────┘
          └───┬───┘              │              ▲
              │ done             └──────────────┘
              └─────────────────────────────────┘
```

## Golden eval (regression gate)

Tasks in `evals/golden_tasks.json` assert:

1. **Tool sequence** — exact ordered list (e.g. `["search", "calculator"]`)
2. **Final answer** — `exact` or `contains` match

```bash
pytest tests/test_eval.py -q
```

If the agent starts skipping tools or changing answers, CI goes red.

## Design notes

- **Prefer zero deps:** the graph is a ~100-line LangGraph-style stand-in so learners can read every line. Pinning `langgraph` is optional later; this demo deliberately stays offline-first.
- **Deterministic planner:** no LLM in the loop — queries are mapped to tool plans with lightweight heuristics so golden tests are stable.
- **Mock tools only:** search / calculator / weather never hit the network.

## License

MIT — see [LICENSE](LICENSE).
