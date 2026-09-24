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
| `src/langgraph_eval_demo/agent.py` | Plan → tools\* → respond agent (+ optional HITL interrupt-lite) |
| `src/langgraph_eval_demo/tools.py` | Mock `search`, `calculator`, `weather` (+ optional fixture replay) |
| `src/langgraph_eval_demo/fixtures.py` | Arg digests + frozen tool-result catalog lookup |
| `evals/tool_fixtures/catalog.json` | Record/replay mock tool outputs keyed by arg digest |
| `evals/golden_tasks.json` | Frozen tasks: tool sequence + answer + optional arg digests |
| `evals/runner.py` | Eval runner + markdown report |
| `tests/` | Unit + golden tests (pytest fails on regression) |
| `ci/github-actions.yml` | GitHub Actions workflow mirror (copy to `.github/workflows/ci.yml` to enable) |

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
3. **`max_tool_calls` budget** — fail the eval/CI if the agent exceeds the per-task tool-call budget (optional related `max_graph_steps` is also supported by the runner)
4. **Optional `expected_tool_args` digests** — SHA-256 of canonicalized (sorted-JSON) resolved tool args; fail on mismatch
5. **Frozen tool-result fixtures** — `evals/tool_fixtures/catalog.json` locks mock outputs; CI fails if a tool returns a different string for a known arg digest

This mirrors a common community wish for agent CI: catch runaway tool loops **and** tool-environment drift (record/replay), even when the answer string still looks fine (see EvalView-style `max_cost` / “freeze the tool results as fixtures” stories on r/LangChain and DEV).

Set `LANGGRAPH_EVAL_USE_FIXTURES=1` to replay catalog outputs from `call_tool` (optional; evals verify fixtures regardless).

```bash
pytest tests/test_eval.py -q
```

If the agent starts skipping tools, looping tools, or changing answers, CI goes red.

> **Honesty:** stdlib LangGraph-*style* graph + mock tools only — not real LangGraph, LangSmith, or employer production eval infra.


## HITL interrupt-lite (approve / reject)

Community agents (LangGraph HITL / OpenAI Agents SDK) pause before side-effecting
tools, bind the decision to the exact tool+args, and resume or abort. This demo
teaches that **contract** in stdlib:

1. `run_agent(query, approval_tools={"weather"})` pauses *before* executing a
   flagged tool and returns `interrupted=True` + `pending_decision` (tool, args,
   args digest).
2. `resume_agent(snapshot, "approve" | "reject")` continues or aborts.
3. Resume is **fail-closed** on digest mismatch (`expected_digest=`).

```bash
python examples/interrupt_demo.py --query "What's the weather in Seattle?"
python examples/interrupt_demo.py --query "What's the weather in Seattle?" --approve
python examples/interrupt_demo.py --query "What's the weather in Paris?" --reject
pytest tests/test_interrupt.py -q
```

```text
plan → tools* ──(flagged tool)──► pending_decision → END (paused)
                     │
                     ├── approve → execute → respond → END
                     └── reject  → abort message → END
```

> **Honesty:** Stdlib pause/resume demo of the HITL *contract* — **not** LangGraph
> `interrupt()`, not ApprovalNode, not durable checkpointers, not employer prod
> approval infra. See [langgraph#8026](https://github.com/langchain-ai/langgraph/issues/8026),
> [LangGraph HITL docs](https://docs.langchain.com/oss/langgraph/human-in-the-loop),
> [OpenAI Agents SDK HITL](https://openai.github.io/openai-agents-python/human_in_the_loop/).

## Design notes

- **Prefer zero deps:** the graph is a ~100-line LangGraph-style stand-in so learners can read every line. Pinning `langgraph` is optional later; this demo deliberately stays offline-first.
- **Deterministic planner:** no LLM in the loop — queries are mapped to tool plans with lightweight heuristics so golden tests are stable.
- **Mock tools only:** search / calculator / weather never hit the network.

## License

MIT — see [LICENSE](LICENSE).

## CI note

The intended GitHub Actions workflow is checked in as [`ci/github-actions.yml`](ci/github-actions.yml) (identical contents).

A fine-grained PAT without the **Workflows** permission cannot create `.github/workflows/ci.yml` on this repo. To enable Actions:

1. Grant the pushing token **Workflows: Read and write** (classic: `workflow` scope), or use the GitHub UI.
2. Copy the file into place and push:

```bash
mkdir -p .github/workflows
cp ci/github-actions.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Add GitHub Actions CI workflow"
git push
```

Until then, run the same checks locally:

```bash
pip install -e ".[dev]"
pytest -q
python evals/runner.py
```