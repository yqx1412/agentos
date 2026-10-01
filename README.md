# agentos

Local-first agent runtime: planner/executor, MCP tools, memory and benchmarks.

**Question this project answers:** How reliably can a local open-weight model plan and carry out multi-step tasks, and which parts of the system make the difference?

Part of the local AI agent ecosystem; see `../ROADMAP.md`.

## Usage

Requires [Ollama](https://ollama.com) running locally with a tool-capable model pulled.

```powershell
uv run agentos run "Read notes.txt, count the words, and write the count to result.txt." `
  --model qwen3:8b --workspace .\demo
```

The final answer goes to stdout; a summary line (stop reason, steps, tool calls, tool errors,
tokens) goes to stderr, and every run writes a JSONL trace to `runs/`.

## MCP servers (A2)

`agentos.toml` lists the MCP servers to connect to over stdio. Their tools are added next to
the built-ins as `<server>__<tool>`; adding a server is a config edit only.

```toml
[mcp_servers.textkit]
command = "{python}"                       # {python} = AgentOS's interpreter
args = ["-m", "agentos.mcp_servers.textkit"]
tools = ["text_stats", "word_frequency"]   # optional allowlist
timeout = 60                               # seconds per call
```

`{workspace}` expands to the `--workspace` directory. The shipped config enables the reference
filesystem server (needs Node.js) and `textkit`, AgentOS's own example server.

```powershell
uv run agentos tools --workspace .\demo      # every tool the agent will see, and its source
uv run agentos run "..." --no-mcp            # built-in tools only
```

Startup is all-or-nothing: a server that fails to start or names an unknown tool in its
allowlist stops the run with an error naming it, so a benchmark never silently runs with
fewer tools. Once running, every MCP failure (server-side validation error, tool error,
timeout, dead server process) is returned to the model as an `ERROR:` tool result. Server
stderr goes to `runs/<trace>.mcp.log`.

## Benchmark (A3)

38 tasks in `benchmarks/tasks/*.yaml`, each run in a fresh temporary workspace and scored by
automatic checks on what the run left behind (files, final answer, tools called). A task
passes only if the agent reached a final answer and every check passes.

```powershell
uv run agentos bench --models qwen3:8b,qwen3:14b,llama3.1:8b
uv run agentos bench --only "mcp,fo-*" --repeats 3     # subset, repeated
uv run agentos bench --list                            # show tasks
```

Results go to `runs/bench/<timestamp>/`: `results.jsonl` (one record per run),
`summary.md`, `meta.json` (git SHA, settings) and a JSONL trace per task. Models are warmed
up before and unloaded after their tasks, so load time and VRAM swapping don't skew timings.

### Baseline: plain loop (A1 agent, no planner, no verification)

Settings: temperature 0, thinking off, `num_ctx` 8192, `num_predict` 2048, RTX 5060 Ti 16 GB.
Full report with every failure: `benchmarks/results/a3-baseline.md`.

| Model | Passed | Success | Avg steps | Tool errors | Avg tokens | Avg time |
|---|---|---|---|---|---|---|
| qwen3:8b | 31/38 | 82% | 3.8 | 35 | 2,656 | 1.9 s |
| qwen3:14b | 33/38 | 87% | 3.5 | 18 | 2,212 | 2.9 s |
| llama3.1:8b | 8/38 | 21% | 2.6 | 27 | 1,333 | 2.9 s |

| Category | Tasks | qwen3:8b | qwen3:14b | llama3.1:8b |
|---|---|---|---|---|
| file_ops | 11 | 10 | 10 | 5 |
| math | 9 | 7 | 8 | 2 |
| chaining | 9 | 8 | 8 | 0 |
| mcp | 6 | 3 | 4 | 0 |
| recovery | 2 | 2 | 2 | 0 |
| safety | 1 | 1 | 1 | 1 |

What the failures show (these are the targets for A4/A5):

- **llama3.1:8b treats tools as an expression language.** Most of its failures write the
  text of a nested call to the output file, e.g. `calculator(expression="read_file(...)")`,
  instead of calling the tools in sequence. 0/9 on chaining.
- **Repeating a failing call.** On `fo-overwrite` both qwen models sent the same invalid
  calculator expression up to 11 times in a row, reading the error each time; qwen3:14b ran
  out of steps. Nothing in the plain loop notices the repetition.
- **Ignoring a tool result.** qwen3:14b's calculator returned `69.0`; it answered `72.0`.
- **Misreading a tool's contract.** textkit tools take `text`; both qwen models passed a
  filename instead (`find_lines(text="code.py")`), got "no matches" back and reported that
  as the answer. Tool design matters as much as the model.
- 14b over 8b: +2 tasks for 1.5x the time per task.

Runs are at temperature 0 but not bit-identical across runs; use `--repeats` before reading
much into a one-task difference.

## Planner + task graph (A4)

`--agent planner` swaps the plain loop for `PlanningAgent`:

1. The planner asks the model for a JSON plan of at most 8 steps with `depends_on` edges,
   validates it (unknown dependencies, cycles) and orders it topologically. An invalid
   plan is sent back once with the error.
2. A 1-step plan runs the plain loop on the original task ("direct"), so simple tasks pay
   only for the planner call. No valid plan at all falls back to the plain loop.
3. Otherwise each step runs the plain loop in a fresh, short conversation that sees the
   task, the finished steps' results and their raw tool outputs.
4. A failed step (it replies `FAILED:` or runs out of turns) triggers a revised plan for
   the rest of the task, at most 2 times. Finally one call writes the answer from the step
   results.

```powershell
uv run agentos run "<task>" --agent planner --show-plan    # plan + per-step results on stderr
uv run agentos bench --models qwen3:8b --agents plain,planner
```

Demo with a 6-step plan (`examples/planner-demo/`, totals 3530 / 4530 / 3250):

```text
$ agentos run "regions.txt lists one CSV file per sales region. ... write totals.json ...
  then report.txt with best: <region> and grand_total: <sum>" --model qwen3:14b --agent planner --show-plan
plan (planned, 0 replans):
  [done] step 1: Read regions.txt and list all CSV files -> north.csv, south.csv, west.csv
  [done] step 2 after [1]: For each CSV file, compute the total sales -> 3530, 4530, 3250
  [done] step 3 after [2]: Write totals.json mapping each region to its total
  [done] step 4 after [3]: Compute the grand total -> 11310
  [done] step 5 after [3]: Identify the region with the highest total -> south
  [done] step 6 after [4, 5]: Write report.txt with best and grand_total
```

qwen3:14b got both files right with the planner. With the plain loop, both qwen models
wrote literal `\n` into `report.txt` (and qwen3:8b wrote escaped, invalid `totals.json`).
qwen3:8b with the planner made up the CSV totals in step 2 without reading the files,
which is what A5's verification is for.

Results (38 tasks x 3 repeats; full write-up in `benchmarks/results/a4-planner.md`):

| Model | Plain | Planner | Avg tokens | Avg time |
|---|---|---|---|---|
| qwen3:8b | 95/114 (83%) | 101/114 (89%) | 2,807 -> 4,919 | 2.0 s -> 4.0 s |
| qwen3:14b | 99/114 (87%) | 108/114 (95%) | 2,220 -> 4,488 | 3.0 s -> 6.9 s |
| llama3.1:8b | 24/114 (21%) | 29/114 (25%) | 1,308 -> 3,758 | 2.5 s -> 6.2 s |

- The planner wins more tasks than it loses on every model (13 flips its way vs 5 against,
  pooled sign test p ~ 0.1). At temperature 0 the repeats are nearly identical, so the
  sample that matters is 38 tasks: consistent, not yet conclusive. The ~2x cost in tokens
  and time is certain.
- It fixes the "repeat the same failing call" failure: `fo-overwrite` goes 0/3 -> 3/3 for
  both qwen models. qwen3:14b's tool errors drop from 18 to 3 per 38 tasks.
- The first version made things worse (qwen3:8b 26/38): it over-planned trivial tasks and
  passed only summaries between steps, losing file contents. The write-up has the details.
- It does not help llama3.1:8b, whose problem is emitting tool calls as text.
- So the plain loop stays the default; the planner is opt-in, and the "direct" path keeps
  its overhead to one call on simple tasks.

## Verification + retries (A5): the ablation

Two more agent kinds build on the planner. `planner-verify` checks every finished step
before trusting it and treats a rejected step as failed, which triggers a replan.
`planner-verify-retry` first re-runs a rejected step up to 2 times, with the rejection
reason in the prompt. The checks (`verifier.py`) run cheapest first:

1. `text_tool_call`: the step wrote a tool call as JSON text and made no real call.
2. `filename_as_content`: a bare file name was passed as `text`/`data`/`body`.
3. `ungrounded`: a tool argument used a number (>= 100) that is in no tool output, in the
   task or in an earlier step, i.e. a made-up input.
4. `reflection`: one model call judges whether the goal was achieved and the result
   follows from the tool outputs. An unparseable verdict counts as a pass.

```powershell
uv run agentos bench --models qwen3:8b --agents plain,planner,planner-verify,planner-verify-retry
uv run agentos run "<task>" --agent planner-verify-retry --show-plan
```

**Headline result of Project 1**: passed runs out of 76 (38 tasks x 2 repeats). Full
write-up: `benchmarks/results/a5-ablation.md`.

| Model | plain | + planner | + verify | + retry | Tokens (plain -> +retry) |
|---|---|---|---|---|---|
| qwen3:8b | 62 (82%) | 63 (83%) | **69 (91%)** | 67 (88%) | 2.6k -> 5.6k |
| qwen3:14b | 64 (84%) | **66 (87%)** | 64 (84%) | 65 (86%) | 2.2k -> 6.3k |
| llama3.1:8b | 15 (20%) | 21 (28%) | 10 (13%) | **22 (29%)** | 1.5k -> 11.0k |

- **What helps depends on the model.** qwen3:8b gains most from verification (+4 tasks,
  none lost), mainly the check that catches `word_frequency(text="essay.txt")`. qwen3:14b
  makes few mistakes the checks can see, so verification only adds cost, and the planner
  alone is its best setup.
- **Verification without retries can hurt.** llama3.1:8b writes tool calls as text. The
  verifier rejects that correctly, but replanning just produces more steps that get
  rejected the same way, so its score drops to 13%. Retries feed the rejection reason back,
  and llama often makes the real call on the next try (+9 tasks), reaching 29%. That's the
  best llama result so far, at 7x the tokens.
- **The first version of the verifier made things worse** (qwen3:14b 36 -> 31). The traces
  showed false rejections: reflection couldn't see earlier steps, and prose mentions
  counted as text calls. They also showed impossible planned steps and replans lost to
  stale dependencies. The write-up lists all five fixes.
- **Known gap:** a file written with malformed contents (escaped JSON) passes, because
  reflection is the same model grading itself. A check that written `.json` files parse is
  the obvious next step.

## Memory (A6)

Long-term memory is one SQLite file (default `~/.agentos/memory.db`, outside any
workspace) with **facts** (stored with the `remember` tool or `agentos memory add`) and
**episodes** (every finished run's task, outcome and answer, recorded automatically).
Lookup is keyword search: FTS5 with Porter stemming, ranked by BM25.

```powershell
uv run agentos run "Remember: the deploy server is build-07, port 8443." --memory auto
uv run agentos run "Write the deploy server as host:port to deploy.txt." --memory auto
uv run agentos memory list          # also: episodes, search QUERY, add TEXT, forget ID
```

`--memory tools` gives the model `remember` / `recall` / `forget`; `--memory auto` also
searches memory with the task text and puts the hits in front of the task. Memory is off by
default. The two commands above are the A6 demo: the second process wrote `build-07:8443`.

Benchmark: 11 multi-session tasks in `benchmarks/memory/` (run with
`--tasks benchmarks/memory --memory none,tools,auto`), 2 repeats:

| Model | none | tools | auto |
|---|---|---|---|
| qwen3:8b | 2/22 | 6/22 | **20/22** |
| qwen3:14b | 2/22 | 4/22 | **20/22** |
| llama3.1:8b | 0/22 | 0/22 | 0/22 |

- **The runtime has to hand memories over.** With tools alone, the qwen models stored facts
  in 24 of 26 setup sessions but rarely called `recall` later; they guessed instead.
- **Keyword search is the limit:** the one `auto` failure stores "manager" and asks about
  "boss". DomainGraph's embedding search is meant to fix that.
- **No measurable cost on the 38 regular tasks** (within one task per model), at 20-30%
  more prompt tokens for the three extra tool schemas.

Short-term memory: once a prompt exceeds `--context-budget` (default 75% of `--num-ctx`),
the plain loop replaces the middle of its conversation with the old tool calls and each
old output clipped to its first and last 300 characters. A model-written summary was tried
first and rejected: qwen3:8b "summarized" file reads it had never been shown. Details in
`benchmarks/results/a6-memory.md`.

## Architecture

| Module | Role |
|---|---|
| `llm.py` | Provider-neutral `Message`/`ToolCall` types, `LLM` protocol, `OllamaLLM` |
| `tools.py` | `Tool` + `ToolRegistry`: Pydantic arg models become JSON schemas; `execute()` never raises |
| `builtin_tools.py` | `read_file`, `write_file` (confined to the workspace), `calculator` (AST-based, no `eval`) |
| `agent.py` | The loop: model -> tool calls -> results fed back -> repeat, capped by `max_steps`; `Tracer` |
| `planner.py` | `Planner`: JSON plan -> validated task graph (`Plan`, `Step`), revision after a failed step |
| `executor.py` | `PlanningAgent`: direct / planned / fallback modes, step prompts, replanning, synthesis, verify + retry |
| `verifier.py` | `Verifier`: text-call, file-name-as-content and ungrounded-number checks, then reflection |
| `memory.py` | `MemoryStore` (SQLite + FTS5 facts and episodes), memory tools, `MemoryAgent`, `compact_messages` |
| `config.py` | `agentos.toml` loading and validation, placeholder expansion |
| `mcp_client.py` | `MCPManager`: stdio connections on a background asyncio loop, MCP tools -> `Tool`s |
| `mcp_servers/textkit.py` | Example MCP server: `text_stats`, `word_frequency`, `find_lines` |
| `bench/` | Benchmark: `tasks.py` (YAML schema), `checks.py`, `runner.py`, `report.py` |
| `cli.py` | `agentos run [--agent KIND] [--memory MODE]`, `agentos tools`, `agentos bench [--agents KIND,...] [--memory MODE,...]`, `agentos memory` |

A malformed tool call (unknown tool, invalid JSON, missing or mistyped arguments) or a tool
exception comes back to the model as an `ERROR: ...` tool message so it can correct itself.

## Development

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```
