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

## Architecture

| Module | Role |
|---|---|
| `llm.py` | Provider-neutral `Message`/`ToolCall` types, `LLM` protocol, `OllamaLLM` |
| `tools.py` | `Tool` + `ToolRegistry`: Pydantic arg models become JSON schemas; `execute()` never raises |
| `builtin_tools.py` | `read_file`, `write_file` (confined to the workspace), `calculator` (AST-based, no `eval`) |
| `agent.py` | The loop: model -> tool calls -> results fed back -> repeat, capped by `max_steps`; `Tracer` |
| `config.py` | `agentos.toml` loading and validation, placeholder expansion |
| `mcp_client.py` | `MCPManager`: stdio connections on a background asyncio loop, MCP tools -> `Tool`s |
| `mcp_servers/textkit.py` | Example MCP server: `text_stats`, `word_frequency`, `find_lines` |
| `bench/` | Benchmark: `tasks.py` (YAML schema), `checks.py`, `runner.py`, `report.py` |
| `cli.py` | `agentos run`, `agentos tools`, `agentos bench` |

A malformed tool call (unknown tool, invalid JSON, missing or mistyped arguments) or a tool
exception comes back to the model as an `ERROR: ...` tool message so it can correct itself.

## Development

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```
