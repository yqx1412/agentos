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
| `cli.py` | `agentos run`, `agentos tools` |

A malformed tool call (unknown tool, invalid JSON, missing or mistyped arguments) or a tool
exception comes back to the model as an `ERROR: ...` tool message so it can correct itself.

## Development

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```
