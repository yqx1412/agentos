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

## Architecture (A1)

| Module | Role |
|---|---|
| `llm.py` | Provider-neutral `Message`/`ToolCall` types, `LLM` protocol, `OllamaLLM` |
| `tools.py` | `Tool` + `ToolRegistry`: Pydantic arg models become JSON schemas; `execute()` never raises |
| `builtin_tools.py` | `read_file`, `write_file` (confined to the workspace), `calculator` (AST-based, no `eval`) |
| `agent.py` | The loop: model -> tool calls -> results fed back -> repeat, capped by `max_steps`; `Tracer` |
| `cli.py` | `agentos run` |

A malformed tool call (unknown tool, invalid JSON, missing or mistyped arguments) or a tool
exception comes back to the model as an `ERROR: ...` tool message so it can correct itself.

## Development

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```
