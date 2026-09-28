"""Command-line entry point: ``agentos run "<task>"``."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from agentos import __version__
from agentos.agent import Agent, Tracer
from agentos.builtin_tools import builtin_tools
from agentos.llm import LLMError, OllamaLLM
from agentos.tools import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentos", description="Local-first agent runtime.")
    parser.add_argument("--version", action="version", version=f"agentos {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run a single task")
    run.add_argument("task", help="Task description in natural language")
    run.add_argument("--model", default="qwen3:8b", help="Ollama model (default: qwen3:8b)")
    run.add_argument("--workspace", type=Path, default=Path.cwd(), help="Directory tools may use")
    run.add_argument("--max-steps", type=int, default=10)
    run.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    run.add_argument("--think", action="store_true", help="Enable model thinking (slower)")
    run.add_argument("--trace-dir", type=Path, default=Path("runs"), help="Where JSONL traces go")
    run.add_argument("--no-trace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 0

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        print(f"workspace does not exist: {workspace}", file=sys.stderr)
        return 2

    trace_path = None
    if not args.no_trace:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        trace_path = args.trace_dir / f"{stamp}-{args.model.replace(':', '_')}.jsonl"

    llm = OllamaLLM(args.model, args.ollama_url, think=args.think)
    agent = Agent(
        llm,
        ToolRegistry(builtin_tools(workspace)),
        max_steps=args.max_steps,
        tracer=Tracer(trace_path),
    )
    try:
        result = agent.run(args.task)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        llm.close()

    print(result.answer)
    print(
        f"\n[{result.stop_reason}] steps={result.steps} tool_calls={result.tool_calls} "
        f"tool_errors={result.tool_errors} tokens={result.prompt_tokens}+{result.completion_tokens}"
        + (f" trace={trace_path}" if trace_path else ""),
        file=sys.stderr,
    )
    return 0 if result.stop_reason == "final_answer" else 3
