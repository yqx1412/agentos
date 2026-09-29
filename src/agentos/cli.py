"""Command-line entry point: ``agentos run "<task>"`` and ``agentos tools``."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TextIO

from agentos import __version__
from agentos.agent import Agent, Tracer
from agentos.builtin_tools import builtin_tools
from agentos.config import AgentOSConfig, ConfigError, load_config
from agentos.llm import LLMError, OllamaLLM
from agentos.mcp_client import MCPError, MCPManager
from agentos.tools import ToolRegistry

DEFAULT_CONFIG = Path("agentos.toml")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", type=Path, default=Path.cwd(), help="Directory tools may use")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Config file (default: ./{DEFAULT_CONFIG} if it exists)",
    )
    p.add_argument("--no-mcp", action="store_true", help="Ignore configured MCP servers")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentos", description="Local-first agent runtime.")
    parser.add_argument("--version", action="version", version=f"agentos {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run a single task")
    run.add_argument("task", help="Task description in natural language")
    run.add_argument("--model", default="qwen3:8b", help="Ollama model (default: qwen3:8b)")
    run.add_argument("--max-steps", type=int, default=10)
    run.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    run.add_argument("--think", action="store_true", help="Enable model thinking (slower)")
    run.add_argument("--trace-dir", type=Path, default=Path("runs"), help="Where JSONL traces go")
    run.add_argument("--no-trace", action="store_true")
    _add_common(run)

    tools = sub.add_parser("tools", help="List the tools the agent would see")
    _add_common(tools)
    return parser


def _load(args: argparse.Namespace) -> AgentOSConfig:
    if args.no_mcp:
        return AgentOSConfig()
    if args.config is not None:
        return load_config(args.config)
    return load_config(DEFAULT_CONFIG) if DEFAULT_CONFIG.is_file() else AgentOSConfig()


@contextlib.contextmanager
def _registry(
    config: AgentOSConfig, workspace: Path, mcp_log: TextIO
) -> Iterator[tuple[ToolRegistry, MCPManager]]:
    with MCPManager(config.enabled_servers(workspace), errlog=mcp_log) as mcp:
        yield ToolRegistry([*builtin_tools(workspace), *mcp.tools()]), mcp


@contextlib.contextmanager
def _open_log(path: Path | None) -> Iterator[TextIO]:
    """Where MCP servers' stderr goes: a log file next to the trace, or the null device."""
    if path is None:
        f = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - closed below
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        f = path.open("a", encoding="utf-8")
    try:
        yield f
    finally:
        f.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command not in ("run", "tools"):
        parser.print_help()
        return 0

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        print(f"workspace does not exist: {workspace}", file=sys.stderr)
        return 2
    try:
        config = _load(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "tools":
        return _cmd_tools(config, workspace)
    return _cmd_run(args, config, workspace)


def _cmd_tools(config: AgentOSConfig, workspace: Path) -> int:
    try:
        with _open_log(None) as log, _registry(config, workspace, log) as (registry, _):
            for tool in registry.tools():
                first_line = (tool.description.strip().splitlines() or [""])[0]
                if len(first_line) > 70:
                    first_line = first_line[:67] + "..."
                print(f"{tool.name:<34} {tool.source:<16} {first_line}")
    except MCPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_run(args: argparse.Namespace, config: AgentOSConfig, workspace: Path) -> int:
    trace_path = log_path = None
    if not args.no_trace:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        trace_path = args.trace_dir / f"{stamp}-{args.model.replace(':', '_')}.jsonl"
        if config.enabled_servers(workspace):
            log_path = trace_path.with_suffix(".mcp.log")

    llm = OllamaLLM(args.model, args.ollama_url, think=args.think)
    try:
        with _open_log(log_path) as log, _registry(config, workspace, log) as (registry, _):
            agent = Agent(llm, registry, max_steps=args.max_steps, tracer=Tracer(trace_path))
            result = agent.run(args.task)
    except (LLMError, MCPError) as exc:
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
