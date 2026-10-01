"""Command-line entry point: ``agentos run "<task>"`` and ``agentos tools``."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TextIO

from agentos import __version__
from agentos.agent import Tracer
from agentos.bench.report import render_markdown, summary_table
from agentos.bench.runner import AGENT_KINDS, MEMORY_MODES, TaskResult, make_agent, run_bench
from agentos.bench.tasks import TaskError, load_tasks, select_tasks
from agentos.builtin_tools import builtin_tools
from agentos.config import AgentOSConfig, ConfigError, load_config
from agentos.llm import LLMError, OllamaLLM
from agentos.mcp_client import MCPError, MCPManager
from agentos.memory import MemoryStore, describe
from agentos.sandbox import sandbox_tools
from agentos.tools import PERMISSION_LEVELS, Policy, Tool, ToolRegistry

DEFAULT_CONFIG = Path("agentos.toml")
DEFAULT_TASKS = Path("benchmarks/tasks")
# Outside any workspace, so the agent's file tools cannot touch the database.
DEFAULT_MEMORY_DB = Path.home() / ".agentos" / "memory.db"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", type=Path, default=Path.cwd(), help="Directory tools may use")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Config file (default: ./{DEFAULT_CONFIG} if it exists)",
    )
    p.add_argument("--no-mcp", action="store_true", help="Ignore configured MCP servers")
    p.add_argument(
        "--sandbox",
        action="store_true",
        help="Add the run_python and run_command tools (permission: dangerous)",
    )
    p.add_argument(
        "--allow",
        choices=list(PERMISSION_LEVELS),
        default="write",
        help="Highest permission level that runs without asking (default: write). Higher "
        "calls are shown for a y/N answer, or denied when stdin is not a terminal",
    )
    p.add_argument("--no-prompt", action="store_true", help="Never ask; deny instead")


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
    run.add_argument("--num-ctx", type=int, default=8192, help="Model context window")
    run.add_argument("--trace-dir", type=Path, default=Path("runs"), help="Where JSONL traces go")
    run.add_argument("--no-trace", action="store_true")
    run.add_argument(
        "--agent",
        choices=list(AGENT_KINDS),
        default="plain",
        help="plain loop, planner (A4), or planner with verification / retries (A5)",
    )
    run.add_argument("--show-plan", action="store_true", help="Print the plan to stderr")
    run.add_argument(
        "--memory",
        choices=list(MEMORY_MODES),
        default="none",
        help="Long-term memory: none, tools (remember/recall/forget), or auto (tools + "
        "relevant memories added to the task)",
    )
    run.add_argument("--memory-db", type=Path, default=DEFAULT_MEMORY_DB, help="SQLite memory file")
    run.add_argument(
        "--context-budget",
        type=int,
        default=None,
        help="Summarize older messages once a prompt exceeds this many tokens "
        "(default: 75%% of --num-ctx; 0 disables; plain agent only)",
    )
    _add_common(run)

    mem = sub.add_parser("memory", help="Inspect or edit long-term memory")
    mem.add_argument("--db", type=Path, default=DEFAULT_MEMORY_DB, help="SQLite memory file")
    mem_sub = mem.add_subparsers(dest="memory_command", required=True)
    mem_sub.add_parser("list", help="List stored facts, newest first")
    mem_sub.add_parser("episodes", help="List past task outcomes, newest first")
    add = mem_sub.add_parser("add", help="Store a fact")
    add.add_argument("text")
    search = mem_sub.add_parser("search", help="Keyword search over facts and episodes")
    search.add_argument("query")
    forget = mem_sub.add_parser("forget", help="Delete a fact by id")
    forget.add_argument("id", type=int)

    tools = sub.add_parser("tools", help="List the tools the agent would see")
    _add_common(tools)

    bench = sub.add_parser("bench", help="Run the benchmark task set against models")
    bench.add_argument(
        "--models", default="qwen3:8b", help="Comma-separated Ollama models (default: qwen3:8b)"
    )
    bench.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help="Task YAML directory")
    bench.add_argument(
        "--only", default=None, help="Comma-separated task id or category globs, e.g. 'mcp,fo-*'"
    )
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument(
        "--agents",
        default="plain",
        help=f"Comma-separated agent kinds to compare: {', '.join(AGENT_KINDS)} (default: plain)",
    )
    bench.add_argument(
        "--memory",
        default="none",
        help="Comma-separated memory modes to compare: none, tools, auto (default: none)",
    )
    bench.add_argument("--out", type=Path, default=Path("runs/bench"), help="Results root")
    bench.add_argument("--list", action="store_true", help="List the selected tasks and exit")
    bench.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    bench.add_argument("--num-ctx", type=int, default=8192, help="Model context window")
    bench.add_argument("--timeout", type=float, default=180.0, help="Seconds per model request")
    bench.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Config with the MCP servers tasks use (default: ./{DEFAULT_CONFIG})",
    )
    return parser


def _load(args: argparse.Namespace) -> AgentOSConfig:
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = load_config(DEFAULT_CONFIG) if DEFAULT_CONFIG.is_file() else AgentOSConfig()
    if getattr(args, "no_mcp", False):
        config = config.model_copy(update={"mcp_servers": {}})
    return config


def ask_approval(tool: Tool, args: dict) -> bool:
    """The human 'yes' for dangerous tools: shows the exact call, defaults to no."""
    shown = json.dumps(args, ensure_ascii=False, indent=2)
    if len(shown) > 3000:
        shown = shown[:3000] + "\n  ... (truncated)"
    print(f"\n[approval] {tool.name} ({tool.permission}) wants to run:\n{shown}", file=sys.stderr)
    try:
        reply = input("Allow this call? [y/N] ")
    except EOFError:
        return False
    return reply.strip().lower() in ("y", "yes")


def _policy(args: argparse.Namespace) -> Policy:
    interactive = sys.stdin.isatty() and not getattr(args, "no_prompt", False)
    return Policy(auto_approve=args.allow, approver=ask_approval if interactive else None)


@contextlib.contextmanager
def _registry(
    config: AgentOSConfig,
    workspace: Path,
    mcp_log: TextIO,
    *,
    sandbox: bool = False,
    policy: Policy | None = None,
) -> Iterator[tuple[ToolRegistry, MCPManager]]:
    with MCPManager(config.enabled_servers(workspace), errlog=mcp_log) as mcp:
        tools = [*builtin_tools(workspace), *mcp.tools()]
        if sandbox:
            tools += sandbox_tools(workspace, config.sandbox)
        yield ToolRegistry(tools, policy=policy), mcp


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
    if args.command == "memory":
        return _cmd_memory(args)
    if args.command not in ("run", "tools", "bench"):
        parser.print_help()
        return 0

    try:
        config = _load(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.command == "bench":
        return _cmd_bench(args, config)

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        print(f"workspace does not exist: {workspace}", file=sys.stderr)
        return 2

    if args.command == "tools":
        return _cmd_tools(args, config, workspace)
    return _cmd_run(args, config, workspace)


def _cmd_bench(args: argparse.Namespace, config: AgentOSConfig) -> int:
    try:
        tasks = select_tasks(load_tasks(args.tasks), _split(args.only))
    except TaskError as exc:
        print(f"task error: {exc}", file=sys.stderr)
        return 2
    if not tasks:
        print(f"no tasks match --only {args.only!r}", file=sys.stderr)
        return 2
    if args.list:
        for t in tasks:
            servers = f" [{', '.join(t.servers)}]" if t.servers else ""
            print(f"{t.id:<22} {t.category:<10}{servers}")
        print(f"\n{len(tasks)} tasks", file=sys.stderr)
        return 0

    models = _split(args.models) or []
    agents = _split(args.agents) or ["plain"]
    memory_modes = _split(args.memory) or ["none"]
    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    settings = {
        "num_ctx": args.num_ctx,
        "num_predict": 2048,
        "temperature": 0.0,
        "think": False,
        "timeout": args.timeout,
        "planner": "step_max_steps=8, max_replans=2, budget=2*max_steps"
        if any(a != "plain" for a in agents)
        else None,
        "verify": "text_tool_call + ungrounded numbers >= 100 + reflection; step_retries=2"
        if any("verify" in a for a in agents)
        else None,
    }

    def factory(model: str) -> OllamaLLM:
        return OllamaLLM(model, args.ollama_url, num_ctx=args.num_ctx, timeout=args.timeout)

    def progress(r: TaskResult, done: int, total: int) -> None:
        status = "PASS" if r.passed else "FAIL"
        why = ""
        if not r.passed:
            why = r.error or next((c["detail"] for c in r.checks if not c["ok"]), "")
            why = why if r.stop_reason in (None, "final_answer") else f"stopped: {r.stop_reason}"
            why = f"  {why[:100]}"
        print(
            f"[{done}/{total}] {r.model:<12} {r.agent:<7} {r.memory:<5} {r.task_id:<22} {status} "
            f"{r.seconds:5.1f}s{why}",
            file=sys.stderr,
            flush=True,
        )

    try:
        results = run_bench(
            models,
            tasks,
            llm_factory=factory,
            config=config,
            out_dir=out_dir,
            repeats=args.repeats,
            settings=settings,
            progress=progress,
            agents=agents,
            memory_modes=memory_modes,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:  # warmup failed: model missing or Ollama down
        print(f"error: {exc}", file=sys.stderr)
        return 1

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    report = render_markdown(results, meta)
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    print(summary_table(results))
    print(f"\nFull report: {out_dir / 'summary.md'}", file=sys.stderr)
    return 0


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _cmd_tools(args: argparse.Namespace, config: AgentOSConfig, workspace: Path) -> int:
    try:
        with (
            _open_log(None) as log,
            _registry(config, workspace, log, sandbox=args.sandbox) as (registry, _),
        ):
            for tool in registry.tools():
                first_line = (tool.description.strip().splitlines() or [""])[0]
                if len(first_line) > 60:
                    first_line = first_line[:57] + "..."
                print(f"{tool.name:<34} {tool.source:<16} {tool.permission:<10} {first_line}")
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

    llm = OllamaLLM(args.model, args.ollama_url, think=args.think, num_ctx=args.num_ctx)
    budget = args.context_budget if args.context_budget is not None else int(0.75 * args.num_ctx)
    store = MemoryStore(args.memory_db) if args.memory != "none" else None
    try:
        with (
            _open_log(log_path) as log,
            _registry(config, workspace, log, sandbox=args.sandbox, policy=_policy(args)) as (
                registry,
                _,
            ),
        ):
            agent = make_agent(
                args.agent,
                llm,
                registry,
                max_steps=args.max_steps,
                tracer=Tracer(trace_path),
                memory=store,
                memory_mode=args.memory,
                context_budget=budget or None,
            )
            result = agent.run(args.task)
    except (LLMError, MCPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        llm.close()
        if store is not None:
            store.close()

    if args.show_plan and result.plan:
        print(
            f"plan ({result.mode}, {result.replans} replans, {result.rejections} rejections, "
            f"{result.retries} retries):",
            file=sys.stderr,
        )
        for s in result.plan:
            deps = f" after {s['depends_on']}" if s["depends_on"] else ""
            res = f" -> {s['result']}" if s.get("result") else ""
            print(
                f"  [{s['status']:<7}] r{s['round']} step {s['id']}{deps}: {s['goal']}{res}",
                file=sys.stderr,
            )
        print(file=sys.stderr)
    print(result.answer)
    print(
        f"\n[{result.stop_reason}] steps={result.steps} tool_calls={result.tool_calls} "
        f"tool_errors={result.tool_errors} tokens={result.prompt_tokens}+{result.completion_tokens}"
        + (f" compactions={result.compactions}" if result.compactions else "")
        + (f" trace={trace_path}" if trace_path else ""),
        file=sys.stderr,
    )
    return 0 if result.stop_reason == "final_answer" else 3


def _cmd_memory(args: argparse.Namespace) -> int:
    with MemoryStore(args.db) as store:
        cmd = args.memory_command
        if cmd == "list":
            for f in store.facts():
                print(f"#{f.id:<4} {f.source:<6} used {f.uses:<3} {f.text}")
        elif cmd == "episodes":
            for e in store.episodes():
                print(
                    f"#{e.id:<4} {e.outcome:<12} {e.model:<12} {e.task[:60]!r} -> {e.answer[:60]!r}"
                )
        elif cmd == "add":
            try:
                fact, created = store.add_fact(args.text, source="user")
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print(f"{'stored' if created else 'already stored'} as fact #{fact.id}")
        elif cmd == "search":
            for f in store.search_facts(args.query, 10):
                print(f"fact #{f.id}: {f.text}")
            for e in store.search_episodes(args.query, 5):
                print(f"episode #{e.id} ({e.outcome}): {e.task[:60]!r} -> {e.answer[:80]!r}")
        elif cmd == "forget":
            if not store.forget(args.id):
                print(f"no fact #{args.id}", file=sys.stderr)
                return 1
            print(f"deleted fact #{args.id}")
        info = describe(store)
        print(
            f"\n{info['facts']} facts, {info['episodes']} episodes in {info['path']}",
            file=sys.stderr,
        )
    return 0
