"""Run benchmark tasks against models and record one result per (model, task, repeat).

Every task runs in a fresh temporary workspace seeded with its ``files``, with the built-in
tools plus the MCP servers it names. A task passes only if the agent reached a final answer
(no backend error, no ``max_steps`` cut-off) *and* every check passes.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentos import __version__
from agentos.agent import Agent, Tracer
from agentos.bench.checks import Outcome
from agentos.bench.tasks import Task
from agentos.builtin_tools import builtin_tools
from agentos.config import AgentOSConfig, ConfigError
from agentos.executor import PlanningAgent
from agentos.llm import LLM, LLMError
from agentos.mcp_client import MCPError, MCPManager
from agentos.memory import MemoryAgent, MemoryStore, with_memory_tools
from agentos.tools import ToolRegistry

LLMFactory = Callable[[str], LLM]
# Agent kinds, cumulative for the A5 ablation: each adds one mechanism to the previous.
AGENT_KINDS: dict[str, dict[str, Any] | None] = {
    "plain": None,  # the A1 loop
    "planner": {},  # + A4 planner / task graph
    "planner-verify": {"verify": True},  # + A5 step verification (rejection -> replan)
    "planner-verify-retry": {"verify": True, "step_retries": 2},  # + retry with feedback
}
# Long-term memory modes (A6): none; memory tools only; tools + automatic injection.
MEMORY_MODES = ("none", "tools", "auto")


def make_agent(
    kind: str,
    llm: LLM,
    registry: ToolRegistry,
    *,
    max_steps: int,
    tracer: Tracer,
    memory: MemoryStore | None = None,
    memory_mode: str = "none",
    context_budget: int | None = None,
) -> Agent | PlanningAgent | MemoryAgent:
    if kind not in AGENT_KINDS:
        raise ConfigError(f"unknown agent kind {kind!r}; choose from {list(AGENT_KINDS)}")
    if memory_mode not in MEMORY_MODES:
        raise ConfigError(f"unknown memory mode {memory_mode!r}; choose from {MEMORY_MODES}")
    if memory_mode != "none":
        if memory is None:
            raise ValueError(f"memory mode {memory_mode!r} needs a MemoryStore")
        with_memory_tools(registry, memory)
    options = AGENT_KINDS[kind]
    inner: Agent | PlanningAgent
    if options is None:
        inner = Agent(
            llm, registry, max_steps=max_steps, tracer=tracer, context_budget=context_budget
        )
    else:
        inner = PlanningAgent(llm, registry, max_steps=max_steps, tracer=tracer, **options)
    if memory_mode == "none" or memory is None:
        return inner
    return MemoryAgent(
        inner,
        memory,
        inject=memory_mode == "auto",
        model=llm.model,
        agent_kind=kind,
        tracer=tracer,
    )


Progress = Callable[["TaskResult", int, int], None]


class TaskResult(BaseModel):
    model: str
    task_id: str
    category: str
    repeat: int
    passed: bool
    stop_reason: str | None
    error: str | None
    checks: list[dict[str, Any]]
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    answer: str = ""
    trace: str | None = None
    agent: str = "plain"
    # PlanningAgent only: plain / direct / planned / fallback, plan size, replans.
    mode: str = "plain"
    plan_steps: int = 0
    replans: int = 0
    rejections: int = 0
    retries: int = 0
    # A6 memory: mode, and how many setup sessions (earlier "conversations") did not finish.
    memory: str = "none"
    sessions: int = 1
    setup_failed: int = 0


def check_servers(tasks: list[Task], config: AgentOSConfig) -> None:
    """Fail before any model runs if a task needs an MCP server that is not configured."""
    enabled = {n for n, c in config.mcp_servers.items() if c.enabled}
    missing = sorted({(t.id, s) for t in tasks for s in t.servers if s not in enabled})
    if missing:
        detail = ", ".join(f"{tid} needs {srv!r}" for tid, srv in missing)
        raise ConfigError(f"MCP servers not configured or disabled: {detail}")


def _seed(workspace: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = workspace / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode("utf-8"))  # bytes: keep LF line endings on Windows


def run_task(
    llm: LLM,
    task: Task,
    *,
    config: AgentOSConfig,
    repeat: int = 0,
    trace_path: Path | None = None,
    mcp_log_path: Path | None = None,
    agent_kind: str = "plain",
    memory_mode: str = "none",
) -> TaskResult:
    """Run one task. A task with ``setup`` runs those prompts first, each as a separate
    session (fresh agent, fresh conversation, fresh MCP servers) sharing one memory store;
    ``setup_files`` exist only during the setup sessions. Checks see the last session only.
    """
    with tempfile.TemporaryDirectory(prefix="agentos-bench-") as tmp:
        # Nest the workspace so a "../x" escape attempt lands inside our temp dir.
        workspace = Path(tmp) / "ws"
        workspace.mkdir()
        _seed(workspace, {**task.files, **task.setup_files})
        servers = {n: c for n, c in config.enabled_servers(workspace).items() if n in task.servers}
        # Outside the workspace, so file tools cannot read the database directly.
        store = MemoryStore(Path(tmp) / "memory.db") if memory_mode != "none" else None
        tracer = Tracer(trace_path)
        sessions = [*task.setup, task.prompt]

        error = stop_reason = None
        answer = ""
        stats: dict[str, Any] = {}
        tools_called: list[str] = []
        p_tok = c_tok = setup_failed = 0
        start = time.perf_counter()
        log = open(mcp_log_path or os.devnull, "a", encoding="utf-8")  # noqa: SIM115
        try:
            for i, prompt in enumerate(sessions):
                main = i == len(sessions) - 1
                if main:
                    for rel in task.setup_files:
                        (workspace / rel).unlink(missing_ok=True)
                if len(sessions) > 1:
                    tracer.emit("session_start", index=i, setup=not main)
                with MCPManager(servers, errlog=log) as mcp:
                    registry = ToolRegistry([*builtin_tools(workspace), *mcp.tools()])
                    agent = make_agent(
                        agent_kind,
                        llm,
                        registry,
                        max_steps=task.max_steps,
                        tracer=tracer,
                        memory=store,
                        memory_mode=memory_mode,
                    )
                    res = agent.run(prompt)
                p_tok += res.prompt_tokens
                c_tok += res.completion_tokens
                if not main:
                    setup_failed += res.stop_reason != "final_answer"
            stop_reason, answer = res.stop_reason, res.answer
            stats = {
                "steps": res.steps,
                "tool_calls": res.tool_calls,
                "tool_errors": res.tool_errors,
                # Every session's tokens: the memory-writing sessions are part of the cost.
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "mode": res.mode,
                "plan_steps": len(res.plan or []),
                "replans": res.replans,
                "rejections": res.rejections,
                "retries": res.retries,
            }
            tools_called = [c.name for m in res.messages for c in m.tool_calls]
        except LLMError as exc:
            error = f"llm: {exc}"
        except MCPError as exc:
            error = f"mcp: {exc}"
        finally:
            log.close()
            if store is not None:
                store.close()  # before the temp dir is removed: Windows locks open files
        seconds = time.perf_counter() - start

        outcome = Outcome(workspace, answer, tools_called, dict(task.files))
        results = [c.evaluate(outcome) for c in task.checks]

    checks = [
        {"type": c.type, "ok": r.ok, "detail": r.detail}
        for c, r in zip(task.checks, results, strict=True)
    ]
    passed = error is None and stop_reason == "final_answer" and all(r.ok for r in results)
    return TaskResult(
        model=llm.model,
        task_id=task.id,
        category=task.category,
        repeat=repeat,
        passed=passed,
        stop_reason=stop_reason,
        error=error,
        checks=checks,
        seconds=round(seconds, 2),
        answer=answer,
        trace=str(trace_path) if trace_path else None,
        agent=agent_kind,
        memory=memory_mode,
        sessions=len(task.setup) + 1,
        setup_failed=setup_failed,
        **stats,
    )


def _git_sha() -> str | None:
    """Short HEAD SHA, suffixed ``-dirty`` when the working tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not sha:
        return None
    return f"{sha}-dirty" if dirty else sha


def run_bench(
    models: list[str],
    tasks: list[Task],
    *,
    llm_factory: LLMFactory,
    config: AgentOSConfig,
    out_dir: Path,
    repeats: int = 1,
    settings: dict[str, Any] | None = None,
    progress: Progress | None = None,
    agents: list[str] | None = None,
    memory_modes: list[str] | None = None,
) -> list[TaskResult]:
    """Run every task ``repeats`` times per model, agent kind and memory mode.

    Results stream to ``results.jsonl``.
    """
    agents = agents or ["plain"]
    memory_modes = memory_modes or ["none"]
    unknown = sorted(set(agents) - set(AGENT_KINDS))
    if unknown:
        raise ConfigError(f"unknown agent kinds {unknown}; choose from {list(AGENT_KINDS)}")
    bad_mem = sorted(set(memory_modes) - set(MEMORY_MODES))
    if bad_mem:
        raise ConfigError(f"unknown memory modes {bad_mem}; choose from {list(MEMORY_MODES)}")
    check_servers(tasks, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "agentos_version": __version__,
        "git_sha": _git_sha(),
        "models": models,
        "agents": agents,
        "memory": memory_modes,
        "tasks": len(tasks),
        "repeats": repeats,
        "settings": settings or {},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    results: list[TaskResult] = []
    total = len(models) * len(agents) * len(memory_modes) * len(tasks) * repeats
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as sink:
        for model in models:
            llm = llm_factory(model)
            safe = model.replace(":", "_").replace("/", "_")
            try:
                if hasattr(llm, "warmup"):
                    llm.warmup()
                for kind, mem, task, r in (
                    (k, m, t, r)
                    for k in agents
                    for m in memory_modes
                    for t in tasks
                    for r in range(repeats)
                ):
                    suffix = f"-r{r}" if repeats > 1 else ""
                    sub = kind if mem == "none" else f"{kind}+mem-{mem}"
                    trace = out_dir / "traces" / safe / sub / f"{task.id}{suffix}.jsonl"
                    result = run_task(
                        llm,
                        task,
                        config=config,
                        repeat=r,
                        trace_path=trace,
                        mcp_log_path=out_dir / "mcp.log" if task.servers else None,
                        agent_kind=kind,
                        memory_mode=mem,
                    )
                    results.append(result)
                    sink.write(result.model_dump_json() + "\n")
                    sink.flush()
                    if progress:
                        progress(result, len(results), total)
            finally:
                # Free VRAM before the next model: two large models resident at once made
                # Ollama stall for minutes on this 16 GB GPU.
                if hasattr(llm, "unload"):
                    llm.unload()
                if hasattr(llm, "close"):
                    llm.close()
    return results
