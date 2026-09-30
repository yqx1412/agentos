"""Summaries of benchmark results as Markdown tables."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from agentos.bench.runner import TaskResult


def _pct(passed: int, n: int) -> str:
    return f"{100 * passed / n:.0f}%" if n else "-"


def _labeler(results: list[TaskResult]):
    """Row label: the model, plus the agent kind when a run compares several."""
    several = len({r.agent for r in results}) > 1
    return (lambda r: f"{r.model} / {r.agent}") if several else (lambda r: r.model)


def summary_table(results: list[TaskResult]) -> str:
    label = _labeler(results)
    by_model: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        by_model[label(r)].append(r)
    lines = [
        "| Model | Passed | Success | Avg steps | Avg tool calls | Tool errors | "
        "Avg tokens | Avg time | Backend errors | Hit max steps |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, rs in by_model.items():
        n, ok = len(rs), sum(r.passed for r in rs)
        lines.append(
            f"| {model} | {ok}/{n} | {_pct(ok, n)} "
            f"| {mean(r.steps for r in rs):.1f} "
            f"| {mean(r.tool_calls for r in rs):.1f} "
            f"| {sum(r.tool_errors for r in rs)} "
            f"| {mean(r.prompt_tokens + r.completion_tokens for r in rs):,.0f} "
            f"| {mean(r.seconds for r in rs):.1f} s "
            f"| {sum(r.error is not None for r in rs)} "
            f"| {sum(r.stop_reason == 'max_steps' for r in rs)} |"
        )
    return "\n".join(lines)


def category_table(results: list[TaskResult]) -> str:
    label = _labeler(results)
    models = list(dict.fromkeys(label(r) for r in results))
    categories = list(dict.fromkeys(r.category for r in results))
    cell: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        cell[(r.category, label(r))].append(r.passed)
    lines = [
        "| Category | Tasks | " + " | ".join(models) + " |",
        "|---|---|" + "---|" * len(models),
    ]
    for cat in categories:
        n_tasks = len({r.task_id for r in results if r.category == cat})
        row = [f"{sum(cell[(cat, m)])}/{len(cell[(cat, m)])}" for m in models]
        lines.append(f"| {cat} | {n_tasks} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def failures(results: list[TaskResult]) -> str:
    label = _labeler(results)
    lines = []
    for r in results:
        if r.passed:
            continue
        if r.error:
            why = r.error
        elif r.stop_reason != "final_answer":
            why = f"stopped: {r.stop_reason}"
        else:
            why = "; ".join(c["detail"] for c in r.checks if not c["ok"])
        lines.append(f"- `{label(r)}` **{r.task_id}**: {why}")
    return "\n".join(lines) or "_none_"


def planner_table(results: list[TaskResult]) -> str:
    """How each planner kind handled tasks: direct (1-step plan), planned, fallback."""
    rs = [r for r in results if r.agent != "plain"]
    if not rs:
        return ""
    by_kind: dict[str, list[TaskResult]] = defaultdict(list)
    for r in rs:
        by_kind[f"{r.model} / {r.agent}"].append(r)
    lines = [
        "| Model / agent | Direct (pass) | Planned (pass) | Fallback (pass) | Avg plan steps "
        "| Replans | Rejections | Retries |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, group in by_kind.items():
        cells = []
        for mode in ("direct", "planned", "fallback"):
            g = [r for r in group if r.mode == mode]
            cells.append(f"{len(g)} ({sum(r.passed for r in g)})")
        planned = [r for r in group if r.mode == "planned"]
        avg = mean(r.plan_steps for r in planned) if planned else 0.0
        lines.append(
            f"| {label} | {' | '.join(cells)} | {avg:.1f} | {sum(r.replans for r in group)} "
            f"| {sum(r.rejections for r in group)} | {sum(r.retries for r in group)} |"
        )
    return "\n".join(lines)


def render_markdown(results: list[TaskResult], meta: dict | None = None) -> str:
    parts = ["# AgentOS benchmark results", ""]
    if meta:
        parts += [
            f"- started: {meta.get('started')}, git: `{meta.get('git_sha')}`, "
            f"tasks: {meta.get('tasks')}, repeats: {meta.get('repeats')}",
            f"- settings: `{meta.get('settings')}`",
            "",
        ]
    parts += [
        "## Summary",
        "",
        summary_table(results),
        "",
        "## By category",
        "",
        category_table(results),
        "",
    ]
    if planner := planner_table(results):
        parts += ["## Planner modes", "", planner, ""]
    parts += [
        "## Failures",
        "",
        failures(results),
        "",
    ]
    return "\n".join(parts)
