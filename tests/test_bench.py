"""Benchmark harness tests: checks, task loading, runner and report. No Ollama needed."""

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from test_agent import ScriptedLLM, assistant  # tests/ is on sys.path under pytest

from agentos.bench.checks import Check, Outcome, numbers_in
from agentos.bench.report import category_table, render_markdown, summary_table
from agentos.bench.runner import check_servers, run_bench, run_task
from agentos.bench.tasks import Task, TaskError, load_tasks, select_tasks
from agentos.cli import main
from agentos.config import AgentOSConfig, ConfigError, MCPServerConfig
from agentos.llm import ChatResponse, LLMError, Message

REPO = Path(__file__).resolve().parents[1]
SHIPPED_TASKS = REPO / "benchmarks" / "tasks"
check_adapter: TypeAdapter = TypeAdapter(Check)


def ev(spec: dict, tmp_path: Path, files: dict[str, str] | None = None, **kw) -> bool:
    for rel, text in (files or {}).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    outcome = Outcome(tmp_path, kw.get("answer", ""), kw.get("tools", []), kw.get("initial", {}))
    return check_adapter.validate_python(spec).evaluate(outcome).ok


# -- checks ------------------------------------------------------------------------------


def test_numbers_in() -> None:
    assert numbers_in("total: 5,000 (up 12.5%) -3 and .5") == [5000, 12.5, -3, 0.5]


@pytest.mark.parametrize(
    ("spec", "content", "ok"),
    [
        ({"type": "file_equals", "value": "otter"}, "otter\n", True),
        ({"type": "file_equals", "value": "otter"}, "Otter", False),
        ({"type": "file_equals", "value": "otter", "case_sensitive": False}, "Otter", True),
        ({"type": "file_number", "value": 19}, "19 words", True),
        ({"type": "file_number", "value": 19}, "The count is 19.", True),
        ({"type": "file_number", "value": 19}, "20", False),
        ({"type": "file_number", "value": 3289.83, "tol": 0.011}, "3,289.83", True),
        ({"type": "file_number", "value": 1}, "none", False),
        ({"type": "file_contains", "values": ["errors: 3"]}, "Errors: 3\n", True),
        ({"type": "file_regex", "pattern": r"^\s*3\s*,\s*7\s*$"}, "3, 7\n", True),
        ({"type": "file_lines", "lines": ["a", "b"]}, "a\n\n b \n", True),
        ({"type": "file_lines", "lines": ["a", "b"]}, "b\na\n", False),
        ({"type": "file_lines", "lines": ["a", "b"], "ordered": False}, "b\na\n", True),
        ({"type": "file_json", "value": {"a": 5}}, '{"a": 5.0}', True),
        ({"type": "file_json", "value": [1, 2]}, "[1, 2", False),
    ],
)
def test_file_checks(tmp_path: Path, spec: dict, content: str, ok: bool) -> None:
    assert ev({**spec, "path": "out.txt"}, tmp_path, {"out.txt": content}) is ok


def test_file_checks_fail_on_missing_file(tmp_path: Path) -> None:
    assert not ev({"type": "file_equals", "path": "nope.txt", "value": "x"}, tmp_path)


def test_absent_and_unchanged(tmp_path: Path) -> None:
    assert ev({"type": "file_absent", "path": "x.txt"}, tmp_path)
    assert not ev({"type": "file_absent", "path": "x.txt"}, tmp_path, {"x.txt": ""})
    spec = {"type": "file_unchanged", "path": "r.txt"}
    assert ev(spec, tmp_path, {"r.txt": "same"}, initial={"r.txt": "same"})
    assert not ev(spec, tmp_path, {"r.txt": "edited"}, initial={"r.txt": "same"})


def test_answer_and_tool_checks(tmp_path: Path) -> None:
    ans = "It went from 48,000 to 57,600, a 20% increase."
    assert ev({"type": "answer_number", "value": 20}, tmp_path, answer=ans)
    assert not ev({"type": "answer_number", "value": 21}, tmp_path, answer=ans)
    anyc = {"type": "answer_contains", "values": ["missing", "not found"], "mode": "any"}
    assert ev(anyc, tmp_path, answer="File NOT FOUND")
    assert not ev({**anyc, "mode": "all"}, tmp_path, answer="File not found")
    assert ev({"type": "tool_used", "name": "calculator"}, tmp_path, tools=["calculator"])
    assert not ev({"type": "tool_used", "name": "calculator"}, tmp_path, tools=[])


def test_unknown_check_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        check_adapter.validate_python({"type": "vibes", "path": "x"})


# -- tasks -------------------------------------------------------------------------------


def test_shipped_tasks_are_valid() -> None:
    tasks = load_tasks(SHIPPED_TASKS)
    assert 30 <= len(tasks) <= 50
    assert {"file_ops", "math", "chaining", "mcp", "recovery", "safety"} <= {
        t.category for t in tasks
    }
    shipped = set(AgentOSConfig.model_validate(_repo_config()).mcp_servers)
    assert {s for t in tasks for s in t.servers} <= shipped


def _repo_config() -> dict:
    import tomllib

    return tomllib.loads((REPO / "agentos.toml").read_text(encoding="utf-8"))


def write_tasks(tmp_path: Path, body: str) -> Path:
    (tmp_path / "t.yaml").write_text(body, encoding="utf-8")
    return tmp_path


def test_task_file_errors(tmp_path: Path) -> None:
    with pytest.raises(TaskError, match=r"no \*\.yaml"):
        load_tasks(tmp_path)
    bad = "category: x\ntasks:\n  - {id: a, prompt: p, checks: []}\n"
    with pytest.raises(TaskError, match="checks"):
        load_tasks(write_tasks(tmp_path, bad))
    dup = (
        "category: x\ntasks:\n"
        "  - {id: a, prompt: p, checks: [{type: file_absent, path: z}]}\n"
        "  - {id: a, prompt: q, checks: [{type: file_absent, path: z}]}\n"
    )
    with pytest.raises(TaskError, match="duplicate"):
        load_tasks(write_tasks(tmp_path, dup))


def test_unchanged_check_needs_a_setup_file() -> None:
    with pytest.raises(ValidationError, match="not in files"):
        Task.model_validate(
            {
                "id": "a",
                "category": "x",
                "prompt": "p",
                "checks": [{"type": "file_unchanged", "path": "r.txt"}],
            }
        )


def test_select_tasks() -> None:
    tasks = load_tasks(SHIPPED_TASKS)
    assert {t.category for t in select_tasks(tasks, ["mcp"])} == {"mcp"}
    assert all(t.id.startswith("fo-") for t in select_tasks(tasks, ["fo-*"]))
    assert select_tasks(tasks, None) == tasks


# -- runner ------------------------------------------------------------------------------

COPY_TASK = Task.model_validate(
    {
        "id": "copy",
        "category": "file_ops",
        "prompt": "copy a.txt to b.txt",
        "files": {"a.txt": "hi\n"},
        "max_steps": 4,
        "checks": [
            {"type": "file_equals", "path": "b.txt", "value": "hi"},
            {"type": "file_unchanged", "path": "a.txt"},
        ],
    }
)


def solve_copy() -> list[Message]:
    return [
        assistant("", ("read_file", {"path": "a.txt"})),
        assistant("", ("write_file", {"path": "b.txt", "content": "hi"})),
        assistant("done"),
    ]


def test_run_task_pass(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    r = run_task(ScriptedLLM(solve_copy()), COPY_TASK, config=AgentOSConfig(), trace_path=trace)
    assert r.passed, r.checks
    assert (r.steps, r.tool_calls, r.tool_errors) == (3, 2, 0)
    assert r.prompt_tokens == 30
    assert trace.is_file()


def test_run_task_fails_a_check(tmp_path: Path) -> None:
    wrong = [assistant("", ("write_file", {"path": "b.txt", "content": "bye"})), assistant("ok")]
    r = run_task(ScriptedLLM(wrong), COPY_TASK, config=AgentOSConfig())
    assert not r.passed
    assert [c["ok"] for c in r.checks] == [False, True]
    assert "expected 'hi'" in r.checks[0]["detail"]


def test_run_task_max_steps_is_a_failure_even_if_checks_pass() -> None:
    loop = [*solve_copy()[:2], *[assistant("", ("read_file", {"path": "a.txt"}))] * 2]
    r = run_task(ScriptedLLM(loop), COPY_TASK, config=AgentOSConfig())
    assert r.stop_reason == "max_steps"
    assert all(c["ok"] for c in r.checks)
    assert not r.passed


class ExplodingLLM:
    model = "boom"

    def chat(self, messages: list[Message], tools: list[dict]) -> ChatResponse:
        raise LLMError("token repeat limit reached")


def test_run_task_backend_error_is_recorded() -> None:
    r = run_task(ExplodingLLM(), COPY_TASK, config=AgentOSConfig())
    assert not r.passed
    assert r.error == "llm: token repeat limit reached"


def test_run_task_seeds_files_with_lf_and_a_fresh_workspace() -> None:
    llm = ScriptedLLM([assistant("", ("read_file", {"path": "a.txt"})), assistant("done")])
    r = run_task(llm, COPY_TASK, config=AgentOSConfig())
    tool_msg = next(m for m in llm.seen[-1] if m.role == "tool")
    assert tool_msg.content.endswith("hi\n")
    assert "\r" not in tool_msg.content
    assert not r.passed  # b.txt from earlier tests does not leak into this workspace


def test_check_servers() -> None:
    task = COPY_TASK.model_copy(update={"servers": ["textkit"]})
    with pytest.raises(ConfigError, match="copy needs 'textkit'"):
        check_servers([task], AgentOSConfig())
    cfg = AgentOSConfig(mcp_servers={"textkit": MCPServerConfig(command="x", enabled=False)})
    with pytest.raises(ConfigError):
        check_servers([task], cfg)


class Lifecycle(ScriptedLLM):
    def __init__(self, model: str, replies: list[Message], log: list[str]) -> None:
        super().__init__(replies)
        self.model, self.log = model, log

    def warmup(self) -> None:
        self.log.append(f"warmup {self.model}")

    def unload(self) -> None:
        self.log.append(f"unload {self.model}")


def test_run_bench_writes_results_and_manages_models(tmp_path: Path) -> None:
    log: list[str] = []
    seen: list[tuple[int, int]] = []
    results = run_bench(
        ["m1", "m2"],
        [COPY_TASK],
        llm_factory=lambda m: Lifecycle(m, solve_copy() + solve_copy(), log),
        config=AgentOSConfig(),
        out_dir=tmp_path,
        repeats=2,
        settings={"num_ctx": 8192},
        progress=lambda r, done, total: seen.append((done, total)),
    )
    assert [r.passed for r in results] == [True] * 4
    assert log == ["warmup m1", "unload m1", "warmup m2", "unload m2"]
    assert seen[-1] == (4, 4)
    lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["task_id"] == "copy"
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["settings"] == {"num_ctx": 8192}
    assert (tmp_path / "traces" / "m1" / "copy-r1.jsonl").is_file()


def test_report_tables(tmp_path: Path) -> None:
    good = run_task(ScriptedLLM(solve_copy()), COPY_TASK, config=AgentOSConfig())
    bad = run_task(ExplodingLLM(), COPY_TASK, config=AgentOSConfig())
    table = summary_table([good, bad])
    assert "| scripted | 1/1 | 100% |" in table
    assert "| boom | 0/1 | 0% |" in table
    assert "| file_ops | 1 | 1/1 | 0/1 |" in category_table([good, bad])
    md = render_markdown([good, bad], {"started": "now", "tasks": 1})
    assert "token repeat limit" in md


def test_cli_bench_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "--list", "--tasks", str(SHIPPED_TASKS), "--only", "mcp"]) == 0
    out = capsys.readouterr().out
    assert "mcp-top-word" in out
    assert "fo-copy" not in out


def test_cli_bench_bad_filter(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "--tasks", str(SHIPPED_TASKS), "--only", "zzz"]) == 2
