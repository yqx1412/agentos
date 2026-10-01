from pathlib import Path

import pytest
from pydantic import ValidationError
from test_agent import ScriptedLLM, assistant  # tests/ is on sys.path under pytest

from agentos.agent import Agent, Tracer
from agentos.bench.report import summary_table
from agentos.bench.runner import make_agent, run_bench, run_task
from agentos.bench.tasks import Task
from agentos.builtin_tools import builtin_tools
from agentos.cli import main
from agentos.config import AgentOSConfig, ConfigError
from agentos.llm import ChatResponse, Message
from agentos.memory import (
    SUMMARY_NOTE,
    MemoryAgent,
    MemoryStore,
    compact_messages,
    fts_query,
    with_memory_tools,
)
from agentos.tools import ToolRegistry

# -- store -----------------------------------------------------------------------------


def test_add_dedupe_forget(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as s:
        f, created = s.add_fact("The deploy server is build-07, port 8443.")
        assert created
        again, created2 = s.add_fact("the  deploy server is BUILD-07 port 8443")
        assert not created2 and again.id == f.id
        assert s.forget(f.id)
        assert not s.forget(f.id)
        assert s.facts() == []
        with pytest.raises(ValueError):
            s.add_fact("   ")
        with pytest.raises(ValueError):
            s.add_fact("x" * 600)


def test_search_stems_ranks_and_ignores_stopwords(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as s:
        s.add_fact("The deploy server is build-07, port 8443.")
        s.add_fact("My cat is called Miso.")
        hits = s.search_facts("which servers do we deploy to?")
        assert [h.text for h in hits] == ["The deploy server is build-07, port 8443."]
        assert s.search_facts("what is the") == []  # stopwords only
        assert fts_query("what is the") is None
        assert fts_query('say "hi" OR NOT') == '"say" OR "hi"'  # quotes/operators neutralized


def test_conflicting_facts_newest_first(tmp_path: Path) -> None:
    clock = iter([100.0, 200.0])
    with MemoryStore(tmp_path / "m.db", clock=lambda: next(clock)) as s:
        s.add_fact("The API port is 8080.")
        s.add_fact("The API port changed to 9090.")
        texts = [f.text for f in s.search_facts("api port")]
        assert texts == ["The API port changed to 9090.", "The API port is 8080."]


def test_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "m.db"
    with MemoryStore(db) as s:
        s.add_fact("Project codename is Bluebird.")
        s.add_episode("sum the sales", "completed", "Total is 1234.", model="m")
    with MemoryStore(db) as s:
        assert [f.text for f in s.facts()] == ["Project codename is Bluebird."]
        assert s.search_episodes("sales total")[0].answer == "Total is 1234."


def test_context_for(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as s:
        assert s.context_for("deploy the app") == ""
        s.add_fact("The deploy server is build-07.")
        s.add_episode("Deploy the docs site", "completed", "Deployed to build-07.")
        block = s.context_for("deploy the app")
        assert "fact #1" in block and "build-07" in block
        assert "past task" in block and "completed" in block
        assert s.facts()[0].uses == 1  # injection counts as a use


# -- tools -----------------------------------------------------------------------------


def test_memory_tools_round_trip(tmp_path: Path) -> None:
    from agentos.llm import ToolCall

    with MemoryStore(tmp_path / "m.db") as s:
        reg = with_memory_tools(ToolRegistry(), s)
        assert reg.names() == ["remember", "recall", "forget"]
        r = reg.execute(ToolCall(name="remember", arguments={"fact": "Wifi password hint: tea."}))
        assert r.ok and "fact #1" in r.output
        r = reg.execute(ToolCall(name="remember", arguments={"fact": "wifi password hint tea"}))
        assert "already stored" in r.output
        r = reg.execute(ToolCall(name="recall", arguments={"query": "wifi hint"}))
        assert "Wifi password hint" in r.output
        assert reg.execute(ToolCall(name="recall", arguments={"query": "zebra"})).output == (
            "no matching memories"
        )
        assert not reg.execute(ToolCall(name="forget", arguments={"id": 99})).ok
        assert reg.execute(ToolCall(name="forget", arguments={"id": 1})).ok


# -- agent wrapper: the A6 "done when" -------------------------------------------------


def test_fact_learned_in_session_1_is_reused_in_session_2(tmp_path: Path) -> None:
    """A6 'done when': a task in session 2 reuses a fact learned in session 1."""
    db = tmp_path / "memory.db"
    ws = tmp_path / "ws"
    ws.mkdir()

    # Session 1: the model stores the fact.
    llm1 = ScriptedLLM(
        [
            assistant("", ("remember", {"fact": "The deploy server is build-07, port 8443."})),
            assistant("Noted."),
        ]
    )
    with MemoryStore(db) as store:
        reg = with_memory_tools(ToolRegistry(builtin_tools(ws)), store)
        MemoryAgent(Agent(llm1, reg), store).run("Remember: deploy server build-07, port 8443.")

    # Session 2: new process-equivalent (new store object, new agent, new conversation).
    llm2 = ScriptedLLM(
        [
            assistant("", ("write_file", {"path": "deploy.txt", "content": "build-07:8443"})),
            assistant("Wrote build-07:8443."),
        ]
    )
    with MemoryStore(db) as store:
        reg = with_memory_tools(ToolRegistry(builtin_tools(ws)), store)
        res = MemoryAgent(Agent(llm2, reg), store).run(
            "Write the deploy server host:port to a file"
        )
        episodes = store.episodes()

    first_prompt = llm2.seen[0][1].content
    assert "build-07, port 8443" in first_prompt  # injected from session 1
    assert first_prompt.rstrip().endswith("Task: Write the deploy server host:port to a file")
    assert res.stop_reason == "final_answer"
    assert (ws / "deploy.txt").read_text() == "build-07:8443"
    assert [e.outcome for e in episodes] == ["completed", "completed"]


def test_memory_agent_without_inject_leaves_prompt_alone(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as store:
        store.add_fact("The deploy server is build-07.")
        llm = ScriptedLLM([assistant("ok")])
        MemoryAgent(Agent(llm, ToolRegistry()), store, inject=False).run("deploy server?")
        assert llm.seen[0][1].content == "deploy server?"
        assert len(store.episodes()) == 1


def test_make_agent_memory_modes(tmp_path: Path) -> None:
    llm = ScriptedLLM([])
    with MemoryStore(tmp_path / "m.db") as store:
        tracer = Tracer(None)
        plain = make_agent("plain", llm, ToolRegistry(), max_steps=3, tracer=tracer)
        assert isinstance(plain, Agent)
        reg = ToolRegistry()
        auto = make_agent(
            "planner", llm, reg, max_steps=3, tracer=tracer, memory=store, memory_mode="auto"
        )
        assert isinstance(auto, MemoryAgent) and auto.inject
        assert "recall" in reg.names()
        with pytest.raises(ConfigError):
            make_agent("plain", llm, ToolRegistry(), max_steps=3, tracer=tracer, memory_mode="x")


# -- short-term memory -----------------------------------------------------------------


class SummarizingLLM(ScriptedLLM):
    """Scripted for the agent; answers summary requests (no tools offered) on its own."""

    def __init__(self, replies: list[Message], prompt_tokens: int) -> None:
        super().__init__(replies)
        self.prompt_tokens = prompt_tokens
        self.summaries = 0

    def chat(self, messages, tools):  # type: ignore[override]
        if not tools:
            self.summaries += 1
            return ChatResponse(message=Message(role="assistant", content="read a.txt: 7"))
        resp = super().chat(messages, tools)
        return ChatResponse(message=resp.message, prompt_tokens=self.prompt_tokens)


def test_compact_messages_keeps_head_tail_and_call_pairs() -> None:
    msgs = [Message(role="system", content="sys"), Message(role="user", content="task")]
    for i in range(4):
        msgs.append(assistant("", ("read_file", {"path": f"{i}.txt"})))
        msgs.append(Message(role="tool", content=f"c{i}", tool_name="read_file"))
    llm = SummarizingLLM([], 0)
    out, _, _ = compact_messages(msgs, llm, keep_last=3)  # type: ignore[misc]
    assert out[:2] == msgs[:2]
    assert out[2].role == "user" and out[2].content.startswith(SUMMARY_NOTE)
    assert out[3].role == "assistant"  # tail starts at a call, never at a tool result
    assert out[3:] == msgs[-4:]
    assert compact_messages(msgs[:4], llm) is None  # nothing worth cutting


def test_agent_compacts_over_budget_and_keeps_full_history(tmp_path: Path) -> None:
    for i in range(4):
        (tmp_path / f"{i}.txt").write_text(str(i), encoding="utf-8")
    replies = [assistant("", ("read_file", {"path": f"{i}.txt"})) for i in range(4)]
    llm = SummarizingLLM([*replies, assistant("done")], prompt_tokens=5000)
    res = Agent(llm, ToolRegistry(builtin_tools(tmp_path)), max_steps=8, context_budget=1000).run(
        "read all"
    )
    assert res.stop_reason == "final_answer"
    assert res.compactions >= 1
    assert llm.summaries == 0  # the loop clips deterministically; no model-written summary
    assert any(SUMMARY_NOTE in m.content for m in llm.seen[-1])  # the model saw the summary
    # The result keeps every tool call, so benchmark checks are unaffected.
    assert sum(len(m.tool_calls) for m in res.messages) == 4


def test_deterministic_compaction_keeps_both_ends_of_long_outputs() -> None:
    long = "HEADER\n" + "filler " * 500 + "\nKEY1 = 111"
    msgs = [Message(role="system", content="sys"), Message(role="user", content="task")]
    for _ in range(3):
        msgs.append(assistant("", ("read_file", {"path": "p.txt"})))
        msgs.append(Message(role="tool", content=long, tool_name="read_file"))
    out, p, c = compact_messages(msgs, keep_last=2)  # type: ignore[misc]
    summary = out[2].content
    assert (p, c) == (0, 0)
    assert summary.count("HEADER") == 2 and summary.count("KEY1 = 111") == 2
    assert "characters omitted" in summary and len(summary) < 2 * len(long)
    # A second compaction keeps the first summary whole instead of clipping it again.
    again, *_ = compact_messages([*out[:3], *msgs[2:]], keep_last=2)  # type: ignore[misc]
    assert again[2].content.count("KEY1 = 111") >= 2


def test_no_budget_means_no_compaction(tmp_path: Path) -> None:
    llm = SummarizingLLM([assistant("done")], prompt_tokens=10**6)
    res = Agent(llm, ToolRegistry(builtin_tools(tmp_path))).run("x")
    assert res.compactions == 0 and llm.summaries == 0


# -- benchmark: multi-session tasks ----------------------------------------------------

MEM_TASK = Task(
    id="mem-port",
    category="memory",
    setup=["Read config.txt and remember the API port."],
    setup_files={"config.txt": "api_port=9443\n"},
    prompt="Write the API port to port.txt.",
    checks=[{"type": "file_number", "path": "port.txt", "value": 9443}],
)


def mem_solution(port: str = "9443") -> list[Message]:
    return [
        assistant("", ("read_file", {"path": "config.txt"})),
        assistant("", ("remember", {"fact": "The API port is 9443."})),
        assistant("Stored."),
        assistant("", ("write_file", {"path": "port.txt", "content": port})),
        assistant("Done."),
    ]


def test_run_task_setup_sessions_share_memory(tmp_path: Path) -> None:
    llm = ScriptedLLM(mem_solution())
    trace = tmp_path / "t.jsonl"
    r = run_task(llm, MEM_TASK, config=AgentOSConfig(), memory_mode="auto", trace_path=trace)
    assert r.passed, r
    assert r.sessions == 2 and r.setup_failed == 0 and r.memory == "auto"
    assert "The API port is 9443." in llm.seen[3][1].content  # injected into session 2
    assert len(llm.seen[3]) == 2  # session 2 started a fresh conversation
    assert r.prompt_tokens == 50  # tokens of both sessions are counted
    assert "session_start" in trace.read_text(encoding="utf-8")


def test_run_task_setup_files_are_gone_in_the_main_session(tmp_path: Path) -> None:
    # Without memory, the model has nothing to go on once config.txt is removed.
    llm = ScriptedLLM(
        [
            assistant("Port is 9443, noted."),
            assistant("", ("read_file", {"path": "config.txt"})),
            assistant("I do not know the port."),
        ]
    )
    r = run_task(llm, MEM_TASK, config=AgentOSConfig(), memory_mode="none")
    assert not r.passed
    tool_msgs = [m for m in llm.seen[2] if m.role == "tool"]
    assert "file not found" in tool_msgs[0].content
    with pytest.raises(ValueError, match="MemoryStore"):
        make_agent("plain", llm, ToolRegistry(), max_steps=2, tracer=None, memory_mode="tools")  # type: ignore[arg-type]


def test_task_setup_validation() -> None:
    with pytest.raises(ValidationError, match="setup prompt"):
        Task(
            id="x",
            category="memory",
            prompt="p",
            setup_files={"a.txt": "1"},
            checks=[{"type": "answer_contains", "values": ["1"]}],
        )
    with pytest.raises(ValidationError, match="both"):
        Task(
            id="x",
            category="memory",
            prompt="p",
            setup=["s"],
            files={"a.txt": "1"},
            setup_files={"a.txt": "1"},
            checks=[{"type": "answer_contains", "values": ["1"]}],
        )


def test_run_bench_memory_modes_and_report(tmp_path: Path) -> None:
    def factory(model: str) -> ScriptedLLM:
        return ScriptedLLM(mem_solution() * 2)

    results = run_bench(
        ["m"],
        [MEM_TASK],
        llm_factory=factory,
        config=AgentOSConfig(),
        out_dir=tmp_path,
        memory_modes=["tools", "auto"],
    )
    assert [r.memory for r in results] == ["tools", "auto"]
    assert "scripted / mem-tools" in summary_table(results)
    with pytest.raises(ConfigError):
        run_bench(
            ["m"],
            [MEM_TASK],
            llm_factory=factory,
            config=AgentOSConfig(),
            out_dir=tmp_path,
            memory_modes=["bogus"],
        )


def test_cli_memory_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "m.db")
    assert main(["memory", "--db", db, "add", "Bluebird is the codename."]) == 0
    assert "fact #1" in capsys.readouterr().out
    assert main(["memory", "--db", db, "search", "codename"]) == 0
    assert "Bluebird" in capsys.readouterr().out
    assert main(["memory", "--db", db, "list"]) == 0
    assert "user" in capsys.readouterr().out
    assert main(["memory", "--db", db, "forget", "1"]) == 0
    assert main(["memory", "--db", db, "forget", "1"]) == 1


def test_shipped_memory_tasks_are_valid() -> None:
    from agentos.bench.tasks import load_tasks

    tasks = load_tasks(Path(__file__).parent.parent / "benchmarks" / "memory")
    assert len(tasks) >= 10
    assert all(t.setup for t in tasks)
