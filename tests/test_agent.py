import json
from pathlib import Path
from typing import Any

from agentos.agent import Agent, Tracer
from agentos.builtin_tools import builtin_tools
from agentos.llm import ChatResponse, Message, ToolCall
from agentos.tools import ToolRegistry


class ScriptedLLM:
    """Returns pre-baked replies in order and records every conversation it was shown."""

    model = "scripted"

    def __init__(self, replies: list[Message]) -> None:
        self.replies = list(replies)
        self.seen: list[list[Message]] = []

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ChatResponse:
        self.seen.append(list(messages))
        return ChatResponse(message=self.replies.pop(0), prompt_tokens=10, completion_tokens=5)


def assistant(content: str = "", *calls: tuple[str, dict | str]) -> Message:
    return Message(
        role="assistant",
        content=content,
        tool_calls=[ToolCall(name=n, arguments=a) for n, a in calls],
    )


def test_final_answer_without_tools(tmp_path: Path) -> None:
    llm = ScriptedLLM([assistant("hello")])
    result = Agent(llm, ToolRegistry(builtin_tools(tmp_path))).run("say hello")
    assert result.answer == "hello"
    assert result.stop_reason == "final_answer"
    assert result.steps == 1
    assert result.tool_calls == 0


def test_word_count_demo_flow(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("the quick brown fox jumps", encoding="utf-8")
    llm = ScriptedLLM(
        [
            assistant("", ("read_file", {"path": "notes.txt"})),
            assistant("", ("write_file", {"path": "result.txt", "content": "5"})),
            assistant("notes.txt has 5 words; wrote 5 to result.txt."),
        ]
    )
    result = Agent(llm, ToolRegistry(builtin_tools(tmp_path))).run("count words")
    assert result.stop_reason == "final_answer"
    assert (tmp_path / "result.txt").read_text() == "5"
    assert result.tool_calls == 2
    assert result.tool_errors == 0
    # The read result, including the word-count header, was fed back to the model.
    tool_msg = llm.seen[1][-1]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_name == "read_file"
    assert "5 words" in tool_msg.content


def test_malformed_call_is_reported_back_and_agent_recovers(tmp_path: Path) -> None:
    """A1 'done when': a malformed tool call reaches the model as an error, not a crash."""
    llm = ScriptedLLM(
        [
            assistant("", ("write_file", {"path": "out.txt"})),  # missing 'content'
            assistant("", ("calculatr", {"expression": "1+1"})),  # unknown tool
            assistant("", ("calculator", '{"expression": ')),  # broken JSON
            assistant("", ("write_file", {"path": "out.txt", "content": "ok"})),
            assistant("done"),
        ]
    )
    result = Agent(llm, ToolRegistry(builtin_tools(tmp_path))).run("write ok to out.txt")

    assert result.stop_reason == "final_answer"
    assert result.tool_calls == 4
    assert result.tool_errors == 3
    assert (tmp_path / "out.txt").read_text() == "ok"
    errors = [m.content for m in result.messages if m.role == "tool"][:3]
    assert all(e.startswith("ERROR:") for e in errors)
    assert "content" in errors[0]
    assert "unknown tool" in errors[1]
    assert "not valid JSON" in errors[2]


def test_multiple_calls_in_one_reply(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            assistant(
                "", ("calculator", {"expression": "2+2"}), ("calculator", {"expression": "3*3"})
            ),
            assistant("4 and 9"),
        ]
    )
    result = Agent(llm, ToolRegistry(builtin_tools(tmp_path))).run("compute")
    tool_msgs = [m.content for m in result.messages if m.role == "tool"]
    assert tool_msgs == ["4", "9"]


def test_max_steps_stops_the_loop(tmp_path: Path) -> None:
    looping = [assistant("", ("calculator", {"expression": "1"})) for _ in range(3)]
    result = Agent(
        llm := ScriptedLLM(looping), ToolRegistry(builtin_tools(tmp_path)), max_steps=3
    ).run("loop forever")
    assert result.stop_reason == "max_steps"
    assert result.steps == 3
    assert len(llm.seen) == 3


def test_trace_is_written_as_jsonl(tmp_path: Path) -> None:
    trace = tmp_path / "runs" / "t.jsonl"
    llm = ScriptedLLM([assistant("", ("calculator", {"expression": "6*7"})), assistant("42")])
    result = Agent(llm, ToolRegistry(builtin_tools(tmp_path)), tracer=Tracer(trace)).run("6*7")
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == [
        "run_start",
        "llm_response",
        "tool_result",
        "llm_response",
        "run_end",
    ]
    assert events[2]["output"] == "42"
    assert events[-1]["prompt_tokens"] == result.prompt_tokens == 20
