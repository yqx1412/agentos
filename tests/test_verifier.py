import json
from pathlib import Path

import pytest
from test_agent import ScriptedLLM, assistant  # tests/ is on sys.path under pytest

from agentos.bench.runner import make_agent
from agentos.builtin_tools import builtin_tools
from agentos.executor import PlanningAgent
from agentos.llm import Message, ToolCall
from agentos.tools import ToolRegistry
from agentos.verifier import Verifier, numbers_in, numbers_used


def registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(builtin_tools(tmp_path))


def ok_verdict(reason: str = "fine") -> Message:
    return assistant(json.dumps({"ok": True, "reason": reason}))


def bad_verdict(reason: str) -> Message:
    return assistant(json.dumps({"ok": False, "reason": reason}))


def plan(*goals: str) -> Message:
    steps = [
        {"id": i, "goal": g, "depends_on": [i - 1] if i > 1 else []} for i, g in enumerate(goals, 1)
    ]
    return assistant(json.dumps({"steps": steps}))


def step_messages(*calls: tuple[str, dict, str]) -> list[Message]:
    """An assistant turn with the given calls, followed by their tool results."""
    tcs = [ToolCall(name=n, arguments=a) for n, a, _ in calls]
    out = [Message(role="assistant", tool_calls=tcs)]
    out += [
        Message(role="tool", content=r, tool_name=tc.name, tool_call_id=tc.id)
        for tc, (_, _, r) in zip(tcs, calls, strict=True)
    ]
    return out


# -- deterministic checks -----------------------------------------------------------------


def test_numbers_in_handles_commas_decimals_and_file_names() -> None:
    assert numbers_in("sales 1,200 and 3.30; q1.txt v2") == {1200.0, 1.0, 200.0, 3.3, 2.0}
    assert numbers_in("jan,1200,1350") == {1200.0, 1350.0}


def test_numbers_used_reads_each_token_once() -> None:
    assert numbers_used("north,1200,1350\ntotal 1,500.5") == {1200.0, 1350.0, 1500.5}


def test_tool_call_written_as_text_is_rejected(tmp_path: Path) -> None:
    answer = 'I will call {"name": "write_file", "parameters": {"path": "total.txt"}}'
    v = Verifier(ScriptedLLM([]), registry(tmp_path)).check(
        task="t", goal="g", evidence="", messages=[], answer=answer
    )
    assert (v.ok, v.check) == (False, "text_tool_call")
    assert "tool interface" in v.reason


def test_function_style_mention_in_prose_goes_to_reflection(tmp_path: Path) -> None:
    llm = ScriptedLLM([ok_verdict()])
    v = Verifier(llm, registry(tmp_path)).check(
        task="t", goal="g", evidence="", messages=[], answer="calculator(9*60+40) gives 580"
    )
    assert v.ok
    assert len(llm.seen) == 1


def test_reflection_sees_earlier_step_results(tmp_path: Path) -> None:
    llm = ScriptedLLM([ok_verdict()])
    Verifier(llm, registry(tmp_path)).check(
        task="t",
        goal="pick the largest",
        evidence="",
        messages=[],
        answer="south",
        context="- step 1 (sum each region): north 3530, south 4530",
    )
    user = llm.seen[0][1].content
    assert user.startswith("Earlier steps' results and tool outputs:\n- step 1")
    assert "(no tool calls)" in user


def test_tool_name_mentioned_in_prose_after_real_calls_is_fine(tmp_path: Path) -> None:
    msgs = step_messages(("write_file", {"path": "a.txt", "content": "x"}, "wrote 1 characters"))
    llm = ScriptedLLM([ok_verdict()])
    v = Verifier(llm, registry(tmp_path)).check(
        task="t", goal="g", evidence="", messages=msgs, answer="used write_file(a.txt)"
    )
    assert v.ok


def test_made_up_numbers_are_rejected_without_calling_the_model(tmp_path: Path) -> None:
    """The A4 demo failure: summing CSV totals that were never read."""
    msgs = step_messages(("calculator", {"expression": "1500 + 2500 + 3000"}, "7000"))
    llm = ScriptedLLM([])  # would raise if the reflection call happened
    v = Verifier(llm, registry(tmp_path)).check(
        task="sum the sales in north.csv", goal="g", evidence="", messages=msgs, answer="7000"
    )
    assert (v.ok, v.check) == (False, "ungrounded")
    assert "1500, 2500, 3000" in v.reason


def test_numbers_from_earlier_outputs_task_and_units_are_grounded(tmp_path: Path) -> None:
    msgs = step_messages(
        ("read_file", {"path": "north.csv"}, "month,sales\njan,1,200\nfeb,1350\n"),
        ("calculator", {"expression": "1200 + 1350 + 980"}, "3530"),
        ("calculator", {"expression": "3530 / 1000 * 3600"}, "12708.0"),
        ("write_file", {"path": "t.json", "content": '{"north": 3530, "x": 12708}'}, "wrote"),
    )
    llm = ScriptedLLM([ok_verdict()])
    v = Verifier(llm, registry(tmp_path)).check(
        task="t", goal="g", evidence="west total 980", messages=msgs, answer="3530"
    )
    assert v.ok, v.reason


def test_file_name_passed_as_text_is_rejected(tmp_path: Path) -> None:
    """The A3/A4 textkit failure: word_frequency(text="essay.txt")."""
    msgs = step_messages(
        ("textkit__word_frequency", {"text": "essay.txt", "top_n": 1}, '{"top": ["essay"]}')
    )
    v = Verifier(ScriptedLLM([]), registry(tmp_path)).check(
        task="t", goal="g", evidence="", messages=msgs, answer="essay"
    )
    assert (v.ok, v.check) == (False, "filename_as_content")
    assert "'essay.txt'" in v.reason


def test_file_name_as_write_content_is_allowed(tmp_path: Path) -> None:
    msgs = step_messages(("write_file", {"path": "out.txt", "content": "big.csv"}, "wrote"))
    v = Verifier(ScriptedLLM([ok_verdict()]), registry(tmp_path)).check(
        task="t", goal="g", evidence="", messages=msgs, answer="wrote big.csv"
    )
    assert v.ok


def test_failed_calls_are_not_checked_for_grounding(tmp_path: Path) -> None:
    msgs = step_messages(
        ("calculator", {"expression": "sum([4444])"}, "ERROR: unsupported syntax"),
        ("calculator", {"expression": "2 + 2"}, "4"),
    )
    v = Verifier(ScriptedLLM([ok_verdict()]), registry(tmp_path)).check(
        task="t", goal="g", evidence="", messages=msgs, answer="4"
    )
    assert v.ok


# -- reflection ---------------------------------------------------------------------------


def test_reflection_rejection_and_prompt(tmp_path: Path) -> None:
    msgs = step_messages(("read_file", {"path": "a.txt"}, "hello"))
    llm = ScriptedLLM([bad_verdict("the goal says to write b.txt; nothing was written")])
    v = Verifier(llm, registry(tmp_path)).check(
        task="t", goal="copy a.txt to b.txt", evidence="", messages=msgs, answer="done"
    )
    assert (v.ok, v.check) == (False, "reflection")
    assert "nothing was written" in v.reason
    assert v.prompt_tokens == 10
    user = llm.seen[0][1].content
    assert "Step goal: copy a.txt to b.txt" in user
    assert '- read_file({"path": "a.txt"}) -> hello' in user
    assert "Reported result: done" in user


def test_unparseable_reflection_counts_as_a_pass(tmp_path: Path) -> None:
    v = Verifier(ScriptedLLM([assistant("looks good to me")]), registry(tmp_path)).check(
        task="t", goal="g", evidence="", messages=[], answer="x"
    )
    assert (v.ok, v.check) == (True, "unparseable")


# -- PlanningAgent with verification ------------------------------------------------------


def test_verified_direct_step_keeps_the_plain_answer(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            plan("compute 6*7"),
            assistant("", ("calculator", {"expression": "6*7"})),
            assistant("42"),
            ok_verdict(),
        ]
    )
    res = PlanningAgent(llm, registry(tmp_path), verify=True).run("What is 6*7?")
    assert (res.mode, res.answer, res.stop_reason) == ("direct", "42", "final_answer")
    assert llm.seen[1][1].content == "What is 6*7?"
    assert "Step goal: What is 6*7?" in llm.seen[3][1].content  # checked against the task
    assert res.rejections == 0


def test_direct_failed_reply_is_a_legitimate_answer(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [plan("read missing.txt"), assistant("FAILED: missing.txt does not exist"), ok_verdict()]
    )
    res = PlanningAgent(llm, registry(tmp_path), verify=True).run("read missing.txt")
    assert res.stop_reason == "final_answer"
    assert res.answer.startswith("FAILED")
    assert res.replans == 0


def test_rejection_without_retries_triggers_a_replan(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            plan("sum the files", "write the total"),
            assistant("", ("calculator", {"expression": "1500 + 2500"})),
            assistant("total 4000"),  # ungrounded -> rejected, no model call
            assistant(json.dumps({"steps": [{"id": 3, "goal": "read the files and sum"}]})),
            assistant("", ("calculator", {"expression": "2 + 3"})),
            assistant("total 5"),
            ok_verdict(),
            assistant("The total is 5."),  # synthesis
        ]
    )
    res = PlanningAgent(llm, registry(tmp_path), verify=True).run("sum")
    assert res.stop_reason == "final_answer"
    assert (res.rejections, res.retries, res.replans) == (1, 0, 1)
    assert "ungrounded" in llm.seen[3][-1].content  # the revise prompt names the rejection
    assert res.plan[0]["rejections"][0].startswith("ungrounded:")


def test_rejection_with_retries_reruns_the_step_with_feedback(tmp_path: Path) -> None:
    (tmp_path / "n.txt").write_text("1500\n2500\n", encoding="utf-8")
    llm = ScriptedLLM(
        [
            plan("sum the numbers in n.txt", "write the total"),
            # attempt 1: writes the tool call as text
            assistant('{"name": "read_file", "parameters": {"path": "n.txt"}}'),
            # attempt 2, prompt carries the rejection
            assistant("", ("read_file", {"path": "n.txt"})),
            assistant("", ("calculator", {"expression": "1500 + 2500"})),
            assistant("total 4000"),
            ok_verdict(),
            # step 2
            assistant("", ("write_file", {"path": "t.txt", "content": "4000"})),
            assistant("wrote 4000"),
            ok_verdict(),
            assistant("The total is 4000."),
        ]
    )
    res = PlanningAgent(llm, registry(tmp_path), verify=True, step_retries=2).run("sum n.txt")
    assert res.stop_reason == "final_answer"
    assert (res.rejections, res.retries, res.replans) == (1, 1, 0)
    assert (tmp_path / "t.txt").read_text() == "4000"
    retry_prompt = llm.seen[2][1].content
    assert "REJECTED" in retry_prompt
    assert "written as text" in retry_prompt
    assert "Current step 1: sum the numbers in n.txt" in retry_prompt
    assert res.plan[0]["attempts"] == 2


def test_retries_are_capped_then_replan(tmp_path: Path) -> None:
    text_call = assistant('{"name": "read_file", "parameters": {"path": "n.txt"}}')
    llm = ScriptedLLM([plan("a", "b"), text_call, text_call, text_call, plan("c")])
    agent = PlanningAgent(llm, registry(tmp_path), verify=True, step_retries=1, max_replans=0)
    res = agent.run("t")
    assert res.stop_reason == "plan_failed"
    assert (res.rejections, res.retries) == (2, 1)


def test_retry_evidence_excludes_the_rejected_attempts_own_arguments(tmp_path: Path) -> None:
    """A made-up value must not become evidence for the retry that repeats it."""
    made_up = assistant("", ("calculator", {"expression": "1500 + 2500"}))
    llm = ScriptedLLM(
        [
            plan("sum the files", "write it"),
            made_up,
            assistant("total 4000"),  # rejected: ungrounded
            made_up,
            assistant("total 4000"),  # still ungrounded, although 1500 was "seen" before
        ]
    )
    agent = PlanningAgent(llm, registry(tmp_path), verify=True, step_retries=1, max_replans=0)
    res = agent.run("sum")
    assert res.stop_reason == "plan_failed"
    assert res.rejections == 2
    assert all(r.startswith("ungrounded") for r in res.plan[0]["rejections"])


def test_step_retries_require_verify(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="verify"):
        PlanningAgent(ScriptedLLM([]), registry(tmp_path), step_retries=1)


def test_make_agent_builds_the_ablation_kinds(tmp_path: Path) -> None:
    from agentos.agent import Agent, Tracer

    kinds = {
        k: make_agent(k, ScriptedLLM([]), registry(tmp_path), max_steps=5, tracer=Tracer(None))
        for k in ("plain", "planner", "planner-verify", "planner-verify-retry")
    }
    assert type(kinds["plain"]) is Agent
    assert kinds["planner"].verifier is None
    assert kinds["planner-verify"].verifier is not None
    assert kinds["planner-verify"].step_retries == 0
    assert kinds["planner-verify-retry"].step_retries == 2
