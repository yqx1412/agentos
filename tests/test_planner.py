import json
from pathlib import Path

import pytest
from test_agent import ScriptedLLM, assistant  # tests/ is on sys.path under pytest

from agentos.agent import Tracer
from agentos.builtin_tools import builtin_tools
from agentos.executor import PlanningAgent
from agentos.llm import Message
from agentos.planner import Plan, PlanError, Planner, extract_json
from agentos.tools import ToolRegistry


def plan_reply(*steps: tuple[int, str, list[int]]) -> Message:
    body = {"steps": [{"id": i, "goal": g, "depends_on": d} for i, g, d in steps]}
    return assistant(json.dumps(body))


def registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(builtin_tools(tmp_path))


# -- Plan / graph -------------------------------------------------------------------------


def test_order_respects_dependencies_and_keeps_model_order_for_ties() -> None:
    plan = Plan.model_validate(
        {
            "steps": [
                {"id": 3, "goal": "c", "depends_on": [1, 2]},
                {"id": 1, "goal": "a"},
                {"id": 2, "goal": "b", "depends_on": [1]},
                {"id": 4, "goal": "d"},
            ]
        }
    )
    assert [s.id for s in plan.order()] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([{"id": 1, "goal": "a", "depends_on": [2]}], "unknown steps"),
        (
            [{"id": 1, "goal": "a", "depends_on": [2]}, {"id": 2, "goal": "b", "depends_on": [1]}],
            "cycle",
        ),
        ([{"id": 1, "goal": "a", "depends_on": [1]}], "itself"),
    ],
)
def test_check_graph_rejects_bad_graphs(steps: list[dict], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Plan.model_validate({"steps": steps}).check_graph()


def test_check_graph_allows_dependencies_on_finished_steps() -> None:
    plan = Plan.model_validate({"steps": [{"id": 3, "goal": "c", "depends_on": [1]}]})
    plan.check_graph(known={1, 2})
    with pytest.raises(ValueError, match="already used"):
        Plan.model_validate({"steps": [{"id": 1, "goal": "x"}]}).check_graph(known={1})


def test_duplicate_step_ids_are_invalid() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Plan.model_validate({"steps": [{"id": 1, "goal": "a"}, {"id": 1, "goal": "b"}]})


@pytest.mark.parametrize(
    "text",
    [
        '{"steps": []}',
        '```json\n{"steps": []}\n```',
        'Here is the plan:\n{"steps": []}\nGood luck.',
    ],
)
def test_extract_json_tolerates_fences_and_prose(text: str) -> None:
    assert extract_json(text) == {"steps": []}


# -- Planner ------------------------------------------------------------------------------


def test_planner_retries_once_with_the_validation_error(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            assistant("I would read the file first."),  # no JSON at all
            plan_reply((1, "read", []), (2, "write", [1])),
        ]
    )
    result = Planner(llm, registry(tmp_path)).plan("task")
    assert result.attempts == 2
    assert [s.id for s in result.plan.steps] == [1, 2]
    retry_prompt = llm.seen[1][-1].content
    assert "invalid" in retry_prompt
    assert "no JSON object" in retry_prompt


def test_planner_gives_up_after_max_attempts(tmp_path: Path) -> None:
    llm = ScriptedLLM([assistant("nope"), assistant('{"steps": [{"id": 1}]}')])
    with pytest.raises(PlanError, match="2 attempts"):
        Planner(llm, registry(tmp_path)).plan("task")


def test_planner_prompt_lists_the_tools(tmp_path: Path) -> None:
    llm = ScriptedLLM([plan_reply((1, "x", []))])
    Planner(llm, registry(tmp_path)).plan("task")
    system = llm.seen[0][0].content
    assert "- read_file:" in system
    assert "- calculator:" in system
    assert '{"steps": [{"id": 1' in system  # format braces survived .format()


def test_revised_plan_may_keep_a_dependency_on_the_failed_step(tmp_path: Path) -> None:
    llm = ScriptedLLM([plan_reply((2, "read x another way", [1]), (3, "use it", [2]))])
    result = Planner(llm, registry(tmp_path)).revise("task", {}, step_id=1, reason="no file")
    assert [(s.id, s.depends_on) for s in result.plan.steps] == [(2, []), (3, [2])]
    assert result.attempts == 1


def test_initial_plan_still_rejects_unknown_dependencies(tmp_path: Path) -> None:
    llm = ScriptedLLM([plan_reply((2, "x", [1])), plan_reply((1, "x", []))])
    assert Planner(llm, registry(tmp_path)).plan("task").attempts == 2


# -- PlanningAgent ------------------------------------------------------------------------


def test_single_step_plan_runs_the_plain_loop_on_the_original_task(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            plan_reply((1, "compute 6*7", [])),
            assistant("", ("calculator", {"expression": "6*7"})),
            assistant("42"),
        ]
    )
    res = PlanningAgent(llm, registry(tmp_path)).run("What is 6*7?")
    assert res.mode == "direct"
    assert res.answer == "42"
    assert res.stop_reason == "final_answer"
    assert llm.seen[1][1].content == "What is 6*7?"  # original prompt, not a step prompt
    assert res.prompt_tokens == 30  # planner call counted too


def test_multi_step_plan_passes_results_forward_and_synthesizes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("7", encoding="utf-8")
    llm = ScriptedLLM(
        [
            plan_reply((1, "Read a.txt and report the number", []), (2, "Write it doubled", [1])),
            # step 1
            assistant("", ("read_file", {"path": "a.txt"})),
            assistant("a.txt contains 7"),
            # step 2
            assistant("", ("write_file", {"path": "out.txt", "content": "14"})),
            assistant("wrote 14 to out.txt"),
            # synthesis
            assistant("out.txt now contains 14."),
        ]
    )
    trace = tmp_path / "t.jsonl"
    res = PlanningAgent(llm, registry(tmp_path), tracer=Tracer(trace)).run("double a.txt")

    assert res.mode == "planned"
    assert res.stop_reason == "final_answer"
    assert res.answer == "out.txt now contains 14."
    assert (tmp_path / "out.txt").read_text() == "14"
    assert res.steps == 4
    assert res.tool_calls == 2
    assert [s["status"] for s in res.plan] == ["done", "done"]

    step2_prompt = llm.seen[3][1].content
    assert "Current step 2: Write it doubled" in step2_prompt
    assert "a.txt contains 7" in step2_prompt  # step 1's result was handed forward
    assert 'read_file({"path": "a.txt"}) returned:' in step2_prompt  # and its raw tool output
    assert "[a.txt: 1 lines, 1 words]" in step2_prompt
    assert "a.txt contains 7" in llm.seen[5][1].content  # and reached the synthesizer
    assert llm.seen[5][0].content.startswith("You write the final reply")

    events = [json.loads(line)["event"] for line in trace.read_text("utf-8").splitlines()]
    assert events[:2] == ["plan_start", "plan"]
    assert events.count("step_start") == 2
    assert events[-2:] == ["synthesis", "plan_end"]


def test_failed_step_triggers_a_revised_plan(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            plan_reply((1, "find the value", []), (2, "use the value", [1])),
            assistant("FAILED: value.txt does not exist"),  # step 1 fails
            plan_reply((3, "compute the value instead", [])),  # revision
            assistant("", ("calculator", {"expression": "2+3"})),
            assistant("the value is 5"),
            assistant("The value is 5."),  # synthesis
        ]
    )
    res = PlanningAgent(llm, registry(tmp_path)).run("get the value")

    assert res.stop_reason == "final_answer"
    assert res.replans == 1
    assert [(s["id"], s["round"], s["status"]) for s in res.plan] == [
        (1, 0, "failed"),
        (2, 0, "dropped"),
        (3, 1, "done"),
    ]
    revise_prompt = llm.seen[2][-1].content
    assert "Step 1 failed: value.txt does not exist" in revise_prompt
    assert "greater than 1" in revise_prompt


def test_step_that_runs_out_of_turns_counts_as_failed(tmp_path: Path) -> None:
    looping = [assistant("", ("calculator", {"expression": "1"})) for _ in range(2)]
    llm = ScriptedLLM([plan_reply((1, "a", []), (2, "b", [1])), *looping])
    res = PlanningAgent(llm, registry(tmp_path), step_max_steps=2, max_replans=0).run("t")
    assert res.stop_reason == "plan_failed"
    assert "ran out of turns" in res.answer


def test_replan_budget_is_enforced(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            plan_reply((1, "a", []), (2, "b", [])),
            assistant("FAILED: no"),
            plan_reply((3, "c", [])),
            assistant("FAILED: still no"),
        ]
    )
    res = PlanningAgent(llm, registry(tmp_path), max_replans=1).run("t")
    assert res.stop_reason == "plan_failed"
    assert res.replans == 1
    assert "step 3 failed" in res.answer


def test_invalid_plan_falls_back_to_the_plain_loop(tmp_path: Path) -> None:
    llm = ScriptedLLM([assistant("no plan"), assistant("still no plan"), assistant("hi")])
    res = PlanningAgent(llm, registry(tmp_path)).run("say hi")
    assert res.mode == "fallback"
    assert res.plan is None
    assert res.answer == "hi"


def test_observations_skip_errors_and_are_capped() -> None:
    from agentos.executor import OBS_PER_CALL, _observations
    from agentos.llm import ToolCall

    ok, err, big = ToolCall(name="t", arguments={}), ToolCall(name="t"), ToolCall(name="t")
    msgs = [
        Message(role="assistant", tool_calls=[ok, err, big]),
        Message(role="tool", content="fine", tool_name="t", tool_call_id=ok.id),
        Message(role="tool", content="ERROR: nope", tool_name="t", tool_call_id=err.id),
        Message(role="tool", content="x" * 5000, tool_name="t", tool_call_id=big.id),
    ]
    out = _observations(msgs)
    assert "fine" in out
    assert "nope" not in out
    assert "x" * OBS_PER_CALL + "..." in out
    assert "x" * (OBS_PER_CALL + 1) not in out


def test_total_budget_is_twice_max_steps(tmp_path: Path) -> None:
    steps = [(i, f"s{i}", []) for i in range(1, 5)]
    replies = [plan_reply(*steps)]
    for _ in range(4):
        replies += [assistant("", ("calculator", {"expression": "1"})), assistant("ok")]
    llm = ScriptedLLM(replies)
    # max_steps=3 -> 6 turns: steps 1-3 use 2 turns each, step 4 finds the budget empty.
    res = PlanningAgent(llm, registry(tmp_path), max_steps=3).run("t")
    assert res.stop_reason == "max_steps"
    assert res.steps == 6
    assert [s["status"] for s in res.plan] == ["done", "done", "done", "pending"]
