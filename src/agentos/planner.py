"""Planner: ask the model to break a task into a small graph of steps with dependencies.

The plan is requested as JSON in the reply text (no tool calling), validated with Pydantic
and checked as a graph (unknown dependencies, cycles). A reply that fails validation is sent
back once with the error, the same recovery idea the agent loop uses for bad tool calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError, model_validator

from agentos.llm import LLM, Message
from agentos.tools import ToolRegistry

MAX_PLAN_STEPS = 8

PLAN_SYSTEM_PROMPT = f"""\
You are the planner of an agent that completes tasks by calling tools.
Break the user's task into the SMALLEST number of steps an executor can do one at a time.

Rules:
- Each step is one concrete sub-goal the executor can finish with a few tool calls,
  e.g. "Read settings.ini and report the host and port values".
- Reading inputs, computing and writing the result for ONE output is a single step.
  Never split "read X" and "write Y" into separate steps.
- If the whole task needs fewer than about 4 tool calls, return exactly ONE step.
- Use several steps only for genuinely separate parts: different outputs, or a result
  that must be found before the next part can even start.
- Every step must be doable with the tools listed below; do not plan steps no tool
  performs (read the tool descriptions: some do more than their name says).
- A step may depend on earlier steps; list their ids in depends_on.
- Say which values a step must report, so later steps can use them.
- Do not add a step for giving the final answer; that happens automatically.
- At most {MAX_PLAN_STEPS} steps.

Available tools:
{{tools}}

Reply with ONLY a JSON object, no prose and no code fence:
{{{{"steps": [{{{{"id": 1, "goal": "...", "depends_on": []}}}}]}}}}"""

REVISE_PROMPT = """\
Step {step_id} failed: {reason}

Steps finished so far:
{done}

Write a NEW plan for the rest of the task only. Do not repeat finished steps; you may depend
on their ids. Try a different approach from the one that failed. New step ids must be
greater than {max_id}. Reply with ONLY the JSON object."""


class PlanError(Exception):
    """The model did not produce a usable plan."""


class Step(BaseModel):
    id: int = Field(ge=1)
    goal: str = Field(min_length=1)
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    steps: list[Step] = Field(min_length=1, max_length=MAX_PLAN_STEPS)

    @model_validator(mode="after")
    def _valid_graph(self) -> Plan:
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate step ids: {ids}")
        return self

    def drop_stale_deps(self, finished: set[int]) -> None:
        """In a revised plan, forget dependencies on steps that did not finish.

        Models often keep ``depends_on: [1]`` after step 1 failed. There is nothing to wait
        for, so rejecting the whole plan for it only burns the replan budget.
        """
        ids = {s.id for s in self.steps}
        for s in self.steps:
            s.depends_on = [d for d in s.depends_on if d in ids or d in finished]

    def check_graph(self, known: set[int] | None = None) -> None:
        """Raise ValueError on unknown dependencies or cycles. ``known`` = finished step ids."""
        known = known or set()
        ids = {s.id for s in self.steps}
        if ids & known:
            raise ValueError(f"step ids {sorted(ids & known)} are already used by finished steps")
        for s in self.steps:
            missing = [d for d in s.depends_on if d not in ids and d not in known]
            if missing:
                raise ValueError(f"step {s.id} depends on unknown steps {missing}")
            if s.id in s.depends_on:
                raise ValueError(f"step {s.id} depends on itself")
        self.order()  # raises on cycles

    def order(self) -> list[Step]:
        """Topological order; ties keep the model's order (Kahn's algorithm)."""
        by_id = {s.id: s for s in self.steps}
        pending = {s.id: {d for d in s.depends_on if d in by_id} for s in self.steps}
        out: list[Step] = []
        while pending:
            ready = [sid for sid in by_id if sid in pending and not pending[sid]]
            if not ready:
                raise ValueError(f"dependency cycle among steps {sorted(pending)}")
            sid = ready[0]
            out.append(by_id[sid])
            del pending[sid]
            for deps in pending.values():
                deps.discard(sid)
        return out


@dataclass
class PlanResult:
    plan: Plan
    prompt_tokens: int
    completion_tokens: int
    attempts: int


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> object:
    """Parse the first JSON object in ``text``, tolerating code fences and surrounding prose."""
    text = _FENCE.sub("", text.strip())
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def describe_tools(tools: ToolRegistry) -> str:
    lines = []
    for t in tools.tools():
        first = (t.description.strip().splitlines() or [""])[0]
        lines.append(f"- {t.name}: {first[:120]}")
    return "\n".join(lines) or "- (none)"


class Planner:
    def __init__(self, llm: LLM, tools: ToolRegistry, *, max_attempts: int = 2) -> None:
        self.llm = llm
        self.system_prompt = PLAN_SYSTEM_PROMPT.format(tools=describe_tools(tools))
        self.max_attempts = max_attempts

    def plan(self, task: str) -> PlanResult:
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=f"Task: {task}"),
        ]
        return self._ask(messages, known=set(), revising=False)

    def revise(
        self, task: str, finished: dict[int, tuple[str, str]], step_id: int, reason: str
    ) -> PlanResult:
        """New plan for the rest of the task. ``finished`` maps step id -> (goal, result)."""
        done = "\n".join(f"- step {i}: {g} -> {r}" for i, (g, r) in finished.items()) or "- none"
        max_id = max([step_id, *finished])
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=f"Task: {task}"),
            Message(
                role="user",
                content=REVISE_PROMPT.format(
                    step_id=step_id, reason=reason, done=done, max_id=max_id
                ),
            ),
        ]
        return self._ask(messages, known=set(finished), revising=True)

    def _ask(self, messages: list[Message], known: set[int], revising: bool) -> PlanResult:
        p_tok = c_tok = 0
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            resp = self.llm.chat(messages, [])
            p_tok += resp.prompt_tokens
            c_tok += resp.completion_tokens
            text = resp.message.content
            try:
                plan = Plan.model_validate(extract_json(text))
                if revising:
                    plan.drop_stale_deps(known)
                plan.check_graph(known)
                return PlanResult(plan, p_tok, c_tok, attempt)
            except (ValueError, ValidationError) as exc:
                # json.JSONDecodeError is a ValueError subclass.
                last_error = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                messages = [
                    *messages,
                    Message(role="assistant", content=text),
                    Message(
                        role="user",
                        content=f"That plan is invalid: {last_error}. Reply with ONLY the "
                        "corrected JSON object.",
                    ),
                ]
        raise PlanError(f"no valid plan after {self.max_attempts} attempts: {last_error}")
