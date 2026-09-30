"""Planning agent: plan -> run each step with the plain loop -> revise the plan on failure.

Flow for one task::

    Planner.plan(task)                    -> Plan (task graph)
      1 step?  -> run the plain loop on the original task ("direct"; no planning overhead
                  beyond the one planner call)
      else     -> for each step in dependency order:
                    plain Agent on a step prompt (task + finished results + this step)
                    step failed?  -> Planner.revise(...) replaces the remaining steps
                  synthesize the final answer from the step results
    planner never produced a valid plan -> plain loop on the original task ("fallback")

Each step runs in a fresh, short conversation. That is the point of the planner: a small
model only has to solve one sub-goal at a time instead of the whole task in one context.
"""

from __future__ import annotations

import json
from textwrap import indent
from typing import Any

from agentos.agent import DEFAULT_SYSTEM_PROMPT, Agent, AgentResult, StopReason, Tracer
from agentos.llm import LLM, Message
from agentos.planner import Plan, PlanError, Planner, Step
from agentos.tools import ToolRegistry
from agentos.verifier import Verifier

STEP_PROMPT = """\
Overall task: {task}

The task has been split into steps. You are doing ONE of them.
{done}
Current step {step_id}: {goal}

Do only the current step, using tools. Tool outputs from finished steps are shown above
exactly as the tools returned them; use them directly instead of repeating those calls.
When the step is done, reply with a short result that states every value later steps
need (numbers, words, file names). If the step cannot be done, reply with "FAILED: "
and the reason."""

# Raw tool output handed to later steps. A model-written summary alone loses data:
# "read source.txt successfully" is useless to a step that has to copy the file.
OBS_PER_CALL = 800
OBS_PER_STEP = 2000

SYNTH_SYSTEM_PROMPT = """\
You write the final reply for a task that an agent has already carried out step by step.
Use only the step results below; do not invent anything. Reply in one or two short
sentences and include every value the task asks for."""

RETRY_NOTE = """\
{prompt}

A previous attempt at this was checked and REJECTED: {reason}
Do it again, fixing that problem. Files written by the previous attempt are still there."""


def _short(text: str, limit: int = 600) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _observations(messages: list[Message]) -> str:
    """The successful tool outputs of one step, verbatim but size-capped."""
    calls = {c.id: c for m in messages for c in m.tool_calls}
    parts: list[str] = []
    used = 0
    for m in messages:
        if m.role != "tool" or m.content.startswith("ERROR:"):
            continue
        call = calls.get(m.tool_call_id or "")
        args = json.dumps(call.arguments, ensure_ascii=False) if call else ""
        body = m.content if len(m.content) <= OBS_PER_CALL else m.content[:OBS_PER_CALL] + "..."
        part = f"    {m.tool_name}({args}) returned:\n" + indent(body, "      ")
        if used + len(part) > OBS_PER_STEP:
            parts.append("    (further tool output omitted)")
            break
        parts.append(part)
        used += len(part)
    return "\n".join(parts)


def _tool_outputs(messages: list[Message]) -> str:
    """Only what the tools returned, uncapped: the verifier's evidence. Never the calls'
    arguments, which may be made up and must not vouch for themselves."""
    return "\n".join(
        m.content for m in messages if m.role == "tool" and not m.content.startswith("ERROR:")
    )


class PlanningAgent:
    """Drop-in alternative to :class:`Agent` with the same ``run(task) -> AgentResult``.

    ``max_steps`` is the plain loop's budget; it is used as-is for direct and fallback runs.
    Planned runs get ``2 * max_steps`` model turns in total, because every step spends one
    extra turn reporting its result.
    """

    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        *,
        max_steps: int = 12,
        step_max_steps: int = 8,
        max_replans: int = 2,
        direct_single_step: bool = True,
        verify: bool = False,
        step_retries: int = 0,
        tracer: Tracer | None = None,
    ) -> None:
        if max_steps < 1 or step_max_steps < 1:
            raise ValueError("max_steps and step_max_steps must be >= 1")
        if step_retries and not verify:
            raise ValueError("step_retries needs verify=True: retries react to rejections")
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.step_max_steps = step_max_steps
        self.max_replans = max_replans
        self.direct_single_step = direct_single_step
        self.step_retries = step_retries
        self.tracer = tracer or Tracer(None)
        self.planner = Planner(llm, tools)
        self.verifier = Verifier(llm, tools) if verify else None

    def run(self, task: str) -> AgentResult:
        self.tracer.emit("plan_start", model=self.llm.model, task=task)
        try:
            pr = self.planner.plan(task)
        except PlanError as exc:
            self.tracer.emit("plan_error", error=str(exc))
            return self._plain(task, mode="fallback", plan=None)

        tokens = (pr.prompt_tokens, pr.completion_tokens)
        self.tracer.emit(
            "plan", attempts=pr.attempts, steps=[s.model_dump() for s in pr.plan.steps]
        )
        if len(pr.plan.steps) == 1 and self.direct_single_step:
            if self.verifier is None:
                return self._plain(task, mode="direct", plan=pr.plan, extra_tokens=tokens)
            # Same run as _plain, but through the step machinery so it gets verified.
            return self._execute(task, pr.plan, tokens, direct=True)
        return self._execute(task, pr.plan, tokens)

    # -- modes -------------------------------------------------------------------------

    def _plain(
        self,
        task: str,
        *,
        mode: str,
        plan: Plan | None,
        extra_tokens: tuple[int, int] = (0, 0),
    ) -> AgentResult:
        res = Agent(self.llm, self.tools, max_steps=self.max_steps, tracer=self.tracer).run(task)
        res.prompt_tokens += extra_tokens[0]
        res.completion_tokens += extra_tokens[1]
        res.mode = mode
        if plan is not None:
            res.plan = [{**s.model_dump(), "round": 0, "status": mode} for s in plan.steps]
        return res

    def _execute(
        self, task: str, plan: Plan, tokens: tuple[int, int], *, direct: bool = False
    ) -> AgentResult:
        """Run the plan. ``direct``: a verified 1-step plan whose step is the task itself.

        A direct step keeps the plain loop's semantics: its prompt is the original task, it
        gets ``max_steps`` turns, running out of turns ends the run with ``max_steps`` and a
        "FAILED:" reply is a legitimate final answer. Only a verifier rejection changes it.
        """
        p_tok, c_tok = tokens
        n_steps = n_calls = n_errors = replans = rejections = retries = 0
        budget = 2 * self.max_steps
        messages: list[Message] = []
        finished: dict[int, tuple[str, str]] = {}
        observations: dict[int, str] = {}
        step_outputs: dict[int, str] = {}  # verifier evidence: what tools returned, only
        records: list[dict[str, Any]] = [
            {**s.model_dump(), "round": 0, "status": "pending"} for s in plan.steps
        ]
        queue: list[Step] = plan.order()
        stop: StopReason | None = None
        answer = direct_answer = ""

        while queue:
            step = queue.pop(0)
            record = next(r for r in records if r["id"] == step.id and r["status"] == "pending")
            is_direct = direct and replans == 0
            if budget <= 0:
                stop, answer = "max_steps", "step budget exhausted"
                break

            self.tracer.emit("step_start", step=step.id, goal=step.goal)
            base_prompt = (
                task if is_direct else self._step_prompt(task, step, finished, observations)
            )
            prompt, attempt, seen_outputs = base_prompt, 0, ""
            while True:
                agent = Agent(
                    self.llm,
                    self.tools,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    max_steps=min(self.max_steps if is_direct else self.step_max_steps, budget),
                    tracer=self.tracer,
                )
                res = agent.run(prompt)
                budget -= res.steps
                n_steps += res.steps
                n_calls += res.tool_calls
                n_errors += res.tool_errors
                p_tok += res.prompt_tokens
                c_tok += res.completion_tokens
                messages += res.messages
                result = res.answer.strip()

                if is_direct and res.stop_reason != "final_answer":
                    break  # plain-loop semantics: out of turns ends the run below
                failed = res.stop_reason != "final_answer" or (
                    not is_direct and result.upper().startswith("FAILED")
                )
                reason = (
                    f"ran out of turns ({res.stop_reason})"
                    if res.stop_reason != "final_answer"
                    else _short(result.split(":", 1)[-1], 300)
                )
                if failed or self.verifier is None:
                    break

                evidence = "\n".join([*step_outputs.values(), seen_outputs])
                verdict = self.verifier.check(
                    task=task,
                    goal=task if is_direct else step.goal,
                    evidence=evidence,
                    messages=res.messages,
                    answer=result,
                    context=self._finished_block(finished, observations),
                )
                p_tok += verdict.prompt_tokens
                c_tok += verdict.completion_tokens
                self.tracer.emit(
                    "verify", step=step.id, attempt=attempt, ok=verdict.ok,
                    check=verdict.check, reason=verdict.reason,
                )  # fmt: skip
                if verdict.ok:
                    break
                rejections += 1
                record.setdefault("rejections", []).append(f"{verdict.check}: {verdict.reason}")
                reason = f"a checker rejected the result ({verdict.check}): {verdict.reason}"
                if attempt >= self.step_retries or budget <= 0:
                    failed = True
                    break
                attempt += 1
                retries += 1
                # Only what the tools RETURNED is evidence; the rejected attempt's own
                # arguments are exactly what is under suspicion.
                seen_outputs += "\n" + _tool_outputs(res.messages)
                prompt = RETRY_NOTE.format(prompt=base_prompt, reason=verdict.reason)
                self.tracer.emit("step_retry", step=step.id, attempt=attempt)

            record["result"] = _short(result, 300)
            record["attempts"] = attempt + 1
            if is_direct and res.stop_reason != "final_answer":
                record["status"] = "failed"
                stop, answer = res.stop_reason, result
                break
            self.tracer.emit("step_end", step=step.id, ok=not failed, result=result)
            if not failed:
                record["status"] = "done"
                finished[step.id] = (step.goal, _short(result))
                observations[step.id] = _observations(res.messages)
                step_outputs[step.id] = _tool_outputs(res.messages)
                if is_direct:
                    direct_answer = result
                continue

            record["status"] = "failed"
            for r in records:  # the rest of the old plan is superseded
                if r["status"] == "pending":
                    r["status"] = "dropped"
            if replans >= self.max_replans:
                stop, answer = "plan_failed", f"step {step.id} failed: {reason}"
                break
            replans += 1
            try:
                pr = self.planner.revise(task, finished, step.id, reason)
            except PlanError as exc:
                self.tracer.emit("replan_error", error=str(exc))
                stop, answer = "plan_failed", f"step {step.id} failed and replanning failed: {exc}"
                break
            p_tok += pr.prompt_tokens
            c_tok += pr.completion_tokens
            self.tracer.emit("replan", round=replans, steps=[s.model_dump() for s in pr.plan.steps])
            records += [
                {**s.model_dump(), "round": replans, "status": "pending"} for s in pr.plan.steps
            ]
            queue = pr.plan.order()

        if stop is None:
            if direct and replans == 0:
                answer = direct_answer  # the plain loop's own final answer, as in _plain
            else:
                answer, sp, sc = self._synthesize(task, finished)
                p_tok += sp
                c_tok += sc
            stop = "final_answer"

        result = AgentResult(
            answer=answer,
            stop_reason=stop,
            steps=n_steps,
            tool_calls=n_calls,
            tool_errors=n_errors,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            messages=messages,
            plan=records,
            replans=replans,
            mode="direct" if direct else "planned",
            rejections=rejections,
            retries=retries,
        )
        self.tracer.emit(
            "plan_end",
            answer=answer,
            stop_reason=stop,
            steps=n_steps,
            replans=replans,
            rejections=rejections,
            retries=retries,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
        )
        return result

    # -- prompts -----------------------------------------------------------------------

    @staticmethod
    def _finished_block(finished: dict[int, tuple[str, str]], observations: dict[int, str]) -> str:
        blocks = []
        for i, (g, r) in finished.items():
            block = f"- step {i} ({g}): {r}"
            if observations.get(i):
                block += "\n  tool outputs:\n" + observations[i]
            blocks.append(block)
        return "\n".join(blocks)

    @classmethod
    def _step_prompt(
        cls,
        task: str,
        step: Step,
        finished: dict[int, tuple[str, str]],
        observations: dict[int, str],
    ) -> str:
        done = (
            f"\nFinished steps:\n{cls._finished_block(finished, observations)}\n"
            if finished
            else ""
        )
        return STEP_PROMPT.format(task=task, done=done, step_id=step.id, goal=step.goal)

    def _synthesize(self, task: str, finished: dict[int, tuple[str, str]]) -> tuple[str, int, int]:
        results = "\n".join(f"- step {i} ({g}): {r}" for i, (g, r) in finished.items())
        resp = self.llm.chat(
            [
                Message(role="system", content=SYNTH_SYSTEM_PROMPT),
                Message(role="user", content=f"Task: {task}\n\nStep results:\n{results}"),
            ],
            [],
        )
        self.tracer.emit("synthesis", content=resp.message.content)
        return resp.message.content.strip(), resp.prompt_tokens, resp.completion_tokens
