"""The basic agent loop: call the model, run requested tools, feed results back, repeat."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agentos.llm import LLM, Message
from agentos.memory import compact_messages
from agentos.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """\
You are AgentOS, an agent that completes tasks by calling tools.
- Use tools to read, compute and write; never guess file contents or arithmetic.
- File paths are relative to the workspace.
- If a tool returns an ERROR, read it, fix the call and try again.
- When the task is complete, reply with a short final answer and no tool calls."""

StopReason = Literal["final_answer", "max_steps", "plan_failed"]


class Tracer:
    """Appends one JSON object per event to a JSONL file (or discards them if path is None)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **data: Any) -> None:
        if self.path is None:
            return
        record = {"ts": round(time.time(), 3), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


@dataclass
class AgentResult:
    answer: str
    stop_reason: StopReason
    steps: int
    tool_calls: int
    tool_errors: int
    prompt_tokens: int
    completion_tokens: int
    messages: list[Message] = field(repr=False)
    # Set by PlanningAgent only.
    plan: list[dict[str, Any]] | None = None
    replans: int = 0
    mode: str = "plain"
    rejections: int = 0  # step results the verifier rejected (A5)
    retries: int = 0  # steps re-run with the verifier's feedback (A5)
    compactions: int = 0  # times the conversation was summarized to fit (A6)


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
        tracer: Tracer | None = None,
        context_budget: int | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.tracer = tracer or Tracer(None)
        # Short-term memory (A6): once a request's prompt exceeds this many tokens, the
        # middle of the conversation is summarized before the next request. None = never.
        self.context_budget = context_budget

    def run(self, task: str) -> AgentResult:
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=task),
        ]
        # ``messages`` is what the model sees; ``history`` keeps every message, so results
        # and benchmark checks still see tool calls that compaction removed from view.
        history = list(messages)

        def add(m: Message) -> None:
            messages.append(m)
            history.append(m)

        schemas = self.tools.schemas()
        n_calls = n_errors = p_tok = c_tok = 0
        self._compactions = 0
        last_prompt = 0
        self.tracer.emit("run_start", model=self.llm.model, task=task, tools=self.tools.names())

        for step in range(1, self.max_steps + 1):
            if self.context_budget is not None and last_prompt > self.context_budget:
                compacted = compact_messages(messages, self.llm)
                if compacted is not None:
                    before = len(messages)
                    messages, sp, sc = compacted
                    p_tok += sp
                    c_tok += sc
                    self._compactions += 1
                    self.tracer.emit(
                        "compact",
                        step=step,
                        prompt_tokens=last_prompt,
                        messages_before=before,
                        messages_after=len(messages),
                        summary=messages[2].content,
                    )
            resp = self.llm.chat(messages, schemas)
            last_prompt = resp.prompt_tokens
            p_tok += resp.prompt_tokens
            c_tok += resp.completion_tokens
            reply = resp.message
            add(reply)
            self.tracer.emit(
                "llm_response",
                step=step,
                content=reply.content,
                tool_calls=[c.model_dump() for c in reply.tool_calls],
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
            )

            if not reply.tool_calls:
                return self._finish(
                    reply.content, "final_answer", step, n_calls, n_errors, p_tok, c_tok, history
                )

            for call in reply.tool_calls:
                n_calls += 1
                result = self.tools.execute(call)
                n_errors += not result.ok
                self.tracer.emit(
                    "tool_result", step=step, call=call.model_dump(), **result.model_dump()
                )
                add(
                    Message(
                        role="tool",
                        content=result.as_message_content(),
                        tool_name=call.name,
                        tool_call_id=call.id,
                    )
                )

        last = next((m.content for m in reversed(history) if m.role == "assistant"), "")
        return self._finish(
            last, "max_steps", self.max_steps, n_calls, n_errors, p_tok, c_tok, history
        )

    def _finish(
        self,
        answer: str,
        reason: StopReason,
        steps: int,
        n_calls: int,
        n_errors: int,
        p_tok: int,
        c_tok: int,
        messages: list[Message],
    ) -> AgentResult:
        result = AgentResult(answer, reason, steps, n_calls, n_errors, p_tok, c_tok, messages)
        result.compactions = getattr(self, "_compactions", 0)
        self.tracer.emit(
            "run_end",
            answer=answer,
            stop_reason=reason,
            steps=steps,
            tool_calls=n_calls,
            tool_errors=n_errors,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
        )
        return result
