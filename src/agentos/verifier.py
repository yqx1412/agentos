"""Step verification (A5): cheap deterministic checks first, then a model "reflection" call.

A step that says it is done is checked before its result is trusted:

1. ``text_tool_call``: the reply contains a tool call written as text, e.g.
   ``{"name": "write_file", ...}`` or ``write_file(path=...)``, and nothing ran it.
2. ``filename_as_content``: a bare file name passed to a parameter that takes content
   (``text``, ``content``, ...), e.g. ``word_frequency(text="essay.txt")``.
3. ``ungrounded``: a tool call used a number that appears nowhere in the evidence: the
   task, earlier steps' tool outputs, or this step's earlier tool outputs. That is a value
   the model made up instead of reading it. Only numbers >= 100 are checked (small numbers
   are too often legitimate constants), minus a few common unit constants.
4. ``reflection``: the model is shown the goal, the tool calls with their outputs and the
   reported result, and answers ``{"ok": ..., "reason": ...}``. An unparseable verdict
   counts as a pass: a broken verifier must not fail work that may be fine.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agentos.llm import LLM, Message
from agentos.planner import extract_json
from agentos.tools import ToolRegistry

MIN_CHECKED_NUMBER = 100
UNIT_CONSTANTS = {100.0, 360.0, 365.0, 1000.0, 1024.0, 1440.0, 3600.0, 86400.0}
_PLAIN = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")
_THOUSANDS = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])")
_ONE = re.compile(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+(?![\d,])|\d+)(?:\.\d+)?")

REFLECT_SYSTEM_PROMPT = """\
You check one step of an agent's work. Be strict about facts, not about wording.

Reply ok=false if ANY of these is true:
- the step goal was not achieved;
- the goal says to write a file and no write tool call succeeded;
- the reported result states a value that follows neither from this step's tool outputs
  nor from the earlier steps' results;
- a tool was given a file NAME where it needed the file's CONTENTS (or the reverse);
- the step claims to have done something that no tool call did.

These are fine and must NOT be rejected:
- a step with no tool calls that only reasons over earlier results (sorting, picking
  the largest, reformatting) when its result follows from them;
- rounding or formatting a number for display (3.3000000000000003 -> 3.30).
Otherwise reply ok=true.

Reply with ONLY a JSON object: {"ok": true, "reason": "..."}"""

MAX_CONTEXT_CHARS = 3000


@dataclass
class Verdict:
    ok: bool
    check: str  # "pass" | "text_tool_call" | "ungrounded" | "reflection" | "unparseable"
    reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


def numbers_in(text: str) -> set[float]:
    """Every number that ``text`` could mean, for evidence: "1,200" counts as 1200 AND as
    1 and 200, because in a CSV row it is two values. Lenient on purpose: a number wrongly
    seen as evidence only weakens the check, a missed one fails a correct step."""
    found = _PLAIN.findall(text) + _THOUSANDS.findall(text)
    return {float(m.replace(",", "")) for m in found}


def numbers_used(text: str) -> set[float]:
    """The numbers a tool call's arguments state, one reading per token."""
    return {float(m.replace(",", "")) for m in _ONE.findall(text)}


def _grounded(value: float, evidence: set[float]) -> bool:
    return any(abs(value - e) <= max(1e-6, 1e-6 * abs(e)) for e in evidence)


# Parameters that take content, not a path. A bare file name there is the A3 textkit
# failure: find_lines(text="code.py") searches the string "code.py" and finds nothing.
# "content" (write_file) is left out: writing a file name into a file is a normal request.
CONTENT_PARAMS = {"text", "data", "body"}
_FILENAME = re.compile(r"^[\w./\\-]+\.(?:txt|md|csv|json|py|log|yaml|yml|toml|ini|html)$", re.I)


def _filename_as_content(args_json: str) -> tuple[str, str] | None:
    try:
        args = json.loads(args_json)
    except ValueError:
        return None
    if not isinstance(args, dict):
        return None
    for key, value in args.items():
        if key.lower() in CONTENT_PARAMS and isinstance(value, str) and _FILENAME.match(value):
            return key, value
    return None


class Verifier:
    def __init__(self, llm: LLM, tools: ToolRegistry, *, reflect: bool = True) -> None:
        self.llm = llm
        self.tool_names = tools.names()
        self.reflect = reflect
        names = "|".join(re.escape(n) for n in sorted(self.tool_names, key=len, reverse=True))
        # Only the JSON shape counts: it is what a model emits when it means to call the
        # tool. "calculator(9*60+40)" in prose is usually an explanation, not an attempt.
        self._text_call = re.compile(rf'"name"\s*:\s*"(?:{names})"') if names else None

    def check(
        self,
        *,
        task: str,
        goal: str,
        evidence: str,
        messages: list[Message],
        answer: str,
        context: str = "",
    ) -> Verdict:
        """``evidence`` = raw tool outputs the step was allowed to take values from before it
        started; ``context`` = earlier steps' results, shown to the reflection call only."""
        calls, outputs = _calls_and_outputs(messages)

        if self._text_call and not calls:
            m = self._text_call.search(answer)
            if m:
                return Verdict(
                    False,
                    "text_tool_call",
                    f"the reply contains a tool call written as text ({m.group(0)!r}) and "
                    "no tool was actually called. Call the tool through the tool interface.",
                )

        known = numbers_in(task) | numbers_in(evidence) | UNIT_CONSTANTS
        for name, args, output in outputs:
            if output is None:  # a failed call did nothing; the model already got the error
                continue
            misuse = _filename_as_content(args)
            if misuse:
                param, value = misuse
                return Verdict(
                    False,
                    "filename_as_content",
                    f"{name} got the file name {value!r} as {param!r}, which expects the "
                    "text itself. Read the file first and pass its contents.",
                )
            used = numbers_used(args)
            made_up = sorted(
                n for n in used if abs(n) >= MIN_CHECKED_NUMBER and not _grounded(n, known)
            )
            if made_up:
                shown = ", ".join(f"{n:g}" for n in made_up[:5])
                return Verdict(
                    False,
                    "ungrounded",
                    f"{name} was called with {shown}, which appear in no file or tool output. "
                    "Read the inputs with a tool and use the values they contain.",
                )
            known |= numbers_in(output)

        if not self.reflect:
            return Verdict(True, "pass")
        return self._reflect(goal, calls, answer, context)

    def _reflect(self, goal: str, calls: list[str], answer: str, context: str) -> Verdict:
        transcript = "\n".join(calls) or "(no tool calls)"
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n..."
        earlier = f"Earlier steps' results and tool outputs:\n{context}\n\n" if context else ""
        resp = self.llm.chat(
            [
                Message(role="system", content=REFLECT_SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=f"{earlier}Step goal: {goal}\n\nTool calls in this step:\n"
                    f"{transcript}\n\nReported result: {answer}",
                ),
            ],
            [],
        )
        tokens = {"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens}
        try:
            data = extract_json(resp.message.content)
            if not isinstance(data, dict) or not isinstance(data.get("ok"), bool):
                raise ValueError("no boolean 'ok'")
        except ValueError:
            return Verdict(True, "unparseable", resp.message.content[:200], **tokens)
        reason = str(data.get("reason") or "")[:500]
        if data["ok"]:
            return Verdict(True, "pass", reason, **tokens)
        return Verdict(False, "reflection", reason or "the checker rejected the step", **tokens)


def _calls_and_outputs(
    messages: list[Message], limit: int = 600
) -> tuple[list[str], list[tuple[str, str, str | None]]]:
    """(transcript lines, [(tool, args json, output or None if it errored)]) in call order."""
    results = {m.tool_call_id: m.content for m in messages if m.role == "tool"}
    lines: list[str] = []
    outputs: list[tuple[str, str, str | None]] = []
    for m in messages:
        for c in m.tool_calls:
            args = (
                c.arguments
                if isinstance(c.arguments, str)
                else json.dumps(c.arguments, ensure_ascii=False)
            )
            out = results.get(c.id, "")
            shown = out if len(out) <= limit else out[:limit] + "..."
            lines.append(f"- {c.name}({args}) -> {shown}")
            ok = not out.startswith("ERROR:")
            outputs.append((c.name, args, out if ok else None))
    return lines, outputs
