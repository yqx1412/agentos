"""Tool registry: Pydantic argument models become JSON tool schemas; calls are validated."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from agentos.llm import ToolCall

# Permission levels (A7), lowest first. ``read`` tools only look; ``write`` tools change
# the workspace or memory; ``dangerous`` tools run code or commands and need a human yes
# unless the policy auto-approves that level.
PERMISSION_LEVELS = ("read", "write", "dangerous")


class ToolError(Exception):
    """Raised by a tool for an expected, user-facing failure (bad path, bad input...)."""


Approver = Callable[["Tool", dict[str, Any]], bool]


@dataclass(frozen=True)
class Policy:
    """Which tool calls run without asking.

    Calls at or below ``auto_approve`` run directly. Higher ones go to ``approver`` (a human
    prompt in the CLI); with no approver they are denied. Default: read and write run,
    dangerous is denied.
    """

    auto_approve: str = "write"
    approver: Approver | None = None

    def __post_init__(self) -> None:
        if self.auto_approve not in PERMISSION_LEVELS:
            raise ValueError(f"auto_approve must be one of {PERMISSION_LEVELS}")

    def allows(self, tool: Tool, args: dict[str, Any]) -> tuple[bool, str]:
        level = PERMISSION_LEVELS.index(tool.permission)
        if level <= PERMISSION_LEVELS.index(self.auto_approve):
            return True, ""
        if self.approver is None:
            return False, (
                f"{tool.name} is a {tool.permission!r} tool and needs approval, which this "
                "run cannot ask for. Do the task with other tools."
            )
        if self.approver(tool, args):
            return True, ""
        return False, f"the user did not approve this {tool.name} call"


class ToolResult(BaseModel):
    ok: bool
    output: str

    def as_message_content(self) -> str:
        return self.output if self.ok else f"ERROR: {self.output}"


@dataclass(frozen=True)
class Tool:
    """A callable tool.

    Local tools set ``args_model``: arguments are validated here and ``fn`` receives the model.
    Remote tools (MCP) set ``args_model=None`` and ``input_schema``: ``fn`` receives the raw
    argument dict and validation is left to the remote side, whose errors come back as results.
    """

    name: str
    description: str
    args_model: type[BaseModel] | None
    fn: Callable[[Any], Any]
    input_schema: dict[str, Any] | None = None
    source: str = "builtin"
    permission: str = "read"

    def __post_init__(self) -> None:
        if self.args_model is None and self.input_schema is None:
            raise ValueError(f"tool {self.name!r} needs args_model or input_schema")
        if self.permission not in PERMISSION_LEVELS:
            raise ValueError(f"tool {self.name!r}: permission must be one of {PERMISSION_LEVELS}")

    def schema(self) -> dict[str, Any]:
        if self.args_model is not None:
            params = self.args_model.model_json_schema()
            params.pop("title", None)
        else:
            params = dict(self.input_schema or {})
            params.setdefault("type", "object")
            params.setdefault("properties", {})
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None, *, policy: Policy | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self.policy = policy or Policy()
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """Run a tool call. Never raises: every failure becomes an error result for the model."""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                ok=False,
                output=f"unknown tool {call.name!r}. Available tools: {', '.join(self.names())}",
            )

        raw = call.arguments
        if isinstance(raw, str):
            try:
                raw = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as exc:
                return ToolResult(ok=False, output=f"arguments are not valid JSON: {exc}")
        if not isinstance(raw, dict):
            return ToolResult(ok=False, output="arguments must be a JSON object")

        args: Any = raw
        if tool.args_model is not None:
            try:
                args = tool.args_model.model_validate(raw)
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                    for e in exc.errors()
                )
                return ToolResult(ok=False, output=f"invalid arguments for {tool.name}: {problems}")

        # Checked after validation, so the approver is shown the arguments that will run.
        shown = args.model_dump() if isinstance(args, BaseModel) else args
        try:
            allowed, why = self.policy.allows(tool, shown)
        except Exception as exc:  # a broken approver denies; it never runs the call
            allowed, why = False, f"approval failed: {type(exc).__name__}: {exc}"
        if not allowed:
            return ToolResult(ok=False, output=f"permission denied: {why}")

        try:
            out = tool.fn(args)
        except ToolError as exc:
            return ToolResult(ok=False, output=str(exc))
        except Exception as exc:  # a tool bug must not kill the agent loop
            return ToolResult(ok=False, output=f"{tool.name} crashed: {type(exc).__name__}: {exc}")
        return ToolResult(ok=True, output=out if isinstance(out, str) else json.dumps(out))
