"""Tool registry: Pydantic argument models become JSON tool schemas; calls are validated."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from agentos.llm import ToolCall


class ToolError(Exception):
    """Raised by a tool for an expected, user-facing failure (bad path, bad input...)."""


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

    def __post_init__(self) -> None:
        if self.args_model is None and self.input_schema is None:
            raise ValueError(f"tool {self.name!r} needs args_model or input_schema")

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
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
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

        try:
            out = tool.fn(args)
        except ToolError as exc:
            return ToolResult(ok=False, output=str(exc))
        except Exception as exc:  # a tool bug must not kill the agent loop
            return ToolResult(ok=False, output=f"{tool.name} crashed: {type(exc).__name__}: {exc}")
        return ToolResult(ok=True, output=out if isinstance(out, str) else json.dumps(out))
