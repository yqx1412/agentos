from pathlib import Path

import pytest
from pydantic import BaseModel

from agentos.builtin_tools import builtin_tools, safe_eval
from agentos.llm import ToolCall
from agentos.tools import Tool, ToolError, ToolRegistry


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(builtin_tools(tmp_path))


def call(name: str, arguments: dict | str) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


def test_schemas_are_openai_style(registry: ToolRegistry) -> None:
    by_name = {s["function"]["name"]: s for s in registry.schemas()}
    assert set(by_name) == {"read_file", "write_file", "calculator"}
    params = by_name["write_file"]["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"path", "content"}
    assert "title" not in params


def test_write_then_read(registry: ToolRegistry, tmp_path: Path) -> None:
    w = registry.execute(call("write_file", {"path": "sub/a.txt", "content": "one two three"}))
    assert w.ok
    assert (tmp_path / "sub" / "a.txt").read_text() == "one two three"
    r = registry.execute(call("read_file", {"path": "sub/a.txt"}))
    assert r.ok
    assert "3 words" in r.output
    assert r.output.endswith("one two three")


@pytest.mark.parametrize("path", ["../escape.txt", "sub/../../escape.txt"])
def test_paths_outside_workspace_are_refused(registry: ToolRegistry, path: str) -> None:
    res = registry.execute(call("write_file", {"path": path, "content": "x"}))
    assert not res.ok
    assert "outside the workspace" in res.output


def test_absolute_path_outside_workspace_is_refused(registry: ToolRegistry, tmp_path: Path) -> None:
    outside = tmp_path.parent / "elsewhere.txt"
    res = registry.execute(call("read_file", {"path": str(outside)}))
    assert not res.ok
    assert "outside the workspace" in res.output


def test_missing_file(registry: ToolRegistry) -> None:
    res = registry.execute(call("read_file", {"path": "nope.txt"}))
    assert not res.ok
    assert res.as_message_content().startswith("ERROR: file not found")


@pytest.mark.parametrize(
    ("expr", "expected"),
    [("2 + 3 * 4", 14), ("(3 + 4) * sqrt(16)", 28.0), ("-2 ** 2", -4), ("max(1, 7, 3)", 7)],
)
def test_calculator(expr: str, expected: float) -> None:
    assert safe_eval(expr) == expected


@pytest.mark.parametrize(
    "expr", ["__import__('os')", "open('x')", "2 ** 100000", "1 / 0", "True + 1", "x"]
)
def test_calculator_rejects_unsafe_or_invalid(expr: str) -> None:
    with pytest.raises(ToolError):
        safe_eval(expr)


# --- malformed tool calls become error results, never exceptions ---


def test_unknown_tool(registry: ToolRegistry) -> None:
    res = registry.execute(call("delete_everything", {}))
    assert not res.ok
    assert "unknown tool" in res.output
    assert "read_file" in res.output  # lists what is available


def test_missing_required_argument(registry: ToolRegistry) -> None:
    res = registry.execute(call("write_file", {"path": "a.txt"}))
    assert not res.ok
    assert "content" in res.output
    assert "Field required" in res.output


def test_wrong_argument_type(registry: ToolRegistry) -> None:
    res = registry.execute(call("calculator", {"expression": ["1", "+", "1"]}))
    assert not res.ok
    assert "expression" in res.output


def test_string_arguments_are_parsed(registry: ToolRegistry) -> None:
    res = registry.execute(call("calculator", '{"expression": "6 * 7"}'))
    assert res.ok
    assert res.output == "42"


def test_invalid_json_arguments(registry: ToolRegistry) -> None:
    res = registry.execute(call("calculator", '{"expression": "6 * 7"'))
    assert not res.ok
    assert "not valid JSON" in res.output


def test_non_object_arguments(registry: ToolRegistry) -> None:
    res = registry.execute(call("calculator", "[1, 2]"))
    assert not res.ok
    assert "JSON object" in res.output


def test_tool_crash_is_contained() -> None:
    class NoArgs(BaseModel):
        pass

    def boom(_: NoArgs) -> str:
        raise RuntimeError("kaboom")

    reg = ToolRegistry([Tool("boom", "always fails", NoArgs, boom)])
    res = reg.execute(call("boom", {}))
    assert not res.ok
    assert "RuntimeError: kaboom" in res.output


def test_duplicate_registration_rejected(tmp_path: Path) -> None:
    reg = ToolRegistry(builtin_tools(tmp_path))
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(builtin_tools(tmp_path)[0])
