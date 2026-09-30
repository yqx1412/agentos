"""Built-in tools: read_file, write_file and calculator.

File tools are confined to a workspace directory; paths that resolve outside it are refused.
"""

from __future__ import annotations

import ast
import math
import operator
from pathlib import Path

from pydantic import BaseModel, Field

from agentos.tools import Tool, ToolError

MAX_READ_BYTES = 200_000


def _resolve(workspace: Path, rel: str) -> Path:
    root = workspace.resolve()
    target = (root / rel).resolve()
    if not target.is_relative_to(root):
        raise ToolError(f"path {rel!r} is outside the workspace")
    return target


class ReadFileArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace")


class WriteFileArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace")
    content: str = Field(description="Full text to write; replaces any existing content")


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '(3 + 4) * sqrt(16)'")


def file_tools(workspace: Path) -> list[Tool]:
    def read_file(args: ReadFileArgs) -> str:
        p = _resolve(workspace, args.path)
        if not p.is_file():
            raise ToolError(f"file not found: {args.path}")
        data = p.read_bytes()
        if len(data) > MAX_READ_BYTES:
            raise ToolError(f"file too large ({len(data)} bytes, limit {MAX_READ_BYTES})")
        text = data.decode("utf-8", errors="replace")
        # Stats are cheap for us and error-prone for a model to compute itself.
        header = f"[{args.path}: {len(text.splitlines())} lines, {len(text.split())} words]"
        return f"{header}\n{text}"

    def write_file(args: WriteFileArgs) -> str:
        p = _resolve(workspace, args.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args.content, encoding="utf-8")
        return f"wrote {len(args.content)} characters to {args.path}"

    return [
        Tool("read_file", "Read a UTF-8 text file from the workspace.", ReadFileArgs, read_file),
        Tool(
            "write_file",
            "Write a UTF-8 text file in the workspace, creating or overwriting it. "
            "Missing parent directories are created automatically.",
            WriteFileArgs,
            write_file,
        ),
    ]


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000


def safe_eval(expression: str) -> float | int:
    """Evaluate arithmetic without ``eval``: only numbers, operators and whitelisted functions."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc

    def ev(node: ast.AST) -> float | int:
        match node:
            case ast.Expression(body=body):
                return ev(body)
            case ast.Constant(value=v) if isinstance(v, int | float) and not isinstance(v, bool):
                return v
            case ast.Name(id=name) if name in _CONSTS:
                return _CONSTS[name]
            case ast.BinOp(left=left, op=op, right=right) if type(op) in _BIN_OPS:
                lv, rv = ev(left), ev(right)
                if isinstance(op, ast.Pow) and abs(rv) > MAX_EXPONENT:
                    raise ToolError(f"exponent too large (limit {MAX_EXPONENT})")
                return _BIN_OPS[type(op)](lv, rv)
            case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY_OPS:
                return _UNARY_OPS[type(op)](ev(operand))
            case ast.Call(func=ast.Name(id=fname), args=args, keywords=[]) if fname in _FUNCS:
                return _FUNCS[fname](*(ev(a) for a in args))
        raise ToolError(f"unsupported syntax: {ast.unparse(node)!r}")

    try:
        return ev(tree)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except (ValueError, TypeError, OverflowError) as exc:
        raise ToolError(f"math error: {exc}") from exc


def calculator_tool() -> Tool:
    return Tool(
        "calculator",
        "Evaluate an arithmetic expression. Supports + - * / // % **, "
        "sqrt, log, exp, abs, round, min, max, pi, e.",
        CalculatorArgs,
        lambda args: str(safe_eval(args.expression)),
    )


def builtin_tools(workspace: Path) -> list[Tool]:
    return [*file_tools(workspace), calculator_tool()]
