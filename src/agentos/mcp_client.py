"""MCP client: connect to MCP servers over stdio and expose their tools to the agent.

The agent loop is synchronous, the MCP SDK is async. ``MCPManager`` owns a private asyncio
loop on a background thread. One long-lived task opens every server connection and keeps
them open until shutdown (anyio cancel scopes must be exited by the task that entered
them); tool calls are submitted to that loop from the agent's thread.

MCP tools are registered as ``<server>__<tool>`` so servers cannot collide with each other
or with built-in tools.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import sys
import threading
from contextlib import AsyncExitStack
from typing import Any, TextIO

import anyio
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client

from agentos.config import MCPServerConfig
from agentos.tools import Tool, ToolError

NAME_SEP = "__"
SHUTDOWN_TIMEOUT = 10.0


class MCPError(Exception):
    """An MCP server could not be started or listed."""


def _describe(exc: BaseException) -> str:
    """Flatten anyio exception groups into the underlying messages."""
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_describe(e) for e in exc.exceptions)
    msg = str(exc)
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def _as_mcp_error(exc: BaseException) -> MCPError:
    if isinstance(exc, MCPError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        found = [e for e in exc.exceptions if isinstance(e, MCPError)]
        if found:
            return found[0]
    return MCPError(_describe(exc))


def render_result(result: Any) -> str:
    """Turn a CallToolResult into text for the model. Raises ToolError for error results."""
    if not isinstance(result, types.CallToolResult):
        raise ToolError(f"unsupported MCP result type: {type(result).__name__}")
    parts: list[str] = []
    for item in result.content:
        if isinstance(item, types.TextContent):
            parts.append(item.text)
        elif isinstance(item, types.ImageContent | types.AudioContent):
            mime = getattr(item, "mime_type", "?")
            parts.append(f"[{item.type} ({mime}) omitted: {len(item.data)} base64 chars]")
        else:
            parts.append(f"[{getattr(item, 'type', type(item).__name__)} content omitted]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False))
    text = "\n".join(parts)
    if result.is_error:
        raise ToolError(text or "the MCP tool reported an error")
    return text


class MCPManager:
    """Context manager that connects to MCP servers and hands out their tools.

    Startup is all-or-nothing: if any enabled server fails to start, ``__enter__`` raises
    ``MCPError`` naming it, so a benchmark never silently runs with fewer tools.
    """

    def __init__(self, servers: dict[str, MCPServerConfig], *, errlog: TextIO | None = None):
        self.servers = servers
        self.errlog = errlog or sys.stderr
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._main: concurrent.futures.Future[None] | None = None
        self._ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._stop = asyncio.Event()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: list[Tool] = []

    # -- lifecycle ---------------------------------------------------------------------

    def __enter__(self) -> MCPManager:
        if not self.servers:
            return self
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="agentos-mcp", daemon=True
        )
        self._thread.start()
        self._main = asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        budget = sum(c.startup_timeout for c in self.servers.values()) + SHUTDOWN_TIMEOUT
        try:
            self._ready.result(timeout=budget)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._stop.set)
        if self._main is not None:
            # Shutdown errors are not actionable for the caller.
            with contextlib.suppress(BaseException):
                self._main.result(timeout=SHUTDOWN_TIMEOUT)
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=SHUTDOWN_TIMEOUT)
        if not loop.is_running():
            loop.close()
        self._loop = None

    async def _run(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                for name, cfg in self.servers.items():
                    try:
                        await self._connect(stack, name, cfg)
                    except MCPError:
                        raise
                    except TimeoutError as exc:
                        raise MCPError(
                            f"MCP server {name!r} did not start within {cfg.startup_timeout}s"
                        ) from exc
                    except Exception as exc:
                        raise MCPError(
                            f"MCP server {name!r} failed to start: {_describe(exc)}"
                        ) from exc
                self._ready.set_result(None)
                await self._stop.wait()
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(_as_mcp_error(exc))
            else:
                raise

    async def _connect(self, stack: AsyncExitStack, name: str, cfg: MCPServerConfig) -> None:
        params = StdioServerParameters(
            command=cfg.command, args=cfg.args, env=cfg.env or None, cwd=cfg.cwd
        )
        # Entering these contexts opens task groups that stay open until shutdown, so no
        # cancel scope may wrap them; the timeout covers the handshake and listing only.
        read, write = await stack.enter_async_context(stdio_client(params, errlog=self.errlog))
        session = await stack.enter_async_context(ClientSession(read, write))
        with anyio.fail_after(cfg.startup_timeout):
            await session.initialize()
            listed = await self._list_tools(session)
        self._sessions[name] = session

        by_name = {t.name: t for t in listed}
        if cfg.tools is not None:
            missing = [t for t in cfg.tools if t not in by_name]
            if missing:
                raise MCPError(
                    f"MCP server {name!r} has no tool(s) {missing}; available: {sorted(by_name)}"
                )
            selected = [by_name[t] for t in cfg.tools]
        else:
            selected = listed
        self._tools.extend(self._make_tool(name, cfg, t) for t in selected)

    # -- tools -------------------------------------------------------------------------

    @staticmethod
    async def _list_tools(session: ClientSession) -> list[types.Tool]:
        listed: list[types.Tool] = []
        cursor: str | None = None
        while True:
            params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
            page = await session.list_tools(params=params)
            listed.extend(page.tools)
            cursor = page.next_cursor
            if not cursor:
                return listed

    def tools(self) -> list[Tool]:
        return list(self._tools)

    def _make_tool(self, server: str, cfg: MCPServerConfig, t: types.Tool) -> Tool:
        def fn(args: dict[str, Any]) -> str:
            return self.call(server, t.name, args, timeout=cfg.timeout)

        return Tool(
            name=f"{server}{NAME_SEP}{t.name}",
            description=t.description or t.title or t.name,
            args_model=None,
            fn=fn,
            input_schema=dict(t.input_schema),
            source=f"mcp:{server}",
        )

    def call(self, server: str, tool: str, args: dict[str, Any], *, timeout: float) -> str:
        """Call a tool synchronously. Every failure surfaces as ToolError for the model."""
        if self._loop is None or server not in self._sessions:
            raise ToolError(f"MCP server {server!r} is not connected")
        session = self._sessions[server]
        fut = asyncio.run_coroutine_threadsafe(
            session.call_tool(tool, args, read_timeout_seconds=timeout), self._loop
        )
        try:
            result = fut.result(timeout=timeout + 5)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            raise ToolError(f"{server}{NAME_SEP}{tool} timed out after {timeout}s") from exc
        except Exception as exc:
            raise ToolError(f"MCP server {server!r} error: {_describe(exc)}") from exc
        return render_result(result)
