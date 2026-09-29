"""A misbehaving MCP server for tests: a slow tool, a crashing tool and a failing tool."""

import time

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("flaky")


@server.tool()
def sleep(seconds: float) -> str:
    """Sleep, then return."""
    time.sleep(seconds)
    return "done"


@server.tool()
def fail(message: str) -> str:
    """Fail with an expected, user-facing error."""
    raise ToolError(message)


@server.tool()
def exit_process() -> str:
    """Kill the server process mid-call."""
    import os

    os._exit(1)


if __name__ == "__main__":
    server.run()
