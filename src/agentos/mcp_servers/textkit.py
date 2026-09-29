"""textkit: a small MCP server with text-analysis tools.

Run it standalone with ``python -m agentos.mcp_servers.textkit`` (stdio transport).
Expected failures raise the SDK's ``ToolError`` so their message reaches the client; the
SDK deliberately hides the text of any other exception.
"""

from __future__ import annotations

import re
from collections import Counter

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("textkit", instructions="Deterministic text statistics and search.")

_WORD = re.compile(r"[A-Za-z0-9']+")


@server.tool()
def text_stats(text: str) -> dict[str, int]:
    """Count the lines, words and characters in a piece of text."""
    return {"lines": len(text.splitlines()), "words": len(text.split()), "chars": len(text)}


@server.tool()
def word_frequency(text: str, top_n: int = 5) -> dict[str, list[dict[str, int | str]]]:
    """Return the most frequent words (case-insensitive), most common first."""
    if top_n < 1:
        raise ToolError("top_n must be >= 1")
    counts = Counter(w.lower() for w in _WORD.findall(text))
    return {"top": [{"word": w, "count": c} for w, c in counts.most_common(top_n)]}


@server.tool()
def find_lines(text: str, pattern: str) -> dict[str, list[str]]:
    """Return the lines matching a regular expression, prefixed with their 1-based number."""
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise ToolError(f"invalid regex: {exc}") from exc
    lines = text.splitlines()
    return {"matches": [f"{i}: {line}" for i, line in enumerate(lines, 1) if rx.search(line)]}


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
