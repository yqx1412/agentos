"""Long-term memory backed by DomainGraph (Project 2, milestone D5) instead of SQLite.

Same interface as :class:`agentos.memory.MemoryStore`, so the ``remember`` / ``recall`` /
``forget`` tools and :class:`~agentos.memory.MemoryAgent` work unchanged. Every call goes to
DomainGraph's MCP server (``add_fact``, ``recall_facts``, ``forget_fact``, ``list_facts``),
which keeps memories in Neo4j with bge-m3 embeddings. The difference that matters: recall is
by meaning, so a fact stored as "my manager is Dana" is found by "who is my boss?", which
SQLite's keyword search misses.

Memories live in a ``scope``. A benchmark run gives every task its own scope and deletes it
afterwards (:meth:`DomainGraphMemory.scoped`), the analogue of a fresh SQLite file per task.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from agentos.config import MCPServerConfig
from agentos.mcp_client import MCPManager
from agentos.memory import MAX_FACT_CHARS, Episode, Fact, context_block
from agentos.tools import ToolError

SERVER = "domaingraph"
EPISODE_CHARS = 1800  # DomainGraph's limit is 2000


class DomainGraphMemory:
    def __init__(
        self,
        mcp: MCPManager,
        *,
        scope: str = "default",
        server: str = SERVER,
        timeout: float = 60.0,
        owns: bool = False,
        clear_on_close: bool = False,
    ) -> None:
        self.mcp = mcp
        self.scope = scope
        self.server = server
        self.timeout = timeout
        self._owns = owns  # close the MCP connection on close()
        self._clear = clear_on_close  # delete this scope's memories on close()

    @classmethod
    def connect(
        cls, cfg: MCPServerConfig, *, scope: str = "default", errlog: Any = None
    ) -> DomainGraphMemory:
        """Start DomainGraph's MCP server and return a memory that owns the connection."""
        mcp = MCPManager({SERVER: cfg}, errlog=errlog).__enter__()
        return cls(mcp, scope=scope, timeout=cfg.timeout, owns=True)

    def scoped(self, prefix: str = "run") -> DomainGraphMemory:
        """A fresh, empty scope on the same connection, deleted when closed."""
        scope = f"{prefix}-{uuid.uuid4().hex[:12]}"
        return DomainGraphMemory(
            self.mcp, scope=scope, server=self.server, timeout=self.timeout, clear_on_close=True
        )

    def __enter__(self) -> DomainGraphMemory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._clear:
            self._clear = False
            self._call("clear_scope", scope=self.scope)
        if self._owns:
            self._owns = False
            self.mcp.close()

    # -- transport ----------------------------------------------------------------------

    def _call(self, tool: str, **args: Any) -> dict[str, Any]:
        text = self.mcp.call(self.server, tool, args, timeout=self.timeout)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolError(f"{self.server} {tool} returned non-JSON: {text[:200]!r}") from exc

    # -- facts ----------------------------------------------------------------------------

    @staticmethod
    def _fact(m: dict[str, Any]) -> Fact:
        return Fact(int(m["id"]), m["text"], "agent", float(m["created"]))

    def add_fact(self, text: str, source: str = "agent") -> tuple[Fact, bool]:
        text = " ".join(text.split())
        if not text:
            raise ValueError("fact is empty")
        if len(text) > MAX_FACT_CHARS:
            raise ValueError(f"fact is too long ({len(text)} chars, limit {MAX_FACT_CHARS})")
        try:
            m = self._call("add_fact", statement=text, scope=self.scope, kind="fact")
        except ToolError as exc:
            raise ValueError(str(exc)) from exc
        return self._fact(m), bool(m.get("new"))

    def get_fact(self, fact_id: int) -> Fact | None:
        return next((f for f in self.facts(1000) if f.id == fact_id), None)

    def forget(self, fact_id: int) -> bool:
        return bool(self._call("forget_fact", id=int(fact_id), scope=self.scope)["deleted"])

    def facts(self, limit: int = 100) -> list[Fact]:
        res = self._call("list_facts", scope=self.scope, kind="fact", limit=limit)
        return [self._fact(m) for m in res["memories"]]

    def search_facts(self, query: str, k: int = 5, *, touch: bool = False) -> list[Fact]:
        if not query.strip():
            return []
        res = self._call("recall_facts", query=query, scope=self.scope, k=k, kind="fact")
        return [self._fact(m) for m in res["memories"]]

    # -- episodes -----------------------------------------------------------------------

    @staticmethod
    def _episode(m: dict[str, Any]) -> Episode:
        d = m.get("data") or {}
        return Episode(
            int(m["id"]),
            d.get("task", m["text"]),
            d.get("outcome", ""),
            d.get("answer", ""),
            d.get("model", ""),
            d.get("agent", ""),
            float(m["created"]),
        )

    def add_episode(
        self, task: str, outcome: str, answer: str, *, model: str = "", agent: str = ""
    ) -> Episode:
        text = f"Task: {' '.join(task.split())}\nResult ({outcome}): {' '.join(answer.split())}"
        if len(text) > EPISODE_CHARS:
            text = text[: EPISODE_CHARS - 3] + "..."
        m = self._call(
            "add_fact",
            statement=text,
            scope=self.scope,
            kind="episode",
            details={
                "task": task,
                "outcome": outcome,
                "answer": answer,
                "model": model,
                "agent": agent,
            },
        )
        return self._episode(m)

    def episodes(self, limit: int = 100) -> list[Episode]:
        res = self._call("list_facts", scope=self.scope, kind="episode", limit=limit)
        return [self._episode(m) for m in res["memories"]]

    def search_episodes(self, query: str, k: int = 3) -> list[Episode]:
        if not query.strip():
            return []
        res = self._call("recall_facts", query=query, scope=self.scope, k=k, kind="episode")
        return [self._episode(m) for m in res["memories"]]

    # -- shared -------------------------------------------------------------------------

    def context_for(self, task: str, *, k_facts: int = 5, k_episodes: int = 3) -> str:
        return context_block(self, task, k_facts=k_facts, k_episodes=k_episodes)

    def describe(self) -> dict[str, Any]:
        return {
            "path": f"domaingraph scope {self.scope!r}",
            "facts": len(self.facts(1000)),
            "episodes": len(self.episodes(1000)),
        }
