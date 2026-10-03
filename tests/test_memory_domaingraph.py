"""DomainGraphMemory against an in-process fake of DomainGraph's MCP tools (the real server
is exercised by domaingraph's own tests and the live runs in the A6/D5 write-up)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from agentos.bench.runner import sqlite_memory
from agentos.memory import MemoryAgent, MemoryStore, memory_tools
from agentos.memory_domaingraph import DomainGraphMemory
from agentos.tools import ToolError


class FakeDomainGraph:
    """Implements add_fact / recall_facts / forget_fact / list_facts / clear_scope like the
    real server, with word-overlap 'similarity' (plus one synonym) instead of embeddings."""

    SYNONYMS: ClassVar[dict[str, str]] = {"boss": "manager"}

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.seq = 0
        self.clock = 1000.0
        self.calls: list[tuple[str, dict]] = []

    def _words(self, text: str) -> set[str]:
        ws = {w.strip(".,?!:").lower() for w in text.split()}
        return {self.SYNONYMS.get(w, w) for w in ws if len(w) > 3}

    def call(self, server: str, tool: str, args: dict, *, timeout: float) -> str:
        self.calls.append((tool, args))
        scope = args.get("scope", "default")
        if tool == "add_fact":
            for r in self.rows:
                if (r["scope"], r["kind"], r["text"]) == (scope, args["kind"], args["statement"]):
                    return json.dumps({**r, "new": False})
            self.seq += 1
            self.clock += 1
            r = {
                "id": self.seq,
                "text": args["statement"],
                "kind": args["kind"],
                "scope": scope,
                "created": self.clock,
            }
            if args.get("details"):
                r["data"] = args["details"]
            self.rows.append(r)
            return json.dumps({**r, "new": True})
        mine = [r for r in self.rows if r["scope"] == scope]
        if args.get("kind"):
            mine = [r for r in mine if r["kind"] == args["kind"]]
        if tool == "recall_facts":
            q = self._words(args["query"])
            hits = [r for r in mine if q & self._words(r["text"])][: args.get("k", 5)]
            return json.dumps({"memories": sorted(hits, key=lambda r: -r["created"])})
        if tool == "list_facts":
            return json.dumps({"memories": sorted(mine, key=lambda r: -r["created"])})
        if tool == "forget_fact":
            before = len(self.rows)
            self.rows = [
                r for r in self.rows if not (r["id"] == args["id"] and r["scope"] == scope)
            ]
            return json.dumps({"deleted": len(self.rows) < before, "id": args["id"]})
        if tool == "clear_scope":
            before = len(self.rows)
            self.rows = [r for r in self.rows if r["scope"] != scope]
            return json.dumps({"deleted": before - len(self.rows)})
        raise ToolError(f"unknown tool {tool}")

    def close(self) -> None:
        pass


def test_facts_roundtrip_and_meaning_search():
    fake = FakeDomainGraph()
    mem = DomainGraphMemory(fake, scope="u")
    f, new = mem.add_fact("My manager is Dana.")
    again, new_again = mem.add_fact("My   manager is Dana.")  # whitespace normalized
    assert new and not new_again and again.id == f.id
    assert [x.text for x in mem.search_facts("Who is my boss?")] == ["My manager is Dana."]
    assert mem.get_fact(f.id).text == "My manager is Dana."
    assert mem.forget(f.id) and not mem.forget(f.id)
    assert mem.facts() == []
    with pytest.raises(ValueError, match="empty"):
        mem.add_fact("   ")


def test_episodes_keep_their_fields():
    mem = DomainGraphMemory(FakeDomainGraph(), scope="u")
    ep = mem.add_episode("Count the words in essay.txt", "completed", "412", model="m", agent="a")
    assert (ep.task, ep.outcome, ep.answer, ep.model, ep.agent) == (
        "Count the words in essay.txt", "completed", "412", "m", "a",
    )  # fmt: skip
    assert [e.answer for e in mem.search_episodes("words essay.txt")] == ["412"]
    assert mem.describe()["episodes"] == 1


def test_scoped_memories_are_isolated_and_cleared_on_close():
    fake = FakeDomainGraph()
    base = DomainGraphMemory(fake, scope="default")
    base.add_fact("The default scope keeps this.")
    with base.scoped("bench-x") as run:
        run.add_fact("Only this run sees this.")
        assert [f.text for f in run.facts()] == ["Only this run sees this."]
        assert run.scope.startswith("bench-x-")
    assert [r["text"] for r in fake.rows] == ["The default scope keeps this."]
    assert ("clear_scope", {"scope": run.scope}) in fake.calls


def test_context_block_and_tools_work_on_either_backend(tmp_path):
    for store in (DomainGraphMemory(FakeDomainGraph(), scope="u"), sqlite_memory(tmp_path, "t")):
        store.add_fact("The deploy server is build-07, port 8443.")
        block = store.context_for("Write the deploy server to deploy.txt")
        assert "build-07" in block and block.startswith("Memory from earlier sessions")
        tools = {t.name: t for t in memory_tools(store)}
        out = tools["recall"].fn(tools["recall"].args_model(query="deploy server"))
        assert "build-07" in out
        store.close()
    assert isinstance(sqlite_memory(tmp_path, "t"), MemoryStore)


def test_memory_agent_records_episode_in_domaingraph():
    class Done:
        def run(self, task):
            from types import SimpleNamespace

            return SimpleNamespace(stop_reason="final_answer", answer=" 42 ")

    fake = FakeDomainGraph()
    mem = DomainGraphMemory(fake, scope="u")
    mem.add_fact("The answer to the riddle is 42.")
    MemoryAgent(Done(), mem, inject=True, model="m", agent_kind="plain").run("Solve the riddle")
    eps = mem.episodes()
    assert [(e.task, e.outcome, e.answer) for e in eps] == [("Solve the riddle", "completed", "42")]


def test_non_json_reply_is_a_tool_error():
    class Broken(FakeDomainGraph):
        def call(self, *a, **k):
            return "oops"

    with pytest.raises(ToolError, match="non-JSON"):
        DomainGraphMemory(Broken()).facts()


def test_cli_needs_a_domaingraph_entry(tmp_path: Path):
    from agentos.cli import _connect_domaingraph
    from agentos.config import AgentOSConfig, ConfigError

    with pytest.raises(ConfigError, match=r"mcp_servers\.domaingraph"):
        _connect_domaingraph(AgentOSConfig(), tmp_path)
