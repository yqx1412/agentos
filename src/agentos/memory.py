"""Memory (A6): long-term facts and past task outcomes in SQLite, plus short-term compaction.

Long-term memory lives in one SQLite file with two tables, each with an FTS5 index:

- ``facts``: self-contained statements ("The deploy server is build-07, port 8443."), written
  by the model through the ``remember`` tool or by the user with ``agentos memory add``.
- ``episodes``: one row per finished run (task, outcome, final answer), written automatically.

Lookup is keyword search (FTS5, Porter stemming, BM25 ranking). It is deliberately simple:
DomainGraph replaces this store with graph + vector retrieval later.

:class:`MemoryAgent` wraps any agent (plain or planning). With ``inject=True`` it searches
memory with the task text and prepends the hits to the prompt before the run; after the run
it records the episode. The memory tools are added to the registry either way, so the model
can also store and look things up itself.

Short-term memory is :func:`compact_messages`: when a conversation grows past a token
budget, the older middle of it is replaced by a model-written summary.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from agentos.llm import LLM, Message
from agentos.tools import Tool, ToolError, ToolRegistry

if TYPE_CHECKING:
    from agentos.agent import AgentResult, Tracer

MAX_FACT_CHARS = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    norm TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL DEFAULT 'agent',
    created REAL NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    text, content='facts', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    task TEXT NOT NULL,
    outcome TEXT NOT NULL,
    answer TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    agent TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    task, answer, content='episodes', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, task, answer) VALUES (new.id, new.task, new.answer);
END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, task, answer)
    VALUES ('delete', old.id, old.task, old.answer);
END;
"""

# Words too common to say anything about relevance. With an OR query, one of these alone
# would match nearly every stored row.
_STOPWORD_TEXT = """
a an and are as at be but by can did do does for from had has have how i if in into is
it its me my no not of on or our so that the their them then there these this to was we
were what when where which who why will with you your write read file txt just only
please tell give use using after before all also any each than
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_WORD = re.compile(r"[A-Za-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def fts_query(text: str) -> str | None:
    """An FTS5 OR-query from free text: quoted terms, stopwords and 1-char words dropped."""
    terms = list(
        dict.fromkeys(w for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS)
    )
    return " OR ".join(f'"{t}"' for t in terms) if terms else None


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


@dataclass(frozen=True)
class Fact:
    id: int
    text: str
    source: str
    created: float
    uses: int = 0


@dataclass(frozen=True)
class Episode:
    id: int
    task: str
    outcome: str
    answer: str
    model: str
    agent: str
    created: float


class MemoryStore:
    """Long-term memory in one SQLite file. Use as a context manager or call :meth:`close`."""

    def __init__(self, path: Path | str, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path) if path != ":memory:" else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        # Monotonic within a process even if the wall clock is coarse: ordering by recency
        # decides which of two conflicting facts is newer.
        self._clock = clock
        self._last = 0.0

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def _now(self) -> float:
        now = max(self._clock(), self._last + 1e-3)
        self._last = now
        return now

    # -- facts -------------------------------------------------------------------------

    def add_fact(self, text: str, source: str = "agent") -> tuple[Fact, bool]:
        """Store a fact. Returns ``(fact, created)``; a duplicate returns the existing one."""
        text = " ".join(text.split())
        if not text:
            raise ValueError("fact is empty")
        if len(text) > MAX_FACT_CHARS:
            raise ValueError(f"fact is too long ({len(text)} chars, limit {MAX_FACT_CHARS})")
        norm = _norm(text)
        row = self._db.execute("SELECT * FROM facts WHERE norm = ?", (norm,)).fetchone()
        if row is not None:
            return _fact(row), False
        with self._db:
            cur = self._db.execute(
                "INSERT INTO facts(text, norm, source, created) VALUES (?, ?, ?, ?)",
                (text, norm, source, self._now()),
            )
        return self.get_fact(int(cur.lastrowid or 0)), True  # type: ignore[return-value]

    def get_fact(self, fact_id: int) -> Fact | None:
        row = self._db.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return _fact(row) if row else None

    def forget(self, fact_id: int) -> bool:
        with self._db:
            cur = self._db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return cur.rowcount > 0

    def facts(self, limit: int = 100) -> list[Fact]:
        rows = self._db.execute(
            "SELECT * FROM facts ORDER BY created DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_fact(r) for r in rows]

    def search_facts(self, query: str, k: int = 5, *, touch: bool = False) -> list[Fact]:
        """Best keyword matches, newest first among the top ``k``."""
        q = fts_query(query)
        if q is None:
            return []
        rows = self._db.execute(
            "SELECT f.* FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
            "WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts) LIMIT ?",
            (q, k),
        ).fetchall()
        hits = sorted((_fact(r) for r in rows), key=lambda f: f.created, reverse=True)
        if touch and hits:
            with self._db:
                self._db.executemany(
                    "UPDATE facts SET uses = uses + 1 WHERE id = ?", [(f.id,) for f in hits]
                )
        return hits

    # -- episodes ----------------------------------------------------------------------

    def add_episode(
        self, task: str, outcome: str, answer: str, *, model: str = "", agent: str = ""
    ) -> Episode:
        with self._db:
            cur = self._db.execute(
                "INSERT INTO episodes(task, outcome, answer, model, agent, created) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (task, outcome, answer, model, agent, self._now()),
            )
        row = self._db.execute("SELECT * FROM episodes WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _episode(row)

    def episodes(self, limit: int = 100) -> list[Episode]:
        rows = self._db.execute(
            "SELECT * FROM episodes ORDER BY created DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_episode(r) for r in rows]

    def search_episodes(self, query: str, k: int = 3) -> list[Episode]:
        q = fts_query(query)
        if q is None:
            return []
        rows = self._db.execute(
            "SELECT e.* FROM episodes_fts JOIN episodes e ON e.id = episodes_fts.rowid "
            "WHERE episodes_fts MATCH ? ORDER BY bm25(episodes_fts) LIMIT ?",
            (q, k),
        ).fetchall()
        return sorted((_episode(r) for r in rows), key=lambda e: e.created, reverse=True)

    # -- prompt context ----------------------------------------------------------------

    def context_for(self, task: str, *, k_facts: int = 5, k_episodes: int = 3) -> str:
        """The memory block prepended to a task, or "" when nothing matches."""
        facts = self.search_facts(task, k_facts, touch=True)
        episodes = self.search_episodes(task, k_episodes)
        if not facts and not episodes:
            return ""
        lines = [
            "Memory from earlier sessions (newest first; it may or may not be relevant, and "
            "when two entries conflict the newer one is correct):"
        ]
        lines += [f"- fact #{f.id} ({_day(f.created)}): {f.text}" for f in facts]
        lines += [
            f"- past task ({_day(e.created)}, {e.outcome}): {_clip(e.task, 200)} "
            f"-> {_clip(e.answer, 300)}"
            for e in episodes
        ]
        return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _fact(r: sqlite3.Row) -> Fact:
    return Fact(r["id"], r["text"], r["source"], r["created"], r["uses"])


def _episode(r: sqlite3.Row) -> Episode:
    return Episode(
        r["id"], r["task"], r["outcome"], r["answer"], r["model"], r["agent"], r["created"]
    )


# -- tools -----------------------------------------------------------------------------


class RememberArgs(BaseModel):
    fact: str = Field(
        description="One self-contained fact, e.g. 'The deploy server is build-07, port 8443.'"
    )


class RecallArgs(BaseModel):
    query: str = Field(description="Keywords to search for, e.g. 'deploy server port'")


class ForgetArgs(BaseModel):
    id: int = Field(description="The fact id shown by recall, e.g. 3")


def memory_tools(store: MemoryStore) -> list[Tool]:
    def remember(args: RememberArgs) -> str:
        try:
            fact, created = store.add_fact(args.fact)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return f"stored as fact #{fact.id}" if created else f"already stored as fact #{fact.id}"

    def recall(args: RecallArgs) -> str:
        facts = store.search_facts(args.query, 8, touch=True)
        episodes = store.search_episodes(args.query, 3)
        if not facts and not episodes:
            return "no matching memories"
        lines = [f"fact #{f.id} ({_day(f.created)}): {f.text}" for f in facts]
        lines += [
            f"past task ({_day(e.created)}, {e.outcome}): {_clip(e.task, 200)} "
            f"-> {_clip(e.answer, 300)}"
            for e in episodes
        ]
        return "\n".join(lines)

    def forget(args: ForgetArgs) -> str:
        if not store.forget(args.id):
            raise ToolError(f"no fact #{args.id}")
        return f"deleted fact #{args.id}"

    return [
        Tool(
            "remember",
            "Save a fact to long-term memory so that future sessions can use it. "
            "Store one self-contained fact per call, with names and numbers spelled out.",
            RememberArgs,
            remember,
            source="memory",
            permission="write",
        ),
        Tool(
            "recall",
            "Search long-term memory: facts saved in earlier sessions and results of past tasks.",
            RecallArgs,
            recall,
            source="memory",
        ),
        Tool(
            "forget",
            "Delete a stored fact by its id, e.g. when it is outdated or wrong.",
            ForgetArgs,
            forget,
            source="memory",
            permission="write",
        ),
    ]


# -- agent wrapper ---------------------------------------------------------------------


class _Runner(Protocol):
    def run(self, task: str) -> AgentResult: ...


MEMORY_TASK = """\
{memory}

Task: {task}"""


class MemoryAgent:
    """Wraps an agent: memory block before the run (``inject``), episode record after it.

    The memory tools must already be in the wrapped agent's registry; see
    :func:`with_memory_tools`.
    """

    def __init__(
        self,
        inner: _Runner,
        store: MemoryStore,
        *,
        inject: bool = True,
        model: str = "",
        agent_kind: str = "",
        tracer: Tracer | None = None,
    ) -> None:
        self.inner = inner
        self.store = store
        self.inject = inject
        self.model = model
        self.agent_kind = agent_kind
        self.tracer = tracer

    def run(self, task: str) -> AgentResult:
        prompt = task
        if self.inject:
            block = self.store.context_for(task)
            if block:
                prompt = MEMORY_TASK.format(memory=block, task=task)
            if self.tracer is not None:
                self.tracer.emit("memory_inject", block=block)
        res = self.inner.run(prompt)
        outcome = "completed" if res.stop_reason == "final_answer" else res.stop_reason
        self.store.add_episode(
            task, outcome, res.answer.strip(), model=self.model, agent=self.agent_kind
        )
        return res


def with_memory_tools(registry: ToolRegistry, store: MemoryStore) -> ToolRegistry:
    for tool in memory_tools(store):
        registry.register(tool)
    return registry


# -- short-term memory -----------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """\
You compress the middle of an agent's working conversation so it fits in the context window.
In at most 120 words, list what has been done so far: each tool call and the values it
returned that may still matter (numbers, names, file paths), copied exactly. Leave out
filler text. Do not add anything that is not in the conversation. No preamble."""

SUMMARY_NOTE = "Summary of the earlier part of this conversation (older messages were removed):"
# Deterministic compaction keeps the start and end of each old tool output: file headers,
# first lines and last lines survive; long middles go.
CLIP_HEAD = 300
CLIP_TAIL = 300


def _clip_output(text: str) -> str:
    if len(text) <= CLIP_HEAD + CLIP_TAIL + 40:
        return text
    omitted = len(text) - CLIP_HEAD - CLIP_TAIL
    return f"{text[:CLIP_HEAD]}\n[... {omitted} characters omitted ...]\n{text[-CLIP_TAIL:]}"


def _render(m: Message) -> str:
    if m.role == "assistant" and m.tool_calls:
        calls = ", ".join(f"{c.name}({c.arguments})" for c in m.tool_calls)
        return f"assistant called: {calls}" + (f"\n  said: {m.content}" if m.content else "")
    if m.role == "tool":
        return f"tool {m.tool_name} returned: {_clip_output(m.content)}"
    if m.role == "user" and m.content.startswith(SUMMARY_NOTE):
        return m.content[len(SUMMARY_NOTE) :].strip()  # an earlier summary: keep it whole
    return f"{m.role}: {m.content}"


def compact_messages(
    messages: list[Message], llm: LLM | None = None, *, keep_last: int = 4
) -> tuple[list[Message], int, int] | None:
    """Replace the middle of a conversation with one summary message.

    Keeps the system prompt, the task and the last ``keep_last`` messages. The cut is moved
    back to an assistant message, so a tool result is never separated from its call.

    Without ``llm`` the summary is the middle itself with every tool output clipped to its
    start and end (deterministic, loses nothing that sits at either end). With ``llm`` the
    model writes the summary from that clipped text; if its summary is not shorter, the
    clipped text is used instead. Returns ``(messages, prompt_tokens, completion_tokens)``,
    or None if nothing can be cut.
    """
    head = 2  # system + task
    cut = len(messages) - keep_last
    while cut > head and messages[cut].role != "assistant":
        cut -= 1
    if cut - head < 2:
        return None
    middle = "\n".join(_render(m) for m in messages[head:cut])
    text, p_tok, c_tok = middle, 0, 0
    if llm is not None:
        resp = llm.chat(
            [
                Message(role="system", content=SUMMARY_SYSTEM_PROMPT),
                Message(role="user", content=f"Task: {messages[1].content}\n\n{middle}"),
            ],
            [],
        )
        p_tok, c_tok = resp.prompt_tokens, resp.completion_tokens
        written = resp.message.content.strip()
        if written and len(written) < len(middle):
            text = written
    summary = Message(role="user", content=f"{SUMMARY_NOTE}\n{text}")
    return [*messages[:head], summary, *messages[cut:]], p_tok, c_tok


def describe(store: MemoryStore) -> dict[str, Any]:
    n_facts = store._db.execute("SELECT count(*) FROM facts").fetchone()[0]
    n_eps = store._db.execute("SELECT count(*) FROM episodes").fetchone()[0]
    return {
        "path": str(store.path) if store.path else ":memory:",
        "facts": n_facts,
        "episodes": n_eps,
    }
