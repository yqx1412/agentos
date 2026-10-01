"""Benchmark task definitions, loaded from YAML files.

One file per category::

    category: file_ops
    tasks:
      - id: fo-copy
        prompt: Copy source.txt to backup/source.txt.
        files: {source.txt: "alpha\\nbeta\\n"}
        servers: []          # MCP servers the task needs (names from agentos.toml)
        max_steps: 12
        checks:
          - {type: file_equals, path: backup/source.txt, value: "alpha\\nbeta"}
"""

from __future__ import annotations

import fnmatch
from collections import Counter
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentos.bench.checks import Check, FileUnchanged

TASK_ID_PATTERN = r"^[a-z0-9][a-z0-9-]*$"


class TaskError(Exception):
    """A task file is unreadable or invalid."""


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=TASK_ID_PATTERN)
    category: str
    prompt: str = Field(min_length=1)
    files: dict[str, str] = Field(default_factory=dict)
    servers: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=12, ge=1)
    checks: list[Check] = Field(min_length=1)
    notes: str | None = None
    # A6 memory tasks: earlier sessions run before ``prompt``, each in a fresh conversation
    # sharing the memory store. ``setup_files`` exist only while those sessions run.
    setup: list[str] = Field(default_factory=list)
    setup_files: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _setup_files_are_separate(self) -> Task:
        if self.setup_files and not self.setup:
            raise ValueError("setup_files needs at least one setup prompt")
        overlap = sorted(set(self.setup_files) & set(self.files))
        if overlap:
            raise ValueError(f"paths in both files and setup_files: {overlap}")
        return self

    @model_validator(mode="after")
    def _unchanged_targets_exist(self) -> Task:
        for c in self.checks:
            if isinstance(c, FileUnchanged) and c.path not in self.files:
                raise ValueError(f"file_unchanged check on {c.path!r}, which is not in files")
        return self


class _TaskFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    tasks: list[dict] = Field(min_length=1)


def load_tasks(directory: Path) -> list[Task]:
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise TaskError(f"no *.yaml task files in {directory}")
    tasks: list[Task] = []
    for f in files:
        try:
            raw = yaml.safe_load(f.read_text(encoding="utf-8"))
            doc = _TaskFile.model_validate(raw)
            for t in doc.tasks:
                tasks.append(Task.model_validate({"category": doc.category, **t}))
        except (yaml.YAMLError, ValidationError) as exc:
            raise TaskError(f"{f.name}: {exc}") from exc
    counts = Counter(t.id for t in tasks)
    dupes = sorted(i for i, n in counts.items() if n > 1)
    if dupes:
        raise TaskError(f"duplicate task ids: {dupes}")
    return tasks


def select_tasks(tasks: list[Task], patterns: list[str] | None) -> list[Task]:
    """Keep tasks whose id or category matches any glob pattern (all tasks if none)."""
    if not patterns:
        return tasks
    return [
        t
        for t in tasks
        if any(fnmatch.fnmatch(t.id, p) or fnmatch.fnmatch(t.category, p) for p in patterns)
    ]
