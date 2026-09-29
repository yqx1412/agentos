"""Automatic checks that decide whether a benchmark task succeeded.

Each check is a Pydantic model tagged by ``type``, so tasks write them as YAML mappings::

    checks:
      - {type: file_number, path: total.txt, value: 260}
      - {type: answer_contains, values: ["not found", "does not exist"], mode: any}

Checks look only at observable outcomes: files in the workspace after the run, the final
answer, and which tools were called. They never inspect the model's reasoning.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def numbers_in(text: str) -> list[float]:
    """All numbers in a text; thousands separators are accepted ("5,000" -> 5000)."""
    out = []
    for m in _NUMBER.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


@dataclass
class Outcome:
    """What a finished run left behind, as seen by the checks."""

    workspace: Path
    answer: str
    tools_called: list[str] = field(default_factory=list)
    initial_files: dict[str, str] = field(default_factory=dict)

    def path(self, rel: str) -> Path:
        return self.workspace / rel

    def read(self, rel: str) -> str | None:
        p = self.path(rel)
        if not p.is_file():
            return None
        return p.read_text(encoding="utf-8", errors="replace")


@dataclass
class CheckResult:
    ok: bool
    detail: str


def _clip(text: str, n: int = 80) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= n else text[: n - 3] + "..."


class _Check(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def evaluate(self, o: Outcome) -> CheckResult:  # pragma: no cover - abstract
        raise NotImplementedError


class _FileCheck(_Check):
    path: str

    def _missing(self) -> CheckResult:
        return CheckResult(False, f"{self.path} does not exist")


class FileEquals(_FileCheck):
    type: Literal["file_equals"]
    value: str
    strip: bool = True
    case_sensitive: bool = True

    def evaluate(self, o: Outcome) -> CheckResult:
        text = o.read(self.path)
        if text is None:
            return self._missing()
        got, want = (text.strip(), self.value.strip()) if self.strip else (text, self.value)
        if not self.case_sensitive:
            got, want = got.lower(), want.lower()
        if got == want:
            return CheckResult(True, f"{self.path} == {_clip(self.value)!r}")
        return CheckResult(False, f"{self.path}: expected {_clip(want)!r}, got {_clip(got)!r}")


class FileContains(_FileCheck):
    type: Literal["file_contains"]
    values: list[str] = Field(min_length=1)
    case_sensitive: bool = False

    def evaluate(self, o: Outcome) -> CheckResult:
        text = o.read(self.path)
        if text is None:
            return self._missing()
        hay = text if self.case_sensitive else text.lower()
        missing = [v for v in self.values if (v if self.case_sensitive else v.lower()) not in hay]
        if missing:
            return CheckResult(False, f"{self.path} lacks {missing}; got {_clip(text)!r}")
        return CheckResult(True, f"{self.path} contains {self.values}")


class FileNumber(_FileCheck):
    """The first number in the file equals ``value`` within ``tol``."""

    type: Literal["file_number"]
    value: float
    tol: float = 1e-6

    def evaluate(self, o: Outcome) -> CheckResult:
        text = o.read(self.path)
        if text is None:
            return self._missing()
        nums = numbers_in(text)
        if not nums:
            return CheckResult(False, f"{self.path} has no number; got {_clip(text)!r}")
        if math.isclose(nums[0], self.value, rel_tol=0, abs_tol=self.tol):
            return CheckResult(True, f"{self.path} == {self.value:g}")
        return CheckResult(False, f"{self.path}: expected {self.value:g}, got {nums[0]:g}")


class FileRegex(_FileCheck):
    type: Literal["file_regex"]
    pattern: str
    flags: Literal["", "i", "m", "im"] = "im"

    def evaluate(self, o: Outcome) -> CheckResult:
        text = o.read(self.path)
        if text is None:
            return self._missing()
        fl = (re.I if "i" in self.flags else 0) | (re.M if "m" in self.flags else 0)
        if re.search(self.pattern, text, fl):
            return CheckResult(True, f"{self.path} matches /{self.pattern}/")
        return CheckResult(False, f"{self.path} !~ /{self.pattern}/; got {_clip(text)!r}")


class FileLines(_FileCheck):
    """Non-blank lines, stripped, equal ``lines`` (in order unless ``ordered: false``)."""

    type: Literal["file_lines"]
    lines: list[str]
    ordered: bool = True
    case_sensitive: bool = True

    def evaluate(self, o: Outcome) -> CheckResult:
        text = o.read(self.path)
        if text is None:
            return self._missing()

        def norm(s: str) -> str:
            s = s.strip()
            return s if self.case_sensitive else s.lower()

        got = [norm(line) for line in text.splitlines() if line.strip()]
        want = [norm(line) for line in self.lines]
        same = got == want if self.ordered else sorted(got) == sorted(want)
        if same:
            return CheckResult(True, f"{self.path} has the {len(want)} expected lines")
        return CheckResult(False, f"{self.path}: expected {want}, got {got}")


class FileJson(_FileCheck):
    """The file parses as JSON equal to ``value`` (numbers compare by value: 5 == 5.0)."""

    type: Literal["file_json"]
    value: Any

    def evaluate(self, o: Outcome) -> CheckResult:
        text = o.read(self.path)
        if text is None:
            return self._missing()
        try:
            got = json.loads(text)
        except json.JSONDecodeError as exc:
            return CheckResult(False, f"{self.path} is not valid JSON: {exc.msg}")
        if got == self.value:
            return CheckResult(True, f"{self.path} JSON matches")
        return CheckResult(False, f"{self.path}: expected {self.value!r}, got {_clip(repr(got))}")


class FileAbsent(_FileCheck):
    type: Literal["file_absent"]

    def evaluate(self, o: Outcome) -> CheckResult:
        if o.path(self.path).exists():
            return CheckResult(False, f"{self.path} exists but must not")
        return CheckResult(True, f"{self.path} absent")


class FileUnchanged(_FileCheck):
    type: Literal["file_unchanged"]

    def evaluate(self, o: Outcome) -> CheckResult:
        if self.path not in o.initial_files:
            return CheckResult(False, f"{self.path} is not a setup file")
        if o.read(self.path) == o.initial_files[self.path]:
            return CheckResult(True, f"{self.path} unchanged")
        return CheckResult(False, f"{self.path} was modified or deleted")


class AnswerContains(_Check):
    type: Literal["answer_contains"]
    values: list[str] = Field(min_length=1)
    mode: Literal["all", "any"] = "all"

    def evaluate(self, o: Outcome) -> CheckResult:
        ans = o.answer.lower()
        hits = [v for v in self.values if v.lower() in ans]
        ok = bool(hits) if self.mode == "any" else len(hits) == len(self.values)
        if ok:
            return CheckResult(True, f"answer contains {hits}")
        return CheckResult(
            False, f"answer lacks {self.mode} of {self.values}; got {_clip(o.answer)!r}"
        )


class AnswerNumber(_Check):
    """Some number in the final answer equals ``value`` within ``tol``."""

    type: Literal["answer_number"]
    value: float
    tol: float = 1e-6

    def evaluate(self, o: Outcome) -> CheckResult:
        nums = numbers_in(o.answer)
        if any(math.isclose(n, self.value, rel_tol=0, abs_tol=self.tol) for n in nums):
            return CheckResult(True, f"answer mentions {self.value:g}")
        return CheckResult(False, f"answer lacks {self.value:g}; got {_clip(o.answer)!r}")


class ToolUsed(_Check):
    type: Literal["tool_used"]
    name: str

    def evaluate(self, o: Outcome) -> CheckResult:
        if self.name in o.tools_called:
            return CheckResult(True, f"called {self.name}")
        return CheckResult(False, f"never called {self.name}; called {sorted(set(o.tools_called))}")


Check = Annotated[
    FileEquals
    | FileContains
    | FileNumber
    | FileRegex
    | FileLines
    | FileJson
    | FileAbsent
    | FileUnchanged
    | AnswerContains
    | AnswerNumber
    | ToolUsed,
    Field(discriminator="type"),
]
