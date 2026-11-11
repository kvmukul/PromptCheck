"""Save a known-good run and diff later runs against it to catch regressions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .runner import SuiteResult


@dataclass
class Diff:
    regressions: list[str] = field(default_factory=list)  # passed before, fails now
    fixed: list[str] = field(default_factory=list)  # failed before, passes now
    new: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # in baseline, not in this run

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)


def snapshot(result: SuiteResult) -> dict:
    return {
        "version": 1,
        "suite": result.suite,
        "model": result.model,
        "timestamp": result.timestamp,
        "cases": {c.name: {"passed": c.passed, "pass_rate": c.pass_rate}
                  for c in result.cases},
    }


def save(result: SuiteResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(snapshot(result), indent=2) + "\n", encoding="utf-8")


def load(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or "cases" not in data:
        raise ValueError(f"{path} is not a promptcheck baseline file")
    return data


def compare(baseline: dict, result: SuiteResult) -> Diff:
    before = baseline["cases"]
    now = {c.name: c.passed for c in result.cases}
    diff = Diff()
    for name, passed in now.items():
        if name not in before:
            diff.new.append(name)
        elif before[name]["passed"] and not passed:
            diff.regressions.append(name)
        elif not before[name]["passed"] and passed:
            diff.fixed.append(name)
    diff.missing = [n for n in before if n not in now]
    return diff
