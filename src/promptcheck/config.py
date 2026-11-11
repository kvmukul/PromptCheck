"""Load and validate TOML test suites."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .assertions import SpecError, validate_spec


class ConfigError(Exception):
    pass


@dataclass
class AssertionSpec:
    type: str
    params: dict


@dataclass
class Case:
    name: str
    prompt: str
    system: str | None
    assertions: list[AssertionSpec]
    mock_output: str | None = None
    runs: int = 1
    min_pass_rate: float = 1.0
    tags: list[str] = field(default_factory=list)


@dataclass
class Suite:
    name: str
    provider: str
    model: str | None
    cases: list[Case]
    max_tokens: int = 1024
    temperature: float = 0.0
    provider_options: dict = field(default_factory=dict)
    judge_provider: str | None = None
    judge_model: str | None = None
    source: str = ""


_VAR = re.compile(r"\{\{\s*([\w.-]+)\s*\}\}")


def render(template: str, variables: dict, where: str) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in variables:
            raise ConfigError(f"{where}: template variable {{{{{key}}}}} has no value")
        return str(variables[key])

    return _VAR.sub(sub, template)


def _specs(raw: list, where: str) -> list[AssertionSpec]:
    out = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict) or "type" not in item:
            raise ConfigError(f"{where}: assertion #{i} needs a `type`")
        params = {k: v for k, v in item.items() if k != "type"}
        try:
            validate_spec(item["type"], params)
        except SpecError as exc:
            raise ConfigError(f"{where}: {exc}") from exc
        out.append(AssertionSpec(item["type"], params))
    return out


def parse_suite(data: dict, base_dir: Path = Path("."), source: str = "<memory>") -> Suite:
    meta = data.get("suite", {})
    shared_vars = meta.get("vars", {})
    defaults = _specs(data.get("defaults", {}).get("assert", []), f"{source} [defaults]")

    raw_cases = data.get("tests")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ConfigError(f"{source}: define at least one [[tests]] table")

    cases: list[Case] = []
    seen: set[str] = set()
    for idx, t in enumerate(raw_cases, 1):
        name = t.get("name")
        if not name:
            raise ConfigError(f"{source}: test #{idx} is missing `name`")
        if name in seen:
            raise ConfigError(f"{source}: duplicate test name {name!r}")
        seen.add(name)
        where = f"{source} [{name}]"

        if "prompt_file" in t:
            try:
                template = (base_dir / t["prompt_file"]).read_text(encoding="utf-8")
            except OSError as exc:
                raise ConfigError(f"{where}: cannot read prompt_file: {exc}") from exc
        elif "prompt" in t:
            template = t["prompt"]
        else:
            raise ConfigError(f"{where}: needs `prompt` or `prompt_file`")

        variables = {**shared_vars, **t.get("vars", {})}
        runs = int(t.get("runs", 1))
        rate = float(t.get("min_pass_rate", 1.0))
        if runs < 1:
            raise ConfigError(f"{where}: `runs` must be >= 1")
        if not 0 < rate <= 1:
            raise ConfigError(f"{where}: `min_pass_rate` must be in (0, 1]")

        system = t.get("system", meta.get("system"))
        cases.append(Case(
            name=name,
            prompt=render(template, variables, where),
            system=render(system, variables, where) if system else None,
            assertions=defaults + _specs(t.get("assert", []), where),
            mock_output=t.get("mock_output"),
            runs=runs,
            min_pass_rate=rate,
            tags=list(t.get("tags", [])),
        ))

    judge = meta.get("judge", {})
    return Suite(
        name=meta.get("name", Path(source).stem),
        provider=meta.get("provider", "mock"),
        model=meta.get("model"),
        cases=cases,
        max_tokens=int(meta.get("max_tokens", 1024)),
        temperature=float(meta.get("temperature", 0.0)),
        provider_options=dict(meta.get("options", {})),
        judge_provider=judge.get("provider"),
        judge_model=judge.get("model"),
        source=source,
    )


def load_suite(path: str | Path) -> Suite:
    p = Path(path)
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"suite file not found: {p}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p}: invalid TOML: {exc}") from exc
    return parse_suite(data, p.parent, str(p))
