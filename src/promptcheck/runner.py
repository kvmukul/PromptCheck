"""Execute a suite concurrently and collect structured results."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .assertions import Context, run_assertion
from .cache import Cache
from .config import Case, Suite
from .providers import Provider, Request, get_provider


@dataclass
class AssertionOutcome:
    type: str
    passed: bool
    message: str


@dataclass
class RunOutcome:
    output: str = ""
    latency_ms: float = 0.0
    cached: bool = False
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    assertions: list[AssertionOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.error is None and all(a.passed for a in self.assertions)


@dataclass
class CaseResult:
    name: str
    prompt: str
    tags: list[str]
    min_pass_rate: float
    runs: list[RunOutcome]

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.runs) / len(self.runs)

    @property
    def passed(self) -> bool:
        return self.pass_rate >= self.min_pass_rate


@dataclass
class SuiteResult:
    suite: str
    provider: str
    model: str
    cases: list[CaseResult]
    duration_s: float
    timestamp: str

    @property
    def passed_count(self) -> int:
        return sum(c.passed for c in self.cases)

    @property
    def failed_count(self) -> int:
        return len(self.cases) - self.passed_count

    @property
    def pass_rate(self) -> float:
        return self.passed_count / len(self.cases) if self.cases else 1.0

    @property
    def total_tokens(self) -> tuple[int, int]:
        runs = [r for c in self.cases for r in c.runs]
        return (sum(r.input_tokens or 0 for r in runs),
                sum(r.output_tokens or 0 for r in runs))

    def to_dict(self) -> dict:
        data = asdict(self)
        for case, raw in zip(self.cases, data["cases"]):
            raw["passed"] = case.passed
            raw["pass_rate"] = case.pass_rate
            for run, rraw in zip(case.runs, raw["runs"]):
                rraw["passed"] = run.passed
        data.update(passed=self.passed_count, failed=self.failed_count,
                    pass_rate=self.pass_rate)
        return data


def _execute(case: Case, run_index: int, suite: Suite, provider: Provider,
             judge: Provider, cache: Cache | None) -> RunOutcome:
    req = Request(case.prompt, case.system, suite.max_tokens, suite.temperature,
                  case.mock_output)
    outcome = RunOutcome()
    try:
        key = None
        completion = None
        if cache is not None and provider.name != "mock":
            key = Cache.key(provider.name, provider.model, case.system, case.prompt,
                            suite.max_tokens, suite.temperature,
                            run_index if case.runs > 1 else 0)
            completion = cache.get(key)
            outcome.cached = completion is not None
        if completion is None:
            completion = provider.complete(req)
            if key is not None:
                cache.put(key, completion)
        outcome.output = completion.text
        outcome.latency_ms = completion.latency_ms
        outcome.input_tokens = completion.input_tokens
        outcome.output_tokens = completion.output_tokens
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome

    def ask_judge(prompt: str) -> str:
        mock = "PASS (mock judge: nothing was really graded)" if judge.name == "mock" else None
        return judge.complete(Request(prompt, None, 300, 0.0, mock)).text

    ctx = Context(outcome.output, outcome.latency_ms, case.prompt, ask_judge)
    for spec in case.assertions:
        ok, msg = run_assertion(spec.type, spec.params, ctx)
        outcome.assertions.append(AssertionOutcome(spec.type, ok, msg))
    return outcome


def run_suite(
    suite: Suite,
    *,
    provider: Provider | None = None,
    judge: Provider | None = None,
    cache: Cache | None = None,
    workers: int = 4,
    name_filter: str | None = None,
    tag: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SuiteResult:
    provider = provider or get_provider(suite.provider, suite.model, suite.provider_options)
    if judge is None:
        judge = (get_provider(suite.judge_provider, suite.judge_model, suite.provider_options)
                 if suite.judge_provider else provider)

    cases = [c for c in suite.cases
             if (not name_filter or name_filter.lower() in c.name.lower())
             and (not tag or tag in c.tags)]

    tasks = [(ci, ri) for ci, c in enumerate(cases) for ri in range(c.runs)]
    outcomes: dict[tuple[int, int], RunOutcome] = {}
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_execute, cases[ci], ri, suite, provider, judge, cache): (ci, ri)
            for ci, ri in tasks
        }
        for done, fut in enumerate(as_completed(futures), 1):
            outcomes[futures[fut]] = fut.result()
            if progress:
                progress(done, len(tasks))

    results = [
        CaseResult(c.name, c.prompt, c.tags, c.min_pass_rate,
                   [outcomes[(ci, ri)] for ri in range(c.runs)])
        for ci, c in enumerate(cases)
    ]
    return SuiteResult(
        suite=suite.name, provider=provider.name, model=provider.model, cases=results,
        duration_s=time.perf_counter() - started,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
