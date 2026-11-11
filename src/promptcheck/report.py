"""Terminal, JSON, JUnit XML and self-contained HTML reports."""

from __future__ import annotations

import html
import json
import os
import xml.etree.ElementTree as ET
from typing import TextIO

from .baseline import Diff
from .runner import CaseResult, SuiteResult


def _supports_color(stream: TextIO) -> bool:
    return stream.isatty() and "NO_COLOR" not in os.environ


def _paint(text: str, code: str, on: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if on else text


def format_terminal(result: SuiteResult, diff: Diff | None = None, *,
                    color: bool = False, verbose: bool = False) -> str:
    g = lambda s: _paint(s, "32", color)  # noqa: E731
    r = lambda s: _paint(s, "31", color)  # noqa: E731
    d = lambda s: _paint(s, "2", color)  # noqa: E731
    b = lambda s: _paint(s, "1", color)  # noqa: E731

    lines = [b(f"promptcheck · {result.suite}") + d(f"  [{result.provider}/{result.model}]"), ""]
    for case in result.cases:
        mark = g("✓") if case.passed else r("✗")
        n = len(case.runs)
        extra = f"{n} runs, {case.pass_rate:.0%} pass" if n > 1 else f"{n} run"
        avg = sum(x.latency_ms for x in case.runs) / n
        lines.append(f" {mark} {case.name}  " + d(f"({extra}, {avg:.0f}ms)"))
        if verbose or not case.passed:
            for i, run in enumerate(case.runs, 1):
                if run.passed and not verbose:
                    continue
                prefix = f"     run {i}: " if n > 1 else "     "
                if run.error:
                    lines.append(prefix + r(f"error: {run.error}"))
                for a in run.assertions:
                    if verbose or not a.passed:
                        sign = g("✓") if a.passed else r("✗")
                        lines.append(f"{prefix}{sign} {a.type}: {a.message}")
                        prefix = " " * len(prefix)
    lines.append("")
    summary = f"{result.passed_count}/{len(result.cases)} passed"
    summary = g(summary) if result.failed_count == 0 else r(summary + f", {result.failed_count} failed")
    lines.append(b("Result: ") + summary + d(f"  in {result.duration_s:.2f}s"))
    tin, tout = result.total_tokens
    if tin or tout:
        lines.append(d(f"Tokens: {tin} in / {tout} out"))
    if diff is not None:
        lines.append("")
        lines.append(b("Baseline comparison"))
        for name in diff.regressions:
            lines.append(r(f"  ▼ regression: {name}"))
        for name in diff.fixed:
            lines.append(g(f"  ▲ fixed: {name}"))
        for name in diff.new:
            lines.append(f"  + new: {name}")
        for name in diff.missing:
            lines.append(d(f"  - not run: {name}"))
        if not (diff.regressions or diff.fixed or diff.new or diff.missing):
            lines.append(g("  no changes vs baseline"))
    return "\n".join(lines)


def to_json(result: SuiteResult, diff: Diff | None = None) -> str:
    data = result.to_dict()
    if diff is not None:
        data["baseline"] = {"regressions": diff.regressions, "fixed": diff.fixed,
                            "new": diff.new, "missing": diff.missing}
    return json.dumps(data, indent=2)


def to_junit(result: SuiteResult) -> str:
    suite = ET.Element("testsuite", name=result.suite, tests=str(len(result.cases)),
                       failures=str(result.failed_count), time=f"{result.duration_s:.3f}")
    for case in result.cases:
        tc = ET.SubElement(suite, "testcase", name=case.name, classname=result.suite,
                           time=f"{sum(x.latency_ms for x in case.runs) / 1000:.3f}")
        if not case.passed:
            msgs = []
            for i, run in enumerate(case.runs, 1):
                if run.error:
                    msgs.append(f"run {i}: {run.error}")
                msgs += [f"run {i}: {a.type}: {a.message}" for a in run.assertions if not a.passed]
            fail = ET.SubElement(tc, "failure", message=msgs[0] if msgs else "failed")
            fail.text = "\n".join(msgs)
    ET.indent(suite)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(suite, encoding="unicode")


_CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--ok:#15803d;--bad:#b91c1c;--okbg:#dcfce7;--badbg:#fee2e2}
@media(prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;--card:#1c1917;--line:#292524;--ok:#4ade80;--bad:#f87171;--okbg:#052e16;--badbg:#450a0a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px}h1{margin:0 0 4px;font-size:24px}
.sub{color:var(--muted);margin-bottom:24px}.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;min-width:140px}
.card b{display:block;font-size:26px}.card span{color:var(--muted);font-size:13px}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;margin-bottom:10px}
summary{cursor:pointer;padding:12px 16px;display:flex;gap:10px;align-items:center;list-style:none}
summary::-webkit-details-marker{display:none}
.badge{font-size:12px;font-weight:600;padding:2px 8px;border-radius:99px}
.pass{background:var(--okbg);color:var(--ok)}.fail{background:var(--badbg);color:var(--bad)}
.meta{color:var(--muted);font-size:13px;margin-left:auto}.body{padding:0 16px 14px;border-top:1px solid var(--line)}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;white-space:pre-wrap;font-size:13px}
ul{padding-left:0;list-style:none;margin:6px 0}li{padding:2px 0}.ok{color:var(--ok)}.no{color:var(--bad)}
h4{margin:14px 0 4px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
"""


def _case_html(case: CaseResult) -> str:
    e = html.escape
    badge = '<span class="badge pass">PASS</span>' if case.passed else '<span class="badge fail">FAIL</span>'
    tags = " ".join(f"#{e(t)}" for t in case.tags)
    parts = [f"<details{'' if case.passed else ' open'}><summary>{badge}<strong>{e(case.name)}</strong>"
             f'<span class="meta">{tags} {len(case.runs)} run(s) · {case.pass_rate:.0%}</span></summary>'
             f'<div class="body"><h4>Prompt</h4><pre>{e(case.prompt)}</pre>']
    for i, run in enumerate(case.runs, 1):
        title = f"Run {i}" if len(case.runs) > 1 else "Output"
        parts.append(f"<h4>{title} · {run.latency_ms:.0f}ms{' · cached' if run.cached else ''}</h4>")
        if run.error:
            parts.append(f'<pre class="no">{e(run.error)}</pre>')
        else:
            parts.append(f"<pre>{e(run.output)}</pre>")
        parts.append("<ul>" + "".join(
            f'<li class="{"ok" if a.passed else "no"}">{"✓" if a.passed else "✗"} '
            f"<code>{e(a.type)}</code> {e(a.message)}</li>" for a in run.assertions) + "</ul>")
    parts.append("</div></details>")
    return "".join(parts)


def to_html(result: SuiteResult, diff: Diff | None = None) -> str:
    e = html.escape
    tin, tout = result.total_tokens
    cards = [
        (f"{result.pass_rate:.0%}", "pass rate"),
        (f"{result.passed_count}/{len(result.cases)}", "tests passed"),
        (f"{result.duration_s:.2f}s", "duration"),
    ]
    if tin or tout:
        cards.append((f"{tin + tout:,}", "tokens"))
    if diff is not None:
        cards.append((str(len(diff.regressions)), "regressions"))
    card_html = "".join(f'<div class="card"><b>{v}</b><span>{l}</span></div>' for v, l in cards)
    ordered = sorted(result.cases, key=lambda c: c.passed)  # failures first
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>promptcheck · {e(result.suite)}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>{e(result.suite)}</h1>"
        f'<div class="sub">{e(result.provider)}/{e(result.model)} · {e(result.timestamp)}</div>'
        f'<div class="cards">{card_html}</div>'
        + "".join(_case_html(c) for c in ordered)
        + "</main></body></html>"
    )
