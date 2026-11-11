"""Command-line interface: `promptcheck run | init | assertions`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, baseline, report
from .assertions import REGISTRY
from .cache import Cache
from .config import ConfigError, load_suite
from .providers import ProviderError, get_provider
from .runner import run_suite

TEMPLATE = '''# promptcheck suite. Run with: promptcheck run promptcheck.toml
[suite]
name = "my first suite"
provider = "mock"          # mock | anthropic | openai
# model = "claude-sonnet-5-5"
# system = "You are a concise assistant."

[suite.vars]
product = "Acme CRM"

[[tests]]
name = "greets politely"
prompt = "Write a one-line greeting for a {{product}} customer."
mock_output = "Welcome to Acme CRM! How can I help today?"   # used by the mock provider
tags = ["smoke"]

  [[tests.assert]]
  type = "contains"
  value = "Acme CRM"

  [[tests.assert]]
  type = "max_words"
  value = 20
'''


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="promptcheck",
                                description="Unit tests and regression checks for LLM prompts.")
    p.add_argument("--version", action="version", version=f"promptcheck {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a test suite")
    run.add_argument("suite", help="path to a .toml suite file")
    run.add_argument("--provider", help="override the suite's provider")
    run.add_argument("--model", help="override the suite's model")
    run.add_argument("-w", "--workers", type=int, default=4, help="parallel requests (default 4)")
    run.add_argument("-k", "--filter", help="only run tests whose name contains this text")
    run.add_argument("--tag", help="only run tests with this tag")
    run.add_argument("--no-cache", action="store_true", help="always call the model")
    run.add_argument("--cache-dir", default=".promptcheck_cache")
    run.add_argument("-v", "--verbose", action="store_true", help="show passing assertions too")
    run.add_argument("--html", metavar="FILE", help="write an HTML report")
    run.add_argument("--json", metavar="FILE", help="write a JSON report")
    run.add_argument("--junit", metavar="FILE", help="write a JUnit XML report")
    run.add_argument("--baseline", metavar="FILE", help="compare against a saved baseline")
    run.add_argument("--save-baseline", metavar="FILE", help="save this run as the baseline")
    run.add_argument("--fail-under", type=float, default=1.0, metavar="RATE",
                     help="minimum fraction of tests that must pass (default 1.0)")

    init = sub.add_parser("init", help="create a starter suite file")
    init.add_argument("path", nargs="?", default="promptcheck.toml")

    sub.add_parser("assertions", help="list available assertion types")
    return p


def cmd_run(args: argparse.Namespace, out, err) -> int:
    suite = load_suite(args.suite)
    if args.provider:
        suite.provider = args.provider
    if args.model:
        suite.model = args.model
    provider = get_provider(suite.provider, suite.model, suite.provider_options)

    cache = None if args.no_cache else Cache(args.cache_dir)
    show_progress = err.isatty()

    def progress(done: int, total: int) -> None:
        if show_progress:
            err.write(f"\r  running {done}/{total}")
            err.flush()

    result = run_suite(suite, provider=provider, cache=cache, workers=args.workers,
                       name_filter=args.filter, tag=args.tag, progress=progress)
    if show_progress:
        err.write("\r" + " " * 24 + "\r")
    if not result.cases:
        err.write("promptcheck: no tests matched the filter\n")
        return 2

    diff = None
    if args.baseline:
        try:
            diff = baseline.compare(baseline.load(args.baseline), result)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read baseline: {exc}") from exc

    out.write(report.format_terminal(result, diff, color=report._supports_color(out),
                                     verbose=args.verbose) + "\n")

    if args.html:
        Path(args.html).write_text(report.to_html(result, diff), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(report.to_json(result, diff), encoding="utf-8")
    if args.junit:
        Path(args.junit).write_text(report.to_junit(result), encoding="utf-8")
    if args.save_baseline:
        baseline.save(result, args.save_baseline)
        out.write(f"baseline saved to {args.save_baseline}\n")

    if diff is not None and diff.has_regressions:
        return 1
    return 0 if result.pass_rate >= args.fail_under else 1


def cmd_init(args: argparse.Namespace, out, err) -> int:
    path = Path(args.path)
    if path.exists():
        err.write(f"promptcheck: {path} already exists, not overwriting\n")
        return 2
    path.write_text(TEMPLATE, encoding="utf-8")
    out.write(f"created {path}. Try: promptcheck run {path}\n")
    return 0


def cmd_assertions(args: argparse.Namespace, out, err) -> int:
    width = max(len(n) for n in REGISTRY)
    for name in sorted(REGISTRY):
        out.write(f"{name.ljust(width)}  {REGISTRY[name][1]}\n")
    return 0


def main(argv: list[str] | None = None, out=None, err=None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    args = build_parser().parse_args(argv)
    handler = {"run": cmd_run, "init": cmd_init, "assertions": cmd_assertions}[args.command]
    try:
        return handler(args, out, err)
    except (ConfigError, ProviderError) as exc:
        err.write(f"promptcheck: error: {exc}\n")
        return 2
