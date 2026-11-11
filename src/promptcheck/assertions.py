"""Assertion registry: every check a test can make about a model's output."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable


class SpecError(ValueError):
    """An assertion in a suite file is malformed."""


@dataclass
class Context:
    """Everything an assertion may need to inspect."""

    output: str
    latency_ms: float = 0.0
    prompt: str = ""
    judge: Callable[[str], str] | None = None  # prompt -> grader reply


Check = Callable[[Context, dict], "tuple[bool, str]"]
REGISTRY: dict[str, tuple[Check, str, tuple[str, ...]]] = {}


def register(name: str, doc: str, required: tuple[str, ...] = ()):
    def deco(fn: Check) -> Check:
        REGISTRY[name] = (fn, doc, required)
        return fn

    return deco


def validate_spec(type_: str, params: dict) -> None:
    if type_ not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise SpecError(f"unknown assertion type {type_!r} (known: {known})")
    for key in REGISTRY[type_][2]:
        if key not in params:
            raise SpecError(f"assertion {type_!r} is missing required key {key!r}")


def run_assertion(type_: str, params: dict, ctx: Context) -> tuple[bool, str]:
    validate_spec(type_, params)
    try:
        return REGISTRY[type_][0](ctx, params)
    except Exception as exc:  # a broken check should fail, not crash the run
        return False, f"assertion raised {type(exc).__name__}: {exc}"


def _short(text: Any, n: int = 80) -> str:
    s = str(text).replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 1] + "…"


def _res(ok: bool, pass_msg: str, fail_msg: str) -> tuple[bool, str]:
    return ok, (pass_msg if ok else fail_msg)


# --------------------------------------------------------------------- text


@register("contains", "Output contains a substring (ignore_case optional).", ("value",))
def _contains(ctx: Context, p: dict):
    needle, hay = str(p["value"]), ctx.output
    if p.get("ignore_case"):
        needle, hay = needle.lower(), hay.lower()
    return _res(
        needle in hay,
        f"contains {_short(p['value'])!r}",
        f"expected output to contain {_short(p['value'])!r}",
    )


@register("not_contains", "Output does NOT contain a substring.", ("value",))
def _not_contains(ctx: Context, p: dict):
    needle, hay = str(p["value"]), ctx.output
    if p.get("ignore_case"):
        needle, hay = needle.lower(), hay.lower()
    return _res(
        needle not in hay,
        f"does not contain {_short(p['value'])!r}",
        f"output must not contain {_short(p['value'])!r}",
    )


@register("contains_all", "Output contains every string in `values`.", ("values",))
def _contains_all(ctx: Context, p: dict):
    hay = ctx.output.lower() if p.get("ignore_case") else ctx.output
    missing = [
        v for v in p["values"]
        if (str(v).lower() if p.get("ignore_case") else str(v)) not in hay
    ]
    return _res(not missing, "contains all values", f"missing: {_short(missing)}")


@register("contains_any", "Output contains at least one string in `values`.", ("values",))
def _contains_any(ctx: Context, p: dict):
    hay = ctx.output.lower() if p.get("ignore_case") else ctx.output
    hit = [
        v for v in p["values"]
        if (str(v).lower() if p.get("ignore_case") else str(v)) in hay
    ]
    return _res(bool(hit), f"contains {_short(hit[0] if hit else '')!r}",
                f"none of {_short(p['values'])} found")


@register("equals", "Output equals `value` exactly (after stripping whitespace).", ("value",))
def _equals(ctx: Context, p: dict):
    got = ctx.output.strip()
    want = str(p["value"]).strip()
    if p.get("ignore_case"):
        got, want = got.lower(), want.lower()
    return _res(got == want, "equals expected value",
                f"expected {_short(want)!r}, got {_short(got)!r}")


@register("one_of", "Output (stripped) equals one of `values`.", ("values",))
def _one_of(ctx: Context, p: dict):
    got = ctx.output.strip()
    return _res(got in [str(v) for v in p["values"]], f"matched {_short(got)!r}",
                f"{_short(got)!r} is not one of {_short(p['values'])}")


@register("starts_with", "Output starts with `value`.", ("value",))
def _starts_with(ctx: Context, p: dict):
    return _res(ctx.output.lstrip().startswith(str(p["value"])),
                f"starts with {_short(p['value'])!r}",
                f"expected output to start with {_short(p['value'])!r}")


@register("regex", "Output matches regex `pattern` (flags: i, m, s).", ("pattern",))
def _regex(ctx: Context, p: dict):
    flags = 0
    for f in str(p.get("flags", "")):
        flags |= {"i": re.I, "m": re.M, "s": re.S}.get(f, 0)
    try:
        ok = re.search(p["pattern"], ctx.output, flags) is not None
    except re.error as exc:
        return False, f"invalid regex: {exc}"
    return _res(ok, f"matches /{p['pattern']}/", f"no match for /{p['pattern']}/")


@register("not_refusal", "Output does not look like a refusal / apology.")
def _not_refusal(ctx: Context, p: dict):
    markers = ("i can't", "i cannot", "i'm sorry", "i am sorry", "i'm unable",
               "as an ai", "i won't be able")
    low = ctx.output.lower()[:200]
    hit = next((m for m in markers if m in low), None)
    return _res(hit is None, "no refusal detected", f"looks like a refusal ({hit!r})")


# ------------------------------------------------------------------- length


@register("max_chars", "Output has at most `value` characters.", ("value",))
def _max_chars(ctx: Context, p: dict):
    n = len(ctx.output)
    return _res(n <= p["value"], f"{n} chars <= {p['value']}", f"{n} chars > {p['value']}")


@register("max_words", "Output has at most `value` words.", ("value",))
def _max_words(ctx: Context, p: dict):
    n = len(ctx.output.split())
    return _res(n <= p["value"], f"{n} words <= {p['value']}", f"{n} words > {p['value']}")


@register("min_words", "Output has at least `value` words.", ("value",))
def _min_words(ctx: Context, p: dict):
    n = len(ctx.output.split())
    return _res(n >= p["value"], f"{n} words >= {p['value']}", f"{n} words < {p['value']}")


@register("latency_ms", "Response arrived within `max` milliseconds.", ("max",))
def _latency(ctx: Context, p: dict):
    return _res(ctx.latency_ms <= p["max"], f"{ctx.latency_ms:.0f}ms <= {p['max']}ms",
                f"{ctx.latency_ms:.0f}ms > {p['max']}ms")


# --------------------------------------------------------------------- JSON

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str, strict: bool = False) -> Any:
    """Parse JSON from a reply, tolerating ```json fences and surrounding prose."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except ValueError:
        if strict:
            raise
    m = _FENCE.search(stripped)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch in "{[":
            try:
                return decoder.raw_decode(stripped[i:])[0]
            except ValueError:
                continue
    raise ValueError("no valid JSON found in output")


def _type_ok(value: Any, t: str) -> bool:
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    py = {"string": str, "boolean": bool, "array": list, "object": dict, "null": type(None)}
    return t in py and isinstance(value, py[t])


def validate_schema(value: Any, schema: dict, path: str = "$") -> list[str]:
    """A small JSON-Schema subset: type, enum, required, properties, items,
    minimum, maximum, minLength, maxLength."""
    errors: list[str] = []
    t = schema.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(value, x) for x in types):
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required property missing")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors += validate_schema(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate_schema(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
    return errors


@register("json_valid", "Output contains parseable JSON (set strict=true to forbid prose/fences).")
def _json_valid(ctx: Context, p: dict):
    try:
        extract_json(ctx.output, strict=bool(p.get("strict")))
    except ValueError as exc:
        return False, f"invalid JSON: {_short(exc)}"
    return True, "valid JSON"


@register("json_schema", "Output JSON satisfies a schema table (JSON-Schema subset).", ("schema",))
def _json_schema(ctx: Context, p: dict):
    try:
        data = extract_json(ctx.output, strict=bool(p.get("strict")))
    except ValueError as exc:
        return False, f"invalid JSON: {_short(exc)}"
    errors = validate_schema(data, p["schema"])
    return _res(not errors, "matches schema", "; ".join(errors[:3]))


# ----------------------------------------------------------------- semantic


@register("similarity", "Fuzzy text similarity to `reference` >= `threshold` (default 0.8).",
          ("reference",))
def _similarity(ctx: Context, p: dict):
    threshold = float(p.get("threshold", 0.8))
    a, b = ctx.output.strip().lower(), str(p["reference"]).strip().lower()
    score = difflib.SequenceMatcher(None, a, b).ratio()
    return _res(score >= threshold, f"similarity {score:.2f} >= {threshold}",
                f"similarity {score:.2f} < {threshold}")


JUDGE_TEMPLATE = """You are a strict evaluator of AI assistant answers.

Criteria the answer must satisfy:
{rubric}

Prompt given to the assistant:
{prompt}

Assistant answer:
{output}

Reply on the first line with exactly PASS or FAIL, then one short sentence explaining why."""


@register("llm_judge", "A grader model checks the output against a plain-English `rubric`.",
          ("rubric",))
def _llm_judge(ctx: Context, p: dict):
    if ctx.judge is None:
        return False, "no judge model available"
    reply = ctx.judge(JUDGE_TEMPLATE.format(rubric=p["rubric"], prompt=ctx.prompt,
                                            output=ctx.output)).strip()
    first = next((ln for ln in reply.splitlines() if ln.strip()), "")
    verdict = first.strip().upper().startswith("PASS")
    return verdict, _short(reply, 120)
