import tempfile
import threading
import unittest

from promptcheck import baseline
from promptcheck.cache import Cache
from promptcheck.config import parse_suite
from promptcheck.providers import Completion, Provider, ProviderError, Request, post_json
from promptcheck.runner import run_suite


class Scripted(Provider):
    """Returns canned replies in order (thread-safe) and counts calls."""

    name = "scripted"

    def __init__(self, replies):
        super().__init__("scripted-1")
        self.replies = list(replies)
        self.calls = 0
        self.lock = threading.Lock()

    def complete(self, req: Request) -> Completion:
        with self.lock:
            i = self.calls
            self.calls += 1
        reply = self.replies[i % len(self.replies)]
        if isinstance(reply, Exception):
            raise reply
        return Completion(reply, 5.0, 10, 20)


def suite(tests, **meta):
    return parse_suite({"suite": {"name": "s", **meta}, "tests": tests})


def case(name, *asserts, **extra):
    return {"name": name, "prompt": f"prompt for {name}", "assert": list(asserts), **extra}


CONTAINS_OK = {"type": "contains", "value": "ok"}


class RunnerTests(unittest.TestCase):
    def test_pass_and_fail(self):
        s = suite([case("good", CONTAINS_OK, mock_output="ok"),
                   case("bad", CONTAINS_OK, mock_output="nope")])
        res = run_suite(s)
        by_name = {c.name: c for c in res.cases}
        self.assertTrue(by_name["good"].passed)
        self.assertFalse(by_name["bad"].passed)
        self.assertEqual((res.passed_count, res.failed_count), (1, 1))

    def test_order_is_preserved_with_many_workers(self):
        names = [f"t{i}" for i in range(20)]
        s = suite([case(n, CONTAINS_OK, mock_output="ok") for n in names])
        res = run_suite(s, workers=8)
        self.assertEqual([c.name for c in res.cases], names)

    def test_provider_error_fails_the_case_without_crashing(self):
        s = suite([case("boom", CONTAINS_OK)])
        res = run_suite(s, provider=Scripted([ProviderError("rate limited")]))
        run = res.cases[0].runs[0]
        self.assertFalse(run.passed)
        self.assertIn("rate limited", run.error)

    def test_multiple_runs_and_pass_rate_threshold(self):
        replies = ["ok", "nope", "ok", "ok"]
        strict = suite([case("flaky", CONTAINS_OK, runs=4)])
        lenient = suite([case("flaky", CONTAINS_OK, runs=4, min_pass_rate=0.7)])
        r1 = run_suite(strict, provider=Scripted(replies), workers=1)
        r2 = run_suite(lenient, provider=Scripted(replies), workers=1)
        self.assertEqual(r1.cases[0].pass_rate, 0.75)
        self.assertFalse(r1.cases[0].passed)
        self.assertTrue(r2.cases[0].passed)

    def test_filter_and_tag(self):
        s = suite([case("alpha", CONTAINS_OK, mock_output="ok", tags=["a"]),
                   case("beta", CONTAINS_OK, mock_output="ok", tags=["b"])])
        self.assertEqual([c.name for c in run_suite(s, name_filter="ALP").cases], ["alpha"])
        self.assertEqual([c.name for c in run_suite(s, tag="b").cases], ["beta"])

    def test_cache_avoids_second_call(self):
        s = suite([case("c", CONTAINS_OK)])
        with tempfile.TemporaryDirectory() as d:
            p = Scripted(["ok"])
            run_suite(s, provider=p, cache=Cache(d))
            res = run_suite(s, provider=p, cache=Cache(d))
            self.assertEqual(p.calls, 1)
            self.assertTrue(res.cases[0].runs[0].cached)

    def test_cache_keeps_repeat_runs_distinct(self):
        s = suite([case("c", CONTAINS_OK, runs=2)])
        with tempfile.TemporaryDirectory() as d:
            p = Scripted(["ok", "nope"])
            res = run_suite(s, provider=p, cache=Cache(d), workers=1)
            self.assertEqual(p.calls, 2)
            self.assertEqual(res.cases[0].pass_rate, 0.5)

    def test_token_totals_and_json_dict(self):
        s = suite([case("c", CONTAINS_OK)])
        res = run_suite(s, provider=Scripted(["ok"]))
        self.assertEqual(res.total_tokens, (10, 20))
        d = res.to_dict()
        self.assertEqual(d["passed"], 1)
        self.assertTrue(d["cases"][0]["runs"][0]["passed"])

    def test_judge_assertion_uses_judge_provider(self):
        s = suite([case("j", {"type": "llm_judge", "rubric": "be nice"})])
        res = run_suite(s, provider=Scripted(["hello"]), judge=Scripted(["PASS\nnice"]))
        self.assertTrue(res.cases[0].passed)
        res = run_suite(s, provider=Scripted(["hello"]), judge=Scripted(["FAIL\nrude"]))
        self.assertFalse(res.cases[0].passed)


class BaselineTests(unittest.TestCase):
    def test_regressions_fixes_new_and_missing(self):
        before = run_suite(suite([
            case("stays", CONTAINS_OK, mock_output="ok"),
            case("will_break", CONTAINS_OK, mock_output="ok"),
            case("will_fix", CONTAINS_OK, mock_output="no"),
            case("removed", CONTAINS_OK, mock_output="ok"),
        ]))
        snap = baseline.snapshot(before)
        after = run_suite(suite([
            case("stays", CONTAINS_OK, mock_output="ok"),
            case("will_break", CONTAINS_OK, mock_output="no"),
            case("will_fix", CONTAINS_OK, mock_output="ok"),
            case("brand_new", CONTAINS_OK, mock_output="ok"),
        ]))
        diff = baseline.compare(snap, after)
        self.assertEqual(diff.regressions, ["will_break"])
        self.assertEqual(diff.fixed, ["will_fix"])
        self.assertEqual(diff.new, ["brand_new"])
        self.assertEqual(diff.missing, ["removed"])
        self.assertTrue(diff.has_regressions)

    def test_save_and_load_roundtrip(self):
        res = run_suite(suite([case("a", CONTAINS_OK, mock_output="ok")]))
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/b.json"
            baseline.save(res, path)
            self.assertEqual(baseline.load(path)["cases"]["a"]["passed"], True)

    def test_load_rejects_other_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/x.json"
            with open(path, "w") as fh:
                fh.write('{"hello": 1}')
            with self.assertRaises(ValueError):
                baseline.load(path)


class RetryTests(unittest.TestCase):
    def test_post_json_gives_up_with_provider_error(self):
        sleeps = []
        with self.assertRaises(ProviderError):
            # port 9 on localhost refuses connections immediately
            post_json("http://127.0.0.1:9/x", {}, {}, timeout=1, retries=2, sleep=sleeps.append)
        self.assertEqual(sleeps, [1, 2])


if __name__ == "__main__":
    unittest.main()
