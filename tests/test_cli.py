import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from promptcheck.cli import main
from promptcheck.report import to_html

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def test_passing_example_exits_zero(self):
        code, out, _ = run_cli("run", str(EXAMPLES / "support_bot.toml"), "--no-cache")
        self.assertEqual(code, 0, out)
        self.assertIn("5/5 passed", out)

    def test_failing_example_exits_one_and_explains(self):
        code, out, _ = run_cli("run", str(EXAMPLES / "failing_demo.toml"), "--no-cache")
        self.assertEqual(code, 1)
        self.assertIn("1/3 passed", out)
        self.assertIn("looks like a refusal", out)

    def test_fail_under_allows_partial_pass(self):
        code, _, _ = run_cli("run", str(EXAMPLES / "failing_demo.toml"),
                             "--no-cache", "--fail-under", "0.3")
        self.assertEqual(code, 0)

    def test_reports_are_written_and_valid(self):
        with tempfile.TemporaryDirectory() as d:
            html, js, xml = f"{d}/r.html", f"{d}/r.json", f"{d}/r.xml"
            run_cli("run", str(EXAMPLES / "failing_demo.toml"), "--no-cache",
                    "--html", html, "--json", js, "--junit", xml)
            self.assertIn("<!doctype html>", Path(html).read_text())
            self.assertEqual(json.loads(Path(js).read_text())["failed"], 2)
            root = ET.fromstring(Path(xml).read_text())
            self.assertEqual(root.get("failures"), "2")
            self.assertEqual(len(root.findall("testcase/failure")), 2)

    def test_html_escapes_model_output(self):
        from promptcheck.config import parse_suite
        from promptcheck.runner import run_suite
        s = parse_suite({"tests": [{"name": "x<b>", "prompt": "p",
                                    "mock_output": "<script>alert(1)</script>",
                                    "assert": [{"type": "contains", "value": "nope"}]}]})
        page = to_html(run_suite(s))
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_baseline_flow_detects_regression(self):
        suite_ok = (
            '[[tests]]\nname="t"\nprompt="p"\nmock_output="hello"\n'
            '[[tests.assert]]\ntype="contains"\nvalue="hello"\n'
        )
        with tempfile.TemporaryDirectory() as d:
            path, base = Path(d) / "s.toml", f"{d}/base.json"
            path.write_text(suite_ok)
            code, out, _ = run_cli("run", str(path), "--save-baseline", base)
            self.assertEqual(code, 0)
            path.write_text(suite_ok.replace('mock_output="hello"', 'mock_output="bye"'))
            code, out, _ = run_cli("run", str(path), "--baseline", base)
            self.assertEqual(code, 1)
            self.assertIn("regression: t", out)

    def test_filter_matching_nothing_is_an_error(self):
        code, _, err = run_cli("run", str(EXAMPLES / "support_bot.toml"), "-k", "zzz-none")
        self.assertEqual(code, 2)
        self.assertIn("no tests matched", err)

    def test_missing_file_is_a_clean_error(self):
        code, _, err = run_cli("run", "/nope.toml")
        self.assertEqual(code, 2)
        self.assertIn("not found", err)

    def test_init_creates_a_runnable_suite_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/promptcheck.toml"
            self.assertEqual(run_cli("init", path)[0], 0)
            self.assertEqual(run_cli("run", path, "--no-cache")[0], 0)
            self.assertEqual(run_cli("init", path)[0], 2)

    def test_assertions_listing(self):
        code, out, _ = run_cli("assertions")
        self.assertEqual(code, 0)
        self.assertIn("json_schema", out)
        self.assertIn("llm_judge", out)


if __name__ == "__main__":
    unittest.main()
