import tempfile
import unittest
from pathlib import Path

from promptcheck.config import ConfigError, load_suite, parse_suite, render


class Rendering(unittest.TestCase):
    def test_render(self):
        self.assertEqual(render("Hi {{ name }}!", {"name": "Ada"}, "t"), "Hi Ada!")

    def test_missing_variable(self):
        with self.assertRaises(ConfigError):
            render("Hi {{name}}", {}, "t")


class Parsing(unittest.TestCase):
    def base(self, **test):
        t = {"name": "t1", "prompt": "hello"}
        t.update(test)
        return {"suite": {"name": "s"}, "tests": [t]}

    def test_minimal(self):
        suite = parse_suite(self.base())
        self.assertEqual(suite.name, "s")
        self.assertEqual(suite.provider, "mock")
        self.assertEqual(suite.cases[0].runs, 1)

    def test_vars_merge_and_test_overrides_suite(self):
        data = self.base(prompt="{{a}}-{{b}}", vars={"b": "test"})
        data["suite"]["vars"] = {"a": "suite", "b": "suite"}
        self.assertEqual(parse_suite(data).cases[0].prompt, "suite-test")

    def test_defaults_are_prepended(self):
        data = self.base(**{"assert": [{"type": "contains", "value": "x"}]})
        data["defaults"] = {"assert": [{"type": "not_refusal"}]}
        types = [a.type for a in parse_suite(data).cases[0].assertions]
        self.assertEqual(types, ["not_refusal", "contains"])

    def test_rejects_unknown_assertion(self):
        with self.assertRaises(ConfigError):
            parse_suite(self.base(**{"assert": [{"type": "bogus"}]}))

    def test_rejects_missing_param(self):
        with self.assertRaises(ConfigError):
            parse_suite(self.base(**{"assert": [{"type": "contains"}]}))

    def test_rejects_duplicates_and_empty(self):
        data = self.base()
        data["tests"].append({"name": "t1", "prompt": "x"})
        with self.assertRaises(ConfigError):
            parse_suite(data)
        with self.assertRaises(ConfigError):
            parse_suite({"suite": {}})

    def test_rejects_bad_pass_rate(self):
        with self.assertRaises(ConfigError):
            parse_suite(self.base(min_pass_rate=0))


class Files(unittest.TestCase):
    def test_load_and_prompt_file(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "p.txt").write_text("Translate {{word}}")
            (Path(d) / "s.toml").write_text(
                '[suite]\nname="x"\n[suite.vars]\nword="cat"\n'
                '[[tests]]\nname="a"\nprompt_file="p.txt"\n'
            )
            suite = load_suite(Path(d) / "s.toml")
            self.assertEqual(suite.cases[0].prompt, "Translate cat")

    def test_errors(self):
        with self.assertRaises(ConfigError):
            load_suite("/nonexistent/file.toml")
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.toml"
            bad.write_text("this is = = not toml")
            with self.assertRaises(ConfigError):
                load_suite(bad)

    def test_shipped_examples_load(self):
        root = Path(__file__).resolve().parent.parent / "examples"
        for f in root.glob("*.toml"):
            self.assertTrue(load_suite(f).cases, f.name)


if __name__ == "__main__":
    unittest.main()
