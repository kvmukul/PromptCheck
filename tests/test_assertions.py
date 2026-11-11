import unittest

from promptcheck.assertions import (Context, SpecError, extract_json, run_assertion,
                                    validate_schema, validate_spec)


def check(type_, output, latency=0.0, judge=None, **params):
    return run_assertion(type_, params, Context(output, latency, "prompt", judge))[0]


class TextAssertions(unittest.TestCase):
    def test_contains(self):
        self.assertTrue(check("contains", "Hello World", value="World"))
        self.assertFalse(check("contains", "Hello World", value="world"))
        self.assertTrue(check("contains", "Hello World", value="world", ignore_case=True))

    def test_not_contains(self):
        self.assertTrue(check("not_contains", "abc", value="z"))
        self.assertFalse(check("not_contains", "abc", value="b"))

    def test_contains_all_any(self):
        self.assertTrue(check("contains_all", "a b c", values=["a", "c"]))
        self.assertFalse(check("contains_all", "a b", values=["a", "c"]))
        self.assertTrue(check("contains_any", "a b", values=["x", "b"]))
        self.assertFalse(check("contains_any", "a b", values=["x", "y"]))

    def test_equals_strips_whitespace(self):
        self.assertTrue(check("equals", "  yes\n", value="yes"))

    def test_one_of_and_starts_with(self):
        self.assertTrue(check("one_of", "b", values=["a", "b"]))
        self.assertFalse(check("one_of", "c", values=["a", "b"]))
        self.assertTrue(check("starts_with", "  Sure, here", value="Sure"))

    def test_regex(self):
        self.assertTrue(check("regex", "order #1234", pattern=r"#\d{4}"))
        self.assertTrue(check("regex", "HELLO", pattern="hello", flags="i"))
        self.assertFalse(check("regex", "abc", pattern="[0-9]"))

    def test_invalid_regex_fails_instead_of_crashing(self):
        ok, msg = run_assertion("regex", {"pattern": "("}, Context("x"))
        self.assertFalse(ok)
        self.assertIn("invalid regex", msg)

    def test_not_refusal(self):
        self.assertTrue(check("not_refusal", "Here is your answer."))
        self.assertFalse(check("not_refusal", "I'm sorry, I can't do that."))


class LengthAndLatency(unittest.TestCase):
    def test_words_and_chars(self):
        self.assertTrue(check("max_words", "one two three", value=3))
        self.assertFalse(check("max_words", "one two three", value=2))
        self.assertTrue(check("min_words", "one two", value=2))
        self.assertTrue(check("max_chars", "abc", value=3))
        self.assertFalse(check("max_chars", "abcd", value=3))

    def test_latency(self):
        self.assertTrue(check("latency_ms", "x", latency=50, max=100))
        self.assertFalse(check("latency_ms", "x", latency=150, max=100))


class JsonAssertions(unittest.TestCase):
    def test_extract_variants(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! Here you go: {"a": [1, 2]} hope it helps'),
                         {"a": [1, 2]})
        with self.assertRaises(ValueError):
            extract_json("no json here")

    def test_strict_rejects_fences(self):
        self.assertTrue(check("json_valid", '```json\n{"a":1}\n```'))
        self.assertFalse(check("json_valid", '```json\n{"a":1}\n```', strict=True))

    def test_schema(self):
        schema = {"type": "object", "required": ["n"],
                  "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 5}}}
        self.assertEqual(validate_schema({"n": 3}, schema), [])
        self.assertTrue(validate_schema({}, schema))
        self.assertTrue(validate_schema({"n": 9}, schema))
        self.assertTrue(validate_schema({"n": True}, schema))  # bool is not an integer
        self.assertTrue(validate_schema({"n": "3"}, schema))

    def test_schema_arrays_and_enums(self):
        schema = {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}}
        self.assertEqual(validate_schema(["a", "b"], schema), [])
        errs = validate_schema(["a", "z"], schema)
        self.assertEqual(len(errs), 1)
        self.assertIn("[1]", errs[0])

    def test_json_schema_assertion(self):
        self.assertTrue(check("json_schema", '{"n": 2}', schema={"type": "object", "required": ["n"]}))
        self.assertFalse(check("json_schema", "not json", schema={"type": "object"}))


class SemanticAssertions(unittest.TestCase):
    def test_similarity(self):
        self.assertTrue(check("similarity", "Refunds within 30 days.",
                              reference="refunds within 30 days", threshold=0.9))
        self.assertFalse(check("similarity", "totally different", reference="abc", threshold=0.8))

    def test_llm_judge_uses_first_line_verdict(self):
        self.assertTrue(check("llm_judge", "x", rubric="r", judge=lambda p: "PASS\nlooks good"))
        self.assertFalse(check("llm_judge", "x", rubric="r", judge=lambda p: "FAIL\nnope"))
        self.assertFalse(check("llm_judge", "x", rubric="r", judge=lambda p: "I think PASS"))

    def test_llm_judge_prompt_contains_rubric_and_output(self):
        seen = []
        check("llm_judge", "THE-ANSWER", rubric="BE-POLITE", judge=lambda p: seen.append(p) or "PASS")
        self.assertIn("BE-POLITE", seen[0])
        self.assertIn("THE-ANSWER", seen[0])

    def test_llm_judge_without_judge_fails(self):
        self.assertFalse(check("llm_judge", "x", rubric="r"))


class Validation(unittest.TestCase):
    def test_unknown_type(self):
        with self.assertRaises(SpecError):
            validate_spec("nope", {})

    def test_missing_required(self):
        with self.assertRaises(SpecError):
            validate_spec("contains", {})


if __name__ == "__main__":
    unittest.main()
