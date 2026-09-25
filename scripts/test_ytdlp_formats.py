"""Regression tests for resilient YouTube format selection."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def functions(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {}
    exec(compile(tree, path, "exec"), namespace)
    return namespace


class YoutubeFormatTests(unittest.TestCase):
    def setUp(self):
        self.ns = functions("core/_1_ytdlp.py")

    def test_resolution_candidates_include_fresh_metadata_fallbacks(self):
        candidates = self.ns["_format_candidates"]("1080")
        self.assertEqual(
            candidates[0],
            "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        )
        self.assertIn("bv*[height<=1080]+ba/b[height<=1080]", candidates)
        self.assertEqual(candidates[-1], "bestvideo+bestaudio/best")

    def test_requested_format_error_is_retryable(self):
        error = RuntimeError("Requested format is not available")
        self.assertTrue(self.ns["_is_format_unavailable_error"](error))
        self.assertFalse(
            self.ns["_is_format_unavailable_error"](RuntimeError("disk full"))
        )


if __name__ == "__main__":
    unittest.main()
