"""Regression tests for repeated phrases and backward subtitle timing."""
import ast
import datetime
from difflib import SequenceMatcher
from pathlib import Path
import re
import unittest
from unittest.mock import Mock, mock_open, patch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def functions(path, **scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in tree.body:
        node.decorator_list = []
    exec(compile(tree, path, "exec"), scope)
    return scope

class TimestampTests(unittest.TestCase):
    def alignment(self):
        return functions("core/_6_gen_sub.py", pd=pd, re=re, SequenceMatcher=SequenceMatcher)

    def test_reviewed_window_prevents_jump_to_later_exact_repetition(self):
        words = pd.DataFrame({"text": ["the", "Wendel", "and", "Werder", "different", "guns", "the", "Werndl", "and", "Werder"],
                              "start": [141, 141.3, 141.6, 142, 143, 143.5, 173, 173.3, 173.6, 174],
                              "end": [141.2, 141.5, 141.9, 142.5, 143.4, 144, 173.2, 173.5, 173.9, 174.5]})
        rows = pd.DataFrame({"Source": ["the Werndl and Werder", "different guns"]})
        result = self.alignment()["get_sentence_timestamps"](words, rows, [(141, 142.5), (143, 144)])
        self.assertLess(result[0][1], 144)
        self.assertEqual(result[1], (143, 144))

    def test_missing_local_text_uses_reviewed_timing_without_consuming_next_row(self):
        words = pd.DataFrame({"text": ["hello", "next", "sentence"], "start": [1, 5, 5.5], "end": [2, 5.4, 6]})
        rows = pd.DataFrame({"Source": ["entirely absent words", "next sentence"]})
        self.assertEqual(self.alignment()["get_sentence_timestamps"](words, rows, [(1, 2), (5, 6)]), [(1, 2), (5, 6)])

    def test_audio_task_merger_rejects_backward_srt(self):
        content = "1\n00:02:53,000 --> 00:02:54,000\nfirst\n\n2\n00:02:23,000 --> 00:02:25,000\nsecond\n"
        ns = functions("core/_8_1_audio_task.py", pd=pd, re=re, datetime=datetime,
                       load_key=lambda key: False, TRANS_SUBS_FOR_AUDIO_FILE="translated.srt",
                       SRC_SUBS_FOR_AUDIO_FILE="source.srt", rprint=Mock(), Panel=Mock())
        with patch("builtins.open", mock_open(read_data=content)):
            with self.assertRaisesRegex(ValueError, "backward subtitle timing"):
                ns["process_srt"]()

if __name__ == "__main__":
    unittest.main()
