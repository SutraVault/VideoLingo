"""Offline regression tests; no model calls and no workbook writes.

Run: python -m unittest discover -s scripts -p test_proofread_context.py -v
"""

import ast
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "proofread_context", ROOT / "core/utils/proofread_context.py"
)
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


def rows(*sources):
    return [{"id": i + 1, "source": source, "translation": f"译文{i + 1}"}
            for i, source in enumerate(sources)]


def load_functions(**namespace):
    # Load the actual production functions without core/__init__ importing GPU,
    # ASR, Streamlit or API clients. Only external dependencies are mocked.
    names = {"_build_prompt", "_valid_proofread", "_proofread_setting",
             "_proofread_translation_unlocked", "_set_proofread_changed_column",
             "_normalized_comparison_text"}
    tree = ast.parse((ROOT / "core/_4_3_proofread_translation.py").read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"json": json, "proofreading_windows": context.proofreading_windows,
             "PROOFREAD_ISSUE_TYPES": context.PROOFREAD_ISSUE_TYPES,
             "validate_proofread_response": context.validate_proofread_response}
    scope.update(namespace)
    exec(compile(tree, "_4_3_proofread_translation.py", "exec"), scope)
    return scope


class WindowTests(unittest.TestCase):
    def test_sparse_risk_expands_sentences_not_gaps(self):
        data = rows("Beginning", "middle", "end.", "Unrelated.", "Next", "end.")
        windows = list(context.proofreading_windows(data, selected_indices={1, 4}))
        self.assertEqual([[r["id"] for r in w["rows"]] for w in windows], [[1, 2, 3], [5, 6]])

    def test_sentence_not_split_at_soft_target(self):
        data = rows("First", "second", "third", "end.", "Next.")
        windows = list(context.proofreading_windows(data, chunk_lines=2))
        self.assertEqual([len(w["rows"]) for w in windows], [4, 1])

    def test_adjacent_sentences_have_separate_redistribution_groups(self):
        window = list(context.proofreading_windows(rows("Start", "end.", "Next.")))[0]
        self.assertEqual(window["semantic_groups"], [["1", "2"], ["3"]])

    def test_context_expands_to_sentence_edges(self):
        data = rows("start", "middle", "end.", "Target.", "next", "middle", "end.")
        window = list(context.proofreading_windows(data, selected_indices={3}, context_lines=1))[0]
        self.assertEqual(len(window["context_before"]), 3)
        self.assertEqual(len(window["context_after"]), 3)
        self.assertEqual([r["id"] for r in window["rows"]], [4])

    def test_no_context_and_edges(self):
        data = rows("First.", "Second.")
        for window in context.proofreading_windows(data, chunk_lines=1, context_lines=0):
            self.assertEqual(window["context_before"], [])
            self.assertEqual(window["context_after"], [])
        window = list(context.proofreading_windows(data))[0]
        self.assertEqual(window["context_before"] + window["context_after"], [])

    def test_missing_punctuation_is_bounded_and_unique(self):
        data = rows(*(["fragment"] * 100))
        windows = list(context.proofreading_windows(data, max_sentence_lines=10))
        self.assertEqual([r["id"] for w in windows for r in w["rows"]], list(range(1, 101)))
        self.assertTrue(all(len(w["rows"]) <= 10 for w in windows))
        self.assertTrue(all(len(w["context_before"]) <= 13 for w in windows))

    def test_quotes_abbreviations_and_chinese(self):
        self.assertEqual(context.sentence_spans(rows("Dr.", "Smith said hello.\"", "你好。", "Next!")),
                         [(0, 2), (2, 3), (3, 4)])

    def test_empty_and_zero_risk(self):
        self.assertEqual(list(context.proofreading_windows([])), [])
        self.assertEqual(list(context.proofreading_windows(rows("A."), selected_indices=set())), [])

    def test_invalid_settings(self):
        for options in ({"chunk_lines": 0}, {"context_lines": -1}, {"max_sentence_lines": 0}):
            with self.assertRaises(ValueError):
                list(context.proofreading_windows(rows("A."), **options))


class ResponseTests(unittest.TestCase):
    def test_complete_response(self):
        response = {"1": {"status": "corrected", "issue_types": ["meaning"],
                          "reason": "根据原文纠正词义。", "proofread": "修订"}}
        self.assertEqual(context.validate_proofread_response(rows("A"), response)["status"], "success")

    def test_ok_can_return_unchanged_without_forced_edits(self):
        response = {"1": {"status": "ok", "issue_types": [],
                          "reason": "原译保留了原文的问候语义。", "proofread": "译文1"}}
        self.assertEqual(context.validate_proofread_response(rows("Hello."), response)["status"], "success")

    def test_needs_review_preserves_uncertain_source(self):
        response = {"1": {"status": "needs_review", "issue_types": ["source_unclear"],
                          "reason": "型号疑似转录错误，需听原音确认。", "proofread": "译文1"}}
        self.assertEqual(context.validate_proofread_response(rows("Unclear model."), response)["status"], "success")

    def test_inconsistent_or_missing_audit_rejected(self):
        good = {"status": "corrected", "issue_types": ["meaning"],
                "reason": "纠正原文词义。", "proofread": "新译文"}
        for update in ({"status": "ok"}, {"status": "needs_review"},
                       {"status": "invented"}, {"proofread": "译文1"},
                       {"issue_types": []}, {"issue_types": "meaning"},
                       {"issue_types": ["unknown"]}, {"issue_types": [{}]},
                       {"reason": " "}, {"reason": None}):
            with self.subTest(update=update):
                response = {"1": dict(good, **update)}
                self.assertEqual(context.validate_proofread_response(rows("Hello."), response)["status"], "error")
        for missing in ("status", "issue_types", "reason"):
            item = good.copy()
            del item[missing]
            self.assertEqual(context.validate_proofread_response(rows("Hello."), {"1": item})["status"], "error")

    def test_issue_status_consistency(self):
        for status, issues in (("ok", ["grammar"]), ("needs_review", [])):
            item = {"status": status, "issue_types": issues,
                    "reason": "检查说明", "proofread": "译文1"}
            self.assertEqual(context.validate_proofread_response(rows("Hello."), {"1": item})["status"], "error")

    def test_partial_or_invalid_response_rejected_without_mutation(self):
        data = rows("A", "B.")
        for response in ({}, {"1": {"proofread": "调整后"}},
                         {"1": {"proofread": "调整后"}, "2": {"proofread": None}},
                         {"1": {"proofread": "调整后"}, "2": {"proofread": 42}},
                         {"1": {"proofread": "调整后"}, "2": {"proofread": "  "}},
                         {"1": "bad", "2": {}}, [],
                         {"1": {"proofread": "a"}, "2": {"proofread": "b"}, "3": {"proofread": "context"}}):
            before = json.dumps(response)
            self.assertEqual(context.validate_proofread_response(data, response)["status"], "error")
            self.assertEqual(json.dumps(response), before)

    def test_prompt_readonly_and_editable_separated(self):
        scope = load_functions(load_key=lambda key: "简体中文" if key == "target_language" else "en")
        data = rows("Before.", "Start", "end.", "After.")
        window = list(context.proofreading_windows(data, selected_indices={1}))[0]
        prompt = scope["_build_prompt"](**window)
        template = json.loads(prompt.split("Output only JSON in this shape:\n")[1])
        self.assertEqual(set(template), {"2", "3"})
        self.assertIn("Before.", prompt)
        self.assertIn("After.", prompt)
        self.assertIn("MAY redistribute meaning", prompt)
        self.assertIn("misplaced punctuation ARE within scope", prompt)
        self.assertIn("Do not force a minimum number of changes", prompt)
        self.assertNotIn("Do not add explanations or comments", prompt)
        self.assertEqual(list(template["2"]), ["status", "issue_types", "reason", "proofread"])


class PipelineTests(unittest.TestCase):
    def setup_pipeline(self, mode="suggest", selected=True):
        import pandas as pd
        frame = pd.DataFrame({"Source": ["Before.", "If this", "then that.", "After."],
                              "Translation": ["前文", "如果这样", "就那样", "后文"],
                              "timestamp": ["t1", "t2", "t3", "t4"]}, index=[10, 20, 30, 40])
        config = {"llm_proofread.chunk_lines": 7, "llm_proofread.mode": mode,
                  "llm_stages.proofread.only_risky": selected,
                  "target_language": "简体中文", "whisper.detected_language": "en"}
        saved = []
        scope = load_functions(
            load_key=config.__getitem__, staged_llm_enabled=lambda: True,
            risky_row_indices=lambda *a, **kw: [20],
            read_excel_with_aliases=lambda *a, **kw: frame.copy(),
            _proofread_api_config=lambda: {},
            _4_3_PROOFREAD_TRANSLATION="proofread.xlsx", _4_2_TRANSLATION="translation.xlsx",
            _safe_to_excel=lambda df, path: saved.append((path, df.copy())),
            _trim_for_subtitle_timing=lambda df: df,
            console=Mock(), os=Mock(),
        )
        scope["os"].path.exists.return_value = False
        def ask(prompt, **kwargs):
            template = json.loads(prompt.split("Output only JSON in this shape:\n")[1])
            result = {key: {"status": "corrected", "issue_types": ["word_order"],
                            "reason": "调整相邻行的条件与结果语序。", "proofread": "修订" + key}
                      for key in template}
            self.assertEqual(kwargs["valid_def"](result)["status"], "success")
            self.assertFalse(kwargs["use_cache"])
            return result
        scope["ask_gpt"] = Mock(side_effect=ask)
        return frame, saved, scope

    def test_risky_pipeline_keeps_context_source_and_timing(self):
        frame, saved, scope = self.setup_pipeline()
        scope["_proofread_translation_unlocked"](force=True)
        self.assertEqual(len(saved), 1)
        result = saved[0][1]
        self.assertEqual(result["LLM Proofread"].tolist(), ["前文", "修订2", "修订3", "后文"])
        self.assertEqual(result["Proofread Reviewed"].tolist(), [False, True, True, False])
        self.assertEqual(result["Proofread Status"].tolist(), ["not_reviewed", "corrected", "corrected", "not_reviewed"])
        self.assertEqual(result["Proofread Issues"].tolist(), ["", "word_order", "word_order", ""])
        self.assertEqual(result.iloc[1]["Proofread Reason"], "调整相邻行的条件与结果语序。")
        for name in ("Source", "Translation", "timestamp", "Original Translation"):
            self.assertEqual(result[name].tolist(), frame["Translation" if name == "Original Translation" else name].tolist())

    def test_full_apply_pipeline(self):
        _, saved, scope = self.setup_pipeline(mode="apply", selected=False)
        scope["_proofread_translation_unlocked"](force=True)
        self.assertEqual([path for path, _ in saved], ["translation.xlsx", "proofread.xlsx"])
        self.assertEqual(saved[0][1]["Translation"].tolist(), ["修订1", "修订2", "修订3", "修订4"])
        self.assertTrue(all(saved[1][1]["Proofread Status"] == "corrected"))

    def test_zero_risk_does_not_call_model(self):
        _, saved, scope = self.setup_pipeline()
        scope["risky_row_indices"] = lambda *a, **kw: []
        scope["_proofread_translation_unlocked"](force=True)
        scope["ask_gpt"].assert_not_called()
        self.assertFalse(saved[0][1]["Proofread Reviewed"].any())
        self.assertTrue(all(saved[0][1]["Proofread Status"] == "not_reviewed"))

    def test_mixed_audit_persists_unchanged_findings(self):
        frame, saved, scope = self.setup_pipeline(selected=False)
        def ask(prompt, **kwargs):
            response = {str(i + 1): {"status": "ok", "issue_types": [],
                                     "reason": "核对后原译保留了原文语义。", "proofread": text}
                        for i, text in enumerate(frame["Translation"])}
            response["2"].update(status="needs_review", issue_types=["source_unclear"],
                                 reason="原文条件含义不明确，需听原音。")
            response["3"].update(status="corrected", issue_types=["grammar"],
                                 reason="修复结果分句的语法。", proofread="那么就这样")
            self.assertEqual(kwargs["valid_def"](response)["status"], "success")
            return response
        scope["ask_gpt"].side_effect = ask
        scope["_proofread_translation_unlocked"](force=True)
        result = saved[0][1]
        self.assertEqual(result["Proofread Status"].tolist(), ["ok", "needs_review", "corrected", "ok"])
        self.assertEqual(result["Proofread Changed"].tolist(), [False, False, True, False])
        self.assertEqual(result.iloc[1]["LLM Proofread"], "如果这样")
        self.assertIn("需听原音", result.iloc[1]["Proofread Reason"])

    def test_no_change_audit_still_has_reasons(self):
        frame, saved, scope = self.setup_pipeline(selected=False)
        def ask(prompt, **kwargs):
            response = {str(i + 1): {"status": "ok", "issue_types": [],
                                     "reason": "核对条件与结果衔接，无具体错误。", "proofread": text}
                        for i, text in enumerate(frame["Translation"])}
            self.assertEqual(kwargs["valid_def"](response)["status"], "success")
            return response
        scope["ask_gpt"].side_effect = ask
        scope["_proofread_translation_unlocked"](force=True)
        result = saved[0][1]
        self.assertFalse(result["Proofread Changed"].any())
        self.assertTrue(result["Proofread Reviewed"].all())
        self.assertTrue(result["Proofread Reason"].str.len().gt(0).all())

    def test_failed_request_does_not_save_partial_batch(self):
        _, saved, scope = self.setup_pipeline()
        scope["ask_gpt"].side_effect = ValueError("validation retries exhausted")
        with self.assertRaises(ValueError):
            scope["_proofread_translation_unlocked"](force=True)
        self.assertEqual(saved, [])


if __name__ == "__main__":
    unittest.main()
