"""Offline tests of production functions without loading ASR/GPU/API clients."""

import ast
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Tuple
import unittest
from unittest.mock import Mock, MagicMock, mock_open, patch


ROOT = Path(__file__).resolve().parents[1]


def functions(path, **scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in tree.body:
        node.decorator_list = []
    scope = dict(json=json, re=re, math=math, **scope)
    exec(compile(tree, path, "exec"), scope)
    return scope


BOUNDARIES = functions("core/utils/proofread_context.py")
CONTEXT = functions("core/utils/translation_context.py", sentence_spans=BOUNDARIES["sentence_spans"],
                    SequenceMatcher=SequenceMatcher)


def config(key):
    return {"target_language": "简体中文", "whisper.detected_language": "en",
            "translation_context_lines": 4, "translation_max_sentence_lines": 28,
            "reflect_translate": False}[key]


def translator(**overrides):
    scope = {key: CONTEXT[key] for key in ("split_translation_chunks", "select_translation_context",
                                         "sentence_split_point", "assemble_translation_results",
                                         "adjacent_duplicate_groups")}
    scope.update(load_key=config, console=Mock(), search_things_to_note_in_prompt=lambda _: "terms")
    scope.update(overrides)
    return functions("core/_4_2_translate.py", **scope)


class ContextTests(unittest.TestCase):
    def test_sentence_survives_both_soft_budgets(self):
        lines = ["Although the vehicle", "carried a badge,", "it was built elsewhere.", "Next."]
        chunks = CONTEXT["split_translation_chunks"](lines, chunk_chars=20, chunk_lines=2)
        self.assertEqual(chunks, ["\n".join(lines[:3]), "Next."])

    def test_missing_punctuation_bounded_and_lossless(self):
        lines = [f"fragment {i}" for i in range(65)]
        chunks = CONTEXT["split_translation_chunks"](lines)
        self.assertEqual([len(c.splitlines()) for c in chunks], [28, 28, 9])
        self.assertEqual("\n".join(chunks).splitlines(), lines)

    def test_context_reaches_across_neighboring_chunks(self):
        ns = translator()
        chunks = ["First.", "Second.", "Third.", "Target.", "Next.", "Last."]
        self.assertEqual(ns["get_previous_content"](chunks, 3), ["First.", "Second.", "Third."])
        self.assertEqual(ns["get_after_content"](chunks, 3), ["Next.", "Last."])

    def test_context_extends_to_sentence_edges(self):
        left, right = CONTEXT["select_translation_context"](
            ["One", "two", "end."], ["Next", "two", "end."], context_lines=1)
        self.assertEqual(left, ["One", "two", "end."])
        self.assertEqual(right, ["Next", "two", "end."])

    def test_zero_context_empty_and_bad_limits(self):
        self.assertEqual(CONTEXT["select_translation_context"](["Hi."], ["Bye."], 0), ([], []))
        self.assertEqual(CONTEXT["split_translation_chunks"]([]), [])
        for options in ({"chunk_chars": 0}, {"chunk_lines": 0}, {"max_sentence_lines": 0}):
            with self.assertRaises(ValueError):
                CONTEXT["split_translation_chunks"](["Hi."], **options)

    def test_oversized_row_not_split(self):
        row = "x" * 5000
        self.assertEqual(CONTEXT["split_translation_chunks"]([row]), [row])

    def test_fallback_supplies_sibling_context(self):
        calls = []
        def translate(chunk, before, after, terms, theme, index):
            calls.append((chunk, before, after))
            if chunk == "First\nends.\nSecond ends.":
                raise ValueError("bad response")
            return "\n".join("译文" for _ in chunk.splitlines()), chunk
        ns = translator(translate_lines=translate)
        result = ns["_translate_chunk_with_fallback"](
            "First\nends.\nSecond ends.", ["Earlier."], ["Later."], "theme", 0)
        self.assertEqual(len(result[0].splitlines()), 3)
        self.assertEqual(calls[1], ("First\nends.", ["Earlier."], ["Second ends.", "Later."]))
        self.assertEqual(calls[2], ("Second ends.", ["Earlier.", "First", "ends."], ["Later."]))

    def test_single_sentence_failure_not_split_into_isolated_rows(self):
        call = Mock(side_effect=ValueError("bad response"))
        ns = translator(translate_lines=call)
        with self.assertRaises(ValueError):
            ns["_translate_chunk_with_fallback"]("If this\nthen that.", [], [], "theme", 0)
        self.assertEqual(call.call_count, 1)

    def test_out_of_order_repeated_source_matches_by_id(self):
        source, result = CONTEXT["assemble_translation_results"](
            ["Same.", "Same."], [(1, "Same.", "译文乙"), (0, "Same.", "译文甲")])
        self.assertEqual(result, ["译文甲", "译文乙"])
        self.assertEqual(source, ["Same.", "Same."])

    def test_invalid_results_fail(self):
        for result in ([], [(0, "Other.", "译文")], [(0, "Same.", "")],
                       [(1, "Same.", "译文")], [(0, "Same.", "译文"), (0, "Same.", "译文")]):
            with self.assertRaises(ValueError):
                CONTEXT["assemble_translation_results"](["Same."], result)

    def test_adjacent_duplicate_screen_finds_phrase_number_and_similarity(self):
        groups = CONTEXT["adjacent_duplicate_groups"](
            ["capturing about 33%", "of the market by 1993."],
            ["占据北美市场约33%的份额", "北美市场约33%的份额"],
        )
        self.assertEqual([(g["start"], g["end"]) for g in groups], [(0, 1)])
        self.assertTrue(any("33%" in reason for reason in groups[0]["reasons"]))

    def test_duplicate_screen_stays_inside_source_sentence(self):
        groups = CONTEXT["adjacent_duplicate_groups"](
            ["First engine.", "Second engine."], ["可靠性非常重要", "可靠性非常重要"])
        self.assertEqual(groups, [])


class PromptAndValidationTests(unittest.TestCase):
    def setUp(self):
        self.prompts = functions("core/prompts.py", sentence_spans=BOUNDARIES["sentence_spans"], load_key=config)
        self.translation = functions("core/translate_lines.py")

    def test_quality_rules_in_both_translation_passes(self):
        lines = "If this\nthen that."
        for prompt in (self.prompts["get_prompt_faithfulness"](lines, "context"),
                       self.prompts["get_prompt_expressiveness"]({"1": {"direct": "如果"}, "2": {"direct": "那么"}}, lines, "context")):
            self.assertIn("newline is NOT a sentence ending", prompt)
            self.assertIn("Every source proposition must appear exactly once", prompt)
            self.assertIn('[["1", "2"]]', prompt)
            self.assertIn("read-only", prompt)

    def test_expressiveness_orders_by_id_not_json_order(self):
        prompt = self.prompts["get_prompt_expressiveness"](
            {"2": {"direct": "乙"}, "1": {"direct": "甲"}}, "First.\nSecond.", "")
        self.assertIn('{"id":"1","source":"First.","direct":"甲"}', prompt)

    def test_context_labeled_read_only(self):
        prompt = self.prompts["generate_shared_prompt"](["Before."], ["After."], "theme", "terms")
        self.assertEqual(prompt.count("do NOT translate into output"), 2)

    def test_nonempty_exact_translation_schema(self):
        validate = self.translation["valid_translate_result"]
        for value in (None, 4, [], {}, " "):
            self.assertEqual(validate({"1": {"direct": value}}, ["1"], ["direct"])["status"], "error")
        self.assertEqual(validate({"1": {"direct": "好"}, "2": {"direct": "多余"}}, ["1"], ["direct"])["status"], "error")
        self.assertEqual(validate({"1": {"translation": "好"}}, ["1"], ["direct"])["status"], "success")

    def test_list_ids_and_duplicate_ids(self):
        normalize = self.translation["normalize_translate_result"]
        result = normalize([{"id": "2", "direct": "乙"}, {"id": "1", "direct": "甲"}], ["direct"])
        self.assertEqual(result["1"]["direct"], "甲")
        self.assertEqual(normalize([{"id": "1", "direct": "甲"}, {"id": "1", "direct": "乙"}], ["direct"]), {})
        self.assertEqual(normalize({"1": "甲", " 1 ": "乙"}, ["direct"]), {})

    def test_one_model_call_and_validator_passed(self):
        call = Mock(return_value={"2": {"direct": "那么"}, "1": {"direct": "如果"}})
        ns = functions("core/translate_lines.py", load_key=config,
                       generate_shared_prompt=self.prompts["generate_shared_prompt"],
                       get_prompt_faithfulness=self.prompts["get_prompt_faithfulness"],
                       stage_api_config=lambda _: {}, ask_gpt=call,
                       Table=Mock(), box=Mock(), console=Mock())
        translated, source = ns["translate_lines"]("If this\nthen that.", [], [], "terms", "theme")
        self.assertEqual(translated, "如果\n那么")
        self.assertEqual(source, "If this\nthen that.")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.kwargs["valid_def"]({})["status"], "error")


class HardPassTests(unittest.TestCase):
    def test_risky_span_expands_and_preserves_other_rows_and_timestamps(self):
        import pandas as pd
        frame = pd.DataFrame({"Source": ["Before.", "If this", "then that.", "After."],
                              "Translation": ["之前", "如果", "那么", "之后"],
                              "timestamp": ["a", "b", "c", "d"]}, index=[5, 10, 15, 20])
        settings = {"llm_stages.hard_translation.enabled": True,
                    "llm_stages.hard_translation.max_ratio": .1,
                    "llm_stages.hard_translation.min_score": 3,
                    "translation_context_lines": 4, "translation_max_sentence_lines": 28}
        def ask(prompt, **kwargs):
            self.assertIn('"id": "1"', prompt)  # Context before.
            self.assertIn('[["2", "3"]]', prompt)
            response = {"2": {"translation": "条件"}, "3": {"translation": "结果"}}
            self.assertEqual(kwargs["valid_def"](response)["status"], "success")
            self.assertEqual(kwargs["valid_def"]({"2": {"translation": None}})["status"], "error")
            return response
        ns = translator(load_key=settings.__getitem__, staged_llm_enabled=lambda: True,
                        risky_row_indices=lambda *a, **kw: [10], sentence_risk_score=lambda *a: 3,
                        stage_api_config=lambda _: {}, proofreading_windows=BOUNDARIES["proofreading_windows"], ask_gpt=ask)
        result = ns["_refine_hard_translations"](frame)
        self.assertEqual(result["Translation"].tolist(), ["之前", "条件", "结果", "之后"])
        self.assertEqual(result["Hard Translation Applied"].tolist(), [False, True, True, False])
        self.assertEqual(result["timestamp"].tolist(), frame["timestamp"].tolist())
        self.assertEqual(result["Initial Translation"].tolist(), frame["Translation"].tolist())
        self.assertEqual(frame["Translation"].tolist(), ["之前", "如果", "那么", "之后"])

    def test_adjacent_duplicate_repair_changes_only_flagged_rows(self):
        import pandas as pd
        frame = pd.DataFrame({
            "Source": ["Before.", "capturing 33%", "of the market by 1993."],
            "Translation": ["之前", "占市场33%份额", "市场33%份额"],
        })
        settings = {
            "translation_duplicate_audit.enabled": True,
            "translation_duplicate_audit.min_phrase_chars": 4,
            "translation_duplicate_audit.similarity_threshold": .72,
            "translation_max_sentence_lines": 28,
        }
        call = Mock(return_value={"2": {"translation": "到1993年占据"},
                                  "3": {"translation": "市场三分之一份额"}})
        ns = translator(load_key=settings.__getitem__, staged_llm_enabled=lambda: True,
                        stage_api_config=lambda _: {}, ask_gpt=call)
        result = ns["_repair_adjacent_duplicates"](frame)
        self.assertEqual(result["Translation"].tolist(), ["之前", "到1993年占据", "市场三分之一份额"])
        self.assertEqual(result["Adjacent Duplicate Repaired"].tolist(), [False, True, True])
        self.assertEqual(call.call_count, 1)


class TTSPreflightTests(unittest.TestCase):
    def test_oversized_final_block_is_shortened_before_inference(self):
        import pandas as pd
        frame = pd.DataFrame([{
            "number": 7, "duration": 2.5, "tol_dur": 2.5,
            "text": "这是非常非常冗长的字幕文本", "lines": ["这是非常非常冗长的字幕文本"],
            "src_lines": ["This is the source."], "real_dur": 0.0, "est_dur": 8.0,
        }])
        settings = {"tts_preflight.enabled": True, "speed_factor.max": 1.55,
                    "tts_preflight.estimator_safety_ratio": 1.15,
                    "tts_preflight.target_fill_ratio": .95}
        estimates = lambda text, _: len(re.sub(r"\W", "", text)) * .5
        ask = Mock(return_value={"result": "精简字幕"})
        fake_path = Mock(return_value=Mock(glob=Mock(return_value=[])))
        ns = functions(
            "core/_10_gen_audio.py", pd=pd, Tuple=Tuple, Path=fake_path,
            contextmanager=lambda fn: fn, load_key=settings.__getitem__,
            init_estimator=lambda: object(), estimate_duration=estimates,
            get_tts_preflight_prompt=lambda *args: "prompt", ask_gpt=ask,
            stage_api_config=lambda _: {}, rprint=Mock(), save_audio_tasks=Mock(),
            _AUDIO_TMP_DIR="tmp", _AUDIO_SEGS_DIR="segs",
        )
        ns["save_audio_tasks"] = Mock()
        result, changed = ns["preflight_tts_tasks"](frame)
        self.assertEqual(changed, [7])
        self.assertEqual(result.loc[0, "lines"], ["精简字幕"])
        self.assertTrue(result.loc[0, "TTS Preflight Trimmed"])
        self.assertEqual(ask.call_count, 1)


class PipelineTests(unittest.TestCase):
    def test_translation_pipeline_preserves_id_order_timing_and_pretrim_text(self):
        import concurrent.futures
        import pandas as pd
        settings = {"translation_chunk_chars": 1600, "translation_chunk_lines": 12,
                    "max_workers": 2, "min_trim_duration": 3.5}
        def timing(frame):
            result = frame.copy()
            result["timestamp"] = ["original timing 1", "original timing 2"]
            result["duration"] = [4.0, 2.0]
            return result
        ns = translator(pd=pd, concurrent=concurrent,
                        load_key=settings.__getitem__, Progress=MagicMock(),
                        SpinnerColumn=Mock(), TextColumn=Mock(),
                        _4_1_TERMINOLOGY="fake_terminology.json", _4_2_TRANSLATION="never_written.xlsx",
                        check_len_then_trim=lambda text, duration: text + "压缩")
        ns["split_chunks_by_chars"] = lambda **kw: ["Same.", "Same."]
        ns["translate_chunk"] = lambda chunk, chunks, theme, index: (index, chunk, ["第一段", "第二段"][index])
        ns["_apply_uploaded_subtitle_timestamps"] = timing
        ns["_refine_hard_translations"] = lambda frame: frame
        with patch("builtins.open", mock_open(read_data='{"theme":"topic"}')), \
                patch.object(pd.DataFrame, "to_excel", autospec=True) as save:
            ns["translate_all"]()
        self.assertEqual(save.call_count, 1)
        result = save.call_args.args[0]
        self.assertEqual(result["Translation"].tolist(), ["第一段压缩", "第二段"])
        self.assertEqual(result["Translation Before Timing Trim"].tolist(), ["第一段", "第二段"])
        self.assertEqual(result["timestamp"].tolist(), ["original timing 1", "original timing 2"])
        self.assertTrue(all(result["Translation Workflow"] == "sentence_context_v1"))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        from collections import Counter
        self.ns = functions("core/utils/proofread_evaluation.py", Counter=Counter,
                            VERDICTS={"fixed", "regression", "mixed", "style_only", "no_issue", "missed", "uncertain"},
                            CHANGED_VERDICTS={"fixed", "regression", "mixed", "style_only"},
                            UNCHANGED_VERDICTS={"no_issue", "missed"})

    def row(self, verdict="", changed=True, reviewed=True):
        return {"Original Translation": "原文", "LLM Proofread": "修改" if changed else "原文",
                "Proofread Reviewed": reviewed, "Proofread Status": "corrected" if changed else "ok",
                "Human Verdict": verdict}

    def test_model_changed_is_not_a_verified_fix(self):
        result = self.ns["summarize_records"]([self.row(), self.row(changed=False)])
        self.assertEqual(result["counts"]["pending_rows"], 2)
        self.assertEqual(result["human_verdicts"]["fixed"], 0)

    def test_human_outcomes_separate_from_model(self):
        result = self.ns["summarize_records"]([self.row("fixed"), self.row("regression"),
                   self.row("style_only"), self.row("mixed"), self.row("no_issue", False),
                   self.row("missed", False), self.row("uncertain", False)])
        self.assertEqual(result["counts"]["labeled_rows"], 7)
        self.assertTrue(all(count == 1 for count in result["human_verdicts"].values()))

    def test_invalid_and_skipped_labels_not_credited(self):
        result = self.ns["summarize_records"]([self.row("fixed", False), self.row("missed"),
                                             self.row("typo"), self.row("missed", False, False)])
        self.assertEqual(result["counts"]["invalid_labels"], 3)
        self.assertEqual(result["counts"]["unreviewed_labels"], 1)
        self.assertEqual(result["counts"]["labeled_rows"], 0)

    def test_nan_legacy_and_actual_text_comparison(self):
        row = self.row(float("nan"))
        row.pop("Proofread Reviewed")
        row["Proofread Changed"] = False
        result = self.ns["summarize_records"]([row])
        self.assertEqual(result["counts"]["unknown_reviewed_rows"], 1)
        self.assertEqual(result["counts"]["changed_rows"], 1)
        self.assertEqual(result["counts"]["pending_rows"], 1)

    def test_aggregate_sums_videos(self):
        summaries = [self.ns["summarize_records"]([self.row("fixed")]),
                     self.ns["summarize_records"]([self.row("regression")])]
        result = self.ns["combine_summaries"](summaries)
        self.assertEqual(result["counts"]["rows"], 2)
        self.assertEqual(result["human_verdicts"]["fixed"], 1)
        self.assertEqual(result["human_verdicts"]["regression"], 1)


if __name__ == "__main__":
    unittest.main()
