"""Regression coverage for IndexTTS2 row recovery and chunk-level fitting."""

import unittest
from unittest.mock import patch

import pandas as pd

import core._10_gen_audio as audio


class IndexTTS2ChunkFitTests(unittest.TestCase):
    def test_adjacent_slack_keeps_an_isolated_overflow_in_its_chunk(self):
        tasks = pd.DataFrame(
            [
                {
                    "number": 12,
                    "start_time": "00:01:02.560",
                    "end_time": "00:01:06.319",
                    "tol_dur": 3.759,
                    "tolerance": 0.0,
                    "gap": 0.0,
                    "real_dur": 6.03,
                    "cut_off": 0,
                },
                {
                    "number": 13,
                    "start_time": "00:01:06.319",
                    "end_time": "00:01:10.239",
                    "tol_dur": 3.92,
                    "tolerance": 0.0,
                    "gap": 0.0,
                    "real_dur": 3.506,
                    "cut_off": 0,
                },
                {
                    "number": 14,
                    "start_time": "00:01:10.239",
                    "end_time": "00:01:15.200",
                    "tol_dur": 4.961,
                    "tolerance": 0.0,
                    "gap": 0.0,
                    "real_dur": 4.249,
                    "cut_off": 1,
                },
            ]
        )

        self.assertGreater(6.03 / (3.759 - 0.1), 1.6275)
        recovered = audio.split_oversized_chunks(
            tasks,
            accept=1.2,
            min_speed=1.0,
            max_speed=1.6275,
        )

        self.assertEqual(recovered["cut_off"].tolist(), [0, 0, 1])
        self.assertLess(
            audio._required_chunk_speed(recovered, accept=1.2, min_speed=1.0),
            1.2,
        )

    def test_row_overflow_is_deferred_to_chunk_timeline(self):
        tasks = pd.DataFrame(
            [
                {
                    "number": 12,
                    "tol_dur": 3.759,
                    "real_dur": 6.03,
                    "lines": ["六十英里外的坎施塔特，戴姆勒向温普夫父子买了马车。"],
                    "est_dur": 5.24,
                    "silence_removed": 0.0,
                    "silence_ratio": 0.0,
                }
            ]
        )

        def load_key(key):
            values = {
                "tts_method": "indextts",
                "indextts.version": "2",
                "indextts.v2.targeted_recovery": {
                    "enabled": True,
                    "max_retries": 0,
                    "min_estimated_duration_ratio": 0.75,
                    "min_abnormal_silence_ms": 600,
                    "keep_abnormal_silence_ms": 220,
                    "keep_edge_silence_ms": 80,
                },
            }
            return values[key]

        with (
            patch.object(audio, "load_key", side_effect=load_key),
            patch.object(audio, "get_audio_duration", return_value=6.03),
            patch.object(
                audio,
                "clean_abnormal_recovery_silence",
                return_value=(6.03, 6.03, 0.0),
            ),
            patch.object(
                audio,
                "trim_recovery_edge_silence",
                return_value=(6.03, 6.03),
            ),
            patch.object(audio, "save_audio_tasks") as save_audio_tasks,
            patch.object(audio, "rprint"),
        ):
            recovered = audio.regenerate_oversized_indextts2_rows(
                tasks,
                max_speed=1.55,
                emergency_max_speed=1.6275,
            )

        self.assertAlmostEqual(recovered.iloc[0]["real_dur"], 6.03)
        save_audio_tasks.assert_called_once()


if __name__ == "__main__":
    unittest.main()
