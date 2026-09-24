import unittest

from core._1_ytdlp import select_creator_subtitle


class YoutubeSubtitleSelectionTests(unittest.TestCase):
    def test_exact_language_wins(self):
        subtitles = {"en-US": [{}], "en": [{}], "fr": [{}]}
        self.assertEqual(select_creator_subtitle(subtitles, "en"), "en")

    def test_regional_creator_track_matches_base_language(self):
        subtitles = {"en-US": [{}], "fr": [{}]}
        self.assertEqual(select_creator_subtitle(subtitles, "en"), "en-US")

    def test_unrelated_language_does_not_replace_whisper(self):
        subtitles = {"fr": [{}], "de": [{}]}
        self.assertIsNone(select_creator_subtitle(subtitles, "en"))

    def test_empty_tracks_are_ignored(self):
        self.assertIsNone(select_creator_subtitle({"en": []}, "en"))


if __name__ == "__main__":
    unittest.main()
