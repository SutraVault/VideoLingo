import unittest

import numpy as np

from core.tts_backend.kokoro_tts import expand_latin_initialisms, trim_kokoro_edge_silence


class KokoroTextTests(unittest.TestCase):
    def test_model_designation_and_acronym_are_spelled(self):
        self.assertEqual(expand_latin_initialisms("绍尔M6"), "绍尔艾姆6")
        self.assertEqual(expand_latin_initialisms("CTDM柴油机"), "西提迪艾姆柴油机")
        self.assertEqual(expand_latin_initialisms("6x6卡车"), "6乘6卡车")

    def test_dotted_initialism_is_spelled(self):
        self.assertEqual(expand_latin_initialisms("C.T.D.M. 柴油机"), "西提迪艾姆 柴油机")

    def test_normal_english_words_are_not_spelled(self):
        self.assertEqual(expand_latin_initialisms("Detroit Diesel 系列"), "Detroit Diesel 系列")

    def test_model_edge_padding_is_trimmed_with_small_margin(self):
        samples = np.concatenate([np.zeros(2400), np.ones(2400) * .2, np.zeros(4800)])
        trimmed = trim_kokoro_edge_silence(samples, sample_rate=24000, keep_ms=50)
        self.assertGreaterEqual(len(trimmed), 2400 + 2 * 1200)
        self.assertLess(len(trimmed), len(samples))


if __name__ == "__main__":
    unittest.main()
