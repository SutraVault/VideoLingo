"""Continuity and cache regression tests for TTS reference selection."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from pydub import AudioSegment
from pydub.generators import Sine

spec = importlib.util.spec_from_file_location(
    'reference_audio', Path(__file__).resolve().parents[1] / 'core/tts_backend/reference_audio.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReferenceTests(unittest.TestCase):
    def test_trims_edges_and_preserves_normal_pause(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'output') as directory:
            root = Path(directory)
            tone = Sine(220).to_audio_segment(duration=3200).apply_gain(-12)
            audio = AudioSegment.silent(7900) + tone + AudioSegment.silent(200) + tone
            source = root / '1.wav'
            audio.export(source, format='wav').close()
            output, report = module.prepare_reference([source], root / 'cache')
            self.assertGreater(report['trimmed_ms'], 7800)
            self.assertGreaterEqual(report['longest_internal_pause_ms'], 190)
            with source.open('rb') as original:
                self.assertEqual(len(AudioSegment.from_wav(original)), len(audio))
            self.assertEqual(module.prepare_reference([source], root / 'cache')[0], output)

    def test_rejects_long_internal_pause_and_uses_continuous_candidate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'output') as directory:
            root = Path(directory)
            tone = Sine(220).to_audio_segment(duration=3200).apply_gain(-12)
            bad, good = root / '1.wav', root / '2.wav'
            (tone + AudioSegment.silent(1200) + tone).export(bad, format='wav').close()
            (tone + tone).export(good, format='wav').close()
            _, report = module.prepare_reference([bad, good], root / 'cache')
            self.assertEqual(Path(report['source']), good)
            self.assertTrue(report['rejected_candidates'])
            # A changed recording must invalidate the prior cached selection.
            AudioSegment.silent(7000).export(good, format='wav').close()
            with self.assertRaisesRegex(ValueError, 'No clean, continuous'):
                module.prepare_reference([bad, good], root / 'cache')

    def test_silence_and_short_speech_do_not_pass(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'output') as directory:
            root = Path(directory)
            for audio in (AudioSegment.silent(10000), Sine(220).to_audio_segment(duration=1000)):
                source = root / '1.wav'
                audio.export(source, format='wav').close()
                with self.assertRaises(ValueError):
                    module.prepare_reference([source], root / 'cache')


if __name__ == '__main__':
    unittest.main()
