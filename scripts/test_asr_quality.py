"""ASR regression checks, with no model download or application imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
TEST_WORK=ROOT/'work/asr-tests'
TEST_WORK.mkdir(parents=True,exist_ok=True)

def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'core/asr_backend'/f'{name}.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

quality=module('quality');cache=module('cache')

class QualityTests(unittest.TestCase):
    def test_repetition_at_another_time_survives(self):
        merged,added=quality.merge_recovery([{'start':0,'end':1,'text':'Hello.'}],[{'start':10,'end':11,'text':'Hello.'}])
        self.assertEqual(len(merged),2);self.assertEqual(len(added),1)

    def test_local_duplicate_is_not_added(self):
        merged,added=quality.merge_recovery([{'start':0,'end':2,'text':'Hello there.'}],[{'start':.1,'end':2.1,'text':'Hello there!'}])
        self.assertFalse(added)

    def test_extended_overlapping_sentence_recovers_missing_text(self):
        merged,added=quality.merge_recovery([{'start':0,'end':10,'text':'Read a book.'}],[{'start':0,'end':10,'text':'Take a break. Read a book.'}])
        self.assertEqual(len(merged),1);self.assertIn('Take a break',merged[0]['text'])

    def test_short_addition_is_not_treated_as_fuzzy_duplicate(self):
        merged,added=quality.merge_recovery([{'start':0,'end':10,'text':'This sentence contains many similar words.'}],[{'start':0,'end':10,'text':'This sentence contains many similar words. Yes.'}])
        self.assertTrue(added);self.assertIn('Yes',merged[0]['text'])

    def test_conflicting_overlap_triggers_retry(self):
        merged,_=quality.merge_recovery([{'start':0,'end':20,'text':'manufacture.'}],[{'start':5,'end':15,'text':'Today is sponsored.'}])
        self.assertTrue(quality.suspicious_windows(merged,30))

    def test_long_word_and_phrase_threshold(self):
        bad={'start':0,'end':20,'words':[{'word':'manufacture.','start':1,'end':17.4}]}
        good={'start':0,'end':4,'words':[{'word':'a complete phrase','start':0,'end':4}]}
        self.assertEqual(quality.suspicious_windows([bad],30),[(0,20.4)])
        self.assertFalse(quality.suspicious_windows([good],30))

    def test_splice_preserves_context(self):
        original=[{'start':0,'end':20,'text':'before wrong after','words':[{'word':'before','start':0,'end':1},{'word':'wrong','start':4,'end':16},{'word':'after','start':19,'end':20}]}]
        replacement=[{'start':3,'end':17,'text':'recovered','words':[{'word':'recovered','start':3,'end':17}]}]
        result=quality.replace_window(original,replacement,2,18)
        self.assertEqual([s['text'] for s in result],['before','recovered','after'])

    def test_zero_duration_words_trigger_retry(self):
        segment={'start':534,'end':537,'words':[{'word':'Republic.','start':536.893,'end':536.893}]}
        self.assertEqual(quality.suspicious_windows([segment],600),[(533.893,539.893)])

    def test_missing_word_timestamp_triggers_retry(self):
        segment={'start':534,'end':537,'words':[{'word':'Republic.'}]}
        self.assertEqual(quality.suspicious_windows([segment],600),[(531,540)])

    def test_retry_expands_context_and_replaces_expanded_bounds(self):
        calls=[]
        def decode(a,b):
            calls.append((a,b))
            end=a if len(calls)==1 else a+1
            return [{'start':a,'end':b,'text':'word','words':[{'word':'word','start':a,'end':end}]}]
        recovered,a,b,diagnostics=quality.recover_with_context(decode,10,20,60)
        self.assertEqual(calls,[(10,20),(7,23)])
        self.assertEqual((a,b),(7,23));self.assertEqual(len(diagnostics),2)
        self.assertIsNotNone(recovered)

    def test_retry_remains_bounded_and_rejects_bad_candidates(self):
        def decode(a,b):
            return [{'start':a,'end':b,'words':[{'word':'word','start':a,'end':a}]}]
        recovered,a,b,diagnostics=quality.recover_with_context(decode,1,9,10)
        self.assertIsNone(recovered);self.assertEqual(len(diagnostics),3)
        self.assertEqual((diagnostics[-1]['start'],diagnostics[-1]['end']),(0,10))

    def test_confident_collapsed_contraction_preserves_observed_span(self):
        words=[{'word':"we're",'start':457.64,'end':457.84},{'word':'going','start':457.84,'end':457.84,'score':.85},{'word':'to','start':457.84,'end':457.98}]
        source=[{'start':457.64,'end':457.98,'words':words}]
        result=quality.repair_isolated_zero_words(source)
        fixed=result[0]['words']
        self.assertTrue(all(w['end']>w['start'] for w in fixed))
        self.assertEqual(fixed[0]['start'],457.64);self.assertAlmostEqual(fixed[-1]['end'],457.98)
        self.assertEqual(words[1]['start'],words[1]['end'])
        self.assertEqual([w['word'] for w in fixed],[w['word'] for w in words])
        self.assertIn('original_timing',fixed[1])

    def test_acoustic_alignment_repairs_terminal_zero_without_redecoding(self):
        bad=[{'start':1,'end':2,'text':'like that','words':[{'word':'like','start':1,'end':2},{'word':'that','start':2,'end':2}]}]
        good=[{'start':1,'end':2,'text':'like that','words':[{'word':'like','start':1,'end':1.5},{'word':'that','start':1.5,'end':2}]}]
        result,a,b,diagnostics=quality.recover_with_context(lambda a,b:bad,0,3,10,realign=lambda *args:good)
        self.assertEqual(result,good);self.assertEqual(len(diagnostics),1)
        self.assertTrue(diagnostics[0]['acoustic_realignment'])

    def test_acoustic_alignment_cannot_drop_the_problem_word(self):
        bad=[{'start':1,'end':2,'text':'like that','words':[{'word':'like','start':1,'end':2},{'word':'that','start':2,'end':2}]}]
        dropped=[{'start':1,'end':2,'text':'like','words':[{'word':'like','start':1,'end':2}]}]
        result,a,b,diagnostics=quality.recover_with_context(lambda a,b:bad,0,3,10,realign=lambda *args:dropped)
        self.assertIsNone(result);self.assertEqual(len(diagnostics),3)
        self.assertIn('dropped',diagnostics[0]['realignment_error'])

    def test_text_without_aligned_words_is_still_an_error(self):
        self.assertTrue(quality.suspicious_windows([{'start':1,'end':2,'text':'like that','words':[]}],10))

    def test_consecutive_zero_words_are_not_interpolated(self):
        source=[{'start':0,'end':.4,'words':[{'word':'in','start':0,'end':.1},{'word':'the','start':.1,'end':.1,'score':.9},{'word':'Czech','start':.1,'end':.1,'score':.9},{'word':'Republic','start':.1,'end':.4}]}]
        self.assertEqual(quality.repair_isolated_zero_words(source),source)

    def test_low_confidence_and_silence_are_not_interpolated(self):
        source=[{'start':0,'end':.8,'words':[{'word':'we','start':0,'end':.2},{'word':'going','start':.2,'end':.2,'score':.2},{'word':'to','start':.2,'end':.4}]}]
        self.assertEqual(quality.repair_isolated_zero_words(source),source)
        source[0]['words'][1]['score']=.9;source[0]['words'][2]['start']=.3
        self.assertEqual(quality.repair_isolated_zero_words(source),source)

class CacheTests(unittest.TestCase):
    def test_fingerprint_tracks_source_and_settings_and_ignores_secret(self):
        with tempfile.TemporaryDirectory(dir=TEST_WORK) as tmp:
            source=Path(tmp)/'video';source.write_bytes(b'one')
            a=cache.fingerprint(source,{'model':'large-v3','api_key':'secret'})
            self.assertNotIn('secret',json.dumps(a))
            self.assertNotEqual(a,cache.fingerprint(source,{'model':'turbo'}))
            source.write_bytes(b'two')
            self.assertNotEqual(a,cache.fingerprint(source,{'model':'large-v3'}))

    def test_stale_archive_preserves_inputs_and_invalidates_downstream(self):
        with tempfile.TemporaryDirectory(dir=TEST_WORK) as tmp:
            root=Path(tmp);(root/'log').mkdir();(root/'audio/segs').mkdir(parents=True)
            for name in ['log/cleaned_chunks.xlsx','log/translation_results.xlsx','log/server.log','audio/raw.mp3','audio/vocal.mp3','audio/segs/1.wav','src.srt','source.mp4']:
                (root/name).write_text('old')
            cache.save_manifest(root,{'version':1})
            self.assertTrue(cache.cache_matches(root,{'version':1}))
            self.assertFalse(cache.cache_matches(root,{'version':2}))
            archive=cache.archive_stale(root,False)
            for name in ['audio/raw.mp3','audio/vocal.mp3','source.mp4','log/server.log']: self.assertTrue((root/name).exists())
            for name in ['log/cleaned_chunks.xlsx','log/translation_results.xlsx','audio/segs/1.wav','src.srt']: self.assertTrue((archive/name).exists());self.assertFalse((root/name).exists())
            cache.archive_stale(root,True)
            self.assertFalse((root/'audio/raw.mp3').exists())

if __name__=='__main__': unittest.main()
