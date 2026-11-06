import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('vl_pronunciation_test',ROOT/'core/utils/pronunciation.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class PronunciationTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'work/pronunciation-tests';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        env=patch.dict(os.environ,{'VL_PRONUNCIATION_GLOSSARY':str(self.root/'glossary.yaml')})
        env.start();self.addCleanup(env.stop)

    def test_rules_apply_once_longest_first_and_leave_other_polyphones_alone(self):
        rules={'弹容量':'dan4容量','dan4':'不应再次替换'}
        self.assertEqual(p.apply_rules('弹容量和弹簧',rules),'dan4容量和弹簧')

    def test_saved_rule_changes_take_effect_without_restarting(self):
        p.save_rules({'弹容量':'dan4容量'})
        self.assertEqual(p.apply_rules('弹容量'),'dan4容量')
        p.save_rules({'弹容量':'dan4 rong2 liang4'})
        self.assertEqual(p.apply_rules('弹容量'),'dan4 rong2 liang4')
        p.save_rules({});self.assertEqual(p.apply_rules('弹容量'),'弹容量')

    def test_only_affected_legacy_audio_is_invalidated(self):
        wav=self.root/'1_temp.wav';wav.write_bytes(b'old wave')
        p.save_rules({'弹容量':'dan4容量'})
        self.assertFalse(p.cache_matches('弹容量',wav))
        self.assertTrue(p.cache_matches('弹簧',wav))
        p.record_audio(wav,'弹容量','dan4容量')
        self.assertTrue(p.cache_matches('弹容量',wav))
        p.save_rules({'弹容量':'dan4容量','其他词':'其他读法'})
        self.assertTrue(p.cache_matches('弹容量',wav))
        p.save_rules({});self.assertFalse(p.cache_matches('弹容量',wav))

    def test_stale_audio_is_preserved_before_regeneration(self):
        wav=self.root/'1_temp.wav';wav.write_bytes(b'old wave')
        p.record_audio(wav,'原文','旧读法')
        p.preserve_stale_audio(wav)
        self.assertFalse(wav.exists())
        backups=list((self.root/'pronunciation_versions').glob('*/1_temp.wav'))
        self.assertEqual(len(backups),1);self.assertEqual(backups[0].read_bytes(),b'old wave')

    def test_invalid_dictionary_never_overwrites_saved_file(self):
        p.save_rules({'弹容量':'dan4容量'})
        before=p.glossary_path().read_bytes()
        with self.assertRaises(ValueError):p.save_rules({'弹容量':''})
        self.assertEqual(p.glossary_path().read_bytes(),before)

    def test_fullwidth_yaml_colon_reports_actionable_error_before_synthesis(self):
        import importlib
        main=importlib.import_module('core.tts_backend.tts_main')
        p.glossary_path().write_text('取弹：\n  zh: 取dan4\n',encoding='utf-8')
        with patch.object(main,'load_key',return_value='indextts'),patch.object(main,'indextts_tts_for_videolingo') as backend:
            with self.assertRaisesRegex(ValueError,'英文半角'):
                main.tts_main('取弹',str(self.root/'bad.wav'),1,None)
        backend.assert_not_called()

    def test_indextts_request_uses_spoken_text_and_records_original(self):
        import core.tts_backend.indextts_tts as backend
        from core.utils import pronunciation as real
        p.save_rules({'弹容量':'dan4容量'})
        wav=self.root/'generated.wav'
        settings={'version':'2','port':9871}
        with patch.object(backend,'_active_settings',return_value=settings),patch.object(backend.requests,'post') as post:
            post.return_value.status_code=200;post.return_value.json.return_value={'ok':True}
            backend.indextts_tts('弹容量和弹簧',wav,self.root/'ref.wav')
        self.assertEqual(post.call_args.kwargs['json']['text'],'dan4容量和弹簧')
        data=json.loads(real.metadata_path(wav).read_text(encoding='utf-8'))
        self.assertEqual(data['source_text'],'弹容量和弹簧')

    def test_tts_main_regenerates_matching_cache_then_reuses_updated_audio(self):
        import importlib
        main=importlib.import_module('core.tts_backend.tts_main')
        from core.utils import pronunciation as real
        wav=self.root/'sentence.wav';wav.write_bytes(b'old wave')
        p.save_rules({'弹容量':'dan4容量'})
        def generate(text,path,*args):
            Path(path).write_bytes(b'new wave')
            real.record_audio(path,text,real.apply_rules(text))
        with patch.object(main,'load_key',return_value='indextts'),patch.object(main,'indextts_tts_for_videolingo',side_effect=generate) as backend,patch.object(main,'get_audio_duration',return_value=1.0):
            main.tts_main('弹容量',str(wav),1,None)
            main.tts_main('弹容量',str(wav),1,None)
        self.assertEqual(backend.call_count,1)
        self.assertEqual(wav.read_bytes(),b'new wave')

    def test_merge_reuse_fast_path_cannot_skip_changed_pronunciation(self):
        import pandas as pd
        import core._10_gen_audio as audio
        book=self.root/'tasks.xlsx';book.write_bytes(b'workbook')
        wav=self.root/'1_0_temp.wav';wav.write_bytes(b'wave')
        tasks=pd.DataFrame([{'number':1,'lines':['弹容量'],'real_dur':1.0,'silence_removed':0.0,'silence_ratio':0.0}])
        p.save_rules({'弹容量':'dan4容量'})
        with patch.object(audio,'_8_1_AUDIO_TASK',str(book)),patch.object(audio,'TEMP_FILE_TEMPLATE',str(self.root/'{}_temp.wav')),patch.object(audio,'load_key',return_value='indextts'):
            self.assertFalse(audio.can_reuse_generated_tts(tasks))
            p.record_audio(wav,'弹容量','dan4容量')
            self.assertTrue(audio.can_reuse_generated_tts(tasks))


if __name__=='__main__':unittest.main()
