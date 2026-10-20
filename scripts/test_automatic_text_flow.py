"""Check automatic draft/full processing plans and safe downstream resume."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

class AutomaticFlowTests(unittest.TestCase):
    def setUp(self):
        self.settings={'pause_after_translate':False,'auto_start_translation_after_video':True}
        source=Path(__file__).resolve().parents[1]/'st.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_get_text_steps','_automatic_text_steps','_should_auto_continue_text')]
        self.scope={'load_key':self.settings.__getitem__,'t':lambda s:s,'SUB_VIDEO':'missing','os':SimpleNamespace(path=SimpleNamespace(exists=lambda p:False))}
        for name,method in [('_2_asr','transcribe'),('_6_gen_sub','align_timestamp_main')]:
            self.scope[name]=SimpleNamespace(**{method:lambda:None})
        self.scope['_5_split_sub']=SimpleNamespace(split_for_sub_main=lambda:None)
        exec(compile(tree,'st.py','exec'),self.scope)

    def test_pause_off_includes_all_steps_and_video_merge(self):
        steps=self.scope['_automatic_text_steps'](True)
        self.assertEqual(len(steps),6)
        self.assertEqual(steps[-1][0],'Merging subtitles into the video')

    def test_pause_on_only_queues_draft(self):
        self.settings['pause_after_translate']=True
        self.assertEqual(len(self.scope['_automatic_text_steps'](True)),3)

    def test_resume_starts_at_subtitle_split_without_repeating_translation(self):
        steps=self.scope['_automatic_text_steps'](True,resume=True)
        self.assertEqual(len(steps),3)
        self.assertEqual(steps[0][0],'Cutting and aligning long subtitles')

    def test_resume_does_not_restart_failed_stopped_or_paused_tasks(self):
        predicate=self.scope['_should_auto_continue_text']
        for state in ['error','stopped','paused','running']:
            self.assertFalse(predicate(state,True,False))
        for state in ['idle','completed']:
            self.assertTrue(predicate(state,True,False))
        self.assertFalse(predicate('idle',True,True))
        self.settings['pause_after_translate']=True
        self.assertFalse(predicate('idle',True,False))

if __name__=='__main__':unittest.main()
