"""Exercise actual Streamlit controls with local fake processing operations."""
from pathlib import Path
import tempfile,time,unittest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
HARNESS=r'''
import streamlit as st
from pathlib import Path
from types import SimpleNamespace as NS
from core.utils.workflow import Project,Options
from core.st_utils import workflow_ui as ui
from core.st_utils import task_runner as tr
root=Path(TEST_DIRECTORY)
source=root/'source.mp4'
source.write_bytes(b'source')
tr.start_run=lambda *a,**k:'test'
tr.start_step=lambda *a,**k:'test'
tr.finish_step=lambda *a,**k:None
tr.finish_run=lambda *a,**k:None
tr.Path=lambda p:root/'output/log'
ui.voice_revision=lambda:'test-voice'
ui.save_preference=lambda options:None
ui._open=lambda path:None
if 'job' not in st.session_state:
    project=Project(source,root/'output',Options(review_translation=True,voice_revision='test-voice'))
    st.session_state['job']=(project,tr.TaskRunner())
project,runner=st.session_state['job']
def write(path,data=b'test'):
    p=project.output/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
def subtitles():
    for p in ['trans.srt','src.srt','audio/src_subs_for_audio.srt','audio/trans_subs_for_audio.srt']:write(p)
def synthesize():
    if project.state.pop('simulate_failure',False):raise RuntimeError('simulated synthesis error')
    write('audio/tmp/1.wav')
host={
 '_2_asr':NS(transcribe=lambda:write('log/cleaned_chunks.xlsx')),
 '_3_1_split_nlp':NS(split_by_spacy=lambda:write('log/split_by_nlp.txt')),
 '_3_2_split_meaning':NS(split_sentences_by_meaning=lambda:write('log/split_by_meaning.txt')),
 '_4_1_summarize':NS(get_summary=lambda:write('log/terminology.json')),
 '_4_2_translate':NS(translate_all=lambda:write('log/translation_results.xlsx')),
 '_4_3_proofread_translation':NS(proofread_translation_if_enabled=lambda:None,is_proofread_running=lambda:False),
 '_5_split_sub':NS(split_for_sub_main=lambda:write('log/translation_results_for_subtitles.xlsx')),
 '_6_gen_sub':NS(align_timestamp_main=subtitles),
 '_7_sub_into_vid':NS(merge_subtitles_to_video=lambda **kw:write('output_sub.mp4',b'original')),
 '_8_1_audio_task':NS(gen_audio_task_main=lambda:write('audio/tts_tasks.xlsx')),
 '_8_2_dub_chunks':NS(gen_dub_chunks=lambda:None),
 '_9_refer_audio':NS(extract_refer_audio_main=lambda:None),
 '_10_gen_audio':NS(gen_audio=synthesize),
 '_11_merge_audio':NS(merge_full_audio=lambda:write('dub.mp3')),
 '_12_dub_to_vid':NS(merge_video_audio=lambda **kw:write('output_dub.mp4',b'dubbed')),
 '_stop_handoff_alarms':lambda:None,'_notify_user_handoff':lambda *a,**k:None,
 'PROOFREAD_TRANSLATION_RESULTS':str(root/'no-proofread.xlsx'),'cleanup':lambda:None,
}
ui.render_controls((project,runner))
ui.render_preview((project,runner))
ui.render_sections((project,runner),host)
'''

class WorkflowUITests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'work/workflow-ui-tests';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.app=AppTest.from_string('TEST_DIRECTORY='+repr(self.temp.name)+'\n'+HARNESS,default_timeout=30).run()
        self.assertFalse(self.app.exception)

    def click(self,key):
        self.app.button(key=key).click().run()
        project,runner=self.app.session_state['job']
        if runner._thread:runner._thread.join(timeout=10)
        self.app.run();self.assertFalse(self.app.exception)
        return project,runner

    def preset(self,name):
        self.app.radio(key='wf_preset').set_value(name).run()
        self.assertFalse(self.app.exception)
        if not self.app.button(key='wf_apply_settings').disabled:
            return self.click('wf_apply_settings')
        return self.app.session_state['job']

    def test_review_then_original_then_append_dubbing(self):
        project,runner=self.click('wf_start')
        self.assertEqual(project.state['status'],'waiting_translation')
        self.assertFalse((project.output/'output_sub.mp4').exists())
        project,runner=self.click('wf_confirm_review')
        self.assertEqual(project.state['status'],'completed')
        self.assertTrue((project.output/'output_sub.mp4').exists())
        translated=(project.output/'log/translation_results.xlsx').stat().st_mtime_ns
        original=(project.output/'output_sub.mp4').read_bytes()
        project,runner=self.click('wf_append_dubbing')
        self.assertEqual(project.state['status'],'completed')
        self.assertEqual((project.output/'log/translation_results.xlsx').stat().st_mtime_ns,translated)
        self.assertEqual((project.output/'output_sub.mp4').read_bytes(),original)
        self.assertTrue((project.output/'output_dub.mp4').exists())
        self.assertTrue(self.app.checkbox(key='wf_option_dubbing').value)

    def test_full_automatic_preset_has_no_review_stop(self):
        self.preset('both')
        project,runner=self.click('wf_start')
        self.assertEqual(project.state['status'],'completed')
        self.assertTrue(project.facts()['original']);self.assertTrue(project.facts()['dubbing'])

    def test_translation_edit_marks_existing_video_outdated(self):
        self.preset('subtitles')
        project,runner=self.click('wf_start')
        (project.output/'log/translation_results.xlsx').write_bytes(b'edited')
        self.app.run()
        self.assertFalse(project.facts()['original'])
        self.assertTrue(any(b.key=='wf_update_original' for b in self.app.button))

    def test_terms_review_precedes_translation_review(self):
        from dataclasses import replace
        project,runner=self.app.session_state['job']
        project.set_options(replace(project.options,review_terms=True))
        self.app.run()
        project,runner=self.click('wf_start')
        self.assertEqual(project.state['status'],'waiting_terms')
        self.assertFalse((project.output/'log/translation_results.xlsx').exists())
        project,runner=self.click('wf_confirm_review')
        self.assertEqual(project.state['status'],'waiting_translation')
        project,runner=self.click('wf_confirm_review')
        self.assertEqual(project.state['status'],'completed')

    def test_retry_dubbing_keeps_original_and_prepared_audio_tasks(self):
        project,runner=self.preset('both')
        project.state['simulate_failure']=True
        project,runner=self.click('wf_start')
        self.assertEqual(project.state['status'],'error')
        tasks=(project.output/'audio/tts_tasks.xlsx').stat().st_mtime_ns
        self.assertTrue((project.output/'output_sub.mp4').exists())
        project,runner=self.click('wf_retry')
        self.assertEqual(project.state['status'],'completed')
        self.assertEqual((project.output/'audio/tts_tasks.xlsx').stat().st_mtime_ns,tasks)

    def test_selecting_preset_only_changes_draft(self):
        project,runner=self.app.session_state['job']
        before=project.options
        self.app.radio(key='wf_preset').set_value('both').run()
        self.assertFalse(self.app.exception)
        self.assertEqual(project.options,before)
        self.assertTrue(self.app.checkbox(key='wf_option_dubbing').value)
        self.assertFalse(self.app.checkbox(key='wf_option_review').value)
        self.assertFalse(self.app.button(key='wf_apply_settings').disabled)
        self.assertIsNone(runner._thread)

    def test_manual_edits_show_custom_until_explicit_apply(self):
        project,runner=self.app.session_state['job']
        self.app.radio(key='wf_preset').set_value('both').run()
        self.app.checkbox(key='wf_option_review').check().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.radio(key='wf_preset').value,'custom')
        self.assertFalse(project.options.dubbing)
        self.click('wf_apply_settings')
        self.assertTrue(project.options.dubbing)
        self.assertTrue(project.options.review_translation)
        self.assertEqual(self.app.radio(key='wf_preset').value,'custom')
        self.assertTrue(self.app.button(key='wf_apply_settings').disabled)

    def test_advanced_options_also_mark_custom(self):
        project,runner=self.preset('both')
        self.app.selectbox(key='wf_option_dmode').set_value('none').run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.radio(key='wf_preset').value,'custom')
        self.assertEqual(project.options.dub_subtitles,'translation')
        self.click('wf_apply_settings')
        self.assertEqual(project.options.dub_subtitles,'none')

if __name__=='__main__':unittest.main()
