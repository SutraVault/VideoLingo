"""Plans, human boundaries, appending outputs and version-preserving reuse."""
from pathlib import Path
from dataclasses import replace
import tempfile,unittest,importlib.util,sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tested_workflow',ROOT/'core/utils/workflow.py')
wf=importlib.util.module_from_spec(spec);sys.modules[spec.name]=wf;spec.loader.exec_module(wf)
Options,Project,plan=wf.Options,wf.Project,wf.plan

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'work/workflow-tests';base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source.mp4';self.source.write_bytes(b'source')
        self.output=self.root/'output';self.output.mkdir()

    def file(self,name,data=b'ready'):
        p=self.output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p

    def ready(self,options=None):
        for name in ['log/cleaned_chunks.xlsx','log/split_by_nlp.txt','log/split_by_meaning.txt',
                     'log/terminology.json','log/translation_results.xlsx','trans.srt','src.srt',
                     'audio/src_subs_for_audio.srt','audio/trans_subs_for_audio.srt']:self.file(name)
        return Project(self.source,self.output,options or Options(review_translation=False))

    def test_original_only_never_queues_dubbing(self):
        steps,boundary=plan(Options(review_translation=False),{}, {})
        self.assertIn('original',steps);self.assertNotIn('synthesis',steps)
        self.assertEqual(boundary,'completed')

    def test_no_video_still_provides_subtitles(self):
        steps,_=plan(Options(original=False,review_translation=False),{}, {})
        self.assertIn('subtitles',steps);self.assertNotIn('original',steps);self.assertNotIn('dubbing',steps)

    def test_review_boundary_not_task_completion(self):
        steps,boundary=plan(Options(),{}, {})
        self.assertEqual(boundary,'waiting_translation');self.assertNotIn('subtitles',steps)
        steps,boundary=plan(Options(review_terms=True),{}, {})
        self.assertEqual(boundary,'waiting_terms');self.assertNotIn('translation',steps)

    def test_append_dubbing_preserves_original_and_reuses_upstream(self):
        p=self.ready();self.file('output_sub.mp4',b'original-video');p.mark_artifact('original')
        p.set_options(replace(p.options,dubbing=True))
        steps,boundary=p.next_plan()
        self.assertEqual(steps,['audio_tasks','references','synthesis','merge_audio','dubbing'])
        self.assertEqual((self.output/'output_sub.mp4').read_bytes(),b'original-video')

    def test_changed_translation_invalidates_outputs_and_review(self):
        p=self.ready(Options(dubbing=True));p.approve('translation')
        for stage in ('original','audio','dubbing'):
            self.file(wf.FILES[stage]);p.mark_artifact(stage)
        self.file('log/translation_results.xlsx',b'edited')
        self.assertFalse(p.approvals()['translation'])
        self.assertTrue(p.facts()['translation'])
        for stage in ('subtitles','original','audio','dubbing'):self.assertFalse(p.facts()[stage])
        self.assertEqual(p.next_plan()[1],'waiting_translation')

    def test_subtitle_style_only_reuses_completed_audio(self):
        p=self.ready(Options(dubbing=True,review_translation=False))
        for stage in ('audio','dubbing'):self.file(wf.FILES[stage]);p.mark_artifact(stage)
        p.set_options(replace(p.options,dub_subtitles='none'))
        self.assertEqual(p.next_plan()[0],['original','dubbing'])

    def test_regeneration_archives_old_video(self):
        p=self.ready();self.file('output_sub.mp4',b'old');p.mark_artifact('original')
        p.wrap('original',lambda:self.file('output_sub.mp4',b'new'))()
        self.assertEqual((self.output/'output_sub.mp4').read_bytes(),b'new')
        self.assertEqual([f.read_bytes() for f in (self.output/'versions').rglob('output_sub.mp4')],[b'old'])

    def test_restart_does_not_automatically_resume_running_job(self):
        p=self.ready();p.state['status']='running';p.save()
        reopened=Project(self.source,self.output)
        self.assertEqual(reopened.state['status'],'interrupted')

    def test_partial_audio_retry_keeps_generated_segments(self):
        p=self.ready();p.prepare('audio_tasks');self.file('audio/tmp/1.wav',b'generated')
        p.prepare('audio_tasks')
        self.assertTrue((self.output/'audio/tmp/1.wav').exists())

    def test_voice_change_requires_audio_rebuild(self):
        p=self.ready(Options(dubbing=True,review_translation=False,voice_revision='old'))
        for stage in ('audio','dubbing'):self.file(wf.FILES[stage]);p.mark_artifact(stage)
        p.set_options(replace(p.options,voice_revision='new'))
        self.assertIn('synthesis',p.next_plan()[0]);self.assertFalse(p.facts()['audio'])

    def test_source_change_cannot_reuse_previous_translation(self):
        p=self.ready();self.source.write_bytes(b'different source')
        new=Project(self.source,self.output)
        self.assertFalse(new.facts()['translation'])
        self.assertTrue(list((self.output/'versions').rglob('translation_results.xlsx')))

    def test_updated_terms_invalidate_translation(self):
        p=self.ready();self.file('log/terminology.json',b'corrected terms')
        self.assertFalse(p.facts()['translation'])

    def test_target_language_change_requires_new_terms_and_translation(self):
        p=self.ready(Options(review_translation=False,translation_revision='Chinese'))
        p.set_options(replace(p.options,translation_revision='Japanese'))
        self.assertFalse(p.facts()['terms']);self.assertFalse(p.facts()['translation'])
        stages,_=p.next_plan()
        self.assertIn('terms',stages);self.assertIn('translation',stages)

    def test_failure_records_state_without_marking_output_complete(self):
        p=self.ready()
        def fail():raise RuntimeError('test synthesis failure')
        with self.assertRaises(RuntimeError):p.wrap('dubbing',fail)()
        self.assertEqual(p.state['status'],'error');self.assertNotIn('dubbing',p.state['artifacts'])

if __name__=='__main__':unittest.main()
