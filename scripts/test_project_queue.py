"""Queue isolation, durable recovery, failure/timeout continuation and real filters."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


q = load('tested_project_queue', 'core/utils/project_queue.py')
paths = load('tested_project_paths', 'core/utils/project_paths.py')
wf = load('tested_workflow', 'core/utils/workflow.py')


class QueueTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'work/project-tests';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.queue=q.Queue(self.root/'Projects')

    def add(self, sources, timeout=30):
        return self.queue.enqueue(sources,'target_language: Chinese\n',
                                  {'original':True,'dubbing':False},timeout)

    def test_reorder_survives_restart_and_controls_claim_without_changing_creation(self):
        ids=self.add(['a','b','c','d'])
        running=self.queue.claim()
        before={j['id']:j['created'] for j in self.queue.jobs()}
        self.queue.reorder(ids[3],'first')
        self.queue.reorder(ids[1],'down')
        self.assertEqual([j['id'] for j in self.queue.jobs()],[ids[0],ids[3],ids[2],ids[1]])
        self.queue.reorder(ids[3],'last')
        self.queue.reorder(ids[3],'up')
        extra=self.add(['e'])[0]
        restored=q.Queue(self.queue.root)
        self.assertEqual([j['id'] for j in restored.jobs() if j['status']=='queued'],[ids[2],ids[3],ids[1],extra])
        self.assertEqual(restored.claim()['id'],ids[2])
        self.assertEqual(restored.get(running['id'])['status'],'running')
        self.assertEqual({key:restored.get(key)['created'] for key in ids},before)
        with self.assertRaises(ValueError):restored.reorder(ids[2],'first')
        with self.assertRaises(ValueError):restored.reorder(ids[0],'last')

    def worker(self):
        worker=self.root/'fake_worker.py'
        worker.write_text('''import importlib.util,json,os,sys,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('queue',sys.argv[0]+'.queue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
queue=m.Queue(sys.argv[2]);job=queue.get(sys.argv[3])
with m.exclusive_file(queue.root/'execution.lock') as acquired:
 assert acquired
 queue.update(job['id'],current='test')
 p=Path(os.environ['VL_OUTPUT_DIR']);p.mkdir(exist_ok=True)
 (p/'binding.json').write_text(json.dumps({'config':os.environ['VL_CONFIG_PATH'],'root':str(p)}))
 if job['source']=='fail':raise RuntimeError('intentional failure')
 if job['source']=='hang':time.sleep(60)
 queue.update(job['id'],status='completed')
''',encoding='utf-8')
        import shutil
        shutil.copy2(ROOT/'core/utils/project_queue.py',str(worker)+'.queue.py')
        return worker

    def test_snapshots_unique_directories_and_persistent_order(self):
        ids=self.add(['https://youtu.be/a','https://youtu.be/b','https://youtu.be/a'])
        self.assertEqual(len(ids),2)
        restored=q.Queue(self.queue.root)
        self.assertEqual([j['id'] for j in restored.jobs()],ids)
        for job in restored.jobs():
            self.assertEqual((Path(job['directory'])/'config.snapshot.yaml').read_text(),'target_language: Chinese\n')
        a=self.queue.rename(ids[0], '同名: video')
        b=self.queue.rename(ids[1], '同名: video')
        self.assertNotEqual(a,b)
        self.assertNotIn(':',a.name)

    def test_claim_is_atomic_and_command_survives(self):
        job=self.add(['a'])[0]
        self.queue.update(job,status='waiting_translation')
        self.queue.submit_command(job,'approve:translation')
        claimed=self.queue.claim()
        self.assertEqual(claimed['command'],'approve:translation')
        self.assertIsNone(self.queue.claim())

    def test_cross_batch_duplicates_use_video_id_and_require_explicit_override(self):
        first=self.add(['https://www.youtube.com/watch?v=example1234'])[0]
        self.queue.update(first,status='completed')
        self.assertEqual(self.add(['https://youtu.be/example1234?si=tracking']),[])
        self.assertEqual(self.queue.last_skipped[0]['id'],first)
        self.assertEqual(len(self.queue.jobs()),1)
        recreated=self.queue.enqueue(['https://www.youtube.com/shorts/example1234'],
                                     'target_language: Chinese\n',{'dubbing':True},allow_existing=True)
        self.assertEqual(len(recreated),1)
        self.assertNotEqual(recreated[0],first)

    def test_uploaded_output_filename_cannot_overwrite_source(self):
        job_id=self.queue.enqueue(['output_sub.mp4'],'target_language: Chinese\n',
                                  {'original':True},uploads={'output_sub.mp4':b'original source'})[0]
        job=self.queue.get(job_id);output=Path(job['directory'])/'output'
        self.assertEqual(job['title'],'output_sub')
        self.assertFalse((output/'output_sub.mp4').exists())
        self.assertEqual((output/job['source']).read_bytes(),b'original source')

    def test_failure_timeout_and_next_project(self):
        ids=self.add(['fail','hang','success'],timeout=1.5)
        q.serve(self.queue,worker_script=self.worker(),max_jobs=3)
        jobs=[self.queue.get(i) for i in ids]
        self.assertEqual([j['status'] for j in jobs],['error','error','completed'])
        self.assertIn('超时',jobs[1]['error'])
        for job in jobs:
            binding=json.loads((Path(job['directory'])/'output/binding.json').read_text())
            self.assertEqual(binding['root'],str(Path(job['directory'])/'output'))
            self.assertEqual(binding['config'],str(Path(job['directory'])/'config.runtime.yaml'))

    def test_restart_marks_interrupted_and_continues_pending(self):
        ids=self.add(['old','success'])
        self.queue.claim()
        q.serve(q.Queue(self.queue.root),worker_script=self.worker(),max_jobs=1)
        self.assertEqual(self.queue.get(ids[0])['status'],'interrupted')
        self.assertEqual(self.queue.get(ids[1])['status'],'completed')
        self.queue.retry(ids[0]);self.assertEqual(self.queue.get(ids[0])['status'],'queued')

    def test_cancel_current_continues_next(self):
        ids=self.add(['hang','success'])
        worker=self.worker()
        thread=threading.Thread(target=q.serve,args=(self.queue,worker,2))
        thread.start()
        deadline=time.time()+15
        while self.queue.get(ids[0])['current']!='test' and time.time()<deadline: time.sleep(.1)
        self.queue.update(ids[0],command='cancel')
        thread.join(20)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.queue.get(ids[0])['status'],'cancelled')
        self.assertEqual(self.queue.get(ids[1])['status'],'completed')

    def test_global_execution_lock(self):
        with q.exclusive_file(self.root/'worker.lock') as first:
            with q.exclusive_file(self.root/'worker.lock') as second:
                self.assertTrue(first);self.assertFalse(second)

    def test_archive_and_restore_preserve_files_review_and_identity(self):
        from dataclasses import asdict
        job_id=self.queue.enqueue(['source.mp4'],'config: true',asdict(wf.Options()),
                                  uploads={'source.mp4':b'video'})[0]
        before=Path(self.queue.get(job_id)['directory']);output=before/'output'
        (output/'log').mkdir()
        (output/'log/translation_results.xlsx').write_bytes(b'translation')
        (output/'output_sub.mp4').write_bytes(b'finished output')
        project=wf.Project(output/'source.mp4',output,wf.Options());project.approve('translation');project.finish('completed')
        self.queue.update(job_id,status='completed')
        (self.queue.root/'logs').mkdir();(self.queue.root/'logs'/(job_id+'.log')).write_bytes(b'run log')
        target=self.queue.archive(job_id)
        self.assertFalse(before.exists())
        self.assertTrue(target.is_relative_to(self.queue.root.parent/'archive/history'))
        self.assertEqual((target/'output/output_sub.mp4').read_bytes(),b'finished output')
        self.assertEqual((target/'logs/runtime.log').read_bytes(),b'run log')
        restored_project=wf.Project(target/'output/source.mp4',target/'output')
        self.assertTrue(restored_project.approvals()['translation'])
        self.assertIsNotNone(self.queue.get(job_id)['archived_at'])
        with self.assertRaises(ValueError):self.queue.submit_command(job_id,'')
        self.queue.retry(job_id);self.assertEqual(self.queue.get(job_id)['status'],'completed')
        original=self.queue.archive(job_id,restore=True)
        self.assertEqual(original,before)
        self.assertIsNone(self.queue.get(job_id)['archived_at'])
        self.assertTrue(wf.Project(original/'output/source.mp4',original/'output').approvals()['translation'])

    def test_archive_rejects_active_jobs_and_remove_cannot_cancel_claimed_job(self):
        job_id=self.add(['a'])[0]
        self.queue.claim()
        with self.assertRaises(ValueError):self.queue.archive(job_id)
        with self.assertRaises(ValueError):self.queue.remove_queued(job_id)
        self.assertEqual(self.queue.get(job_id)['status'],'running')

    def test_stopped_record_can_be_hidden_and_restored_without_deleting_files(self):
        job_id=self.add(['a'])[0]
        directory=Path(self.queue.get(job_id)['directory'])
        (directory/'output/keep.txt').write_bytes(b'keep project files')
        self.queue.remove_queued(job_id)
        self.assertIsNotNone(self.queue.get(job_id)['removed_at'])
        self.queue.remove_record(job_id,restore=True)
        self.queue.remove_record(job_id)
        self.assertIsNotNone(self.queue.get(job_id)['removed_at'])
        self.assertEqual((directory/'output/keep.txt').read_bytes(),b'keep project files')
        with self.assertRaises(ValueError):self.queue.submit_command(job_id,'')
        self.queue.retry(job_id);self.assertEqual(self.queue.get(job_id)['status'],'cancelled')
        self.queue.remove_record(job_id,restore=True)
        self.assertIsNone(self.queue.get(job_id)['removed_at'])
        self.assertEqual(self.queue.get(job_id)['status'],'cancelled')

    def test_active_record_cannot_be_hidden(self):
        job_id=self.add(['a'])[0]
        self.queue.claim()
        with self.assertRaises(ValueError):self.queue.remove_record(job_id)
        self.assertIsNone(self.queue.get(job_id)['removed_at'])

    def test_locked_directory_uses_verified_copy_and_keeps_original(self):
        from unittest.mock import patch
        job_id=self.add(['a'])[0]
        source=Path(self.queue.get(job_id)['directory'])
        (source/'output/output_sub.mp4').write_bytes(b'finished video')
        self.queue.update(job_id,status='completed')
        rename=Path.rename
        def deny_source(path,target):
            if path==source:raise PermissionError('Windows denied directory move')
            return rename(path,target)
        with patch.object(Path,'rename',deny_source):target=self.queue.archive(job_id)
        self.assertEqual((target/'output/output_sub.mp4').read_bytes(),b'finished video')
        self.assertEqual((source/'output/output_sub.mp4').read_bytes(),b'finished video')
        job=self.queue.get(job_id)
        self.assertIsNotNone(job['archived_at'])
        self.assertEqual(json.loads(job['archive_retained_sources']),[str(source)])
        self.assertIsNone(job['archive_pending_directory'])

        events=[json.loads(line) for line in (self.queue.root/'logs/archive.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([event['event'] for event in events],['started','rename_denied','copy_started','copy_verified','completed'])
        self.assertIn('probes',events[1])
        self.assertIn('file_lock_owners',events[1])
        self.assertGreater(events[3]['metrics']['bytes'],0)
        self.assertEqual(events[-1]['method'],'verified_copy')

    def test_bad_archive_copy_never_commits_or_removes_original(self):
        from unittest.mock import patch
        job_id=self.add(['a'])[0];source=Path(self.queue.get(job_id)['directory'])
        (source/'output/output_sub.mp4').write_bytes(b'finished video')
        self.queue.update(job_id,status='completed')
        rename=Path.rename;copy=q.shutil.copytree
        def deny_source(path,target):
            if path==source:raise PermissionError('locked')
            return rename(path,target)
        def corrupt(src,dst,*args,**kwargs):
            result=copy(src,dst,*args,**kwargs)
            if Path(src)==source:(Path(dst)/'output/output_sub.mp4').write_bytes(b'corrupt')
            return result
        with patch.object(Path,'rename',deny_source),patch.object(q.shutil,'copytree',corrupt):
            with self.assertRaises(OSError):self.queue.archive(job_id)
        self.assertIsNone(self.queue.get(job_id)['archived_at'])
        self.assertIsNone(self.queue.get(job_id)['archive_pending_directory'])
        self.assertEqual((source/'output/output_sub.mp4').read_bytes(),b'finished video')

    def test_import_legacy_keeps_review_versions_and_source(self):
        from dataclasses import asdict
        old=self.root/'old_output';(old/'log').mkdir(parents=True)
        source=old/'source.mp4';source.write_bytes(b'video')
        (old/'log/translation_results.xlsx').write_bytes(b'translation')
        (old/'versions/v1').mkdir(parents=True)
        (old/'versions/v1/output_sub.mp4').write_bytes(b'old result')
        project=wf.Project(source,old,wf.Options())
        project.approve('translation');project.finish('waiting_translation')
        job_id=self.queue.import_output(old,'target_language: Chinese\n',asdict(wf.Options()))
        job=self.queue.get(job_id);output=Path(job['directory'])/'output'
        restored=wf.Project(output/'source.mp4',output)
        self.assertEqual(job['status'],'waiting_translation')
        self.assertTrue(restored.approvals()['translation'])
        self.assertEqual((output/'versions/v1/output_sub.mp4').read_bytes(),b'old result')
        self.assertEqual(source.read_bytes(),b'video')

    def test_review_and_version_preserved_per_project(self):
        ids=self.add(['a','b'])
        projects=[]
        for job_id in ids:
            output=Path(self.queue.get(job_id)['directory'])/'output'
            source=output/'source.mp4';source.write_bytes(b'video')
            (output/'log').mkdir()
            (output/'log/translation_results.xlsx').write_bytes(b'translation')
            project=wf.Project(source,output,wf.Options())
            project.approve('translation');projects.append(project)
        projects[0].archive(['log/translation_results.xlsx'])
        self.assertTrue(projects[1].approvals()['translation'])
        self.assertTrue(list((projects[0].output/'versions').rglob('translation_results.xlsx')))

    def test_ffmpeg_subtitles_with_windows_and_title_characters(self):
        folder=self.root/"中文 title's [demo], test"
        folder.mkdir()
        subtitle=folder/'trans.srt'
        subtitle.write_text('1\n00:00:00,000 --> 00:00:01,000\nHello\n',encoding='utf-8')
        result=subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=s=160x90:d=0.1',
                               '-vf','subtitles='+paths.subtitle_filter_path(subtitle),'-f','null','-'],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))


if __name__=='__main__': unittest.main()
