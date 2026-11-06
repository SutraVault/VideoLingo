"""Real worker/FFmpeg integration with cached synthetic upstream artifacts, no APIs."""
from dataclasses import asdict
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ruamel.yaml import YAML

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


q=load('integration_queue','core/utils/project_queue.py')
wf=load('integration_workflow','core/utils/workflow.py')


class WorkerTests(unittest.TestCase):
    def test_failed_stage_resume_keeps_completed_audio_preparation(self):
        base=ROOT/'work/project-worker-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root=Path(temp);queue=q.Queue(root/'Projects')
            options=wf.Options(dubbing=True,review_translation=False)
            job_id=queue.enqueue(['source.mp4'],(ROOT/'config.yaml').read_text(encoding='utf-8'),
                                 asdict(options),uploads={'source.mp4':b'source'})[0]
            directory=Path(queue.get(job_id)['directory']);output=directory/'output'
            harness=root/'controlled_worker.py'
            harness.write_text('''import importlib.util,json,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);directory=Path(sys.argv[4])
os.environ['VL_OUTPUT_DIR']=str(directory/'output')
os.environ['VL_CONFIG_PATH']=str(directory/'config.runtime.yaml')
os.environ['VL_TERMS_PATH']=str(directory/'custom_terms.xlsx')
from core.utils import pipeline
output=directory/'output'
def operation(stage):
 def run():
  path=directory/'calls.json'
  counts=json.loads(path.read_text()) if path.exists() else {}
  counts[stage]=counts.get(stage,0)+1;path.write_text(json.dumps(counts))
  if stage=='synthesis' and counts[stage]==1:raise RuntimeError('test synthesis failure')
  names={'asr':['log/cleaned_chunks.xlsx'],'nlp':['log/split_by_nlp.txt'],
   'meaning':['log/split_by_meaning.txt'],'terms':['log/terminology.json'],
   'translation':['log/translation_results.xlsx'],
   'subtitles':['trans.srt','src.srt','audio/src_subs_for_audio.srt','audio/trans_subs_for_audio.srt'],
   'original':['output_sub.mp4'],'audio_tasks':['audio/tts_tasks.xlsx'],
   'merge_audio':['dub.mp3'],'dubbing':['output_dub.mp4']}
  for name in names.get(stage,[]):
   p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'ready')
 return run
pipeline.operations=lambda host,options:{stage:operation(stage) for stage in
 ['asr','nlp','meaning','terms','translation','split','subtitles','original','audio_tasks',
  'references','synthesis','merge_audio','dubbing']}
spec=importlib.util.spec_from_file_location('worker',Path(sys.argv[1])/'core/utils/project_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
queue=worker.queue_module.Queue(sys.argv[2])
try:worker.run_job(queue,sys.argv[3])
except Exception as exc:queue.update(sys.argv[3],status='error',error=str(exc));sys.exit(1)
''',encoding='utf-8')
            def run():
                queue.claim()
                return subprocess.run([sys.executable,str(harness),str(ROOT),str(queue.root),job_id,str(directory)],
                                      cwd=ROOT,capture_output=True,timeout=60,
                                      env=dict(os.environ,PYTHONIOENCODING='utf-8'))
            self.assertEqual(run().returncode,1)
            self.assertEqual(queue.get(job_id)['current'],'synthesis')
            self.assertTrue((output/'output_sub.mp4').exists())
            tasks=(output/'audio/tts_tasks.xlsx').stat().st_mtime_ns
            queue.retry(job_id)
            result=run()
            self.assertEqual(result.returncode,0,(result.stdout+result.stderr).decode(errors='replace')[-3000:])
            self.assertEqual(queue.get(job_id)['status'],'completed')
            self.assertEqual((output/'audio/tts_tasks.xlsx').stat().st_mtime_ns,tasks)
            counts=json.loads((directory/'calls.json').read_text())
            self.assertEqual(counts['synthesis'],2)
            self.assertEqual(counts['audio_tasks'],1)
            self.assertEqual(counts['references'],1)
            queue.submit_command(job_id,'refresh_pronunciation')
            refreshed=run()
            self.assertEqual(refreshed.returncode,0,(refreshed.stdout+refreshed.stderr).decode(errors='replace')[-3000:])
            self.assertEqual(queue.get(job_id)['status'],'completed')
            self.assertEqual((output/'audio/tts_tasks.xlsx').stat().st_mtime_ns,tasks)
            counts=json.loads((directory/'calls.json').read_text())
            self.assertEqual(counts['audio_tasks'],1)
            self.assertEqual(counts['references'],1)
            self.assertEqual(counts['synthesis'],3)
            self.assertEqual(counts['translation'],1)

    def test_isolated_outputs_review_resume_and_version(self):
        base=ROOT/'work/project-worker-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root=Path(temp)
            video=root/'test.mp4'
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=320x180:r=10:d=1',
                            '-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','libx264','-c:a','aac',str(video)],check=True)
            yaml=YAML();config=yaml.load((ROOT/'config.yaml').read_text(encoding='utf-8'))
            config['ffmpeg_gpu']=False
            config.setdefault('watermark',{})['enabled']=False
            text=io.StringIO();yaml.dump(config,text)
            queue=q.Queue(root/'Projects')
            options=wf.Options(original=True,dubbing=True,review_translation=True)
            job_id=queue.enqueue(["中文 title's [demo], test.mp4"],text.getvalue(),asdict(options),
                                 uploads={"中文 title's [demo], test.mp4":video.read_bytes()})[0]
            directory=Path(queue.get(job_id)['directory']);output=directory/'output'
            subtitle='1\n00:00:00,000 --> 00:00:00,900\nHello world\n'
            for name in ['log/cleaned_chunks.xlsx','log/split_by_nlp.txt','log/split_by_meaning.txt',
                         'log/terminology.json','log/translation_results.xlsx','trans.srt','src.srt',
                         'audio/src_subs_for_audio.srt','audio/trans_subs_for_audio.srt','dub.srt']:
                path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(subtitle,encoding='utf-8')
            # WAV content is supported by pydub even with the pipeline's mp3 suffix.
            for name in ['dub.mp3','audio/background.mp3']:
                subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=220:duration=1',
                                '-f','wav',str(output/name)],check=True)
            def run(command=''):
                if queue.get(job_id)['status'] in q.TERMINAL: queue.submit_command(job_id,command)
                queue.claim()
                env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
                result=subprocess.run([sys.executable,str(ROOT/'core/utils/project_worker.py'),
                                       'job',str(queue.root),job_id],cwd=ROOT,env=env,capture_output=True,timeout=90)
                self.assertEqual(result.returncode,0,(result.stdout+result.stderr).decode(errors='replace')[-9000:])
                return queue.get(job_id)
            job=run()
            self.assertEqual(job['status'],'waiting_translation')
            self.assertFalse((output/'output_sub.mp4').exists())
            job=run('approve:translation')
            self.assertEqual(job['status'],'completed')
            for name in ['output_sub.mp4','output_dub.mp4']:
                self.assertGreater((output/name).stat().st_size,1000)
            original=(output/'output_sub.mp4').read_bytes()
            # Changing output settings rebuilds only that video, preserving versions.
            from dataclasses import replace
            queue.submit_command(job_id,'',asdict(replace(options,original_subtitles='bilingual')))
            job=run()
            self.assertEqual(job['status'],'completed')
            self.assertTrue(any(p.read_bytes()==original for p in (output/'versions').rglob('output_sub.mp4')))
            self.assertEqual((directory/'config.snapshot.yaml').read_text(encoding='utf-8'),text.getvalue())


if __name__=='__main__':unittest.main()
