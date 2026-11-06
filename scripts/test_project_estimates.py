import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


est=load('estimates','core/utils/project_estimates.py')
q=load('timing_queue','core/utils/project_queue.py')


class EstimateTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'work/estimate-tests';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.queue=q.Queue(self.temp.name)

    def test_execution_time_accumulates_without_queue_or_review_wait(self):
        from unittest.mock import patch
        job_id=self.queue.enqueue(['a'],'config: true',{'dubbing':False})[0]
        with patch.object(q.time,'time',return_value=100):self.queue.claim()
        self.assertEqual(est.elapsed(self.queue.get(job_id),130),30)
        with patch.object(q.time,'time',return_value=140):self.queue.update(job_id,status='waiting_translation')
        self.assertEqual(est.elapsed(self.queue.get(job_id),900),40)
        self.queue.retry(job_id)
        with patch.object(q.time,'time',return_value=1000):self.queue.claim()
        with patch.object(q.time,'time',return_value=1010):self.queue.update(job_id,status='completed')
        self.assertEqual(est.elapsed(self.queue.get(job_id),2000),50)

    def test_eta_scales_by_video_length_and_subtracts_current_stage(self):
        ids=self.queue.enqueue(['a','b'],'same: config',{'original':False,'dubbing':True})
        for index,job_id in enumerate(ids):
            directory=Path(self.queue.get(job_id)['directory'])
            (directory/'source_metadata.json').write_text(json.dumps({'duration':100*(index+1)}))
            logs=directory/'output/log';logs.mkdir()
            steps=[{'label':s,'status':'completed','duration':10} for s in ('synthesis','merge_audio','dubbing')] if index==0 else [
                {'label':'synthesis','status':'running','start':90}]
            (logs/'pipeline_timing.json').write_text(json.dumps({'steps':steps}))
        self.queue.update(ids[0],status='completed')
        self.queue.update(ids[1],status='running',current='synthesis')
        times=est.table_times(self.queue.jobs(),now=100)
        self.assertEqual(times[ids[1]]['预计剩余'],'约 00:50')
        self.queue.update(ids[1],current='download')
        self.assertEqual(est.table_times(self.queue.jobs(),now=100)[ids[1]]['预计剩余'],'暂无法估算')

    def test_missing_history_is_not_a_fabricated_estimate(self):
        job_id=self.queue.enqueue(['a'],'config: true',{})[0]
        self.assertEqual(est.table_times(self.queue.jobs())[job_id]['预计剩余'],'暂无法估算')

    def test_old_worker_completion_is_backfilled_only_once(self):
        job_id=self.queue.enqueue(['a'],'config: true',{})[0]
        with self.queue.connect() as db:
            db.execute("UPDATE jobs SET status='completed',process_started=100,attempt_started=100,updated=140 WHERE id=?",(job_id,))
        for _ in range(2):
            restored=q.Queue(self.temp.name)
            self.assertEqual(est.elapsed(restored.get(job_id),900),40)


if __name__=='__main__':unittest.main()
