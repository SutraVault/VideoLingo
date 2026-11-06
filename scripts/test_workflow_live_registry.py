"""A live worker must survive losing the UI registry during source reload."""
import tempfile
import threading
import unittest
from pathlib import Path
from core.st_utils import workflow_ui
from core.st_utils.task_runner import TaskRunner
from core.utils.workflow import Project, Options


class RegistryTests(unittest.TestCase):
    def test_adopts_live_worker_without_restarting(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'source.mp4';source.write_bytes(b'video')
            project=Project(source,root/'output',Options())
            runner=TaskRunner()
            entered=threading.Event();release=threading.Event()
            def operation():
                entered.set();release.wait(5)
                path=project.output/'log/terminology.json'
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            runner.start([('terms',project.wrap('terms',operation))])
            try:
                self.assertTrue(entered.wait(2))
                recovered=workflow_ui.job_for(source)
                self.assertIs(recovered[0],project)
                self.assertIs(recovered[1],runner)
                self.assertTrue(workflow_ui.busy(recovered[1]))
            finally:
                release.set();runner._thread.join(5)
                for key,value in list(workflow_ui._jobs.items()):
                    if value[1] is runner:del workflow_ui._jobs[key]


if __name__=='__main__':unittest.main()
