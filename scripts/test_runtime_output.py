"""Concurrent tasks must not display another task's console output."""
import tempfile
import threading
import unittest
from pathlib import Path
from core.st_utils.runtime_output import capture_worker_output


class OutputTests(unittest.TestCase):
    def test_concurrent_workers_have_separate_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/f'{i}.log' for i in range(2)]
            barrier=threading.Barrier(2)
            def run(i):
                with capture_worker_output(paths[i]):
                    barrier.wait(timeout=5)
                    print(f'video-{i}')
            threads=[threading.Thread(target=run,args=(i,)) for i in range(2)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(10)
            for i,path in enumerate(paths):
                self.assertEqual(path.read_text(encoding='utf-8'),f'video-{i}\n')


if __name__=='__main__':unittest.main()
