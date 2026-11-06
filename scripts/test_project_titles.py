import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[1]


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


q=load('title_queue','core/utils/project_queue.py')
titles=load('queued_titles','core/utils/project_titles.py')


class TitleTests(unittest.TestCase):
    def setUp(self):
        root=ROOT/'work/title-tests';root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.temp.cleanup)
        self.queue=q.Queue(self.temp.name)
        self.ids=self.queue.enqueue(['https://youtu.be/first','https://youtu.be/second'],'config: true',{})

    def test_queued_titles_resolve_while_another_video_runs_without_moving_directories(self):
        self.queue.claim()
        before=self.queue.get(self.ids[1])
        titles.fill_titles(self.queue,fetch=lambda source:'YouTube title')
        after=self.queue.get(self.ids[1])
        self.assertEqual(after['title'],'YouTube title')
        self.assertEqual(after['directory'],before['directory'])
        self.assertEqual(after['status'],'queued')
        self.assertTrue(self.queue.get(self.ids[0])['title'].startswith('pending_'))

    def test_failure_does_not_stop_queue_and_retries_are_throttled(self):
        calls=[]
        def fail(source):calls.append(source);raise TimeoutError()
        titles.fill_titles(self.queue,fetch=fail)
        titles.fill_titles(self.queue,fetch=fail)
        self.assertEqual(len(calls),2)
        self.assertTrue(all(job['status']=='queued' for job in self.queue.jobs()))

    def test_lookup_cannot_overwrite_worker_title_after_claim(self):
        def fetch(source):
            self.queue.claim();self.queue.update(self.ids[0],title='Worker title')
            return 'Stale lookup title'
        titles.fill_titles(self.queue,fetch=fetch)
        self.assertEqual(self.queue.get(self.ids[0])['title'],'Worker title')

    def test_oembed_401_falls_back_to_metadata_without_downloading(self):
        source='https://www.youtube.com/watch?v=Si4romekvi0'
        config=Path(self.queue.get(self.ids[0])['directory'])/'config.snapshot.yaml'
        with patch.object(titles,'_fetch_oembed',side_effect=HTTPError(source,401,'Unauthorized',None,None)),patch('yt_dlp.YoutubeDL') as ytdlp:
            extractor=ytdlp.return_value.__enter__.return_value
            extractor.extract_info.return_value={'title':'Why Russia Keeps Losing the Su 34'}
            result=titles.fetch_title(source,config_path=config)
        self.assertEqual(result,'Why Russia Keeps Losing the Su 34')
        extractor.extract_info.assert_called_once_with(source,download=False)
        self.assertTrue(ytdlp.call_args.args[0]['skip_download'])

    def test_successful_oembed_does_not_invoke_fallback(self):
        with patch.object(titles,'_fetch_oembed',return_value='Public title'),patch('yt_dlp.YoutubeDL') as ytdlp:
            self.assertEqual(titles.fetch_title('https://youtu.be/test'),'Public title')
        ytdlp.assert_not_called()


if __name__=='__main__':unittest.main()
