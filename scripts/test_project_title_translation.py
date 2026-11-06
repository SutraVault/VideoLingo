import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


q=load('title_translation_queue','core/utils/project_queue.py')
titles=load('project_title_translation','core/utils/project_title_translation.py')


class TitleTranslationTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'work/title-translation-tests';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.queue=q.Queue(self.temp.name)
        self.key=self.queue.enqueue(['https://youtu.be/title'],'config: true',{},
                                   source_titles={'https://youtu.be/title':'How a Modern Revolver Works'})[0]

    def test_translation_is_persistent_cached_and_does_not_change_processing_state(self):
        calls=[];before=self.queue.get(self.key)
        def ask(prompt,**kwargs):
            calls.append(prompt)
            return {'chinese_title':'现代左轮手枪的工作原理'}
        result=titles.translate_project_title(self.queue,self.key,ask=ask)
        self.assertEqual(result,'现代左轮手枪的工作原理')
        titles.translate_project_title(q.Queue(self.queue.root),self.key,ask=ask)
        self.assertEqual(len(calls),1)
        after=self.queue.get(self.key)
        for field in ('title','directory','status','current','options','updated'):
            self.assertEqual(after[field],before[field])
        titles.translate_project_title(self.queue,self.key,force=True,ask=ask)
        self.assertEqual(len(calls),2)

    def test_invalid_response_does_not_overwrite_previous_translation(self):
        self.queue.save_chinese_title(self.key,'How a Modern Revolver Works','原有中文标题')
        with self.assertRaises(ValueError):
            titles.translate_project_title(self.queue,self.key,force=True,ask=lambda *a,**k:{'chinese_title':''})
        self.assertEqual(self.queue.get(self.key)['chinese_title'],'原有中文标题')

    def test_title_change_during_request_is_not_saved_under_wrong_source(self):
        def ask(*args,**kwargs):
            self.queue.update(self.key,title='Updated title')
            return {'chinese_title':'旧标题的译文'}
        with self.assertRaises(ValueError):titles.translate_project_title(self.queue,self.key,ask=ask)
        self.assertEqual(self.queue.get(self.key)['chinese_title'],'')
