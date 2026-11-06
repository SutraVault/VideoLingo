"""Exercise the real queue form and controls without starting background work."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

HARNESS='''
from pathlib import Path
from core.st_utils import project_ui as ui
from core.utils.project_queue import Queue
import streamlit as st
# AppTest cannot replay dataframe row-selection events on subsequent button
# clicks. Inject the browser event payload while still rendering the real table.
def test_dataframe(*args,_original=st.dataframe,**kwargs):
    result=_original(*args,**kwargs)
    if kwargs.get('on_select') and '_test_selected_rows' in st.session_state:
        rows=st.session_state.pop('_test_selected_rows')
        if callable(kwargs['on_select']):kwargs['on_select'](rows=rows)
        return {'selection':{'rows':rows,'columns':[]}}
    return result
st.dataframe=test_dataframe
queue=Queue(TEST_ROOT)
queue.ensure_service=lambda:None
ui.Queue=lambda:queue
ui.prefetch_titles=lambda queue:None
ui.preview_title=lambda source:'Title: '+source
from core.utils.project_estimates import table_times as real_table_times
ui.table_times=lambda jobs:real_table_times(jobs,now=st.session_state.get('_test_clock'))
if st.session_state.pop('_test_update_titles',False):ui.update_batch_titles()
ui.render()
'''


class ProjectUITests(unittest.TestCase):
    def setUp(self):
        from core.utils.project_queue import Queue
        base=ROOT/'work/project-ui-tests';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.queue=Queue(self.temp.name)
        self.app=self.new_app()

    def new_app(self):
        app=AppTest.from_string('TEST_ROOT='+repr(self.temp.name)+'\n'+HARNESS,default_timeout=45).run()
        self.assertFalse(app.exception)
        return app

    def view(self,name):
        self.app.radio(key='workspace_view').set_value(name).run()
        self.assertFalse(self.app.exception)

    def submit(self,rows):
        self.view('待提交')
        self.app.session_state['_batch_input_rows']=rows
        self.app.session_state['batch_links']={'edited_rows':{},'added_rows':[],'deleted_rows':[]}
        self.app.run()
        self.app.button(key='batch_submit').click().run()
        self.assertFalse(self.app.exception)

    def add(self,sources):
        return self.queue.enqueue(sources,'target_language: Chinese',{'original':True,'dubbing':False})

    def select(self,rows):
        self.app.session_state['_test_selected_rows']=rows
        self.app.run()
        self.assertFalse(self.app.exception)

    def test_default_is_queue_without_input_or_disabled_action_wall(self):
        self.assertEqual(self.app.radio(key='workspace_view').value,'任务队列')
        self.assertFalse(self.app.dataframe)
        self.assertFalse(any(b.key=='batch_submit' for b in self.app.button))
        self.assertFalse(any(b.key=='queue_retry_selected' for b in self.app.button))
        self.app.button(key='workspace_add').click().run()
        self.assertEqual(self.app.radio(key='workspace_view').value,'待提交')

    def test_empty_saved_draft_keeps_columns_and_can_submit(self):
        (self.queue.root/'drafts.json').write_text('[]',encoding='utf-8')
        self.app=self.new_app();self.view('待提交')
        table=self.app.dataframe[0].value
        self.assertEqual(list(table.columns),['本次加入','YouTube 链接','最终输出','标题'])
        self.assertEqual(len(table),2)
        self.submit([{'YouTube 链接':'https://youtu.be/afterempty','最终输出':'字幕版'}])
        self.assertEqual(len(self.queue.jobs()),1)

    def test_deleting_all_input_rows_restores_editable_blank_rows(self):
        self.view('待提交')
        self.app.session_state['batch_links']={'edited_rows':{},'added_rows':[],'deleted_rows':[0,1]}
        self.app.session_state['_test_update_titles']=True
        self.app.run()
        self.assertFalse(self.app.exception)
        table=self.app.dataframe[0].value
        self.assertEqual(len(table),2)
        self.assertIn('YouTube 链接',table.columns)
        self.assertEqual(list(table['YouTube 链接']),['',''])

    def test_selected_submission_preserves_drafts_and_titles_across_restart(self):
        self.submit([{'本次加入':False,'YouTube 链接':'https://youtu.be/later','最终输出':'配音版','标题':'Later'},
                     {'本次加入':True,'YouTube 链接':'https://youtu.be/now','最终输出':'字幕+配音版','标题':'Now'}])
        jobs=self.queue.jobs();self.assertEqual([j['source'] for j in jobs],['https://youtu.be/now'])
        self.assertEqual(jobs[0]['title'],'Now')
        self.assertTrue(json.loads(jobs[0]['options'])['dubbing'])
        self.assertEqual(self.app.radio(key='workspace_view').value,'任务队列')
        self.app=self.new_app();self.view('待提交')
        table=self.app.dataframe[0].value
        self.assertEqual(list(table['YouTube 链接']),['https://youtu.be/later'])
        self.assertEqual(list(table['标题']),['Later'])
        self.assertEqual(list(table['最终输出']),['配音版'])
        self.app.button(key='batch_submit').click().run()
        self.assertEqual(len(self.queue.jobs()),2)
        self.view('待提交')
        self.assertEqual(list(self.app.dataframe[0].value['YouTube 链接']),['',''])

    def test_unselected_invalid_draft_does_not_block_selected_video(self):
        self.submit([{'本次加入':False,'YouTube 链接':'unfinished'},
                     {'本次加入':True,'YouTube 链接':'https://youtu.be/now','最终输出':'配音版'}])
        self.assertEqual(len(self.queue.jobs()),1)
        self.view('待提交')
        self.assertEqual(list(self.app.dataframe[0].value['YouTube 链接']),['unfinished'])

    def test_invalid_input_creates_nothing_and_remains_in_editor(self):
        self.submit([{'YouTube 链接':'https://example.com/bad'}])
        self.assertEqual(self.queue.jobs(),[])
        self.assertTrue(self.app.error)
        self.assertEqual(list(self.app.dataframe[0].value['YouTube 链接'])[0],'https://example.com/bad')

    def test_duplicate_keeps_draft_and_shows_location(self):
        key=self.add(['https://youtu.be/a'])[0];self.queue.remove_queued(key)
        self.submit([{'YouTube 链接':'https://youtu.be/a','最终输出':'字幕版'}])
        self.assertEqual(len(self.queue.jobs()),1)
        self.assertTrue(self.app.warning)
        self.assertEqual(self.app.radio(key='workspace_view').value,'待提交')
        self.assertEqual(list(self.app.dataframe[0].value['YouTube 链接'])[0],'https://youtu.be/a')
        self.assertTrue(any('位于回收站' in element.value for element in self.app.caption))
        self.app.button(key='duplicate_retry_'+key).click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.queue.get(key)['status'],'queued')
        self.assertIsNone(self.queue.get(key)['removed_at'])

    def test_global_running_status_visible_in_empty_recycle_bin(self):
        self.add(['https://youtu.be/a','https://youtu.be/b']);self.queue.claim()
        self.view('回收站')
        text=' '.join(m.value for m in self.app.markdown)
        self.assertIn('正在处理 1',text);self.assertIn('等待 1',text)
        self.assertFalse(self.app.dataframe)
        self.assertTrue(any('回收站为空' in m.value for m in self.app.info))
        self.app.button(key='queue_pause').click().run()
        self.assertEqual(self.queue.meta('paused'),'1')
        self.assertTrue(any('当前' in m.value for m in self.app.caption))

    def test_retry_clears_frozen_status_and_actions_are_contextual(self):
        key=self.add(['https://youtu.be/a'])[0];self.queue.update(key,status='cancelled')
        self.app.run();self.select([0])
        self.assertTrue(any(b.key=='queue_retry_selected' for b in self.app.button))
        self.assertFalse(any(b.key=='queue_archive_selected' for b in self.app.button))
        self.app.button(key='queue_retry_selected').click().run()
        self.assertEqual(self.queue.get(key)['status'],'queued')
        self.assertEqual(list(self.app.dataframe[0].value['状态']),['排队中'])
        self.assertNotIn('_queue_frozen_table',self.app.session_state)

    def test_reorder_and_remove_then_restore_and_queue(self):
        ids=self.add(['https://youtu.be/a','https://youtu.be/b','https://youtu.be/c'])
        self.app.run();self.select([2])
        self.app.button(key='queue_order_first').click().run()
        self.assertEqual(self.queue.jobs()[0]['id'],ids[2])
        self.select([0]);self.app.button(key='queue_remove_selected').click().run()
        self.assertIsNotNone(self.queue.get(ids[2])['removed_at'])
        self.view('回收站');self.select([0])
        self.app.button(key='queue_unhide_retry').click().run()
        self.assertEqual(self.queue.get(ids[2])['status'],'queued')
        self.assertIsNone(self.queue.get(ids[2])['removed_at'])

    def test_completed_archive_restore_and_results_detail(self):
        key=self.add(['https://youtu.be/a'])[0];self.queue.update(key,status='completed')
        original=Path(self.queue.get(key)['directory'])
        (original/'output/output_sub.mp4').write_bytes(b'video')
        self.addCleanup(lambda:self.queue.archive(key,restore=True) if self.queue.get(key)['archived_at'] else None)
        self.app.run();self.assertFalse(self.app.dataframe)
        self.view('已完成');self.select([0])
        self.assertTrue(any(b.key=='project_result_output_sub.mp4' for b in self.app.button))
        self.app.button(key='queue_archive_selected').click().run()
        self.assertIsNotNone(self.queue.get(key)['archived_at']);self.assertFalse(self.app.dataframe)
        self.app.radio(key='completed_scope').set_value('已归档').run();self.select([0])
        self.app.button(key='queue_restore_selected').click().run()
        self.assertIsNone(self.queue.get(key)['archived_at']);self.assertTrue(original.exists())

    def test_missing_project_can_recreate(self):
        import shutil
        key=self.add(['https://youtu.be/missing'])[0];self.queue.update(key,status='error')
        shutil.rmtree(self.queue.get(key)['directory'])
        self.app.run();self.select([0])
        self.assertFalse(any(b.key=='queue_retry_selected' for b in self.app.button))
        self.app.button(key='queue_recreate_selected').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.queue.jobs()),2)
        self.assertIsNotNone(self.queue.get(key)['removed_at'])

    def test_titles_update_and_draft_edit_is_saved_before_submission(self):
        self.view('待提交')
        self.app.session_state['batch_links']={'edited_rows':{0:{'YouTube 链接':'https://youtu.be/a'}},'added_rows':[],'deleted_rows':[]}
        self.app.session_state['_test_update_titles']=True
        self.app.run();self.assertFalse(self.app.exception)
        self.assertEqual(list(self.app.dataframe[0].value['标题'])[0],'Title: https://youtu.be/a')
        self.app=self.new_app();self.view('待提交')
        self.assertEqual(list(self.app.dataframe[0].value['标题'])[0],'Title: https://youtu.be/a')

    def test_search_filters_without_hiding_global_status(self):
        self.add(['https://youtu.be/apple','https://youtu.be/banana']);self.queue.claim()
        self.app.run()
        self.app.text_input(key='queue_search').set_value('banana').run()
        self.assertEqual(len(self.app.dataframe[0].value),1)
        self.assertIn('正在处理 1',' '.join(m.value for m in self.app.markdown))

    def test_detail_can_translate_and_display_persistent_chinese_title(self):
        from unittest.mock import patch
        key=self.add(['https://youtu.be/title'])[0]
        self.queue.update(key,title='Original English title')
        self.app.run();self.select([0])
        def translate(queue,job_id,force=False):
            queue.save_chinese_title(job_id,queue.get(job_id)['title'],'中文视频标题')
            return '中文视频标题'
        with patch('core.utils.project_title_translation.translate_project_title',side_effect=translate) as request:
            self.app.button(key='project_translate_title').click().run()
            self.assertFalse(self.app.exception)
            self.assertEqual(request.call_count,1)
            self.assertTrue(any(code.value=='中文视频标题' for code in self.app.code))
            self.app.run()
            self.assertEqual(request.call_count,1)
        self.assertEqual(self.queue.get(key)['chinese_title'],'中文视频标题')

    def test_pronunciation_dictionary_can_be_edited_from_header(self):
        import os
        from unittest.mock import patch
        from core.utils.pronunciation import save_rules,load_rules
        path=self.queue.root/'test_glossary.yaml'
        save_rules({'弹容量':'dan4容量'},path)
        with patch.dict(os.environ,{'VL_PRONUNCIATION_GLOSSARY':str(path)}):
            self.app.button(key='workspace_pronunciation').click().run()
            self.assertFalse(self.app.exception)
            self.assertTrue(any('dan4容量' in code.value for code in self.app.code))
            # AppTest does not replay nested dialog fragment events. Exercise
            # the same editor body directly for save/validation events.
            editor=AppTest.from_string('from core.st_utils.project_ui import render_pronunciation_editor\nrender_pronunciation_editor()',default_timeout=45).run()
            editor.session_state['pronunciation_rules']={'edited_rows':{0:{'配音读法':'dan4 rong2 liang4'}},'added_rows':[],'deleted_rows':[]}
            editor.button(key='pronunciation_save').click().run()
            self.assertFalse(editor.exception)
            self.assertEqual(load_rules(path),{'弹容量':'dan4 rong2 liang4'})

    def test_completed_dubbing_project_can_request_pronunciation_refresh(self):
        key=self.queue.enqueue(['https://youtu.be/pronunciation'],'config: true',{'original':False,'dubbing':True})[0]
        job=self.queue.get(key)
        (Path(job['directory'])/'output/audio').mkdir()
        (Path(job['directory'])/'output/audio/tts_tasks.xlsx').write_bytes(b'task placeholder')
        self.queue.update(key,status='completed')
        self.view('已完成');self.select([0])
        self.app.button(key='project_refresh_pronunciation').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.queue.get(key)['status'],'queued')
        self.assertEqual(self.queue.get(key)['command'],'refresh_pronunciation')

    def test_ticking_clock_does_not_recreate_selectable_table_or_styles(self):
        key=self.add(['https://youtu.be/clock'])[0]
        self.queue.claim()
        started=self.queue.get(key)['attempt_started']
        self.app.session_state['_test_clock']=started+10
        self.app.run()
        first=self.app.dataframe[0].proto
        first_id=first.id;first_style=first.styler.uuid
        self.assertNotIn('处理时间',self.app.dataframe[0].value.columns)
        self.assertNotIn('预计剩余',self.app.dataframe[0].value.columns)
        self.assertTrue(any('已用 00:10' in c.value for c in self.app.caption))
        self.app.session_state['_test_clock']=started+40
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.dataframe[0].proto.id,first_id)
        self.assertEqual(self.app.dataframe[0].proto.styler.uuid,first_style)
        self.assertTrue(any('已用 00:40' in c.value for c in self.app.caption))

    def test_completed_table_keeps_total_processing_time_without_remaining_estimate(self):
        key=self.add(['https://youtu.be/completed'])[0]
        self.queue.update(key,status='completed')
        with self.queue.connect() as db:
            db.execute('UPDATE jobs SET elapsed_seconds=125 WHERE id=?',(key,))
        self.view('已完成')
        table=self.app.dataframe[0].value
        self.assertEqual(list(table['总处理时间']),['02:05'])
        self.assertNotIn('处理时间',table.columns)
        self.assertNotIn('预计剩余',table.columns)


if __name__=='__main__':unittest.main()
