"""Project library and serial queue UI. Never changes process cwd or config binding."""
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
from core.utils.project_queue import Queue, ROOT, TERMINAL
from core.utils.project_estimates import table_times
from core.utils.project_titles import prefetch_titles, fetch_title
from core.utils.workflow import Options, FILES
from core.st_utils import workflow_ui

STATUS = {'queued':'排队中','running':'处理中','completed':'完成','error':'失败',
          'interrupted':'意外中断','cancelled':'已停止','waiting_terms':'等待术语审核',
          'waiting_translation':'等待翻译审核'}
OUTPUT_MODES = {'字幕版':(True,False),'配音版':(False,True),'字幕+配音版':(True,True)}


def draft_rows(rows):
    """Keep the editor schema and editable seed rows when the draft is empty."""
    defaults={'本次加入':False,'YouTube 链接':'','最终输出':'字幕版','标题':''}
    return [{**defaults,**row} for row in rows] if rows else [dict(defaults),dict(defaults)]


def save_drafts(root,rows):
    import uuid
    path=Path(root)/'drafts.json'
    temporary=path.with_name('drafts.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def navigate_workspace(view):
    st.session_state['workspace_view']=view
    clear_queue_selection()


def render():
    queue=Queue();queue.ensure_service()
    st.session_state['_draft_store_root']=str(queue.root)
    if st.session_state.get('_drafts_loaded_for')!=str(queue.root):
        if '_batch_input_rows' not in st.session_state and (queue.root/'drafts.json').exists():
            try:
                rows=json.loads((queue.root/'drafts.json').read_text(encoding='utf-8'))
                if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):raise ValueError('无效的草稿格式')
                st.session_state['_batch_input_rows']=draft_rows(rows)
            except (OSError,ValueError) as exc:st.warning('待提交清单读取失败，原文件未修改：'+str(exc))
        st.session_state['_drafts_loaded_for']=str(queue.root)
    next_view=st.session_state.pop('_workspace_next_view',None)
    if next_view:navigate_workspace(next_view)
    # Reset the editor before rendering any widgets, even when returning to queue.
    if st.session_state.pop('_batch_reset_after_submit',False):
        st.session_state.pop('batch_links',None)
        remaining=st.session_state.pop('_batch_remaining_rows',[])
        st.session_state['_batch_input_rows']=remaining or [
            {'YouTube 链接':'','最终输出':'字幕版','标题':''},
            {'YouTube 链接':'','最终输出':'字幕版','标题':''}]
    title,dictionary,add=st.columns([4,1,1])
    title.subheader('VideoLingo')
    dictionary.button('发音词典',key='workspace_pronunciation',on_click=pronunciation_editor)
    add.button('＋ 添加视频',key='workspace_add',type='primary',on_click=navigate_workspace,args=('待提交',))
    render_global_status(queue)
    view=st.radio('工作区',['任务队列','待提交','已完成','回收站'],horizontal=True,
                  key='workspace_view',label_visibility='collapsed',on_change=clear_queue_selection)
    for kind,message in st.session_state.pop('_batch_submit_notices',[]):
        if kind=='success':st.toast(message)
        else:st.session_state['_batch_duplicate_notice']=message
    if st.session_state.get('_batch_duplicate_notice'):
        with st.expander('上次提交有重复视频',expanded=True):
            st.warning(st.session_state['_batch_duplicate_notice'])
            for key in st.session_state.get('_batch_duplicate_ids',()):
                job=queue.get(key)
                location='回收站' if job.get('removed_at') else '已完成' if job['status']=='completed' else '任务队列'
                st.caption(job['title']+' · 位于'+location+' · '+STATUS.get(job['status'],job['status']))
                if location=='回收站' and (Path(job['directory'])/'config.snapshot.yaml').exists():
                    st.button('恢复并排队',key='duplicate_retry_'+key,on_click=perform_queue_action,args=(queue,'unhide_retry',(key,)))
                st.button('前往'+location,key='duplicate_locate_'+key,on_click=navigate_workspace,args=(location,))
            st.button('关闭',key='batch_dismiss_duplicate',on_click=lambda:st.session_state.pop('_batch_duplicate_notice',None))
    if view=='待提交':
        render_action_feedback()
        render_drafts(queue)
    else:render_queue(queue)


@st.dialog('发音词典',width='large')
def pronunciation_editor():
    render_pronunciation_editor()


def render_pronunciation_editor():
    from core.utils.pronunciation import load_rules,save_rules,apply_rules,glossary_path
    st.caption('仅用于 IndexTTS 配音，字幕保持原文。使用完整词语，例如“弹容量 → dan4容量”，避免影响“弹簧”。')
    try:rules=load_rules()
    except (OSError,ValueError) as exc:
        st.error('词典读取失败：'+str(exc));return
    rows=st.data_editor([{'词语':word,'配音读法':value} for word,value in rules.items()] or
                        [{'词语':'','配音读法':''}],num_rows='dynamic',hide_index=True,
                        width='stretch',key='pronunciation_rules')
    preview=st.text_input('试读文本预览',value='弹容量较大，但弹簧没有变化。',key='pronunciation_preview')
    try:
        edited={};seen=set()
        for row in rows:
            word=(row.get('词语') or '').strip();spoken=(row.get('配音读法') or '').strip()
            if not word and not spoken:continue
            if not word or not spoken:raise ValueError('词语和配音读法请填写完整。')
            if word.casefold() in seen:raise ValueError('词语重复：'+word)
            seen.add(word.casefold());edited[word]=spoken
        st.code(apply_rules(preview,edited),language=None)
        if st.button('保存发音词典',key='pronunciation_save',type='primary'):
            save_rules(edited)
            st.session_state['_queue_action_notice']='发音词典已保存，将用于后续 IndexTTS 生成，无需重启模型。'
            st.rerun()
    except (OSError,ValueError) as exc:st.error(str(exc))
    st.caption('词典文件：'+str(glossary_path()))
    st.caption('已有成片不会自动改动。项目详情中的“按当前词典更新配音”可重新排队，仅重生成读法发生变化的音频缓存。')


def render_action_feedback():
    if st.session_state.get('_queue_action_notice'):
        st.toast(st.session_state.pop('_queue_action_notice'))
    for key,kind in [('_queue_action_error','error'),('_queue_action_warning','warning')]:
        if st.session_state.get(key):
            getattr(st,kind)(st.session_state[key])
            st.button('关闭提示',key='dismiss'+key,on_click=lambda key=key:st.session_state.pop(key,None))


@st.fragment(run_every=2)
def render_global_status(queue):
    jobs=[j for j in queue.jobs() if not j.get('removed_at') and not j.get('archived_at')]
    active=next((j for j in jobs if j['status']=='running'),None)
    pending=sum(j['status']=='queued' for j in jobs)
    attention=sum(j['status'] in ('error','cancelled','interrupted','waiting_terms','waiting_translation') for j in jobs)
    paused=queue.meta('paused')=='1'
    left,right=st.columns([5,1])
    left.markdown(f"**{'后续任务已暂停' if paused else '队列运行中' if active or pending else '队列空闲'}** · 正在处理 {int(active is not None)} · 等待 {pending} · 需要处理 {attention}")
    if active:
        timing=table_times(queue.jobs())[active['id']]
        left.caption('当前：'+active['title']+' · '+workflow_ui.LABELS.get(active['current'],active['current'])+
                     ' · 已用 '+timing['处理时间']+' · 预计剩余 '+timing['预计剩余'])
    if right.button('继续后续任务' if paused else '暂停后续任务',key='queue_pause'):
        queue.meta('paused','0' if paused else '1');st.rerun()
    if paused:left.caption('正在运行的视频继续处理；后续任务暂不启动。')
    if active:render_log(queue,active,expanded=False,title='运行日志')


def render_selection_actions(queue,selected_jobs,all_jobs):
    st.caption(f'已选 {len(selected_jobs)} 项 · 列表暂定格以保留选择，全局运行状态继续更新。')
    stopped=[j for j in selected_jobs if j['status'] in ('cancelled','error','interrupted') and not j.get('removed_at')]
    removed=[j for j in selected_jobs if j.get('removed_at')]
    pending=[j for j in selected_jobs if j['status']=='queued']
    completed=[j for j in selected_jobs if j['status']=='completed' and not j.get('archived_at')]
    archived=[j for j in selected_jobs if j.get('archived_at')]
    actions=[]
    retryable=[j for j in stopped if (Path(j['directory'])/'config.snapshot.yaml').exists()]
    missing=[j for j in stopped if not Path(j['directory']).exists()]
    if retryable:actions.append(('重新排队','retry','queue_retry_selected',retryable))
    if missing:actions.append(('重新创建并排队（使用当前设置）','recreate','queue_recreate_selected',missing))
    if removed:actions.append(('恢复并排队','unhide_retry','queue_unhide_retry',removed))
    if completed:actions.append(('归档','archive','queue_archive_selected',completed))
    if archived:actions.append(('恢复到项目库','restore','queue_restore_selected',archived))
    with st.container(border=True):
        columns=st.columns(max(2,len(actions)+2))
        for column,(label,action,key,targets) in zip(columns,actions):
            column.button(f'{label}（{len(targets)}）',key=key,type='primary',on_click=perform_queue_action,
                          args=(queue,action,tuple(j['id'] for j in targets)))
        with columns[-2].popover('更多操作'):
            if pending:st.button('移除排队任务',key='queue_remove_selected',on_click=perform_queue_action,args=(queue,'remove',tuple(j['id'] for j in pending)))
            if stopped:st.button('移到回收站',key='queue_hide_selected',on_click=perform_queue_action,args=(queue,'hide',tuple(j['id'] for j in stopped)))
            if removed:st.button('仅恢复记录',key='queue_unhide_selected',on_click=perform_queue_action,args=(queue,'unhide',tuple(j['id'] for j in removed)))
            if len(selected_jobs)==1 and pending:
                ordered=[j['id'] for j in all_jobs if j['status']=='queued']
                index=ordered.index(pending[0]['id'])
                for direction,label in [('up','↑ 上移'),('down','↓ 下移'),('first','移到队首'),('last','移到队尾')]:
                    disabled=index==0 if direction in ('up','first') else index==len(ordered)-1
                    st.button(label,key='queue_order_'+direction,disabled=disabled,on_click=reorder_queue,args=(queue,pending[0]['id'],direction))
        columns[-1].button('取消选择',key='queue_clear_selection',on_click=clear_queue_selection)
    if missing:st.warning('原项目文件缺失。重新创建将保留输出版本选择，使用当前共享配置。')


def status_style(value):
    palette={
        '完成':('#dcfce7','#166534'),
        '处理中':('#dbeafe','#1e40af'),
        '排队中':('#fef3c7','#92400e'),
        '等待术语审核':('#fef3c7','#92400e'),
        '等待翻译审核':('#fef3c7','#92400e'),
        '已停止':('#fee2e2','#991b1b'),
        '失败':('#fee2e2','#991b1b'),
        '意外中断':('#fee2e2','#991b1b'),
    }
    if value not in palette:return ''
    background,foreground=palette[value]
    return f'background-color: {background}; color: {foreground}; font-weight: bold'


def clear_queue_selection():
    st.session_state['_queue_action_ids']=()
    for key in ('_queue_frozen_table','_queue_frozen_row_ids','_queue_frozen_view'):
        st.session_state.pop(key,None)
    st.session_state['_queue_selection_epoch']=st.session_state.get('_queue_selection_epoch',0)+1


def perform_queue_action(queue,action,job_ids):
    """Execute from the button callback, before table refresh can reset selection."""
    done=[];retained=[]
    st.session_state.pop('_queue_action_notice',None)
    st.session_state.pop('_queue_action_error',None)
    st.session_state.pop('_queue_action_warning',None)
    try:
        for job_id in job_ids:
            if action=='remove':queue.remove_queued(job_id)
            elif action=='unhide_retry':
                job=queue.get(job_id)
                if not (Path(job['directory'])/'config.snapshot.yaml').exists():
                    raise ValueError('原项目文件缺失，请先仅恢复记录，再选择重新创建并排队')
                queue.remove_record(job_id,restore=True)
                queue.retry(job_id)
                if queue.get(job_id)['status']!='queued':raise ValueError('无法重新排队，请查看任务状态')
            elif action=='retry':
                job=queue.get(job_id)
                if not (Path(job['directory'])/'config.snapshot.yaml').exists():
                    raise ValueError('原项目文件缺失，无法复用。请点击“重新创建并排队（使用当前设置）”')
                queue.retry(job_id)
                if queue.get(job_id)['status']!='queued':raise ValueError('任务状态已改变，请刷新后检查。')
            elif action=='recreate':
                job=queue.get(job_id)
                if Path(job['directory']).exists():raise ValueError('原项目目录仍存在，请使用重新排队。')
                if not job['source'].startswith(('http://','https://')):raise ValueError('本地视频文件缺失，请重新上传。')
                queue.enqueue([job['source']],(ROOT/'config.yaml').read_text(encoding='utf-8'),
                              json.loads(job['options']),timeout=job['timeout'],allow_existing=True,
                              source_titles={job['source']:job['title']})
                queue.remove_record(job_id)
            elif action in ('hide','unhide'):queue.remove_record(job_id,restore=action=='unhide')
            else:
                queue.archive(job_id,restore=action=='restore')
                retained.extend(json.loads(queue.get(job_id).get('archive_retained_sources') or '[]'))
            done.append(job_id)
    except Exception as exc:
        st.session_state['_queue_action_error']=f'操作未完成：{exc}。已成功处理 {len(done)} 个项目，其余项目仍保留原状态。'
    if done:
        st.session_state['_queue_action_notice']=(f'已将 {len(done)} 个项目重新排队。' if action in ('retry','unhide_retry') else
                                                 f'已使用当前共享配置重新创建并排队 {len(done)} 个项目，原缺失记录已移到已移除记录。' if action=='recreate' else
                                                 f'已归档 {len(done)} 个项目，并从当前列表移除。在“已完成 → 已归档”中查看。' if action=='archive' else
                                                 f'已从列表移除 {len(done)} 条记录，文件仍保留。可在“回收站”恢复。' if action=='hide' else
                                                 f'已恢复 {len(done)} 条记录到当前列表，不会自动重新处理。' if action=='unhide' else
                                                 f'已恢复 {len(done)} 个项目到当前项目列表。' if action=='restore' else
                                                 f'已取消并从列表移除 {len(done)} 个排队任务，文件已保留。可在“回收站”恢复。')
    if retained:
        st.session_state['_queue_action_warning']='归档已完成。Windows 拒绝移动部分目录，已完整复制并校验，原目录额外保留：'+'；'.join(dict.fromkeys(retained))+'。诊断日志：'+str(queue.root/'logs/archive.jsonl')
    clear_queue_selection()


def switch_queue_view(selected_key,other_key):
    if st.session_state.get(selected_key):st.session_state[other_key]=False
    clear_queue_selection()


def reorder_queue(queue,job_id,direction):
    try:
        queue.reorder(job_id,direction)
        st.session_state['_queue_action_notice']='已更新排队顺序，正在处理的视频继续运行。'
        st.session_state.pop('_queue_action_warning',None)
        if queue.meta('queue_order_supported')!='1':
            st.session_state['_queue_action_warning']='当前后台服务尚未加载排序功能。请在当前视频完成后暂停队列并重启 VL 后台服务；新顺序已保存，服务重启后生效。'
        st.session_state.pop('_queue_action_error',None)
    except Exception as exc:
        st.session_state['_queue_action_error']=str(exc)
    clear_queue_selection()


def output_label(options):
    flags=(options.get('original',True),options.get('dubbing',False))
    return next((label for label,value in OUTPUT_MODES.items() if value==flags),'字幕文件')


def batch_row_options(rows, base_options):
    """Validate the entire form before creating any projects."""
    sources=[];settings={}
    for index,row in enumerate(rows,1):
        source=(row.get('YouTube 链接') or '').strip()
        if not source:continue
        links=youtube_links(source)
        if len(links)!=1:raise ValueError(f'第 {index} 行请只填写一个视频链接。')
        mode=row.get('最终输出') or '字幕版'
        if mode not in OUTPUT_MODES:raise ValueError(f'第 {index} 行的输出版本无效。')
        original,dubbing=OUTPUT_MODES[mode]
        options=asdict(replace(base_options,original=original,dubbing=dubbing))
        if source in settings and settings[source]!=options:
            raise ValueError(f'第 {index} 行链接重复且输出选择不同；需要两种版本时请选择“字幕+配音版”。')
        if source not in settings:sources.append(source)
        settings[source]=options
    return sources,settings


def render_log(queue, job, expanded=False,title=None):
    with st.expander(title or '后台输出与错误详情：'+job['title'],expanded=expanded):
        path = queue.root/'logs'/(job['id']+'.log')
        st.caption('后台日志独立于启动终端，随所在区域自动更新。'+str(path))
        if not path.exists():
            st.caption('项目尚未启动，暂无后台输出。'); return
        with path.open('rb') as stream:
            # Windows may cache directory-entry size while another process writes.
            size=stream.seek(0,os.SEEK_END)
            stream.seek(max(0,size-18000))
            text=stream.read(18000).decode('utf-8',errors='replace')
        st.caption(f'已写入 {size:,} 字节')
        text=re.sub(r'sk-[A-Za-z0-9_-]+','[redacted]',text)
        text=re.sub(r'\x1b\[[0-9;]*m','',text).replace('\r\n','\n').replace('\r','\n')
        with st.container(height=300):
            st.code('\n'.join(text.splitlines()[-80:]) if text else '工作进程已启动，等待第一条输出…',language=None)


def open_file(path):
    if os.name == 'nt': os.startfile(str(Path(path).resolve()))
    else: st.info(str(path))


def youtube_links(text):
    result = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        parsed = urlparse(line)
        if parsed.scheme not in ('http','https') or parsed.hostname not in (
            'youtube.com','www.youtube.com','m.youtube.com','youtu.be','www.youtu.be'):
            raise ValueError('每行请填写一个 YouTube 视频链接：'+line)
        if parsed.path == '/playlist': raise ValueError('请提供视频链接，暂不支持整个播放列表。')
        if line not in result: result.append(line)
    return result


@st.cache_data(ttl=60,show_spinner=False)
def preview_title(source):
    try:
        if len(youtube_links(source))!=1:return '请每行填写一个视频链接'
        return fetch_title(source)
    except Exception:
        return '暂时无法获取标题（仍可提交）'


def update_batch_titles():
    """Apply editor deltas once, then bind fetched titles to the edited URLs."""
    rows=draft_rows(st.session_state.get('_batch_input_rows'))
    changes=st.session_state.get('batch_links',{})
    for index,values in changes.get('edited_rows',{}).items():
        rows[int(index)].update(values)
    rows=[row for index,row in enumerate(rows) if index not in changes.get('deleted_rows',[])]
    rows.extend(changes.get('added_rows',[]))
    rows=draft_rows(rows)
    from concurrent.futures import ThreadPoolExecutor
    sources=list(dict.fromkeys((row.get('YouTube 链接') or '').strip() for row in rows))
    sources=[source for source in sources if source]
    with ThreadPoolExecutor(max_workers=4) as pool:
        titles=dict(zip(sources,pool.map(preview_title,sources)))
    for row in rows:
        row.setdefault('本次加入',False)
        row['标题']=titles.get((row.get('YouTube 链接') or '').strip(),'')
    st.session_state['_batch_input_rows']=rows
    if st.session_state.get('_draft_store_root'):save_drafts(Path(st.session_state['_draft_store_root']),rows)
    # Reset deltas after rebinding data, so edits/deletions are never applied twice.
    st.session_state.pop('batch_links',None)


def render_drafts(queue):
    if st.session_state.pop('_batch_reset_after_submit',False):
        st.session_state.pop('batch_links',None)
        remaining=st.session_state.pop('_batch_remaining_rows',[])
        if remaining:st.session_state['_batch_input_rows']=remaining
        else:st.session_state.pop('_batch_input_rows',None)
    st.subheader('待提交清单')
    st.caption('链接自动保存；选中本次要处理的视频，其余留待下次。')
    with st.expander('上传本地视频'):
        files = st.file_uploader('上传本地视频（可多选，上传后可逐个选择输出版本）',
                                type=['mp4','mkv','mov','webm','avi','flv','m4v'],accept_multiple_files=True)
    with st.container(border=True):
        st.caption('每行填写一个 YouTube 链接，并分别选择最终输出版本。可粘贴多行链接，或在表格底部添加行。')
        input_rows=draft_rows(st.session_state.get('_batch_input_rows'))
        st.session_state['_batch_input_rows']=input_rows
        rows=st.data_editor(input_rows,
                            column_config={
                                '本次加入':st.column_config.CheckboxColumn('本次加入',default=False),
                                'YouTube 链接':st.column_config.TextColumn('YouTube 链接',width='large'),
                                '最终输出':st.column_config.SelectboxColumn('最终输出',options=list(OUTPUT_MODES),default='字幕版',required=True),
                                '标题':st.column_config.TextColumn('标题',width='large')},
                            disabled=['标题'],on_change=update_batch_titles,
                            num_rows='dynamic',hide_index=True,width='stretch',key='batch_links')
        st.caption('输入链接后按 Enter 或离开单元格，自动获取标题；获取失败不影响加入队列。')
        checked=[i for i,row in enumerate(rows) if row.get('本次加入') is True]
        selected_indices=checked if checked else list(range(len(rows)))
        selected_rows=[rows[i] for i in selected_indices]
        count=sum(bool((row.get('YouTube 链接') or '').strip()) for row in selected_rows)
        st.caption(f'本次提交 {count} 个链接：'+('仅勾选行；未勾选的链接留待下次。' if checked else '未勾选任何行，将提交全部非空链接。'))
        local_modes={file.name:st.selectbox(file.name+' — 最终输出',list(OUTPUT_MODES),key=f'local_mode_{i}')
                     for i,file in enumerate(files)}
        with st.expander('高级选项'):
            review = st.checkbox('翻译后等待人工审核（其他项目继续处理）',value=False)
            allow_existing=st.checkbox('重新创建已有视频的项目（会重新处理）',value=False,
                                       help='默认跳过项目库中已有的视频。需要补生成输出时，可在原项目中调整输出并继续。')
            timeout = st.number_input('单个阶段最长运行时间（分钟）',min_value=1,max_value=1440,value=120,
                                      help='包含下载、翻译、配音等阶段；超时会终止当前项目及其子进程，继续下一个。长视频可调高。')
        label=f'加入选中的 {count} 个视频' if checked else f'加入全部 {count} 个视频'
        st.button(label,type='primary',key='batch_submit',disabled=not count and not files,
                  on_click=submit_drafts,args=(queue,rows,files,review,allow_existing,timeout,local_modes))
    if st.session_state.get('_batch_submit_error'):st.error(st.session_state['_batch_submit_error'])
    with st.expander('共享工作流与视频输出设置'):
        workflow_ui.render_controls()
    if (ROOT/'output').is_dir():
        with st.expander('导入旧 output 项目'):
            st.caption('复制旧 output 的源视频、审核状态和历史版本到 Projects，保留原目录。可能需要较多磁盘空间。')
            if st.button('复制导入旧 output',disabled=workflow_ui.any_active()):
                try:
                    with st.spinner('正在复制旧项目…'):
                        queue.import_output(ROOT/'output',(ROOT/'config.yaml').read_text(encoding='utf-8'),asdict(workflow_ui.preference()))
                    st.success('导入完成。可在项目列表中查看或恢复。')
                except Exception as exc:st.error(str(exc))


def submit_drafts(queue,rows,files,review,allow_existing,timeout,local_modes):
    st.session_state.pop('_batch_submit_error',None)
    rows=st.session_state.get('_batch_input_rows',rows)
    checked=[i for i,row in enumerate(rows) if row.get('本次加入') is True]
    selected_indices=checked if checked else list(range(len(rows)))
    selected_rows=[rows[i] for i in selected_indices]
    try:
        uploads = {file.name:file.getvalue() for file in files}
        options = replace(workflow_ui.preference(),review_translation=review,auto_start=True,
                          voice_revision=workflow_ui.voice_revision(),
                          translation_revision=workflow_ui.translation_revision())
        links,source_options=batch_row_options(selected_rows,options)
        for name,mode in local_modes.items():
            original,dubbing=OUTPUT_MODES[mode]
            source_options[name]=asdict(replace(options,original=original,dubbing=dubbing))
        if not links and not uploads: raise ValueError('请提供链接或视频文件。')
        source_titles={(row.get('YouTube 链接') or '').strip():row['标题'] for row in selected_rows
                       if row.get('标题') and row['标题'] not in (
                           '暂时无法获取标题（仍可提交）','请每行填写一个视频链接')}
        created=queue.enqueue(links+list(uploads), (ROOT/'config.yaml').read_text(encoding='utf-8'),
                      asdict(options),timeout=timeout*60,uploads=uploads,source_options=source_options,
                      allow_existing=allow_existing,source_titles=source_titles)
        notices=[]
        if created:
            queue.meta('paused','0')
            notices.append(('success',f'已将 {len(created)} 个新项目加入队列。'))
        if queue.last_skipped:
            existing=[queue.get(j['id']) for j in queue.last_skipped]
            st.session_state['_batch_duplicate_ids']=tuple(j['id'] for j in existing)
            notices.append(('warning','本次未重复添加：'+'；'.join(j['title']+'（原项目：'+STATUS.get(j['status'],j['status'])+'）' for j in existing)+
                       '。本次提交未停止或更改原任务，原项目仍按队列状态执行。需要补生成版本时，请在原项目中调整输出；需要独立重做时，勾选“重新创建已有视频的项目”。'))
        st.session_state['_batch_submit_notices']=notices
        submitted_sources={queue.get(key)['source'] for key in created}
        remaining=[dict(row) for i,row in enumerate(rows)
            if (row.get('YouTube 链接') or '').strip() and
            (i not in selected_indices or (row.get('YouTube 链接') or '').strip() not in submitted_sources)]
        st.session_state['_batch_remaining_rows']=remaining
        save_drafts(queue.root,remaining)
        if created:st.session_state['_workspace_next_view']='任务队列'
        st.session_state['_batch_reset_after_submit']=True
    except Exception as exc:st.session_state['_batch_submit_error']=str(exc)


@st.fragment(run_every=5)
def render_queue(queue):
    queue.ensure_service()
    prefetch_titles(queue)
    all_jobs = queue.jobs()
    view=st.session_state.get('workspace_view','任务队列')
    removed_view=view=='回收站'
    archived_view=False
    if view=='已完成':
        scope=st.radio('归档范围',['未归档','已归档','全部'],horizontal=True,key='completed_scope',label_visibility='collapsed',on_change=clear_queue_selection)
        archived_view=scope=='已归档'
    else:scope='未归档'
    view_key=view+':'+scope
    jobs=[j for j in all_jobs if (bool(j.get('removed_at')) if removed_view else
          not j.get('removed_at') and (j['status']=='completed' and
          (scope=='全部' or bool(j.get('archived_at'))==archived_view) if view=='已完成' else
          not j.get('archived_at') and j['status']!='completed'))]
    render_action_feedback()
    query=st.text_input('搜索标题或来源链接',key='queue_search',placeholder='搜索视频…',on_change=clear_queue_selection)
    if query.strip():jobs=[j for j in jobs if query.casefold() in (j['title']+' '+j['source']).casefold()]
    if not jobs:
        clear_queue_selection()
        st.info('没有匹配的任务。' if query else '回收站为空。' if removed_view else '这里还没有已完成的项目。' if view=='已完成' else '没有待处理任务，点击右上角“添加视频”开始。')
        return
    times=table_times(jobs) if view=='已完成' else {}
    st.caption(f'{len(jobs)} 个项目'+(' · 总处理时间不含排队和人工审核等待。' if view=='已完成' else ''))
    table=pd.DataFrame([{'项目':j['title'],'输出':output_label(json.loads(j['options'])),
                   '状态':STATUS.get(j['status'],j['status']),
                   '阶段':workflow_ui.LABELS.get(j['current'],j['current']),
                   **({'总处理时间':times[j['id']]['处理时间']} if view=='已完成' else {}), '问题':j['error'],
                   '来源 URL / 文件':j['source'],
                   '加入时间':datetime.fromtimestamp(j['created']).strftime('%Y-%m-%d %H:%M:%S')} for j in jobs])
    row_ids=tuple(j['id'] for j in jobs)
    previous_ids=st.session_state.get('_queue_table_row_ids')
    if previous_ids is not None and (previous_ids!=row_ids or st.session_state.get('_queue_table_view')!=view_key):
        st.session_state['_queue_selection_epoch']=st.session_state.get('_queue_selection_epoch',0)+1
        st.session_state.pop('_queue_frozen_table',None)
        st.session_state['_queue_action_ids']=()
    st.session_state['_queue_table_row_ids']=row_ids
    st.session_state['_queue_table_view']=view_key
    frozen=st.session_state.get('_queue_frozen_table')
    if frozen is not None and (st.session_state.get('_queue_frozen_row_ids')!=row_ids or
                               st.session_state.get('_queue_frozen_view')!=view_key or
                               tuple(frozen.columns)!=tuple(table.columns)):
        clear_queue_selection();frozen=None
    if frozen is not None:table=frozen
    table_key='queue_table_'+view_key+'_'+str(st.session_state.get('_queue_selection_epoch',0))

    def on_table_select(rows=None):
        if rows is None:rows=st.session_state[table_key]['selection']['rows']
        st.session_state['_queue_action_ids']=tuple(row_ids[i] for i in rows if 0<=i<len(row_ids))
        if rows:
            st.session_state['_queue_frozen_table']=table.copy()
            st.session_state['_queue_frozen_row_ids']=row_ids
            st.session_state['_queue_frozen_view']=view_key
        else:st.session_state.pop('_queue_frozen_table',None)

    styled=table.style.set_uuid('vl_queue').apply(lambda column:[status_style(value) for value in column],subset=['状态'])
    st.dataframe(styled,
                 hide_index=True,width='stretch',
                 key=table_key,on_select=on_table_select,selection_mode='multi-row',
                 column_order=['项目','输出','状态']+(['总处理时间'] if view=='已完成' else ['阶段'])+['问题'],
                 column_config={'项目':st.column_config.TextColumn(width='large')})
    selected_jobs=[j for j in jobs if j['id'] in st.session_state.get('_queue_action_ids',())]
    if not selected_jobs:
        st.caption('选择一行查看详情；勾选多行可批量操作。')
        return
    render_selection_actions(queue,selected_jobs,all_jobs)
    if len(selected_jobs)!=1:return
    selected=selected_jobs[0]['id']
    if selected_jobs[0]['error']:st.error(selected_jobs[0]['error'][:700])
    with st.expander('查看详情与成果 · '+selected_jobs[0]['title'],expanded=False):
        render_project_details(queue,selected)


def translate_title_action(queue,selected):
    from core.utils.project_title_translation import translate_project_title
    error_key='_title_translation_error_'+selected
    st.session_state.pop(error_key,None)
    try:translate_project_title(queue,selected,force=True)
    except Exception as exc:st.session_state[error_key]='中文标题翻译失败：'+str(exc)


def render_project_details(queue,selected):
    job = queue.get(selected)
    directory = Path(job['directory'])
    output = directory/'output'
    st.markdown('**原始标题**')
    st.write(job['title'])
    st.markdown('**中文标题**')
    if job.get('chinese_title') and job.get('chinese_title_source')==job['title']:
        st.code(job['chinese_title'],language=None)
    else:st.caption('尚未翻译')
    st.button('重新翻译中文标题' if job.get('chinese_title') else '翻译成中文',key='project_translate_title',
              on_click=translate_title_action,args=(queue,selected))
    if st.session_state.get('_title_translation_error_'+selected):
        st.error(st.session_state['_title_translation_error_'+selected])
    st.caption(str(directory))
    st.markdown('**来源 URL / 文件**')
    st.code(job['source'],language=None)
    st.caption('加入时间：'+datetime.fromtimestamp(job['created']).strftime('%Y-%m-%d %H:%M:%S')+
               ' · 批次：'+str(job['batch_id'])+' · 任务 ID：'+job['id'])
    if st.button('打开项目目录',key='project_open'): open_file(directory)
    if job.get('removed_at'):
        st.info('此记录已从当前列表移除，项目文件仍保留。恢复记录后才可重试。')
        st.button('恢复这条记录',key='project_unhide',on_click=perform_queue_action,args=(queue,'unhide',(selected,)))
    elif job['status'] in ('cancelled','error','interrupted'):
        st.button('从列表移除此记录（保留文件）',key='project_hide',on_click=perform_queue_action,args=(queue,'hide',(selected,)))
    if job.get('archived_at'):
        st.info('项目已归档到历史目录。恢复到项目库后可以追加或更新输出。')
        st.button('恢复到项目库',key='project_restore',on_click=perform_queue_action,args=(queue,'restore',(selected,)))
    elif job['status']=='completed':
        st.button('归档到 archive/history',key='project_archive',on_click=perform_queue_action,args=(queue,'archive',(selected,)))
    retained_sources=json.loads(job.get('archive_retained_sources') or '[]')
    if retained_sources:st.warning('归档时无法移动源目录；这些原目录仍额外保留：'+'；'.join(retained_sources))
    if job['error']: st.error(job['error'])
    if job['status']=='running':
        st.info('正在处理：'+workflow_ui.LABELS.get(job['current'],job['current']))
        if st.button('停止当前项目，继续下一个',key='project_cancel'):
            queue.update(selected,command='cancel');st.rerun()
    elif job['status']=='queued':
        if st.button('从队列移除（保留项目文件）',key='project_remove'):
            queue.remove_queued(selected)
            st.rerun()
    elif job['status'] in TERMINAL and not job.get('archived_at') and not job.get('removed_at'):
        if json.loads(job['options']).get('dubbing') and (output/'audio/tts_tasks.xlsx').exists():
            st.button('按当前词典更新配音',key='project_refresh_pronunciation',
                      on_click=refresh_pronunciation_action,args=(queue,selected))
        if job['status'].startswith('waiting_'):
            kind = job['status'][len('waiting_'):]
            review_path = output/FILES[kind]
            st.warning('请打开审核文件，保存修改后确认继续。')
            if st.button('打开审核文件',key='project_review'): open_file(review_path)
            if kind=='translation' and (output/'log/translation_results_proofread.xlsx').exists():
                if st.button('应用 LLM 校对结果',key='project_apply'):
                    queue.submit_command(selected,'apply_proofread');st.rerun()
            if st.button('确认审核并继续',key='project_approve',type='primary'):
                queue.submit_command(selected,'approve:'+kind);st.rerun()
        elif job['status']!='completed':
            st.button('重试 / 恢复（复用有效成果）',key='project_retry',on_click=perform_queue_action,
                      args=(queue,'retry',(selected,)))
        with st.expander('调整本项目输出 / 更新成果'):
            options = Options.from_dict(json.loads(job['options']))
            original = st.checkbox('生成字幕版',value=options.original,key='project_original_'+selected)
            dubbing = st.checkbox('生成配音版',value=options.dubbing,key='project_dubbing_'+selected)
            omode = st.selectbox('字幕版字幕',['translation','bilingual'],index=['translation','bilingual'].index(options.original_subtitles),key='project_omode_'+selected)
            dmode = st.selectbox('配音版字幕',['translation','bilingual','none'],index=['translation','bilingual','none'].index(options.dub_subtitles),key='project_dmode_'+selected)
            if st.button('保存并继续 / 更新输出',key='project_update'):
                updated = replace(options,original=original,dubbing=dubbing,original_subtitles=omode,dub_subtitles=dmode)
                queue.submit_command(selected,'',asdict(updated));st.rerun()
    with st.expander('成果、字幕与历史版本',expanded=job['status']=='completed'):
        for name in ('output_sub.mp4','output_dub.mp4'):
            path = output/name
            if path.exists():
                if st.button('打开 '+name,key='project_result_'+name): open_file(path)
                if st.checkbox('预览 '+name,key='project_preview_'+selected+name): st.video(str(path))
        for relative in ('trans.srt','src.srt','src_trans.srt','log/translation_results.xlsx','log/terminology.json'):
            path = output/relative
            if path.is_file(): st.download_button(path.name,path.read_bytes(),file_name=path.name,key='project_file_'+relative)
        if (output/'versions').exists():
            if st.button('打开历史版本目录',key='project_versions'): open_file(output/'versions')
    render_log(queue,job)


def refresh_pronunciation_action(queue,selected):
    try:
        queue.submit_command(selected,'refresh_pronunciation')
        st.session_state['_queue_action_notice']='已重新排队更新配音；复用未受词典影响的音频。'
    except Exception as exc:st.session_state['_queue_action_error']=str(exc)
    clear_queue_selection()
