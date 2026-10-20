"""One task plan shared by upload, subtitle processing, dubbing and outputs."""
from dataclasses import asdict,replace
from pathlib import Path
import hashlib,json,os,threading
import streamlit as st
from core.utils.workflow import Options,Project,FILES
from core.utils.config_utils import load_key,update_key,lock,yaml,CONFIG_PATH,invalidate_config_cache
from core.st_utils.task_runner import TaskRunner
from translations.translations import translate as t

_jobs={}
_lock=threading.RLock()
LABELS={'asr':'Speech recognition','nlp':'Sentence segmentation','meaning':'Semantic segmentation',
        'terms':'Extract terminology','translation':'Translate and proofread','split':'Split subtitles',
        'subtitles':'Generate subtitle files','original':'Original-audio subtitle video',
        'audio_tasks':'Prepare dubbing','references':'Extract reference audio','synthesis':'Generate dubbing',
        'merge_audio':'Merge dubbing audio','dubbing':'Dubbed video'}


def preference():
    try:return Options.from_dict(dict(load_key('workflow')))
    except KeyError:
        return Options(dubbing=load_key('auto_start_dubbing_after_subtitles'),
                       review_translation=load_key('pause_after_translate'),
                       review_terms=load_key('pause_before_translate'),
                       auto_start=load_key('auto_start_translation_after_video'))


def save_preference(options):
    with lock:
        with open(CONFIG_PATH,encoding='utf-8') as stream:data=yaml.load(stream)
        data['workflow']=asdict(options)
        data['pause_after_translate']=options.review_translation
        data['pause_before_translate']=options.review_terms
        data['auto_start_translation_after_video']=options.auto_start
        data['auto_start_dubbing_after_subtitles']=options.dubbing
        with open(CONFIG_PATH,'w',encoding='utf-8') as stream:yaml.dump(data,stream)
    invalidate_config_cache()


def voice_revision():
    settings={}
    for key in ('tts_method','indextts','gpt_sovits','kokoro','edge_tts','sf_fish_tts','openai_tts','azure_tts','fish_tts'):
        try:settings[key]=load_key(key)
        except KeyError:pass
    return hashlib.sha256(json.dumps(settings,sort_keys=True,default=str).encode()).hexdigest()


def translation_revision():
    return hashlib.sha256(str(load_key('target_language')).encode()).hexdigest()


def busy(runner):
    return bool(runner and (runner.is_active or (runner._thread and runner._thread.is_alive())))


def job_for(source):
    with _lock:
        p=Path(source).resolve();s=p.stat()
        key=(str(p),s.st_size,s.st_mtime_ns)
        if key not in _jobs:
            project=Project(p,options=replace(preference(),voice_revision=voice_revision(),translation_revision=translation_revision()))
            if not project.state.get('options'):project.set_options(preference())
            _jobs[key]=(project,TaskRunner())
        return _jobs[key]


def any_active():return any(busy(r) for _,r in _jobs.values())


_PRESETS = {
    'subtitles': dict(original=True,dubbing=False,review_translation=False,review_terms=False,auto_start=True,original_subtitles='translation',dub_subtitles='translation'),
    'review_dub': dict(original=False,dubbing=True,review_translation=True,review_terms=False,auto_start=True,original_subtitles='translation',dub_subtitles='translation'),
    'both': dict(original=True,dubbing=True,review_translation=False,review_terms=False,auto_start=True,original_subtitles='translation',dub_subtitles='translation'),
}
_PRESET_LABELS = {'subtitles':'Subtitles, fully automatic','review_dub':'Review, then dub',
                  'both':'Both videos, fully automatic','custom':'Custom'}
_DRAFT_FIELDS = {'original':'wf_option_original','dubbing':'wf_option_dubbing',
                 'original_subtitles':'wf_option_omode','dub_subtitles':'wf_option_dmode',
                 'review_translation':'wf_option_review','review_terms':'wf_option_terms','auto_start':'wf_option_auto'}


def _select_draft_preset():
    selected=st.session_state['wf_preset']
    for field,value in _PRESETS.get(selected,{}).items():
        st.session_state[_DRAFT_FIELDS[field]]=value


def _customize_draft():
    st.session_state['wf_preset']='custom'


def _draft_identity(project,options):
    return 'draft-v2:'+(project.identity if project else 'preferences')+json.dumps(asdict(options),sort_keys=True)


@st.fragment
def render_controls(job=None):
    st.header(t('Workflow control'))
    project,runner=job if job else (None,None)
    options=project.options if project else preference()
    is_busy=busy(runner)
    if project and not is_busy and (options.voice_revision!=voice_revision() or options.translation_revision!=translation_revision()):
        options=replace(options,voice_revision=voice_revision(),translation_revision=translation_revision());project.set_options(options)
    identity=_draft_identity(project,options)
    if st.session_state.get('_workflow_form_source')!=identity:
        baseline={key:getattr(options,field) for field,key in _DRAFT_FIELDS.items()}
        baseline['wf_option_open']=load_key('auto_open_translation_results')
        for key,value in baseline.items():st.session_state[key]=value
        st.session_state['_wf_draft_baseline']=baseline
        st.session_state['wf_preset']=next((name for name,changes in _PRESETS.items()
            if all(getattr(options,k)==v for k,v in changes.items())),'custom')
        st.session_state['_workflow_form_source']=identity
    with st.container(border=True):
        st.write(t('Choose your outputs, review points and start mode.'))
        st.markdown("""<style>
        .st-key-wf_preset [role="radiogroup"] {gap: .65rem; flex-wrap: wrap;}
        .st-key-wf_preset [role="radiogroup"] > label {
            border: 2px solid rgba(128,128,128,.4); border-radius: .6rem;
            padding: .6rem .85rem; margin: 0;}
        .st-key-wf_preset label:has(input:checked) {
            border-color: #ff4b4b; background: rgba(255,75,75,.16);
            box-shadow: inset 0 2px 4px rgba(0,0,0,.2);}
        .st-key-wf_preset label:has(input:checked)::after {content: "✓"; font-weight: 700;}
        </style>""",unsafe_allow_html=True)
        st.radio(t('Quick setup — choose a starting point'),list(_PRESET_LABELS),
                 format_func=lambda value:t(_PRESET_LABELS[value]),key='wf_preset',
                 horizontal=True,on_change=_select_draft_preset,disabled=is_busy)
        st.caption(t('Selecting a preset fills in the settings below. You can adjust them before applying.'))
        st.markdown('**'+t('Review and customize')+'**')
        with st.container(border=True):
            st.markdown('**'+t('Requested outputs')+'**')
            col1,col2=st.columns(2)
            original=col1.checkbox(t('Original-audio subtitle video'),key='wf_option_original',on_change=_customize_draft,disabled=is_busy)
            dubbing=col2.checkbox(t('Dubbed video'),key='wf_option_dubbing',on_change=_customize_draft,disabled=is_busy)
            st.caption(t('Translation workbook and subtitle files are always provided. No video selected means files only.'))
            modes={'translation':t('Translation only'),'bilingual':t('Bilingual'),'none':t('No subtitles')}
            with st.expander(t('Video output settings')):
                omode=st.selectbox(t('Original-video subtitles'),['translation','bilingual'],format_func=modes.get,key='wf_option_omode',on_change=_customize_draft,disabled=is_busy)
                dmode=st.selectbox(t('Dubbed-video subtitles'),list(modes),format_func=modes.get,key='wf_option_dmode',on_change=_customize_draft,disabled=is_busy)
            review=st.checkbox(t('Wait for my review after translation'),key='wf_option_review',on_change=_customize_draft,disabled=is_busy)
            with st.expander(t('Additional review points')):
                terms=st.checkbox(t('Wait for my terminology review'),key='wf_option_terms',on_change=_customize_draft,disabled=is_busy)
                auto_open=st.checkbox(t('Auto-open translation results after translation completes'),key='wf_option_open',on_change=_customize_draft,disabled=is_busy)
            auto=st.checkbox(t('Start automatically when the video is ready'),key='wf_option_auto',on_change=_customize_draft,disabled=is_busy)
            outputs=[t(label) for selected,label in ((original,'Original-audio subtitle video'),(dubbing,'Dubbed video')) if selected]
            stops=t('Review terminology and translation') if terms and review else t('Terminology review') if terms else t('Translation review') if review else t('No review stops')
            st.markdown('**'+t('Settings preview')+'**')
            st.write(' · '.join([' + '.join(outputs) if outputs else t('Translation workbook and subtitle files'),stops,t('Starts after upload') if auto else t('Start manually')]))
            baseline=st.session_state['_wf_draft_baseline']
            dirty=any(st.session_state[key]!=value for key,value in baseline.items())
            if dirty:st.warning(t('Unsaved changes. The current task still uses the applied settings.'))
            else:st.caption(t('Settings are applied.'))
            if st.button(t('Apply settings'),key='wf_apply_settings',type='primary',disabled=is_busy or not dirty):
                updated=replace(options,original=original,dubbing=dubbing,original_subtitles=omode,dub_subtitles=dmode,review_translation=review,review_terms=terms,auto_start=auto)
                save_preference(updated)
                if auto_open!=load_key('auto_open_translation_results'):update_key('auto_open_translation_results',auto_open)
                if project:project.set_options(updated)
                st.session_state['_wf_draft_baseline']={key:st.session_state[key] for key in baseline}
                st.session_state['_workflow_form_source']=_draft_identity(project,updated)
                st.rerun()
        st.caption(t('Running settings are locked. Stop before changing the remaining plan.') if is_busy else
                   t('Changes take effect after you apply them. Adding an output never starts it without confirmation.'))


@st.fragment(run_every=2)
def render_preview(job=None):
    st.caption(t('Applied workflow'))
    if not job:
        options=preference();st.info(t('Waiting for a video. Your workflow settings are ready.'))
        _preview_line(options);return
    project,runner=job
    _preview_line(project.options,project)
    status=project.state['status']
    if runner.state=='paused':
        st.warning(t('Paused. Resume to run the next operation.') if project.state.get('current_status')=='completed' else t('Pause requested: the current operation will finish before the next step waits.'))
    elif runner.state=='stopped' and busy(runner):
        st.info(t('Stopping after the current operation. Please wait before changing or resuming the plan.'))
    elif runner.is_active:
        st.info(t('Running')+': '+t(LABELS.get(project.state.get('current'),'Subtitle processing')))
        st.progress(runner.progress)
    elif status in ('waiting_translation','waiting_terms'):
        enabled=project.options.review_translation if status=='waiting_translation' else project.options.review_terms
        if enabled:st.warning(t('Waiting for your translation review') if status=='waiting_translation' else t('Waiting for your terminology review'))
        else:st.info(t('Review pause is disabled. Confirm to continue automatically.'))
    elif status=='error':st.error(t('Processing failed. Completed outputs are preserved.')+' '+project.state.get('last_error',''))
    elif status in ('stopped','interrupted'):st.warning(t('Task stopped or interrupted. Confirm to resume.'))
    elif status=='completed':
        pending,_=project.next_plan()
        st.success(t('All requested outputs are complete.')) if not pending else st.info(t('The plan has new or outdated outputs. Confirm below to continue.'))
    else:st.info(t('Ready to start.'))


def _preview_line(options,project=None):
    facts=project.facts() if project else {}
    approvals=project.approvals() if project else {}
    def node(label,stage):
        return ('✓ ' if facts.get(stage) else '▶ ' if project and project.state.get('current')==stage and project.state['status']=='running' else '')+t(label)
    nodes=[t('Video'),node('Speech recognition','asr')]
    if options.review_terms:nodes += [node('Extract terminology','terms'),('✓ ' if approvals.get('terms') else '⏸ ')+t('Terminology review')]
    nodes += [node('Translation','translation')]
    if options.review_translation:nodes += [('✓ ' if approvals.get('translation') else '⏸ ')+t('Translation review')]
    nodes += [node('Subtitle files','subtitles')]
    st.markdown(' → '.join(nodes))
    if options.original:st.write('↳ '+node('Original-audio subtitle video','original'))
    if options.dubbing:st.write('↳ '+t('Generate dubbing')+' → '+node('Dubbed video','dubbing'))
    if not options.original and not options.dubbing:st.caption(t('Finish after generating the translation workbook and subtitle files.'))


def operations(host,options):
    return {
        'asr':host['_2_asr'].transcribe,'nlp':host['_3_1_split_nlp'].split_by_spacy,
        'meaning':host['_3_2_split_meaning'].split_sentences_by_meaning,
        'terms':host['_4_1_summarize'].get_summary,
        'translation':lambda:(host['_4_2_translate'].translate_all(),host['_4_3_proofread_translation'].proofread_translation_if_enabled()),
        'split':host['_5_split_sub'].split_for_sub_main,'subtitles':host['_6_gen_sub'].align_timestamp_main,
        'original':lambda:host['_7_sub_into_vid'].merge_subtitles_to_video(force_burn=True,subtitle_mode=options.original_subtitles),
        'audio_tasks':lambda:(host['_8_1_audio_task'].gen_audio_task_main(),host['_8_2_dub_chunks'].gen_dub_chunks()),
        'references':host['_9_refer_audio'].extract_refer_audio_main,'synthesis':host['_10_gen_audio'].gen_audio,
        'merge_audio':host['_11_merge_audio'].merge_full_audio,
        'dubbing':lambda:host['_12_dub_to_vid'].merge_video_audio(subtitle_mode=options.dub_subtitles),
    }


def start(job,host):
    project,runner=job
    if busy(runner):return
    stages,boundary=project.next_plan()
    host['_stop_handoff_alarms']()
    if not stages:project.finish(boundary);return
    ops=operations(host,project.options)
    steps=[(t(LABELS[s]),project.wrap(s,ops[s])) for s in stages]
    steps.append((t('Finish current plan'),lambda:project.finish(boundary)))
    project.state.update(status='running',last_error='');project.save()
    runner.start(steps,run_label=t('Workflow processing'))


def _open(path):
    path=Path(path).resolve()
    if os.name=='nt':os.startfile(str(path))
    else:st.info(str(path))


@st.fragment(run_every=2)
def render_sections(job,host):
    project,runner=job
    # Update the full page once when work finishes, so controls/sidebar unlock.
    # Ordinary draft edits only rerender render_controls, not the whole page.
    busy_key='_wf_was_busy_'+project.identity
    was_busy=st.session_state.get(busy_key,False)
    is_busy=busy(runner)
    st.session_state[busy_key]=is_busy
    if was_busy and not is_busy:st.rerun(scope='app')
    # Only a fresh upload event may auto-start; never revive errors/stops/reviews.
    if st.session_state.pop('_auto_start_translation_pending',False) and project.options.auto_start and project.state['status']=='ready':
        start(job,host);st.rerun()
    st.header(t('b. Translate and Generate Subtitles'))
    facts=project.facts()
    for key in ('asr','meaning','terms','translation','subtitles'):
        active=runner.is_active and project.state.get('current')==key
        st.write(('▶ ' if active else '✓ ' if facts.get(key) else '○ ')+t(LABELS[key]))
    state=project.state['status']
    if state in ('waiting_translation','waiting_terms') and not runner.is_active:
        kind='translation' if state=='waiting_translation' else 'terms'
        path=project.output/FILES[kind]
        enabled=project.options.review_translation if kind=='translation' else project.options.review_terms
        if enabled:st.warning(t('Waiting for your translation review') if kind=='translation' else t('Waiting for your terminology review'))
        else:st.info(t('Review pause is disabled. Confirm to continue automatically.'))
        host['_notify_user_handoff']('workflow_review_'+project.identity+'_'+kind+'_'+str(path.stat().st_mtime_ns), 'VideoLingo',t('Review the file, save your changes, then confirm to continue.'))
        left,right=st.columns(2)
        if left.button(t('Open review file'),key='wf_open_review'):_open(path)
        proofreader=host['_4_3_proofread_translation']
        reviewing=kind=='translation' and proofreader.is_proofread_running()
        if kind=='translation' and Path(host['PROOFREAD_TRANSLATION_RESULTS']).exists():
            if st.button(t('Apply LLM Proofread'),disabled=reviewing):
                proofreader.apply_proofread_to_translation();st.rerun()
        if right.button(t('Confirm review and continue') if enabled else t('Continue automatically'),disabled=reviewing,key='wf_confirm_review'):
            project.approve(kind);start(job,host);st.rerun()
    elif busy(runner) and runner.state=='stopped':
        st.info(t('Stopping after the current operation. Please wait before changing or resuming the plan.'))
    elif busy(runner):
        left,right=st.columns(2)
        if runner.state=='paused':
            if left.button(t('Resume processing'),key='wf_resume'):
                runner.resume();st.rerun()
        elif left.button(t('Pause after current operation'),key='wf_pause'):
            runner.pause();st.rerun()
        if right.button(t('Stop processing'),key='wf_stop'):
            runner.stop();project.state['status']='stopped';project.save();st.rerun()
    elif state=='error':
        st.error(project.state.get('last_error',''))
        host['_notify_user_handoff']('workflow_error_'+project.state.get('failure_id',project.identity),'VideoLingo',project.state.get('last_error',''),kind='error')
        if st.button(t('Retry failed step'),key='wf_retry'):
            # Preserve the runner's failed-step index, including already completed
            # audio tasks; rebuilding a plan here would discard successful audio.
            host['_stop_handoff_alarms']()
            if runner.state=='error':runner.retry_from_failed_step()
            else:start(job,host)
            st.rerun()
    elif (project.next_plan()[0] or project.next_plan()[1]!='completed') and st.button(t('Start / continue workflow'),key='wf_start'):
        start(job,host);st.rerun()
    st.header(t('c. Dubbing'))
    if project.options.dubbing:
        st.write(t('Dubbing runs after subtitle files are ready and required reviews are confirmed.'))
        if runner.is_active and project.state.get('current') in ('audio_tasks','references','synthesis','merge_audio','dubbing'):
            st.info(t(LABELS[project.state['current']]))
    else:st.caption(t('Not requested. You can add a dubbed video after subtitle processing.'))
    _results(job,host)


def _results(job,host):
    project,runner=job
    st.header(t('Outputs'))
    for stage in ('original','dubbing'):
        meta=project.state['artifacts'].get(stage)
        if not meta:continue
        path=project.output/meta['path']
        if not path.exists():continue
        fresh=project.facts()[stage]
        with st.container(border=True):
            st.write(('✓ ' if fresh else '↻ ')+t(LABELS[stage]))
            if not fresh:st.warning(t('This output uses earlier translation or output settings. The saved file is preserved.'))
            left,right=st.columns(2)
            if left.button(t('Play'),key='wf_play_'+stage):st.session_state['wf_preview_'+stage]=not st.session_state.get('wf_preview_'+stage,False)
            if right.button(t('Open file'),key='wf_open_'+stage):_open(path)
            if st.session_state.get('wf_preview_'+stage):st.video(str(path))
            if not fresh and st.button(t('Update this output'),key='wf_update_'+stage,disabled=busy(runner)):
                project.set_options(replace(project.options,**{stage:True}));start(job,host);st.rerun()
    with st.expander(t('Translation workbook and subtitle files')):
        for relative in ('log/translation_results.xlsx','trans.srt','src.srt','src_trans.srt'):
            path=project.output/relative
            if path.exists():st.download_button(path.name,path.read_bytes(),file_name=path.name,key='wf_download_'+path.name)
    if project.facts()['subtitles'] and not project.facts()['dubbing'] and not busy(runner):
        with st.expander(t('Add a dubbed video'),expanded=not project.options.dubbing):
            st.write(t('Reuse existing recognition, translation and subtitles. Your original-audio video will be kept.'))
            mode=st.selectbox(t('Dubbed-video subtitles'),['translation','bilingual','none'],index=['translation','bilingual','none'].index(project.options.dub_subtitles),format_func=lambda v:t({'translation':'Translation only','bilingual':'Bilingual','none':'No subtitles'}[v]),key='wf_append_mode')
            if st.button(t('Start generating dubbed video'),key='wf_append_dubbing'):
                project.set_options(replace(project.options,dubbing=True,dub_subtitles=mode))
                start(job,host);st.rerun()
    if not busy(runner) and project.state['status']=='completed':
        if st.button(t("Archive to 'history'"),key='wf_archive'):
            host['cleanup']();st.rerun()
