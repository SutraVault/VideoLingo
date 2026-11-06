"""Standalone service/worker entry point. Bind paths before importing core."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Importing core eagerly loads pipeline constants; the supervisor must avoid that.
spec = importlib.util.spec_from_file_location('vl_queue', Path(__file__).with_name('project_queue.py'))
queue_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue_module)


def run_job(queue, job_id):
    job = queue.get(job_id)
    directory = Path(job['directory'])
    source = job['source']
    if source.startswith(('https://', 'http://')) and not (directory/'source_metadata.json').exists():
        queue.update(job_id, current='metadata')
        from ruamel.yaml import YAML
        import yt_dlp
        config = YAML(typ='safe').load((directory/'config.snapshot.yaml').read_text(encoding='utf-8'))
        settings = {'quiet': True, 'noplaylist': True, 'socket_timeout': 30, 'retries': 2}
        cookies = config.get('youtube', {}).get('cookies_path')
        if cookies: settings['cookiefile'] = str((ROOT / cookies).resolve())
        with yt_dlp.YoutubeDL(settings) as downloader:
            metadata = downloader.extract_info(source, download=False)
        if metadata.get('_type') == 'playlist': raise ValueError('请提供单个视频链接，不支持播放列表。')
        directory = queue.rename(job_id, metadata.get('title') or metadata.get('id') or 'video')
        (directory/'source_metadata.json').write_text(json.dumps(
            {'title':metadata.get('title'),'id':metadata.get('id'),'duration':metadata.get('duration')},ensure_ascii=False),encoding='utf-8')
    if not (directory/'config.runtime.yaml').exists():
        import shutil
        shutil.copy2(directory/'config.snapshot.yaml',directory/'config.runtime.yaml')
    for key, value in {'VL_OUTPUT_DIR':directory/'output',
                       'VL_CONFIG_PATH':directory/'config.runtime.yaml',
                       'VL_TERMS_PATH':directory/'custom_terms.xlsx'}.items():
        os.environ[key] = str(value)
    job = queue.get(job_id)
    manifest = directory/'project.json'
    manifest.write_text(json.dumps({'id':job_id,'source':source,'title':job['title'],
                                        'created':job['created'],'batch_id':job['batch_id']},
                                       ensure_ascii=False,indent=2),encoding='utf-8')
    import core
    from core._1_ytdlp import find_video_files, download_video_ytdlp
    from core.utils.config_utils import load_key
    from core.utils.workflow import Project, Options
    if source.startswith(('https://', 'http://')):
        try:
            find_video_files()
        except ValueError:
            queue.update(job_id, current='download')
            result = download_video_ytdlp(source, resolution=load_key('ytb_resolution'),
                                         prefer_creator_subtitles=True,
                                         subtitle_language=load_key('whisper.language'))
            subtitle = result.get('creator_subtitle') if result else None
            if subtitle:
                from core.utils.srt_import import import_srt_to_cleaned_chunks
                import_srt_to_cleaned_chunks(Path(subtitle['path']).read_text(encoding='utf-8-sig'))
    source_path = find_video_files()
    project = Project(source_path, directory/'output', Options.from_dict(json.loads(job['options'])))
    project.set_options(Options.from_dict(json.loads(job['options'])))
    command = job['command']
    if command.startswith('approve:'):
        project.approve(command.split(':',1)[1])
    if command == 'apply_proofread':
        core._4_3_proofread_translation.apply_proofread_to_translation()
        queue.update(job_id,status=project.state['status'],command='')
        return
    from core.utils.pipeline import operations
    ops = operations(vars(core),project.options)
    stages, boundary = project.next_plan()
    # Resume an interrupted plan only if its inputs/options have not changed.
    resume_path = directory/'remaining_plan.json'
    signature = {'options':json.loads(job['options']), 'translation':project.revision(),
                 'inputs':project.translation_inputs()}
    try:
        previous = json.loads(resume_path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        previous = {}
    resume_valid=(previous.get('signature')==signature and previous.get('command','')==command and previous.get('stages'))
    if resume_valid:
        stages = previous['stages']; boundary = previous['boundary']
    if command=='refresh_pronunciation' and not resume_valid:
        if not (directory/'output/audio/tts_tasks.xlsx').is_file():
            raise ValueError('配音任务文件缺失，请使用普通重试重新准备项目。')
        project.archive(['dub.mp3','dub.srt','normalized_dub.wav'])
        stages=['synthesis','merge_audio']+(['dubbing'] if project.options.dubbing else [])
        boundary='completed'
    from core.utils.timing import start_run, finish_run, start_step, finish_step
    run_id = start_run('Project queue')
    for index, stage in enumerate(stages):
        queue.update(job_id,current=stage)
        resume_path.write_text(json.dumps({'signature':{'options':json.loads(job['options']),
                               'translation':project.revision(),'inputs':project.translation_inputs()},
                               'stages':stages[index:], 'boundary':boundary,'command':command}),encoding='utf-8')
        step_id = start_step(stage,run_id=run_id)
        try:
            project.wrap(stage,ops[stage])()
        except Exception as exc:
            finish_step(step_id,status='error',error=str(exc))
            finish_run(run_id,status='error',error=str(exc))
            raise
        finish_step(step_id)
    project.finish(boundary)
    finish_run(run_id,status=boundary)
    resume_path.unlink(missing_ok=True)
    queue.update(job_id,status=boundary,current='',error='',command='')


if __name__ == '__main__':
    queue = queue_module.Queue(sys.argv[2])
    if sys.argv[1] == 'service':
        queue_module.serve(queue)
    else:
        with queue_module.exclusive_file(queue.root/'execution.lock') as acquired:
            if not acquired:
                queue.update(sys.argv[3],status='interrupted',error='另一个工作进程尚未释放资源，请稍后重试。')
            else:
                try:
                    run_job(queue,sys.argv[3])
                except BaseException as exc:
                    traceback.print_exc()
                    queue.update(sys.argv[3],status='error',error=str(exc))
                    sys.exit(1)
