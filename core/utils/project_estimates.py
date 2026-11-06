"""Elapsed execution time and conservative duration-scaled stage estimates."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

STAGES=('asr','nlp','meaning','terms','translation','split','subtitles','original',
        'audio_tasks','references','synthesis','merge_audio','dubbing')


def read_json(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}


def elapsed(job,now):
    value=job.get('elapsed_seconds',0) or 0
    if job['status']=='running':
        started=job.get('attempt_started') or job.get('process_started')
        if started:value+=max(0,now-started)
    return value


def duration_text(seconds):
    if seconds is None:return '—'
    seconds=max(0,int(seconds));hours,seconds=divmod(seconds,3600);minutes,seconds=divmod(seconds,60)
    return f'{hours}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'


@lru_cache(maxsize=256)
def probe(path,stamp):
    try:
        result=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                               '-of','default=noprint_wrappers=1:nokey=1',path],
                              capture_output=True,text=True,timeout=5,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        return float(result.stdout) if result.returncode==0 else None
    except (OSError,ValueError,subprocess.TimeoutExpired):return None


def video_duration(directory):
    duration=read_json(directory/'source_metadata.json').get('duration')
    if isinstance(duration,(int,float)) and duration>0:return duration
    output=directory/'output'
    try:
        source=next(p for p in output.iterdir() if p.suffix.lower() in ('.mp4','.mkv','.mov','.webm','.avi','.flv','.m4v')
                    and p.name.lower() not in ('output_sub.mp4','output_dub.mp4'))
        return probe(str(source),source.stat().st_mtime_ns)
    except (OSError,StopIteration):return None


def table_times(jobs,now=None):
    now=time.time() if now is None else now
    samples={};details={}
    for job in jobs:
        directory=Path(job['directory'])
        log=read_json(directory/'output/log/pipeline_timing.json')
        try:config=hashlib.sha256((directory/'config.snapshot.yaml').read_bytes()).hexdigest()
        except OSError:config=None
        duration=video_duration(directory)
        details[job['id']]=(log,config,duration)
        if job['status']!='completed' or not duration or not config:continue
        # Only the latest successful measurement for each stage in a project.
        stages={s.get('label'):s for s in log.get('steps',[]) if s.get('status')=='completed' and s.get('duration') is not None}
        for stage,step in stages.items():
            samples.setdefault((config,stage),[]).append(step['duration']/duration)
    result={}
    for job in jobs:
        used=elapsed(job,now)
        estimate='—' if job['status']=='completed' else '暂无法估算'
        if job['status'] in ('running','queued'):
            log,config,duration=details[job['id']]
            options=json.loads(job['options'])
            stages=[s for s in STAGES if (s!='original' or options.get('original',True)) and
                    (s not in STAGES[8:] or options.get('dubbing'))]
            current=job.get('current')
            if job['status']=='running' and current in stages:stages=stages[stages.index(current):]
            if duration and config and (job['status']=='queued' or current in stages) and all(samples.get((config,s)) for s in stages):
                remaining=sum(statistics.median(samples[(config,s)])*duration for s in stages)
                active=next((s for s in reversed(log.get('steps',[])) if s.get('label')==current and s.get('status')=='running'),None)
                if active:remaining-=max(0,now-active['start'])
                estimate='约 '+duration_text(remaining) if remaining>0 else '超出预估，仍在处理'
        result[job['id']]={'处理时间':duration_text(used) if used or job['status']=='running' else '—',
                           '预计剩余':estimate}
    return result
