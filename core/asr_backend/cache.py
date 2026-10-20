"""Versioned ASR cache; stale outputs are archived, never deleted."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

VERSION = 2


def fingerprint(source, settings):
    source=Path(source)
    digest=hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
    code=hashlib.sha256()
    base=Path(__file__).parent
    for p in [base/'quality.py',base/'whisperX_local.py',base/'audio_preprocess.py',base.parent/'_2_asr.py',Path(__file__)]:
        code.update(p.read_bytes())
    safe={k:v for k,v in settings.items() if 'key' not in k}
    return {'version':VERSION,'source_sha256':digest.hexdigest(),'settings':safe,'code_sha256':code.hexdigest()}


def cache_matches(output, expected):
    output=Path(output)
    try:
        return (output/'log/cleaned_chunks.xlsx').exists() and json.loads((output/'log/asr_manifest.json').read_text())==expected
    except (OSError,ValueError): return False


def archive_stale(output, source_changed=False):
    output=Path(output).resolve()
    backup=output/'backups'/('asr-stale-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    paths=[output/'gpt_log']
    # Leave running server logs/locks in place; invalidate pipeline data only.
    log=output/'log'
    if log.exists():
        paths += [p for p in log.iterdir() if p.suffix in ('.xlsx','.txt','.json')]
        paths += [log/'asr_quality']
    paths += list(output.glob('*.srt'))
    paths += [output/n for n in ['dub.mp3','normalized_dub.wav','output_sub.mp4','output_dub.mp4']]
    audio=output/'audio'
    if audio.exists():
        paths += [p for p in audio.iterdir() if source_changed or p.name not in ('raw.mp3','vocal.mp3','background.mp3')]
    for p in paths:
        if p.exists():
            dest=backup/p.relative_to(output)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(p),str(dest))
    return backup


def save_manifest(output, expected):
    path=Path(output)/'log/asr_manifest.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(expected,sort_keys=True,indent=2),encoding='utf-8')
    temp.replace(path)
