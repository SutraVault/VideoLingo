"""VL-owned pronunciation rules and text-aware IndexTTS cache metadata."""
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import uuid
from ruamel.yaml import YAML

ROOT=Path(__file__).resolve().parents[2]


def glossary_path():
    return Path(os.environ.get('VL_PRONUNCIATION_GLOSSARY',ROOT/'glossary.yaml'))


def validate_rules(value):
    if value is None:return {}
    if not isinstance(value,dict):raise ValueError('发音词典必须是词语到读法的映射。')
    rules={}
    for word,entry in value.items():
        spoken=entry.get('zh') if isinstance(entry,dict) else entry
        if not isinstance(word,str) or not word.strip() or not isinstance(spoken,str) or not spoken.strip():
            raise ValueError('词语和配音读法都不能为空。')
        if '\n' in word or '\n' in spoken:raise ValueError('每条发音规则只能占一行。')
        if word.casefold() in {key.casefold() for key in rules}:raise ValueError('存在重复词语：'+word)
        rules[word]=spoken.strip()
    return rules


@lru_cache(maxsize=8)
def _load(path,mtime,size):
    from ruamel.yaml.error import YAMLError
    text=Path(path).read_text(encoding='utf-8')
    try:value=YAML(typ='safe').load(text)
    except YAMLError as exc:
        hint='词条后的冒号请使用英文半角 :（例如“取弹:”），不能使用中文全角 ：。' if re.search(r'：\s*$',text,re.MULTILINE) else ''
        raise ValueError('发音词典 YAML 格式错误：'+hint+str(exc)) from exc
    return validate_rules(value)


def load_rules(path=None):
    path=Path(path) if path else glossary_path()
    try:stat=path.stat()
    except FileNotFoundError:return {}
    return dict(_load(str(path.resolve()),stat.st_mtime_ns,stat.st_size))


def save_rules(rules,path=None):
    rules=validate_rules(rules)
    path=Path(path) if path else glossary_path()
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('w',encoding='utf-8') as stream:
            YAML().dump({word:{'zh':spoken} for word,spoken in rules.items()},stream)
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)
    _load.cache_clear()


def apply_rules(text,rules=None):
    rules=load_rules() if rules is None else rules
    if not rules:return str(text)
    # Longest match first, one pass: replacements cannot trigger other rules.
    lookup={word.casefold():value for word,value in rules.items()}
    pattern=re.compile('|'.join(re.escape(word) for word in sorted(rules,key=len,reverse=True)),re.IGNORECASE)
    return pattern.sub(lambda match:lookup[match.group().casefold()],str(text))


def metadata_path(wav):return Path(str(wav)+'.pronunciation.json')


def cache_matches(text,wav):
    wav=Path(wav)
    if not wav.exists():return False
    desired={'source_text':str(text),'spoken_text':apply_rules(text)}
    sidecar=metadata_path(wav)
    if not sidecar.exists():return desired['source_text']==desired['spoken_text']
    try:stored=json.loads(sidecar.read_text(encoding='utf-8'))
    except (OSError,ValueError):return False
    return all(stored.get(key)==value for key,value in desired.items())


def record_audio(wav,source,spoken):
    path=metadata_path(wav);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_text(json.dumps({'source_text':source,'spoken_text':spoken},ensure_ascii=False),encoding='utf-8')
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def preserve_stale_audio(wav):
    wav=Path(wav)
    if not wav.exists():return
    destination=wav.parent/'pronunciation_versions'/uuid.uuid4().hex
    destination.mkdir(parents=True)
    # Copy before removing the active cache: retain the previous complete file.
    shutil.copy2(wav,destination/wav.name)
    sidecar=metadata_path(wav)
    if sidecar.exists():shutil.copy2(sidecar,destination/sidecar.name)
    wav.unlink()
    sidecar.unlink(missing_ok=True)
