"""Nonblocking title lookup for queued public YouTube videos."""
import builtins
import json
import threading
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if not hasattr(builtins,'_vl_title_lookup_runtime'):
    builtins._vl_title_lookup_runtime=({},threading.Lock())
_threads,_lock=builtins._vl_title_lookup_runtime


def _fetch_oembed(source):
    endpoint='https://www.youtube.com/oembed?'+urlencode({'url':source,'format':'json'})
    request=Request(endpoint,headers={'User-Agent':'Mozilla/5.0'})
    with urlopen(request,timeout=10) as response:
        payload=json.loads(response.read(65536).decode('utf-8'))
    title=payload.get('title')
    if not isinstance(title,str) or not title.strip():raise ValueError('YouTube 未返回视频标题')
    return title.strip()


def fetch_title(source,config_path=None):
    try:return _fetch_oembed(source)
    except Exception:
        # Some otherwise accessible videos return 401 from oEmbed. Reuse the
        # normal metadata extractor; this must never download video or audio.
        import yt_dlp
        from ruamel.yaml import YAML
        root=Path(__file__).resolve().parents[2]
        config_file=Path(config_path) if config_path else root/'config.yaml'
        config=YAML(typ='safe').load(config_file.read_text(encoding='utf-8')) if config_file.exists() else {}
        options={'quiet':True,'no_warnings':True,'noplaylist':True,'skip_download':True,
                 'socket_timeout':10,'retries':0,'extractor_retries':0}
        cookies=(config or {}).get('youtube',{}).get('cookies_path')
        if cookies:options['cookiefile']=str((root/cookies).resolve())
        with yt_dlp.YoutubeDL(options) as extractor:
            metadata=extractor.extract_info(source,download=False)
        title=(metadata or {}).get('title')
        if not isinstance(title,str) or not title.strip():raise ValueError('未能获取视频标题')
        return title.strip()


def pending_titles(queue):
    now=time.time()
    return [job for job in queue.jobs() if job['status']=='queued' and
            job['source'].startswith(('https://','http://')) and job['title'].startswith('pending_') and
            now-float(queue.meta('title_retry:'+job['id']) or 0)>60]


def fill_titles(queue,fetch=None):
    for job in pending_titles(queue):
        queue.meta('title_retry:'+job['id'],time.time())
        try:
            title=(fetch(job['source']) if fetch else
                   fetch_title(job['source'],config_path=Path(job['directory'])/'config.snapshot.yaml'))
            # A worker may claim the project during the request. Never rename or
            # alter the paths/configuration underneath a live worker.
            with queue.connect() as db:
                db.execute("UPDATE jobs SET title=? WHERE id=? AND status='queued' AND title LIKE 'pending_%'",
                           (title,job['id']))
            queue.meta('title_error:'+job['id'],'')
        except Exception as exc:
            queue.meta('title_error:'+job['id'],str(exc)[:500])
            # Metadata is fetched again by the real worker with project cookies.
            # A title lookup failure must never fail a processing job.
            continue


def prefetch_titles(queue):
    key=str(queue.root)
    with _lock:
        current=_threads.get(key)
        if current and current.is_alive():return
        if not pending_titles(queue):return
        thread=threading.Thread(target=fill_titles,args=(queue,),daemon=True,name='VL-YouTube-titles')
        _threads[key]=thread
        thread.start()
