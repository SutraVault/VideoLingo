"""Durable serial queue; this module deliberately has no pipeline/UI imports."""
from contextlib import contextmanager
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ROOT / 'Projects'
TERMINAL = {'completed', 'error', 'interrupted', 'cancelled', 'waiting_translation', 'waiting_terms'}


def source_identity(source):
    """Identify YouTube videos across watch/short/share URLs and tracking params."""
    parsed=urlparse(source.strip())
    if parsed.hostname not in ('youtube.com','www.youtube.com','m.youtube.com','youtu.be','www.youtu.be'):
        return None
    parts=parsed.path.strip('/').split('/')
    video_id=(parts[0] if parsed.hostname in ('youtu.be','www.youtu.be') else
              parse_qs(parsed.query).get('v',[''])[0] if parsed.path=='/watch' else
              parts[1] if len(parts)>1 and parts[0] in ('shorts','embed','live') else '')
    return 'youtube:'+video_id if video_id and re.fullmatch(r'[A-Za-z0-9_-]+',video_id) else None


def copy_verified(source, destination):
    """Archive around a locked directory without removing any source files."""
    entries=list(source.rglob('*'))
    if any(p.is_symlink() or not p.resolve().is_relative_to(source) for p in entries):
        raise ValueError('项目中存在链接到外部的文件，无法安全复制归档。')
    files=[p for p in entries if p.is_file()]
    needed=sum(p.stat().st_size for p in files)
    if shutil.disk_usage(destination.parent).free < needed+16*1024*1024:
        raise OSError('可用空间不足，无法复制归档被占用的项目。')
    started=time.monotonic()
    shutil.copytree(source,destination)
    copied_at=time.monotonic()
    def digest(path):
        result=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
        return result.digest()
    for path in files:
        copied=destination/path.relative_to(source)
        if path.stat().st_size!=copied.stat().st_size or digest(path)!=digest(copied):
            raise OSError('归档副本校验失败，原项目仍保留：'+str(path))
    if {p.relative_to(source) for p in source.rglob('*') if p.is_file()}!={p.relative_to(source) for p in files}:
        raise OSError('复制期间项目文件发生变化，原项目仍保留，请停止编辑后重试。')
    return {'files':len(files),'bytes':needed,'copy_seconds':round(copied_at-started,3),
            'verify_seconds':round(time.monotonic()-copied_at,3)}


@contextmanager
def exclusive_file(path):
    """OS releases the lock on process exit, including crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    try:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b'0'); stream.flush(); stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError):
        stream.close()
        yield False
        return
    try:
        yield True
    finally:
        stream.close()


class Queue:
    def __init__(self, root=PROJECTS):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'queue.sqlite3'
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, batch_id TEXT, directory TEXT, title TEXT,
                    source TEXT, options TEXT, status TEXT, current TEXT DEFAULT '',
                    error TEXT DEFAULT '', created REAL, updated REAL,
                    timeout REAL DEFAULT 7200, command TEXT DEFAULT '',
                    pid INTEGER, process_started REAL);
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            ''')
            db.execute('BEGIN IMMEDIATE')
            columns={row[1] for row in db.execute('PRAGMA table_info(jobs)')}
            for name,definition in (('elapsed_seconds','REAL NOT NULL DEFAULT 0'),('attempt_started','REAL'),('archived_at','REAL'),
                                    ('archive_pending_directory','TEXT'),('archive_retained_sources',"TEXT NOT NULL DEFAULT '[]'"),('removed_at','REAL'),('queue_order','REAL'),
                                    ('chinese_title',"TEXT NOT NULL DEFAULT ''"),('chinese_title_source',"TEXT NOT NULL DEFAULT ''")):
                if name not in columns: db.execute(f'ALTER TABLE jobs ADD COLUMN {name} {definition}')
            # Backfill jobs started by the previous service without restarting it.
            db.execute("UPDATE jobs SET attempt_started=process_started WHERE status='running' AND attempt_started IS NULL AND process_started IS NOT NULL")
            db.execute("UPDATE jobs SET elapsed_seconds=elapsed_seconds+MAX(0,updated-attempt_started), attempt_started=NULL WHERE status IN ('completed','error','cancelled','interrupted','waiting_terms','waiting_translation') AND attempt_started IS NOT NULL")
            db.execute("UPDATE jobs SET elapsed_seconds=MAX(0,updated-process_started) WHERE status IN ('completed','error','cancelled','interrupted','waiting_terms','waiting_translation') AND elapsed_seconds=0 AND attempt_started IS NULL AND process_started IS NOT NULL")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def jobs(self):
        with self.connect() as db:
            rows=[dict(row) for row in db.execute('SELECT * FROM jobs ORDER BY created, rowid')]
            pending=iter(sorted((row for row in rows if row['status']=='queued'),
                                key=lambda row:row['queue_order'] if row['queue_order'] is not None else row['created']))
            return [next(pending) if row['status']=='queued' else row for row in rows]

    def reorder(self,job_id,direction):
        """Serialize with claim so a running task can never be moved."""
        if direction not in ('up','down','first','last'):raise ValueError('无效的排序操作。')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows=db.execute("SELECT id,created FROM jobs WHERE status='queued' AND removed_at IS NULL AND archived_at IS NULL ORDER BY COALESCE(queue_order,created),created,rowid").fetchall()
            ids=[row['id'] for row in rows]
            if job_id not in ids:raise ValueError('该任务已开始运行或不在排队中，请刷新后检查。')
            index=ids.index(job_id)
            target={'up':max(0,index-1),'down':min(len(ids)-1,index+1),'first':0,'last':len(ids)-1}[direction]
            ids.insert(target,ids.pop(index))
            # Explicit ranks precede timestamps of newly submitted videos.
            for rank,key in enumerate(ids):
                db.execute('UPDATE jobs SET queue_order=? WHERE id=?',(rank,key))

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row is None: raise KeyError(job_id)
            return dict(row)

    def save_chinese_title(self,job_id,source,title):
        with self.connect() as db:
            result=db.execute('UPDATE jobs SET chinese_title=?,chinese_title_source=? WHERE id=? AND title=?',
                              (title,source,job_id,source))
            if result.rowcount!=1:raise ValueError('视频原始标题已变化，请重新翻译。')

    def remove_queued(self, job_id):
        with self.connect() as db:
            now=time.time()
            result=db.execute("UPDATE jobs SET status='cancelled',updated=?,removed_at=? WHERE id=? AND status='queued' AND archived_at IS NULL AND removed_at IS NULL",(now,now,job_id))
            if result.rowcount!=1:raise ValueError('该任务已开始运行或不在排队中，请刷新后检查。')

    def remove_record(self,job_id,restore=False):
        with self.connect() as db:
            if restore:
                result=db.execute('UPDATE jobs SET removed_at=NULL WHERE id=? AND removed_at IS NOT NULL',(job_id,))
            else:
                result=db.execute("UPDATE jobs SET removed_at=? WHERE id=? AND status IN ('cancelled','error','interrupted') AND archived_at IS NULL AND archive_pending_directory IS NULL AND removed_at IS NULL",(time.time(),job_id))
            if result.rowcount!=1:raise ValueError('只有已停止、失败或中断的记录可以从列表移除；请刷新后检查状态。')

    def archive(self, job_id, restore=False):
        # Import without core.__init__: this module is also loaded standalone.
        import importlib.util
        spec=importlib.util.spec_from_file_location('vl_archive_diagnostics',ROOT/'core/utils/archive_diagnostics.py')
        diagnostics=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostics)
        log=diagnostics.record_archive
        archive_root=(self.root.parent/'archive/history').resolve()
        if not archive_root.is_relative_to(self.root.parent):
            raise ValueError('归档目录超出了项目库所在目录的范围。')
        with exclusive_file(self.root/'submission.lock') as acquired:
            if not acquired:raise ValueError('项目库正在更新，请稍后重试。')
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                row=db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
                if not row or row['status']!='completed':raise ValueError('只有已完成的项目可以归档或恢复。')
                if bool(row['archived_at'])!=restore:raise ValueError('项目归档状态已改变，请刷新后检查。')
                source=Path(row['directory']).resolve()
                allowed=archive_root if restore else self.root
                if source==allowed or not source.is_relative_to(allowed):raise ValueError('项目目录不在允许的范围内。')
                destination_root=self.root if restore else archive_root
                destination_root.mkdir(parents=True,exist_ok=True)
                destination=(destination_root/source.name).resolve();suffix=2
                while destination.exists():
                    destination=(destination_root/(source.name+'_'+str(suffix))).resolve();suffix+=1
                if not destination.is_relative_to(destination_root) or destination.is_relative_to(source):
                    raise ValueError('无效的归档目录。')
                state_path=source/'output/workflow_state.json'
                original_state=state_path.read_bytes() if state_path.exists() else None
                db.execute('UPDATE jobs SET archive_pending_directory=? WHERE id=?',(str(destination),job_id))
            # Copying large projects must not hold the DB write lock and stall
            # the processing service heartbeat or other job state updates.
            moved=False;retained=json.loads(row['archive_retained_sources'] or '[]')
            started=time.monotonic()
            log(self.root,'started',source,destination,job_id=job_id,restore=restore)
            try:
                try:source.rename(destination);moved=True
                except PermissionError as exc:
                    log(self.root,'rename_denied',source,destination,job_id=job_id,
                        error=str(exc),errno=exc.errno,winerror=getattr(exc,'winerror',None))
                    log(self.root,'copy_started',source,destination,job_id=job_id)
                    metrics=copy_verified(source,destination)
                    log(self.root,'copy_verified',source,destination,job_id=job_id,metrics=metrics)
                    if str(source) not in retained:retained.append(str(source))
                if original_state:
                    state=json.loads(original_state.decode('utf-8'))
                    old_source=Path(state['source']).resolve()
                    if not old_source.is_relative_to(source):raise ValueError('工作流源文件不在项目目录内。')
                    new_source=destination/old_source.relative_to(source);stat=new_source.stat()
                    state['source']=str(new_source)
                    state['source_id']=hashlib.sha256(f'{new_source}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()
                    (destination/'output/workflow_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
                runtime=self.root/'logs'/(job_id+'.log')
                if not restore and runtime.exists():
                    (destination/'logs').mkdir(exist_ok=True)
                    shutil.copy2(runtime,destination/'logs/runtime.log')
                with self.connect() as db:
                    db.execute('UPDATE jobs SET directory=?,archived_at=?,archive_pending_directory=NULL,archive_retained_sources=? WHERE id=?',
                               (str(destination),None if restore else time.time(),json.dumps(retained),job_id))
                log(self.root,'completed',source,destination,job_id=job_id,method='rename' if moved else 'verified_copy',
                    seconds=round(time.monotonic()-started,3))
            except BaseException as exc:
                log(self.root,'failed',source,destination,job_id=job_id,error=str(exc),
                    seconds=round(time.monotonic()-started,3))
                if moved:
                    if original_state:(destination/'output/workflow_state.json').write_bytes(original_state)
                    destination.rename(source)
                with self.connect() as db:db.execute('UPDATE jobs SET archive_pending_directory=NULL WHERE id=?',(job_id,))
                raise
        return destination

    def update(self, job_id, **values):
        allowed = {'directory','title','status','current','error','command','options','pid','process_started'}
        if not values.keys() <= allowed: raise ValueError('Invalid queue field')
        values['updated'] = time.time()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
            if row and row['status']=='running' and values.get('status') in TERMINAL:
                started=row['attempt_started'] or row['process_started']
                if started:
                    values['elapsed_seconds']=row['elapsed_seconds']+max(0,values['updated']-started)
                values['attempt_started']=None
            db.execute('UPDATE jobs SET '+','.join(k+'=?' for k in values)+' WHERE id=?',
                       (*values.values(), job_id))

    def meta(self, key, value=None):
        with self.connect() as db:
            if value is not None:
                db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key,str(value)))
                return str(value)
            row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
            return row[0] if row else ''

    @staticmethod
    def safe_title(title):
        title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', title).strip(' .')[:70].rstrip(' .')
        if title.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:
            title = '_'+title
        return title or 'video'

    def rename(self, job_id, title):
        job = self.get(job_id)
        directory = Path(job['directory'])
        # Rename only before a workflow exists: absolute source identity stays stable.
        if (directory/'output/workflow_state.json').exists(): return directory
        name = datetime.fromtimestamp(job['created']).strftime('%Y-%m-%d')+'_'+self.safe_title(title)
        target = self.root / name
        suffix = 2
        while target.exists() and target != directory:
            target = self.root / (name+'_'+str(suffix)); suffix += 1
        if directory != target:
            directory.rename(target)
        self.update(job_id, directory=str(target), title=title)
        return target

    def enqueue(self, sources, config, options, timeout=7200, uploads=None, initial_status='queued', source_options=None, allow_existing=False, source_titles=None):
        self.last_skipped=[]
        # Serialize duplicate detection and insertion across browser sessions.
        with exclusive_file(self.root/'submission.lock') as acquired:
            if not acquired:raise ValueError('另一个页面正在提交任务，请稍后再试。')
            existing={source_identity(j['source']):j for j in self.jobs() if source_identity(j['source'])}
            accepted=[];seen={}
            for source in dict.fromkeys(s.strip() for s in sources if s.strip()):
                identity=source_identity(source)
                selected=(source_options or {}).get(source,options)
                if identity and identity in seen:
                    if seen[identity]!=selected:raise ValueError('同一视频的链接选择了不同输出版本，请合并为一行。')
                    continue
                if identity:seen[identity]=selected
                if identity in existing and not allow_existing:
                    self.last_skipped.append(existing[identity]);continue
                accepted.append(source)
            return self._enqueue(accepted,config,options,timeout,uploads,initial_status,source_options,source_titles)

    def _enqueue(self, sources, config, options, timeout, uploads, initial_status, source_options,source_titles=None):
        if timeout < 1: raise ValueError('Timeout must be positive')
        batch = uuid.uuid4().hex
        ids = []
        for source in dict.fromkeys(s.strip() for s in sources if s.strip()):
            job_options = (source_options or {}).get(source,options)
            display_title = Path(source).stem if not source.startswith('http') else (source_titles or {}).get(source)
            job_id = uuid.uuid4().hex
            directory = self.root / (datetime.now().strftime('%Y-%m-%d')+'_pending_'+job_id[:8])
            directory.mkdir()
            (directory/'output').mkdir()
            # Keep secrets out of the DB. Snapshots have the same local trust as config.yaml.
            (directory/'config.snapshot.yaml').write_text(config, encoding='utf-8')
            (directory/'config.runtime.yaml').write_text(config, encoding='utf-8')
            if (ROOT/'custom_terms.xlsx').exists():
                shutil.copy2(ROOT/'custom_terms.xlsx', directory/'custom_terms.xlsx')
            if uploads and source in uploads:
                filename = self.safe_title(Path(source).stem)+Path(source).suffix.lower()
                if filename.lower() in ('output_sub.mp4','output_dub.mp4'):
                    filename = 'source_'+filename
                (directory/'output'/filename).write_bytes(uploads[source])
                source = filename
            now = time.time()
            with self.connect() as db:
                db.execute('''INSERT INTO jobs
                    (id,batch_id,directory,title,source,options,status,created,updated,timeout)
                    VALUES (?,?,?,?,?,?,?,?,?,?)''',
                    (job_id,batch,str(directory),source,source,json.dumps(job_options), 'preparing',now,now,timeout))
            self.rename(job_id, display_title if display_title is not None else 'pending_'+job_id[:8])
            job = self.get(job_id)
            (Path(job['directory'])/'project.json').write_text(json.dumps(
                {'id':job_id,'source':source,'title':job['title'],'created':now,'batch_id':batch},
                ensure_ascii=False,indent=2),encoding='utf-8')
            self.update(job_id,status=initial_status)
            ids.append(job_id)
        return ids

    def claim(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY COALESCE(queue_order,created),created,rowid LIMIT 1").fetchone()
            if not row: return None
            now=time.time()
            db.execute("UPDATE jobs SET status='running', error='', attempt_started=?, updated=? WHERE id=?", (now,now,row['id']))
            return dict(row, status='running')

    def import_output(self, source_output, config, options):
        """Copy a legacy workspace, keeping the original and all saved revisions."""
        source_output = Path(source_output).resolve()
        candidates = [p for p in source_output.iterdir() if p.is_file() and
                      p.suffix.lower() in ('.mp4','.mkv','.mov','.webm','.avi','.flv','.m4v') and
                      p.name not in ('output_sub.mp4','output_dub.mp4')]
        if len(candidates) != 1: raise ValueError('旧 output 中需要有且只有一个源视频。')
        try:
            state = json.loads((source_output/'workflow_state.json').read_text(encoding='utf-8'))
        except (OSError,ValueError): state = {}
        job_id = self.enqueue([candidates[0].name],config,state.get('options') or options,initial_status='interrupted')[0]
        directory = Path(self.get(job_id)['directory'])
        # The new entry must stay unclaimable until its files and state are consistent.
        self.update(job_id,status='interrupted')
        try:
            shutil.copytree(source_output,directory/'output',dirs_exist_ok=True)
            source = (directory/'output'/candidates[0].name).resolve(); stat=source.stat()
            if state:
                state['source_id']=hashlib.sha256(f'{source}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()
                state['source']=str(source)
                if state.get('status') == 'running': state['status']='interrupted'
                (directory/'output/workflow_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
            status=state.get('status','interrupted')
            self.update(job_id,status=status if status in TERMINAL else 'interrupted')
        except BaseException as exc:
            self.update(job_id,status='error',error='导入失败：'+str(exc)); raise
        return job_id

    def retry(self, job_id):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='queued',command='',error='',updated=? WHERE id=? AND archived_at IS NULL AND archive_pending_directory IS NULL AND removed_at IS NULL AND status IN ('error','interrupted','cancelled','waiting_terms','waiting_translation','completed')", (time.time(),job_id))

    def submit_command(self, job_id, command, options=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job = db.execute('SELECT status,options,archived_at,archive_pending_directory,removed_at FROM jobs WHERE id=?',(job_id,)).fetchone()
            if job is None or job['status'] not in TERMINAL or job['archived_at'] or job['archive_pending_directory'] or job['removed_at']:
                raise ValueError('请等待当前任务停止后再操作。')
            db.execute("UPDATE jobs SET status='queued',command=?,options=?,error='',updated=? WHERE id=?",
                       (command,json.dumps(options) if options is not None else job['options'],time.time(),job_id))

    def ensure_service(self):
        if time.time()-float(self.meta('heartbeat') or 0) < 10: return
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with (self.root/'service.log').open('ab') as log:
            subprocess.Popen([sys.executable, str(ROOT/'core/utils/project_worker.py'),
                              'service', str(self.root)], cwd=ROOT, stdout=log, stderr=log,
                             creationflags=flags, start_new_session=os.name != 'nt')


def attach_process_tree(process):
    """Use a Windows Job Object: taskkill depends on process-enumeration rights."""
    if os.name != 'nt': return
    import ctypes
    from ctypes import wintypes
    class BasicLimits(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit',ctypes.c_int64),('PerJobUserTimeLimit',ctypes.c_int64),
                    ('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),
                    ('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',wintypes.DWORD),
                    ('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]
    class IoCounters(ctypes.Structure):
        _fields_ = [(name,ctypes.c_uint64) for name in ('ReadOperationCount','WriteOperationCount',
                    'OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
    class ExtendedLimits(ctypes.Structure):
        _fields_ = [('BasicLimitInformation',BasicLimits),('IoInfo',IoCounters),
                    ('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),
                    ('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p,wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE,wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None,None)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits(); limits.BasicLimitInformation.LimitFlags = 0x2000
    if not kernel.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(handle,int(process._handle)):
        error = ctypes.get_last_error();kernel.CloseHandle(handle)
        process.kill();process.wait()
        raise ctypes.WinError(error)
    process._vl_job = handle


def close_process_tree(process):
    if os.name == 'nt' and getattr(process,'_vl_job',None):
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        kernel.CloseHandle(process._vl_job);process._vl_job=None


def terminate_tree(process):
    if process.poll() is not None: return
    if os.name == 'nt':
        if getattr(process,'_vl_job',None):
            import ctypes
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.TerminateJobObject.argtypes=[ctypes.c_void_p,ctypes.c_uint]
            if not kernel.TerminateJobObject(process._vl_job,1):
                raise ctypes.WinError(ctypes.get_last_error())
        else: process.kill()
    else:
        import signal
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=30)


def serve(queue, worker_script=None, max_jobs=None):
    with exclusive_file(queue.root/'service.lock') as acquired:
        if not acquired: return
        queue.meta('queue_order_supported','1')
        # A surviving worker holds execution.lock. Do not restart it or overlap it.
        for job in queue.jobs():
            if job['status'] in ('running','preparing'):
                queue.update(job['id'], status='interrupted', error='后台执行器意外中断；确认后可从有效成果继续。')
        processed = 0
        while True:
            if max_jobs is not None and processed >= max_jobs: return
            queue.meta('heartbeat', time.time())
            if queue.meta('paused') == '1':
                time.sleep(1); continue
            with exclusive_file(queue.root/'execution.lock') as available:
                pass
            if not available:
                time.sleep(1); continue
            job = queue.claim()
            if not job:
                time.sleep(1); continue
            directory = Path(job['directory'])
            env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                       VL_OUTPUT_DIR=str(directory/'output'),
                       VL_CONFIG_PATH=str(directory/'config.runtime.yaml'),
                       VL_TERMS_PATH=str(directory/'custom_terms.xlsx'))
            (queue.root/'logs').mkdir(exist_ok=True)
            with (queue.root/'logs'/(job['id']+'.log')).open('ab') as log:
                flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                process = subprocess.Popen([sys.executable,str(worker_script or ROOT/'core/utils/project_worker.py'),
                                            'job',str(queue.root),job['id']], cwd=ROOT, env=env,
                                           stdout=log, stderr=log, creationflags=flags,
                                           start_new_session=os.name != 'nt')
                try:
                    attach_process_tree(process)
                except Exception as exc:
                    queue.update(job['id'],status='error',error='无法启动隔离工作进程：'+str(exc))
                    processed += 1
                    continue
                queue.update(job['id'], pid=process.pid, process_started=time.time())
                started = time.monotonic()
                stage = None
                while process.poll() is None:
                    queue.meta('heartbeat', time.time())
                    current = queue.get(job['id'])
                    if current['current'] != stage:
                        stage = current['current']; started = time.monotonic()
                    cancelled = current['command'] == 'cancel'
                    expired = time.monotonic()-started > job['timeout']
                    if cancelled or expired:
                        terminate_tree(process)
                        queue.update(job['id'],status='cancelled' if cancelled else 'error',
                                     error='用户停止当前项目' if cancelled else f'阶段 {stage or "启动"} 超时，已终止进程并继续队列。')
                        break
                    time.sleep(1)
                current = queue.get(job['id'])
                if current['status'] == 'running':
                    queue.update(job['id'],status='error',error=f'工作进程异常退出 ({process.returncode})，请展开后台输出与错误详情。')
                close_process_tree(process)
                processed += 1
