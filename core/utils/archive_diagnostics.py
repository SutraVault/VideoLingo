"""Best-effort archive diagnostics; never changes permissions or closes handles."""
import json
import os
import time
from pathlib import Path


def directory_probe(path):
    result={'path':str(path),'exists':path.exists()}
    if os.name!='nt' or not path.exists():return result
    import ctypes
    from ctypes import wintypes as w
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,ctypes.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
    kernel.CreateFileW.restype=w.HANDLE
    kernel.CloseHandle.argtypes=[w.HANDLE]
    kernel.GetFileAttributesW.argtypes=[w.LPCWSTR]
    kernel.GetFileAttributesW.restype=w.DWORD
    result['attributes']=kernel.GetFileAttributesW(str(path))
    for name,access in [('read_attributes',0x80),('delete_access',0x10000)]:
        # Opening with DELETE access does not delete anything. Share all access.
        handle=kernel.CreateFileW(str(path),access,7,None,3,0x02000000,None)
        if handle==ctypes.c_void_p(-1).value:
            code=ctypes.get_last_error()
            result[name]={'ok':False,'winerror':code,'message':ctypes.FormatError(code).strip()}
        else:
            kernel.CloseHandle(handle)
            result[name]={'ok':True}
    return result


def record_archive(root,event,source,destination,**details):
    """Diagnostics must never turn a successful archive into a failed one."""
    try:
        payload={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'event':event,
                 'pid':os.getpid(),'cwd':os.getcwd(),'database':str(root/'queue.sqlite3'),
                 'source':str(source),'destination':str(destination),**details}
        if event=='rename_denied':
            payload['probes']=[directory_probe(path) for path in (source,source.parent,destination.parent)]
            payload['open_handles']=open_path_handles(source)
            if payload['open_handles'].get('matches'):
                # Restart Manager does not report these directory handles.
                # Avoid a slow full-tree scan after an owner is already found.
                payload['file_lock_owners']={'skipped':'Open directory/file handles identified'}
                payload['blocked_descendants']={'skipped':'Open directory/file handles identified'}
            else:
                payload['file_lock_owners']=file_lock_owners(source)
                payload['blocked_descendants']=blocked_descendants(source)
        path=root/'logs/archive.jsonl'
        path.parent.mkdir(exist_ok=True)
        with path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(payload,ensure_ascii=False)+'\n')
    except Exception:
        pass


def open_path_handles(source):
    if os.name!='nt':return {'supported':False}
    try:
        import importlib.util
        script=Path(__file__).resolve().parents[2]/'scripts/inspect_archive_handles.py'
        spec=importlib.util.spec_from_file_location('vl_archive_handles',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.inspect([source])
    except Exception as exc:
        return {'error':str(exc)}


def blocked_descendants(source):
    """Probe nested directory/file handles that can prevent a parent rename."""
    if os.name!='nt':return {'supported':False}
    from itertools import islice
    blocked=[];checked=0
    for path in islice(source.rglob('*'),4096):
        checked+=1
        probe=directory_probe(path)
        if not probe.get('delete_access',{}).get('ok',True):blocked.append(probe)
    return {'checked':checked,'limit':4096,'blocked':blocked}


def file_lock_owners(source):
    """Restart Manager reports file users; directory handles may be absent."""
    if os.name!='nt':return {'supported':False}
    import ctypes
    from ctypes import wintypes as w
    class Process(ctypes.Structure):
        _fields_=[('pid',w.DWORD),('started',w.FILETIME)]
    class Info(ctypes.Structure):
        _fields_=[('process',Process),('app',w.WCHAR*256),('service',w.WCHAR*64),
                  ('type',ctypes.c_int),('status',w.ULONG),('session',w.DWORD),('restartable',w.BOOL)]
    api=ctypes.WinDLL('Rstrtmgr')
    api.RmStartSession.argtypes=[ctypes.POINTER(w.DWORD),w.DWORD,w.LPWSTR]
    api.RmRegisterResources.argtypes=[w.DWORD,w.UINT,ctypes.POINTER(w.LPCWSTR),w.UINT,ctypes.c_void_p,w.UINT,ctypes.c_void_p]
    api.RmGetList.argtypes=[w.DWORD,ctypes.POINTER(w.UINT),ctypes.POINTER(w.UINT),ctypes.POINTER(Info),ctypes.POINTER(w.DWORD)]
    api.RmEndSession.argtypes=[w.DWORD]
    session=w.DWORD();key=ctypes.create_unicode_buffer(33)
    code=api.RmStartSession(ctypes.byref(session),0,key)
    if code:return {'error_code':code,'step':'start'}
    try:
        from itertools import islice
        paths=[str(path) for path in islice((p for p in source.rglob('*') if p.is_file()),4096)]
        if not paths:return {'files_checked':0,'owners':[]}
        code=api.RmRegisterResources(session,len(paths),(w.LPCWSTR*len(paths))(*paths),0,None,0,None)
        if code:return {'error_code':code,'step':'register'}
        needed=w.UINT();count=w.UINT();reason=w.DWORD()
        code=api.RmGetList(session,ctypes.byref(needed),ctypes.byref(count),None,ctypes.byref(reason))
        owners=[]
        if code==234 and needed.value:
            count.value=needed.value;items=(Info*count.value)()
            code=api.RmGetList(session,ctypes.byref(needed),ctypes.byref(count),items,ctypes.byref(reason))
            if code==0:
                owners=[{'pid':item.process.pid,'app':item.app,'service':item.service} for item in items[:count.value]]
        return {'files_checked':len(paths),'error_code':code,'owners':owners,
                'limitation':'No owners does not exclude directory handles or permission restrictions.'}
    finally:
        api.RmEndSession(session)
