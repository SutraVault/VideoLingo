"""Read-only handle inspection for project directories (Windows, same-user apps).

Duplicates handles only for querying names. Never closes another process's
handles, changes permissions, or terminates processes. Run outside the sandbox
when process enumeration is denied there.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import time


def inspect(roots,pid=None):
    kernel=c.WinDLL('kernel32',use_last_error=True)
    nt=c.WinDLL('ntdll')
    kernel.GetCurrentProcess.restype=w.HANDLE
    kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
    kernel.CloseHandle.argtypes=[w.HANDLE]
    kernel.DuplicateHandle.argtypes=[w.HANDLE,w.HANDLE,w.HANDLE,c.POINTER(w.HANDLE),w.DWORD,w.BOOL,w.DWORD]
    kernel.GetFileType.argtypes=[w.HANDLE];kernel.GetFileType.restype=w.DWORD
    kernel.GetFinalPathNameByHandleW.argtypes=[w.HANDLE,w.LPWSTR,w.DWORD,w.DWORD]
    kernel.GetFinalPathNameByHandleW.restype=w.DWORD
    kernel.QueryFullProcessImageNameW.argtypes=[w.HANDLE,w.DWORD,w.LPWSTR,c.POINTER(w.DWORD)]
    kernel.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,c.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
    kernel.CreateFileW.restype=w.HANDLE
    nt.NtQuerySystemInformation.argtypes=[w.ULONG,c.c_void_p,w.ULONG,c.POINTER(w.ULONG)]
    nt.NtQuerySystemInformation.restype=c.c_long
    class Entry(c.Structure):
        _fields_=[('object',c.c_void_p),('pid',c.c_size_t),('handle',c.c_size_t),
                  ('access',w.ULONG),('trace',w.USHORT),('type',w.USHORT),('attributes',w.ULONG),('reserved',w.ULONG)]
    probe=kernel.CreateFileW(str(roots[0]),0,7,None,3,0x02000000,None)
    if probe==c.c_void_p(-1).value:raise c.WinError(c.get_last_error())
    try:
        size=1024*1024
        while True:
            buf=c.create_string_buffer(size);needed=w.ULONG()
            status=nt.NtQuerySystemInformation(64,buf,size,c.byref(needed))
            if status==0:break
            if status & 0xffffffff!=0xc0000004:raise OSError(hex(status & 0xffffffff))
            size=max(size*2,needed.value+65536)
            if size>256*1024*1024:raise RuntimeError('Handle snapshot exceeds diagnostic limit')
        count=c.c_size_t.from_buffer(buf).value
        entries=(Entry*count).from_buffer(buf,2*c.sizeof(c.c_size_t))
        file_type=next(e.type for e in entries if e.pid==os.getpid() and e.handle==probe)
        processes={};matches=[];checked=0;denied=0;started=time.monotonic()
        prefixes=[str(p).casefold().rstrip('\\') for p in roots]
        names={'python.exe','pythonw.exe','ffmpeg.exe','ffprobe.exe','explorer.exe','firefox.exe',
               'msedge.exe','chrome.exe','code.exe','everything.exe','dllhost.exe','searchindexer.exe'}
        current=kernel.GetCurrentProcess()
        try:
            for entry in entries:
                if entry.type!=file_type or entry.pid==os.getpid() or (pid and entry.pid!=pid):continue
                if entry.pid not in processes:
                    handle=kernel.OpenProcess(0x1040,False,entry.pid)
                    image=c.create_unicode_buffer(32768);length=w.DWORD(len(image))
                    if not handle:
                        processes[entry.pid]=(None,'');denied+=1;continue
                    kernel.QueryFullProcessImageNameW(handle,0,image,c.byref(length))
                    name=Path(image.value).name
                    if not pid and name.casefold() not in names:
                        kernel.CloseHandle(handle);processes[entry.pid]=(None,name);continue
                    processes[entry.pid]=(handle,name)
                process,name=processes[entry.pid]
                if not process:continue
                duplicate=w.HANDLE()
                if not kernel.DuplicateHandle(process,entry.handle,current,c.byref(duplicate),0,False,2):continue
                try:
                    if kernel.GetFileType(duplicate)!=1:continue
                    checked+=1;path=c.create_unicode_buffer(32768)
                    n=kernel.GetFinalPathNameByHandleW(duplicate,path,len(path),0)
                    if not 0<n<len(path):continue
                    value=path.value.removeprefix('\\\\?\\')
                    if any(value.casefold()==root or value.casefold().startswith(root+'\\') for root in prefixes):
                        matches.append({'pid':entry.pid,'process':name,'path':value,
                                        'handle':hex(entry.handle),'access':hex(entry.access)})
                finally:kernel.CloseHandle(duplicate)
        finally:
            for handle,_ in processes.values():
                if handle:kernel.CloseHandle(handle)
        return {'matches':matches,'file_handles_checked':checked,'processes_denied':denied,
                'seconds':round(time.monotonic()-started,3)}
    finally:kernel.CloseHandle(probe)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+');parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    print(json.dumps(inspect([Path(p).resolve() for p in args.directories],args.pid),ensure_ascii=False))
