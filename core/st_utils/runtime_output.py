"""Route worker-thread console output to its own log while keeping the terminal."""
import builtins
import sys
import threading
from contextlib import contextmanager


class ThreadOutput:
    def __init__(self, original, routes, lock):
        self.original,self.routes,self.lock=original,routes,lock

    def write(self, text):
        with self.lock:
            target=self.routes.get(threading.get_ident())
            if target is not None:
                target.write(text)
                target.flush()
        try:
            return self.original.write(text)
        except UnicodeEncodeError:
            encoding=getattr(self.original,'encoding',None) or 'utf-8'
            self.original.write(text.encode(encoding,errors='replace').decode(encoding))
            return len(text)

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original,name)


@contextmanager
def capture_worker_output(path):
    # Persist across Streamlit module reloads; never stack stream wrappers.
    if not hasattr(builtins,'_videolingo_output_routes'):
        builtins._videolingo_output_routes=({},threading.RLock())
    routes,lock=builtins._videolingo_output_routes
    with lock:
        for name in ('stdout','stderr'):
            stream=getattr(sys,name)
            if not isinstance(stream,ThreadOutput) and getattr(stream,'routes',None) is not routes:
                setattr(sys,name,ThreadOutput(stream,routes,lock))
    with path.open('a',encoding='utf-8') as stream:
        with lock:routes[threading.get_ident()]=stream
        try:
            yield
        finally:
            with lock:routes.pop(threading.get_ident(),None)
