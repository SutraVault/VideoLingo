"""Windows directory-handle reproduction and archive diagnosis fallback."""
import ctypes as c
from ctypes import wintypes as w
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archive_diag_test',ROOT/'core/utils/archive_diagnostics.py')
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)


class ArchiveDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'work/archive-diagnostics-tests';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()

    @unittest.skipUnless(os.name=='nt','Windows directory handle semantics')
    def test_child_directory_handle_blocks_parent_move_until_closed(self):
        source=self.root/'project';destination=self.root/'moved'
        (source/'output').mkdir(parents=True)
        kernel=c.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,c.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
        kernel.CreateFileW.restype=w.HANDLE
        kernel.CloseHandle.argtypes=[w.HANDLE]
        handle=kernel.CreateFileW(str(source/'output'),0x100081,7,None,3,0x02000000,None)
        self.assertNotEqual(handle,c.c_void_p(-1).value)
        try:
            with self.assertRaises(PermissionError) as caught:source.rename(destination)
            self.assertEqual(caught.exception.winerror,5)
            self.assertTrue(diag.directory_probe(source/'output')['delete_access']['ok'])
        finally:kernel.CloseHandle(handle)
        self.assertTrue(source.resolve().is_relative_to(self.root))
        self.assertTrue(destination.resolve().is_relative_to(self.root))
        source.rename(destination)
        self.assertTrue((destination/'output').is_dir())

    def test_identified_handles_skip_slow_full_tree_probes(self):
        source=self.root/'project';source.mkdir()
        matches={'matches':[{'pid':123,'process':'explorer.exe','path':str(source/'output')}]}
        with patch.object(diag,'open_path_handles',return_value=matches),patch.object(diag,'file_lock_owners') as rm,patch.object(diag,'blocked_descendants') as walk:
            diag.record_archive(self.root,'rename_denied',source,self.root/'target',winerror=5)
        entry=json.loads((self.root/'logs/archive.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(entry['open_handles'],matches)
        rm.assert_not_called();walk.assert_not_called()


if __name__=='__main__':unittest.main()
