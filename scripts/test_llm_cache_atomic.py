import json
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core.utils.ask_gpt import _atomic_json


class AtomicCacheTests(unittest.TestCase):
    def test_failed_replace_preserves_previous_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cache.json';path.write_text('[1]')
            with patch.object(importlib.import_module('core.utils.ask_gpt').os,'replace',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):_atomic_json(str(path),[2])
            self.assertEqual(json.loads(path.read_text()),[1])
            self.assertEqual(len(list(Path(folder).iterdir())),1)
            _atomic_json(str(path),[2])
            self.assertEqual(json.loads(path.read_text()),[2])


if __name__=='__main__':unittest.main()
