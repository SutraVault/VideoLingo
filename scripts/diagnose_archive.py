"""Read-only Windows archive probes; does not move or remove project files."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archive_diagnostics',ROOT/'core/utils/archive_diagnostics.py')
diagnostics=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostics)
for argument in sys.argv[1:]:
    source=Path(argument).resolve()
    print(json.dumps({'directory':diagnostics.directory_probe(source),
                      'parent':diagnostics.directory_probe(source.parent),
                      'blocked_descendants':diagnostics.blocked_descendants(source),
                      'file_lock_owners':diagnostics.file_lock_owners(source)},ensure_ascii=False))
