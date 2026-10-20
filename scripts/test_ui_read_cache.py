"""UI caching must avoid repeated parsing without hiding configuration edits."""
import builtins,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from core.utils import config_utils as cfg
from translations import translations as messages

ROOT=Path(__file__).resolve().parents[1]

class ReadCacheTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'work/ui-cache-tests';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'config.yaml'
        self.path.write_text('display_language: en\nnumber: 1\nnested:\n  values: [1, 2]\n',encoding='utf-8')
        mock=patch.object(cfg,'CONFIG_PATH',str(self.path));mock.start();self.addCleanup(mock.stop)
        cfg.invalidate_config_cache();self.addCleanup(cfg.invalidate_config_cache)

    def test_repeated_reads_parse_yaml_once(self):
        with patch.object(cfg.yaml,'load',wraps=cfg.yaml.load) as parser:
            for _ in range(50):self.assertEqual(cfg.load_key('number'),1)
            self.assertEqual(parser.call_count,1)

    def test_update_is_immediately_visible(self):
        self.assertEqual(cfg.load_key('number'),1)
        cfg.update_key('number',2)
        self.assertEqual(cfg.load_key('number'),2)

    def test_external_same_size_edit_is_detected(self):
        self.assertEqual(cfg.load_key('number'),1)
        previous=self.path.stat().st_mtime_ns
        self.path.write_text(self.path.read_text().replace('number: 1','number: 9'),encoding='utf-8')
        os.utime(self.path,ns=(previous+1000000000,previous+1000000000))
        self.assertEqual(cfg.load_key('number'),9)

    def test_mutable_value_does_not_modify_shared_cache(self):
        value=cfg.load_key('nested');value['values'].append(3)
        self.assertEqual(cfg.load_key('nested.values'),[1,2])

    def test_missing_key_still_raises(self):
        with self.assertRaises(KeyError):cfg.load_key('missing')

    def test_translation_files_cached_and_external_edits_detected(self):
        root=Path(self.temp.name);path=root/'en.json';path.write_text(json.dumps({'hello':'Hello'}))
        with patch.object(messages,'TRANSLATIONS_DIR',root):
            with patch.object(messages,'open',wraps=builtins.open,create=True) as opened:
                for _ in range(25):self.assertEqual(messages.load_translations('en')['hello'],'Hello')
                self.assertEqual(opened.call_count,1)
            path.write_text(json.dumps({'hello':'Changed'}))
            self.assertEqual(messages.load_translations('en')['hello'],'Changed')
            value=messages.load_translations('en');value['hello']='mutated'
            self.assertEqual(messages.load_translations('en')['hello'],'Changed')

if __name__=='__main__':unittest.main()
