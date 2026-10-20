from ruamel.yaml import YAML
import threading
from pathlib import Path
import copy

CONFIG_PATH = 'config.yaml'
lock = threading.Lock()

yaml = YAML()
yaml.preserve_quotes = True
_cached_config = None
_cached_stamp = None


def _read_config_locked():
    """Reuse parsed YAML until the underlying file changes. Caller holds lock."""
    global _cached_config, _cached_stamp
    path=Path(CONFIG_PATH).resolve()
    stat=path.stat()
    stamp=(str(path),stat.st_mtime_ns,stat.st_size,stat.st_ino)
    if _cached_stamp != stamp:
        with path.open('r',encoding='utf-8') as stream:
            data=yaml.load(stream)
        _cached_config=data
        _cached_stamp=stamp
    return _cached_config


def invalidate_config_cache():
    global _cached_stamp
    with lock:_cached_stamp=None

# -----------------------
# load & update config
# -----------------------

def load_key(key):
    with lock:
        value=_read_config_locked()
        for k in key.split('.'):
            if isinstance(value, dict) and k in value:
                value=value[k]
            else:
                raise KeyError(f"Key '{k}' not found in configuration")
        # Do not let a caller mutate the shared parsed config accidentally.
        return copy.deepcopy(value) if isinstance(value,(dict,list)) else value

def update_key(key, new_value):
    global _cached_stamp
    with lock:
        data = copy.deepcopy(_read_config_locked())

        keys = key.split('.')
        current = data
        for k in keys[:-1]:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False

        if isinstance(current, dict) and keys[-1] in current:
            current[keys[-1]] = new_value
            with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
                yaml.dump(data, file)
            _cached_stamp=None
            return True
        else:
            raise KeyError(f"Key '{keys[-1]}' not found in configuration")
        
# basic utils
def get_joiner(language):
    if language in load_key('language_split_with_space'):
        return " "
    elif language in load_key('language_split_without_space'):
        return ""
    else:
        raise ValueError(f"Unsupported language code: {language}")

if __name__ == "__main__":
    print(load_key('language_split_with_space'))
