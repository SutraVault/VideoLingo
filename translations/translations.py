import json
from functools import lru_cache
from pathlib import Path

TRANSLATIONS_DIR=Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def _read_translations(path,mtime_ns,size):
    with open(path,'r',encoding='utf-8') as stream:
        return json.load(stream)

DISPLAY_LANGUAGES = {
    "🇬🇧 English": "en",
    "🇨🇳 简体中文": "zh-CN",
    "🇭🇰 繁体中文": "zh-HK",
    "🇯🇵 日本語": "ja",
    "🇪🇸 Español": "es",
    "🇷🇺 Русский": "ru",
    "🇫🇷 Français": "fr",
}

# Load the language file based on user selection
def load_translations(language="en"):
    path=TRANSLATIONS_DIR/f'{language}.json'
    stat=path.stat()
    return dict(_read_translations(str(path),stat.st_mtime_ns,stat.st_size))

# Function to fetch the translation
def translate(key):
    from core.utils.config_utils import load_key
    try:
        display_language = load_key("display_language")
        translations = load_translations(display_language)
        translation = translations.get(key)
        if translation is None:
            print(f"Warning: Translation not found for key '{key}' in language '{display_language}'")
            return key
        return translation
    except:
        return key
