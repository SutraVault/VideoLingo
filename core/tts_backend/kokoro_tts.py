"""Local fixed-voice Kokoro TTS backend."""

from pathlib import Path
from threading import Lock
import re

import numpy as np
import soundfile as sf

from core.utils import load_key


_PIPELINES = {}
_PIPELINE_LOCK = Lock()
_INFERENCE_LOCK = Lock()

_LATIN_LETTER_NAMES = {
    "A": "诶", "B": "比", "C": "西", "D": "迪", "E": "伊", "F": "艾弗",
    "G": "吉", "H": "艾尺", "I": "艾", "J": "杰", "K": "开", "L": "艾勒",
    "M": "艾姆", "N": "艾恩", "O": "欧", "P": "批", "Q": "丘", "R": "阿尔",
    "S": "艾丝", "T": "提", "U": "优", "V": "维", "W": "达布流",
    "X": "艾克斯", "Y": "歪", "Z": "贼德",
}


def expand_latin_initialisms(text):
    """Spell uppercase model codes for Kokoro's Mandarin-only G2P path."""
    text = str(text or "")
    text = re.sub(r"(?<=\d)[xX×](?=\d)", "乘", text)
    # Treat dotted initialisms (C.T.D.M.) as one code before expanding them.
    text = re.sub(
        r"(?<![A-Za-z])(?:[A-Z]\.){2,}(?![A-Za-z])",
        lambda match: match.group(0).replace(".", ""),
        text,
    )

    def spell(match):
        return "".join(_LATIN_LETTER_NAMES[letter] for letter in match.group(0))

    # Do not alter normal title-case words such as Detroit or Series.
    return re.sub(r"(?<![A-Za-z])[A-Z]+(?![a-z])", spell, text)


def trim_kokoro_edge_silence(samples, sample_rate=24000, keep_ms=60):
    """Remove Kokoro's large model padding while preserving natural edges."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(audio):
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    # Relative threshold handles quiet voices; the absolute floor ignores
    # floating-point noise in otherwise silent model padding.
    threshold = max(10 ** (-55 / 20), peak * 10 ** (-42 / 20))
    voiced = np.flatnonzero(np.abs(audio) >= threshold)
    if not len(voiced):
        return audio
    keep = max(0, int(sample_rate * keep_ms / 1000))
    start = max(0, int(voiced[0]) - keep)
    end = min(len(audio), int(voiced[-1]) + keep + 1)
    trimmed = audio[start:end].copy()
    fade = min(int(sample_rate * 0.005), len(trimmed) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        trimmed[:fade] *= ramp
        trimmed[-fade:] *= ramp[::-1]
    return trimmed


def _settings():
    settings = load_key("kokoro_tts") or {}
    device = str(settings.get("device", "auto")).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("kokoro_tts.device must be auto, cpu, or cuda")
    return {
        "voice": str(settings.get("voice", "zm_yunyang")),
        "speed": float(settings.get("speed", 1.0)),
        "device": None if device == "auto" else device,
        "spell_latin_letters": bool(settings.get("spell_latin_letters", True)),
    }


def _get_pipeline(device):
    # Import lazily so other TTS methods do not require Kokoro to be installed.
    from kokoro import KPipeline

    key = device or "auto"
    with _PIPELINE_LOCK:
        if key not in _PIPELINES:
            _PIPELINES[key] = KPipeline(
                lang_code="z",
                repo_id="hexgrad/Kokoro-82M",
                device=device,
            )
        return _PIPELINES[key]


def kokoro_tts(text, save_path):
    """Synthesize Mandarin locally with a fixed Kokoro voice."""
    settings = _settings()
    if not 0.5 <= settings["speed"] <= 2.0:
        raise ValueError("kokoro_tts.speed must be between 0.5 and 2.0")
    if not settings["voice"].startswith(("zf_", "zm_")):
        raise ValueError("Choose a Mandarin Kokoro voice beginning with zf_ or zm_")

    synthesis_text = (
        expand_latin_initialisms(text) if settings["spell_latin_letters"] else str(text)
    )
    if synthesis_text != text:
        print(f"Kokoro letter pronunciation: {text} -> {synthesis_text}")

    pipeline = _get_pipeline(settings["device"])
    # A shared PyTorch model is not safe to invoke concurrently. Kokoro is still
    # substantially faster than cloning a new reference voice for every subtitle.
    with _INFERENCE_LOCK:
        parts = []
        for result in pipeline(synthesis_text, voice=settings["voice"], speed=settings["speed"]):
            if result.audio is not None and result.audio.numel():
                parts.append(result.audio.detach().float().cpu().numpy())

    if not parts:
        raise RuntimeError(f"Kokoro produced no audio for: {synthesis_text}")

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = trim_kokoro_edge_silence(np.concatenate(parts), 24000)
    sf.write(output, samples, 24000, subtype="PCM_16")
    print(f"Kokoro audio saved to {output}")
