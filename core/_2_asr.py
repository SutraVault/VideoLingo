from core.utils import *
from core.asr_backend.demucs_vl import demucs_audio
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, normalize_audio_volume
from core._1_ytdlp import find_video_files
from core.utils.models import *
from core.asr_backend.cache import fingerprint, cache_matches, archive_stale, save_manifest
from pathlib import Path
import json

def transcribe():
    # Imported creator/SRT subtitles are intentional inputs, not ASR caches.
    if Path(_SOURCE_SUBTITLE_SEGMENTS).exists() and Path(_2_CLEANED_CHUNKS).exists():
        rprint('[green]Using imported subtitle transcription; WhisperX skipped.[/green]')
        return
    # 1. video to audio
    video_file = find_video_files()
    settings = dict(load_key('whisper'))
    settings['demucs'] = load_key('demucs')
    settings['model_dir'] = load_key('model_dir')
    expected = fingerprint(video_file, settings)
    if cache_matches('output', expected):
        rprint('[green]ASR source, settings and code unchanged; using validated cache.[/green]')
        return
    manifest = Path('output/log/asr_manifest.json')
    try:
        old = json.loads(manifest.read_text())
        source_changed = old.get('source_sha256') != expected['source_sha256']
    except (OSError, ValueError):
        source_changed = True
    backup = archive_stale('output', source_changed)
    rprint(f'[yellow]ASR cache changed or unversioned; archived old intermediates to {backup}[/yellow]')
    convert_video_to_audio(video_file)

    # 2. Demucs vocal separation:
    if load_key("demucs"):
        demucs_audio()
        vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
    else:
        vocal_audio = _RAW_AUDIO_FILE

    # 3. Extract audio
    segments = split_audio(_RAW_AUDIO_FILE)
    
    # 4. Transcribe audio by clips
    all_results = []
    runtime = load_key("whisper.runtime")
    if runtime == "local":
        from core.asr_backend.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with local model...[/cyan]")
    elif runtime == "cloud":
        from core.asr_backend.whisperX_302 import transcribe_audio_302 as ts
        rprint("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")
    elif runtime == "elevenlabs":
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts
        rprint("[cyan]🎤 Transcribing audio with ElevenLabs API...[/cyan]")

    for start, end in segments:
        result = ts(_RAW_AUDIO_FILE, vocal_audio, start, end)
        all_results.append(result)
    
    # 5. Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # 6. Process df
    df = process_transcription(combined_result)
    save_results(df)
    # Language auto-detection may update the config during transcription.
    settings = dict(load_key('whisper'))
    settings['demucs'] = load_key('demucs')
    settings['model_dir'] = load_key('model_dir')
    save_manifest('output', fingerprint(video_file, settings))
        
if __name__ == "__main__":
    transcribe()
