import os
import warnings
import time
import subprocess
import torch
import functools

warnings.filterwarnings("ignore")

# =============================================================================
# Compatibility shim — applied BEFORE importing whisperx
# =============================================================================

# torch.load: default weights_only=False for pyannote checkpoints
# PyTorch >=2.6 changed torch.load default to weights_only=True.
# pyannote checkpoints contain omegaconf objects that fail the safety check.
# Monkey-patch torch.load to default to weights_only=False (matching <2.6
# behavior).  This is safe here because all model files come from trusted
# sources (HuggingFace / pyannote).
_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# =============================================================================
# Now safe to import whisperx and the rest of the application
# =============================================================================
import whisperx
from core.asr_backend.quality import merge_recovery, suspicious_windows, replace_window, recover_with_context
from whisperx.audio import load_audio as _whisperx_load_audio, SAMPLE_RATE as _WHISPERX_SR
from rich import print as rprint
from core.utils import *
MODEL_DIR = load_key("model_dir")


def _optional_key(key, default):
    try:
        return load_key(key)
    except KeyError:
        return default


def _merge_recovery_segments(primary_segments, recovery_segments, max_overlap_ratio):
    """Compatibility entry point; merge by local text instead of overlap ratio."""
    return merge_recovery(primary_segments, recovery_segments)

@except_handler("failed to check hf mirror", default_return=None)
def check_hf_mirror():
    mirrors = {'Official': 'huggingface.co', 'Mirror': 'hf-mirror.com'}
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float('inf')
    rprint("[cyan]🔍 Checking HuggingFace mirrors...[/cyan]")
    for name, domain in mirrors.items():
        if os.name == 'nt':
            cmd = ['ping', '-n', '1', '-w', '3000', domain]
        else:
            cmd = ['ping', '-c', '1', '-W', '3', domain]
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        response_time = time.time() - start
        if result.returncode == 0:
            if response_time < best_time:
                best_time = response_time
                fastest_url = f"https://{domain}"
            rprint(f"[green]✓ {name}:[/green] {response_time:.2f}s")
    if best_time == float('inf'):
        rprint("[yellow]⚠️ All mirrors failed, using default[/yellow]")
    rprint(f"[cyan]🚀 Selected mirror:[/cyan] {fastest_url} ({best_time:.2f}s)")
    return fastest_url

@except_handler("WhisperX processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    os.environ['HF_ENDPOINT'] = check_hf_mirror()
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rprint(f"🚀 Starting WhisperX using device: {device} ...")
    
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = 16 if gpu_mem > 8 else 2
        compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
        rprint(f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB, [cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
    else:
        batch_size = 1
        compute_type = "int8"
        rprint(f"[cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
    rprint(f"[green]▶️ Starting WhisperX for segment {start:.2f}s to {end:.2f}s...[/green]")
    
    if WHISPER_LANGUAGE == 'zh':
        model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
        local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct-fasterwhisper")
    else:
        model_name = load_key("whisper.model")
        local_model = os.path.join(MODEL_DIR, model_name)
        
    if os.path.exists(local_model):
        rprint(f"[green]📥 Loading local WHISPER model:[/green] {local_model} ...")
        model_name = local_model
    else:
        rprint(f"[green]📥 Using WHISPER model from HuggingFace:[/green] {model_name} ...")

    vad_options = {
        "vad_onset": float(_optional_key("whisper.vad_onset", 0.35)),
        "vad_offset": float(_optional_key("whisper.vad_offset", 0.25)),
    }
    asr_options = {
        "temperatures": list(_optional_key(
            "whisper.temperatures", [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )),
        "initial_prompt": "",
    }
    whisper_language = None if 'auto' in WHISPER_LANGUAGE else WHISPER_LANGUAGE
    rprint("[bold yellow] You can ignore warning of `Model was trained with torch 1.10.0+cu102, yours is 2.0.0+cu118...`[/bold yellow]")
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=whisper_language, vad_options=vad_options, asr_options=asr_options, download_root=MODEL_DIR)

    def load_audio_segment(audio_file, start, end):
        # Use whisperx's ffmpeg-based loader instead of librosa.load() which
        # deadlocks inside Streamlit's ScriptRunner thread.
        full_audio = _whisperx_load_audio(audio_file, sr=_WHISPERX_SR)
        start_sample = int(start * _WHISPERX_SR)
        end_sample = int(end * _WHISPERX_SR)
        return full_audio[start_sample:end_sample]

    raw_audio_segment = load_audio_segment(raw_audio_file, start, end)
    vocal_audio_segment = load_audio_segment(vocal_audio_file, start, end)
    
    # -------------------------
    # 1. transcribe raw audio
    # -------------------------
    transcribe_start_time = time.time()
    rprint("[bold green]Note: You will see Progress if working correctly ↓[/bold green]")
    result = model.transcribe(raw_audio_segment, batch_size=batch_size, print_progress=True)
    transcribe_time = time.time() - transcribe_start_time
    rprint(f"[cyan]⏱️ time transcribe:[/cyan] {transcribe_time:.2f}s")

    # Free GPU resources
    del model
    torch.cuda.empty_cache()

    # A second pass with a more sensitive VAD can recover quiet speech that
    # the primary pass omitted. Only uncovered segments are merged, so the
    # second pass does not replace or duplicate successful primary output.
    if _optional_key("whisper.recovery_enabled", True):
        recovery_vad_options = {
            "vad_onset": float(_optional_key("whisper.recovery_vad_onset", 0.20)),
            "vad_offset": float(_optional_key("whisper.recovery_vad_offset", 0.15)),
        }
        rprint(
            "[cyan]🔎 Running sensitive WhisperX recovery pass "
            f"(VAD {recovery_vad_options['vad_onset']}/{recovery_vad_options['vad_offset']})...[/cyan]"
        )
        recovery_model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            language=whisper_language,
            vad_options=recovery_vad_options,
            asr_options=asr_options,
            download_root=MODEL_DIR,
        )
        recovery_result = recovery_model.transcribe(
            vocal_audio_segment,
            batch_size=batch_size,
            print_progress=True,
        )
        del recovery_model
        torch.cuda.empty_cache()
        max_overlap_ratio = float(
            _optional_key("whisper.recovery_max_overlap_ratio", 0.25)
        )
        result["segments"], recovered = _merge_recovery_segments(
            result["segments"], recovery_result["segments"], max_overlap_ratio
        )
        rprint(f"[green]🔎 Recovery pass added {len(recovered)} missing segment(s).[/green]")

    # Save language
    update_key("whisper.language", result['language'])
    if result['language'] == 'zh' and WHISPER_LANGUAGE != 'zh':
        raise ValueError("Please specify the transcription language as zh and try again!")

    # -------------------------
    # 2. align by vocal audio
    # -------------------------
    align_start_time = time.time()
    detected_language = result['language']
    # Align timestamps using vocal audio
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    conflict_windows = suspicious_windows(result['segments'], len(raw_audio_segment) / _WHISPERX_SR)
    result = whisperx.align(result["segments"], model_a, metadata, vocal_audio_segment, device, return_char_alignments=False)
    align_time = time.time() - align_start_time
    rprint(f"[cyan]⏱️ time align:[/cyan] {align_time:.2f}s")

    # Keep the small acoustic model until local recovery has also been aligned.

    if _optional_key('whisper.alignment_repair_enabled', True):
        threshold = float(_optional_key('whisper.max_word_seconds', 2.5))
        windows = suspicious_windows(result['segments'], len(raw_audio_segment) / _WHISPERX_SR, threshold)
        windows += conflict_windows
        # Merge windows, including conflicts discovered before alignment.
        windows = suspicious_windows([{'start':a,'end':b,'recovery_conflict':True} for a,b in windows], len(raw_audio_segment)/_WHISPERX_SR, threshold, padding=0)
        if windows:
            from faster_whisper import WhisperModel
            retry_model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=MODEL_DIR)
            def decode_retry(a,b):
                retry_segments, _ = retry_model.transcribe(raw_audio_segment[int(a*_WHISPERX_SR):int(b*_WHISPERX_SR)], language=detected_language, beam_size=5, vad_filter=False, word_timestamps=True, condition_on_previous_text=False)
                recovered=[]
                for s in retry_segments:
                    words=[{'word':w.word.strip(),'start':w.start+a,'end':w.end+a,'score':w.probability} for w in s.words or [] if w.word.strip()]
                    if words:
                        recovered.append({'start':s.start+a,'end':s.end+a,'text':s.text,'words':words})
                return recovered
            def realign_retry(segments,a,b):
                # Recovery timestamps are relative to this full audio block.
                # Use raw audio, matching the independently decoded speech.
                return whisperx.align(segments,model_a,metadata,raw_audio_segment,device,return_char_alignments=False)['segments']
            for a,b in windows:
                recovered,repair_start,repair_end,diagnostics = recover_with_context(decode_retry,a,b,len(raw_audio_segment)/_WHISPERX_SR,threshold,realign=realign_retry)
                # Persist the exact failed words/crop bounds, including successful
                # later attempts. Segment times in this log are clip-relative.
                import json
                from pathlib import Path
                diagnostic_dir = Path('output/log/asr_quality')
                diagnostic_dir.mkdir(parents=True,exist_ok=True)
                diagnostic_path = diagnostic_dir/f'recovery_{start+a:.3f}_{start+b:.3f}.json'
                diagnostic_path.write_text(json.dumps({'clip_offset':start,'resolved':recovered is not None,'attempts':diagnostics},ensure_ascii=False,indent=2),encoding='utf-8')
                if recovered is None:
                    raise ValueError(f'ASR quality check unresolved at {start+a:.2f}-{start+b:.2f}s; refusing to save faulty alignment')
                result['segments'] = replace_window(result['segments'],recovered,repair_start,repair_end)
                rprint(f'[green]Recovered suspicious alignment at {start+repair_start:.2f}-{start+repair_end:.2f}s ({len(diagnostics)} attempt(s))[/green]')
            del retry_model
            torch.cuda.empty_cache()

    del model_a
    torch.cuda.empty_cache()
    # Adjust timestamps
    for segment in result['segments']:
        segment['start'] += start
        segment['end'] += start
        for word in segment['words']:
            if 'start' in word:
                word['start'] += start
            if 'end' in word:
                word['end'] += start
    return result
