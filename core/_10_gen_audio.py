import os
import math
import time
import shutil
import subprocess
import threading
from contextlib import contextmanager
from typing import Tuple

import pandas as pd
from pydub import AudioSegment
from pydub.silence import detect_silence
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import *
from core.utils.models import *
from core.utils.timing import timed_step
from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.tts_main import tts_main
from core.tts_backend.indextts_tts import (
    get_indextts_duration_stats,
    regenerate_indextts_to_duration,
    reset_indextts_duration_stats,
)

console = Console()

TEMP_FILE_TEMPLATE = f"{_AUDIO_TMP_DIR}/{{}}_temp.wav"
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"
WARMUP_SIZE = 5
MAX_CHUNK_TRUNCATE_OVERFLOW = 1.2
TTS_GENERATION_LOCK = "output/log/tts_generation.lock"


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, ValueError):
        return False


@contextmanager
def tts_generation_lock():
    """Prevent duplicate Streamlit threads from generating into one cache."""
    os.makedirs(os.path.dirname(TTS_GENERATION_LOCK), exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(TTS_GENERATION_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
            break
        except FileExistsError:
            try:
                with open(TTS_GENERATION_LOCK, encoding="ascii") as lock_file:
                    owner_pid = int(lock_file.read().strip())
            except (OSError, ValueError):
                owner_pid = 0
            if owner_pid and _pid_is_running(owner_pid):
                raise RuntimeError(
                    f"Another TTS generation task is already running in process {owner_pid}. "
                    "Wait for it to finish instead of starting a duplicate task."
                )
            try:
                os.remove(TTS_GENERATION_LOCK)
            except FileNotFoundError:
                pass
    else:
        raise RuntimeError("Unable to acquire the TTS generation lock")
    try:
        yield
    finally:
        try:
            os.remove(TTS_GENERATION_LOCK)
        except FileNotFoundError:
            pass


def save_audio_tasks(tasks_df: pd.DataFrame) -> None:
    """Atomically replace the task workbook so readers never see a partial ZIP."""
    temp_path = f"{_8_1_AUDIO_TASK}.{os.getpid()}.{threading.get_ident()}.tmp.xlsx"
    try:
        tasks_df.to_excel(temp_path, index=False)
        os.replace(temp_path, _8_1_AUDIO_TASK)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _silence_cleanup_settings() -> dict:
    """Return optional post-generation cleanup settings for cloned speech."""
    if load_key("tts_method") != "indextts":
        return {"enabled": False}
    settings = load_key("indextts")
    return settings.get("silence_cleanup", {"enabled": False})


def clean_generated_silence(audio_file: str) -> tuple[float, float, float]:
    """Trim edge silence and shorten only clearly abnormal internal pauses."""
    audio = AudioSegment.from_wav(audio_file)
    original_ms = len(audio)
    settings = _silence_cleanup_settings()
    if not settings.get("enabled", False) or original_ms == 0:
        return original_ms / 1000, original_ms / 1000, 0.0

    threshold = float(settings.get("silence_threshold_dbfs", -38))
    min_internal_ms = int(settings.get("min_internal_silence_ms", 350))
    keep_internal_ms = int(settings.get("keep_internal_silence_ms", 180))
    keep_edge_ms = int(settings.get("keep_edge_silence_ms", 120))
    ranges = detect_silence(
        audio,
        min_silence_len=min_internal_ms,
        silence_thresh=threshold,
        seek_step=5,
    )
    if not ranges:
        return original_ms / 1000, original_ms / 1000, 0.0

    total_silence_ms = sum(end - start for start, end in ranges)
    rebuilt = AudioSegment.empty()
    cursor = 0
    last_range = len(ranges) - 1
    for range_index, (start, end) in enumerate(ranges):
        rebuilt += audio[cursor:start]
        is_leading = range_index == 0 and start == 0
        is_trailing = range_index == last_range and end >= original_ms
        keep_ms = keep_edge_ms if is_leading or is_trailing else keep_internal_ms
        rebuilt += AudioSegment.silent(duration=min(keep_ms, end - start), frame_rate=audio.frame_rate)
        cursor = end
    rebuilt += audio[cursor:]
    rebuilt = rebuilt.set_channels(audio.channels).set_sample_width(audio.sample_width)
    rebuilt.export(audio_file, format="wav")
    return original_ms / 1000, len(rebuilt) / 1000, total_silence_ms / original_ms

def parse_df_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    hours, minutes, seconds = time_str.strip().split(':')
    seconds, milliseconds = seconds.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def adjust_audio_speed(input_file: str, output_file: str, speed_factor: float) -> None:
    """Adjust audio speed and handle edge cases"""
    # If the speed factor is close to 1, directly copy the file
    if abs(speed_factor - 1.0) < 0.001:
        shutil.copy2(input_file, output_file)
        return
        
    atempo = speed_factor
    cmd = ['ffmpeg', '-i', input_file, '-filter:a', f'atempo={atempo}', '-y', output_file]
    input_duration = get_audio_duration(input_file)
    max_retries = 2
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            output_duration = get_audio_duration(output_file)
            expected_duration = input_duration / speed_factor
            diff = output_duration - expected_duration
            # If the output duration exceeds the expected duration, but the input audio is less than 3 seconds, and the error is within 0.1 seconds, truncate to the expected length
            if output_duration >= expected_duration * 1.02 and input_duration < 3 and diff <= 0.1:
                audio = AudioSegment.from_wav(output_file)
                trimmed_audio = audio[:(expected_duration * 1000)]  # pydub uses milliseconds
                trimmed_audio.export(output_file, format="wav")
                print(f"✂️ Trimmed to expected duration: {expected_duration:.2f} seconds")
                return
            elif output_duration >= expected_duration * 1.02:
                raise Exception(f"Audio duration abnormal: input file={input_file}, output file={output_file}, speed factor={speed_factor}, input duration={input_duration:.2f}s, output duration={output_duration:.2f}s")
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                rprint(f"[yellow]⚠️ Audio speed adjustment failed, retrying in 1s ({attempt + 1}/{max_retries})[/yellow]")
                time.sleep(1)
            else:
                rprint(f"[red]❌ Audio speed adjustment failed, max retries reached ({max_retries})[/red]")
                raise e

def process_row(row: pd.Series, tasks_df: pd.DataFrame) -> Tuple[int, float, float, float]:
    """Helper function for processing single row data"""
    number = row['number']
    lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
    real_dur = 0
    original_dur = 0
    weighted_silence = 0
    for line_index, line in enumerate(lines):
        temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
        tts_main(line, temp_file, number, tasks_df)
        before, after, silence_ratio = clean_generated_silence(temp_file)
        original_dur += before
        real_dur += after
        weighted_silence += silence_ratio * before
    silence_ratio = weighted_silence / original_dur if original_dur else 0.0
    return number, real_dur, original_dur - real_dur, silence_ratio

def generate_tts_audio(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Generate TTS audio sequentially and calculate actual duration"""
    tasks_df['real_dur'] = 0
    tasks_df['silence_removed'] = 0.0
    tasks_df['silence_ratio'] = 0.0
    if load_key("tts_method") == "indextts":
        reset_indextts_duration_stats()
    rprint("[bold green]🎯 Starting TTS audio generation...[/bold green]")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]🔄 Generating TTS audio...", total=len(tasks_df))
        
        # warm up for first 5 rows
        warmup_size = min(WARMUP_SIZE, len(tasks_df))
        for _, row in tasks_df.head(warmup_size).iterrows():
            try:
                number, real_dur, silence_removed, silence_ratio = process_row(row, tasks_df)
                tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                tasks_df.loc[tasks_df['number'] == number, 'silence_removed'] = silence_removed
                tasks_df.loc[tasks_df['number'] == number, 'silence_ratio'] = silence_ratio
                progress.advance(task)
            except Exception as e:
                rprint(f"[red]❌ Error in warmup: {str(e)}[/red]")
                raise e
        
        # Local cloning engines keep large models in memory and should run serially.
        serial_tts_methods = {"gpt_sovits", "indextts"}
        max_workers = load_key("max_workers") if load_key("tts_method") not in serial_tts_methods else 1
        # parallel processing for remaining tasks
        if len(tasks_df) > warmup_size:
            remaining_tasks = tasks_df.iloc[warmup_size:].copy()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_row, row, tasks_df.copy())
                    for _, row in remaining_tasks.iterrows()
                ]
                
                for future in as_completed(futures):
                    try:
                        number, real_dur, silence_removed, silence_ratio = future.result()
                        tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                        tasks_df.loc[tasks_df['number'] == number, 'silence_removed'] = silence_removed
                        tasks_df.loc[tasks_df['number'] == number, 'silence_ratio'] = silence_ratio
                        progress.advance(task)
                    except Exception as e:
                        rprint(f"[red]❌ Error: {str(e)}[/red]")
                        raise e

    if load_key("tts_method") == "indextts":
        stats = get_indextts_duration_stats()
        rprint(
            "[cyan]IndexTTS duration planning: "
            f"{stats['first_passes']} first passes, "
            f"{stats['predicted_first_passes']} predicted native factors, "
            f"{stats['corrective_retries']} corrective retries, "
            f"estimator calibration {stats['calibration_ratio']:.3f}x.[/cyan]"
        )
    rprint("[bold green]✨ TTS audio generation completed![/bold green]")
    return tasks_df

def process_chunk(chunk_df: pd.DataFrame, accept: float, min_speed: float) -> tuple[float, bool]:
    """Process audio chunk and calculate speed factor"""
    chunk_durs = chunk_df['real_dur'].sum()
    tol_durs = chunk_df['tol_dur'].sum()
    durations = tol_durs - chunk_df.iloc[-1]['tolerance']
    all_gaps = chunk_df['gap'].sum() - chunk_df.iloc[-1]['gap']
    
    keep_gaps = True
    speed_var_error = 0.1

    if (chunk_durs + all_gaps) / accept < durations:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / (durations-speed_var_error))
    elif chunk_durs / accept < durations:
        speed_factor = max(min_speed, chunk_durs / (durations-speed_var_error))
        keep_gaps = False
    elif (chunk_durs + all_gaps) / accept < tol_durs:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / (tol_durs-speed_var_error))
    else:
        speed_factor = chunk_durs / (tol_durs-speed_var_error)
        keep_gaps = False
        
    return round(speed_factor, 3), keep_gaps

def fit_chunk_to_timeline(
    chunk_df: pd.DataFrame,
    speed_factor: float,
    keep_gaps: bool,
    available_duration: float,
    accept: float,
    max_speed: float,
) -> tuple[float, bool]:
    """Ensure a chunk fits its real timeline, including overlapping subtitles."""
    # Keep a small margin for ffmpeg duration rounding and codec padding.
    target_duration = available_duration - 0.1
    if target_duration <= 0:
        raise ValueError(f"Invalid chunk timeline duration: {available_duration:.3f}s")

    audio_duration = chunk_df['real_dur'].sum()
    gap_duration = chunk_df['gap'].iloc[:-1].sum()
    required_speed = (audio_duration + gap_duration) / target_duration if keep_gaps else audio_duration / target_duration

    # Prefer dropping gaps over forcing speech beyond the accepted speed.
    speed_without_gaps = audio_duration / target_duration
    if keep_gaps and required_speed > accept and speed_without_gaps <= accept:
        keep_gaps = False
        required_speed = speed_without_gaps

    if required_speed > speed_factor:
        # Round upward so three-decimal ffmpeg speed values cannot under-fit.
        speed_factor = math.ceil(required_speed * 1000) / 1000

    if speed_factor > max_speed:
        numbers = ", ".join(str(number) for number in chunk_df['number'].tolist())
        raise ValueError(
            f"TTS audio for subtitle(s) {numbers} requires {speed_factor:.3f}x speed, "
            f"above speed_factor.max={max_speed:.3f}. Shorten/re-split the translation "
            "or regenerate the affected speech instead of forcing low-quality audio."
        )

    return speed_factor, keep_gaps


def _required_chunk_speed(chunk_df: pd.DataFrame, accept: float, min_speed: float) -> float:
    """Calculate final required speed without producing audio files."""
    speed_factor, keep_gaps = process_chunk(chunk_df.reset_index(drop=True), accept, min_speed)
    chunk_start_time = parse_df_srt_time(chunk_df.iloc[0]['start_time'])
    chunk_end_time = (
        parse_df_srt_time(chunk_df.iloc[-1]['end_time'])
        + float(chunk_df.iloc[-1]['tolerance'])
    )
    target_duration = chunk_end_time - chunk_start_time - 0.1
    if target_duration <= 0:
        return float("inf")
    audio_duration = chunk_df['real_dur'].sum()
    gap_duration = chunk_df['gap'].iloc[:-1].sum()
    required_speed = (
        (audio_duration + gap_duration) / target_duration
        if keep_gaps else audio_duration / target_duration
    )
    speed_without_gaps = audio_duration / target_duration
    if keep_gaps and required_speed > accept and speed_without_gaps <= accept:
        required_speed = speed_without_gaps
    return max(speed_factor, required_speed)


def split_oversized_chunks(
    tasks_df: pd.DataFrame,
    accept: float,
    min_speed: float,
    max_speed: float,
) -> pd.DataFrame:
    """Split oversized multi-subtitle chunks at safe subtitle boundaries."""
    tasks_df = tasks_df.copy()
    chunk_start = 0
    original_ends = [
        index for index, row in tasks_df.iterrows() if int(row['cut_off']) == 1
    ]
    if not original_ends or original_ends[-1] != len(tasks_df) - 1:
        original_ends.append(len(tasks_df) - 1)

    for chunk_end in original_ends:
        chunk_df = tasks_df.iloc[chunk_start:chunk_end + 1]
        required_speed = _required_chunk_speed(chunk_df, accept, min_speed)
        if len(chunk_df) > 1 and required_speed > max_speed:
            tasks_df.loc[chunk_start:chunk_end, 'cut_off'] = 1
            numbers = ", ".join(str(number) for number in chunk_df['number'].tolist())
            rprint(
                f"[yellow]Chunk containing subtitle(s) {numbers} requires "
                f"{required_speed:.3f}x; splitting at subtitle boundaries.[/yellow]"
            )
        chunk_start = chunk_end + 1
    return tasks_df


def can_reuse_generated_tts(tasks_df: pd.DataFrame) -> bool:
    """Return True when a failed merge can safely reuse its completed TTS files."""
    required_columns = {"real_dur", "silence_removed", "silence_ratio"}
    if not required_columns.issubset(tasks_df.columns):
        return False
    if tasks_df.empty or (tasks_df["real_dur"] <= 0).any():
        return False

    task_mtime = os.path.getmtime(_8_1_AUDIO_TASK)
    for _, row in tasks_df.iterrows():
        lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
        for line_index in range(len(lines)):
            temp_file = TEMP_FILE_TEMPLATE.format(f"{row['number']}_{line_index}")
            if not os.path.exists(temp_file):
                return False
            # The workbook is saved immediately after TTS completes. A much newer
            # workbook indicates that its task definitions may have changed.
            if os.path.getmtime(temp_file) > task_mtime + 5:
                return False
    return True


def refresh_cached_tts_durations(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile workbook duration metadata with the cached WAV files."""
    tasks_df = tasks_df.copy()
    repaired = []
    for index, row in tasks_df.iterrows():
        lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
        durations = []
        for line_index in range(len(lines)):
            temp_file = TEMP_FILE_TEMPLATE.format(f"{row['number']}_{line_index}")
            if not os.path.exists(temp_file):
                durations = []
                break
            durations.append(get_audio_duration(temp_file))
        if not durations:
            continue
        actual_duration = sum(durations)
        recorded_duration = float(row.get('real_dur', 0) or 0)
        if abs(actual_duration - recorded_duration) > 0.02:
            tasks_df.at[index, 'real_dur'] = actual_duration
            repaired.append((row['number'], recorded_duration, actual_duration))
    if repaired:
        details = ", ".join(
            f"{number}: {old:.3f}s→{new:.3f}s" for number, old, new in repaired[:8]
        )
        suffix = f" and {len(repaired) - 8} more" if len(repaired) > 8 else ""
        rprint(f"[yellow]Repaired cached TTS duration metadata ({details}{suffix}).[/yellow]")
    return tasks_df


def regenerate_oversized_indextts_rows(
    tasks_df: pd.DataFrame,
    max_speed: float,
    emergency_max_speed: float | None = None,
) -> pd.DataFrame:
    """Regenerate only cached IndexTTS2.5 rows that cannot fit at max_speed."""
    if load_key("tts_method") != "indextts" or str(load_key("indextts.version")) != "2.5":
        return tasks_df
    if not bool(load_key("indextts.v2_5.auto_duration.enabled")):
        rprint(
            "[cyan]IndexTTS2.5 native duration regeneration is disabled to preserve "
            "natural word boundaries; oversized rows will use uniform final fitting.[/cyan]"
        )
        return tasks_df

    tasks_df = tasks_df.copy()
    emergency_max_speed = float(emergency_max_speed or max_speed)
    for index, row in tasks_df.iterrows():
        available = float(row['tol_dur']) - 0.1
        if available <= 0 or float(row['real_dur']) / available <= max_speed:
            continue

        lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
        number = row['number']
        current_durations = [
            get_audio_duration(TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}"))
            for line_index in range(len(lines))
        ]
        total_duration = sum(current_durations)
        allowed_total = available * max_speed * 0.98
        rprint(
            f"[yellow]Subtitle {number} is too long ({total_duration:.3f}s); "
            "regenerating only this subtitle with stronger native duration control.[/yellow]"
        )

        new_total = 0.0
        original_total = 0.0
        weighted_silence = 0.0
        for line_index, (line, current_duration) in enumerate(zip(lines, current_durations)):
            temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
            line_allowance = allowed_total * current_duration / total_duration
            regenerate_indextts_to_duration(line, temp_file, number, line_allowance)
            before, after, silence_ratio = clean_generated_silence(temp_file)
            original_total += before
            new_total += after
            weighted_silence += silence_ratio * before

        tasks_df.at[index, 'real_dur'] = new_total
        tasks_df.at[index, 'silence_removed'] = original_total - new_total
        tasks_df.at[index, 'silence_ratio'] = weighted_silence / original_total if original_total else 0.0
        if new_total / available > max_speed and len(lines) > 1:
            language = str(load_key("indextts.v2_5.language")).upper()
            separator = "，" if language == "ZH" else " "
            combined_line = separator.join(str(line).strip(" ，,.") for line in lines)
            combined_file = TEMP_FILE_TEMPLATE.format(f"{number}_0")
            rprint(
                f"[yellow]Subtitle {number} still exceeds the limit after per-line regeneration; "
                "combining its short lines into one TTS inference to remove repeated edge overhead.[/yellow]"
            )
            regenerate_indextts_to_duration(combined_line, combined_file, number, allowed_total)
            before, after, silence_ratio = clean_generated_silence(combined_file)
            new_total = after
            tasks_df.at[index, 'lines'] = [combined_line]
            src_lines = row.get('src_lines', [])
            if isinstance(src_lines, str):
                try:
                    src_lines = eval(src_lines)
                except (SyntaxError, ValueError):
                    src_lines = [src_lines]
            tasks_df.at[index, 'src_lines'] = [" ".join(str(line) for line in src_lines)]
            tasks_df.at[index, 'real_dur'] = after
            tasks_df.at[index, 'silence_removed'] = before - after
            tasks_df.at[index, 'silence_ratio'] = silence_ratio
        required_speed = new_total / available
        if max_speed < required_speed <= emergency_max_speed:
            rprint(
                f"[yellow]Subtitle {number} requires emergency final fitting at "
                f"{required_speed:.3f}x (target {max_speed:.3f}x); continuing with a quality warning.[/yellow]"
            )
        elif required_speed > emergency_max_speed:
            raise ValueError(
                f"IndexTTS2.5 subtitle {number} is still too long after targeted regeneration: "
                f"{required_speed:.3f}x > emergency limit {emergency_max_speed:.3f}x. "
                "Shorten its translation."
            )
    return tasks_df

def merge_chunks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Merge audio chunks and adjust timeline"""
    rprint("[bold blue]🔄 Starting audio chunks processing...[/bold blue]")
    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")
    max_speed = load_key("speed_factor.max")
    emergency_ratio = float(load_key("speed_factor.emergency_overflow_ratio"))
    emergency_max_speed = max_speed * emergency_ratio
    tasks_df = refresh_cached_tts_durations(tasks_df)
    tasks_df = regenerate_oversized_indextts_rows(
        tasks_df, max_speed, emergency_max_speed
    )
    save_audio_tasks(tasks_df)
    tasks_df = split_oversized_chunks(
        tasks_df, accept, min_speed, emergency_max_speed
    )
    chunk_start = 0
    
    tasks_df['new_sub_times'] = None
    tasks_df['speed_factor'] = 0.0
    tasks_df['quality_warning'] = ''
    
    for index, row in tasks_df.iterrows():
        if row['cut_off'] == 1:
            chunk_df = tasks_df.iloc[chunk_start:index+1].reset_index(drop=True)
            speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)
            
            # 🎯 Step1: Start processing new timeline
            chunk_start_time = parse_df_srt_time(chunk_df.iloc[0]['start_time'])
            chunk_end_time = parse_df_srt_time(chunk_df.iloc[-1]['end_time']) + chunk_df.iloc[-1]['tolerance'] # 加上tolerance才是这一块的结束
            speed_factor, keep_gaps = fit_chunk_to_timeline(
                chunk_df,
                speed_factor,
                keep_gaps,
                chunk_end_time - chunk_start_time,
                accept,
                emergency_max_speed,
            )
            cur_time = chunk_start_time
            for i, row in chunk_df.iterrows():
                # If i is not 0, which is not the first row of the chunk, cur_time needs to be added with the gap of the previous row, remember to divide by speed_factor
                if i != 0 and keep_gaps:
                    cur_time += chunk_df.iloc[i-1]['gap']/speed_factor
                new_sub_times = []
                number = row['number']
                lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
                for line_index, line in enumerate(lines):
                    # 🔄 Step2: Start speed change and save as OUTPUT_FILE_TEMPLATE
                    temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    adjust_audio_speed(temp_file, output_file, speed_factor)
                    ad_dur = get_audio_duration(output_file)
                    new_sub_times.append([cur_time, cur_time+ad_dur])
                    cur_time += ad_dur
                # 🔄 Step3: Find corresponding main DataFrame index and update new_sub_times
                main_df_idx = tasks_df[tasks_df['number'] == row['number']].index[0]
                tasks_df.at[main_df_idx, 'new_sub_times'] = new_sub_times
                tasks_df.at[main_df_idx, 'speed_factor'] = speed_factor
                if speed_factor > accept:
                    tasks_df.at[main_df_idx, 'quality_warning'] = (
                        f"accelerated above accepted speed ({speed_factor:.3f}x > {accept:.3f}x)"
                    )
                # 🎯 Step4: Choose emoji based on speed_factor and accept comparison
                emoji = "⚡" if speed_factor <= accept else "⚠️"
                rprint(f"[cyan]{emoji} Processed chunk {chunk_start} to {index} with speed factor {speed_factor}[/cyan]")
            # 🔄 Step5: Check if the last row exceeds the range
            if cur_time > chunk_end_time:
                time_diff = cur_time - chunk_end_time
                if time_diff <= MAX_CHUNK_TRUNCATE_OVERFLOW:
                    rprint(f"[yellow]⚠️ Chunk {chunk_start} to {index} exceeds by {time_diff:.3f}s, truncating last audio[/yellow]")
                    # Get the last audio file
                    last_number = tasks_df.iloc[index]['number']
                    last_lines = eval(tasks_df.iloc[index]['lines']) if isinstance(tasks_df.iloc[index]['lines'], str) else tasks_df.iloc[index]['lines']
                    last_line_index = len(last_lines) - 1
                    last_file = OUTPUT_FILE_TEMPLATE.format(f"{last_number}_{last_line_index}")
                    
                    # Calculate the duration to keep
                    audio = AudioSegment.from_wav(last_file)
                    original_duration = len(audio) / 1000  # Convert to seconds
                    new_duration = original_duration - time_diff
                    if new_duration <= 0.2:
                        raise Exception(
                            f"Chunk {chunk_start} to {index} exceeds by {time_diff:.3f}s, "
                            f"but truncating would leave only {new_duration:.3f}s of audio"
                        )
                    trimmed_audio = audio[:(new_duration * 1000)]  # pydub uses milliseconds
                    trimmed_audio.export(last_file, format="wav")
                    
                    # Update the last timestamp
                    last_times = tasks_df.at[index, 'new_sub_times']
                    last_times[-1][1] = chunk_end_time
                    tasks_df.at[index, 'new_sub_times'] = last_times
                else:
                    raise Exception(f"Chunk {chunk_start} to {index} exceeds the chunk end time {chunk_end_time:.2f} seconds with current time {cur_time:.2f} seconds")
            chunk_start = index+1
    
    rprint("[bold green]✅ Audio chunks processing completed![/bold green]")
    return tasks_df

def _gen_audio_unlocked() -> None:
    """Main function: Generate audio and process timeline"""
    rprint("[bold magenta]🚀 Starting audio generation process...[/bold magenta]")
    
    # 🎯 Step1: Create necessary directories
    os.makedirs(_AUDIO_TMP_DIR, exist_ok=True)
    os.makedirs(_AUDIO_SEGS_DIR, exist_ok=True)
    
    # 📝 Step2: Load task file
    tasks_df = pd.read_excel(_8_1_AUDIO_TASK)
    rprint("[green]📊 Loaded task file successfully[/green]")
    
    # 🔊 Step3: Generate TTS audio, or reuse a complete cache after a merge-only failure.
    if can_reuse_generated_tts(tasks_df):
        rprint("[bold green]♻️ Reusing completed TTS audio; retrying speed adjustment and merge.[/bold green]")
    else:
        with timed_step("TTS audio generation", category="detail"):
            tasks_df = generate_tts_audio(tasks_df)
        save_audio_tasks(tasks_df)
    
    # 🔄 Step4: Merge audio chunks
    with timed_step("Audio speed adjustment and chunk merge", category="detail"):
        tasks_df = merge_chunks(tasks_df)
    
    # 💾 Step5: Save results
    save_audio_tasks(tasks_df)
    rprint("[bold green]🎉 Audio generation completed successfully![/bold green]")


def gen_audio() -> None:
    with tts_generation_lock():
        _gen_audio_unlocked()

if __name__ == "__main__":
    gen_audio()
