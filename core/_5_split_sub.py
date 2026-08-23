import pandas as pd
from typing import List, Tuple
import concurrent.futures
import json
import os
import threading
import time

from core._3_2_split_meaning import split_sentence
from core.prompts import get_align_prompt
from core.utils.excel_utils import read_excel_with_aliases
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.models import *
from core.utils.timing import record_event
console = Console()

SUBTITLE_SPLIT_TIMING_LOG = "output/log/subtitle_split_timing.json"
_split_timing_lock = threading.Lock()


def _write_split_timing(summary: dict, attempts: list[dict]) -> None:
    os.makedirs(os.path.dirname(SUBTITLE_SPLIT_TIMING_LOG), exist_ok=True)
    payload = {
        "summary": summary,
        "attempts": attempts,
    }
    with open(SUBTITLE_SPLIT_TIMING_LOG, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

def safe_text(value) -> str:
    if pd.isna(value):
        return ''
    return str(value)

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)

def split_text_by_weights(text: str, weights: List[float]) -> List[str]:
    """Fallback splitter used when the LLM returns empty alignment parts."""
    text = str(text).strip()
    if not weights:
        return [text] if text else []

    if not text:
        return [""] * len(weights)

    total_weight = sum(max(weight, 1) for weight in weights)
    target_lengths = [max(1, round(len(text) * max(weight, 1) / total_weight)) for weight in weights]

    parts = []
    start = 0
    for i, target_length in enumerate(target_lengths):
        remaining_parts = len(weights) - i
        if remaining_parts == 1:
            parts.append(text[start:].strip())
            break

        remaining_chars = len(text) - start
        cut = start + min(target_length, remaining_chars - (remaining_parts - 1))
        cut = max(start + 1, cut)

        search_start = max(start + 1, cut - 8)
        search_end = min(len(text) - (remaining_parts - 1), cut + 8)
        candidates = [
            pos + 1
            for pos in range(search_start - 1, search_end)
            if text[pos] in " ,，.。;；:：!?！？、"
        ]
        if candidates:
            cut = min(candidates, key=lambda pos: abs(pos - cut))

        parts.append(text[start:cut].strip())
        start = cut

    return [part if part else text[:1] for part in parts]

def normalize_align_data(response_data, tr_sub: str, src_parts: List[str]) -> List[str]:
    align_data = response_data.get('align', [])
    tr_parts = []
    for i, item in enumerate(align_data):
        key = f'target_part_{i+1}'
        value = item.get(key, item.get('target_part', item.get('target', '')))
        tr_parts.append(str(value).strip())

    if len(tr_parts) == len(src_parts) and all(tr_parts):
        return tr_parts

    console.print(
        "[yellow]Warning: LLM subtitle alignment returned empty or incomplete target parts. "
        "Using local proportional fallback split.[/yellow]"
    )
    return split_text_by_weights(tr_sub, [calc_len(part) for part in src_parts])

def align_subs(
    src_sub: str,
    tr_sub: str,
    src_part: str,
    attempt_tracker: dict | None = None,
) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    src_parts = src_part.split('\n')
    expected_parts = len(src_parts)
    
    def valid_align(response_data):
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if len(response_data['align']) != expected_parts:
            return {
                "status": "error",
                "message": f"Align returned {len(response_data['align'])} parts, expected {expected_parts}",
            }
        if not all(isinstance(item, dict) for item in response_data['align']):
            return {"status": "error", "message": "`align` must be a list of JSON objects"}
        return {"status": "success", "message": "Align completed"}

    try:
        parsed = ask_gpt(
            align_prompt,
            resp_type='json',
            valid_def=valid_align,
            log_title='align_subs_v2',
            attempt_tracker=attempt_tracker,
        )
    except Exception as e:
        console.print(
            "[yellow]Warning: LLM subtitle alignment failed. "
            f"Using local proportional fallback split. Details: {e}[/yellow]"
        )
        parsed = {'align': []}

    tr_parts = normalize_align_data(parsed, tr_sub, src_parts)
    
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)
    tr_remerged = joiner.join(tr_parts)
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)
    
    return src_parts, tr_parts, tr_remerged

def split_align_subs(src_lines: List[str], tr_lines: List[str], timing_context: list | None = None):
    src_lines = [safe_text(item) for item in src_lines]
    tr_lines = [safe_text(item) for item in tr_lines]
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()
    
    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > MAX_SUB_LENGTH or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)
    
    @except_handler("Error in split_align_subs")
    def process(i):
        started_at = time.time()
        split_tracker = {}
        align_tracker = {}
        line_timing = {
            "line_index": i,
            "source_length": len(str(src_lines[i])),
            "translation_weighted_length": calc_len(str(tr_lines[i])),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "running",
        }
        try:
            split_src = split_sentence(
                src_lines[i],
                num_parts=2,
                attempt_tracker=split_tracker,
            ).strip()
            src_parts, tr_parts, tr_remerged = align_subs(
                src_lines[i],
                tr_lines[i],
                split_src,
                attempt_tracker=align_tracker,
            )
            src_lines[i] = src_parts
            tr_lines[i] = tr_parts
            remerged_tr_lines[i] = tr_remerged
            line_timing["source_parts"] = len(src_parts)
            line_timing["translation_parts"] = len(tr_parts)
            line_timing["status"] = "completed"
        except Exception as exc:
            line_timing["status"] = "error"
            line_timing["error"] = str(exc)
            raise
        finally:
            line_timing["duration"] = time.time() - started_at
            line_timing["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            line_timing["split_requests"] = split_tracker.get("requests", 0)
            line_timing["split_cache_hits"] = split_tracker.get("cache_hits", 0)
            line_timing["split_validation_retries"] = split_tracker.get("validation_errors", 0)
            line_timing["align_requests"] = align_tracker.get("requests", 0)
            line_timing["align_cache_hits"] = align_tracker.get("cache_hits", 0)
            line_timing["align_validation_retries"] = align_tracker.get("validation_errors", 0)
            line_timing["validation_retry_count"] = (
                line_timing["split_validation_retries"]
                + line_timing["align_validation_retries"]
            )
            if timing_context is not None:
                with _split_timing_lock:
                    timing_context.append(line_timing)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        list(executor.map(process, to_split))
    
    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    if len(src_lines) != len(tr_lines):
        raise ValueError(
            f"Subtitle split alignment produced mismatched lengths: "
            f"{len(src_lines)} source rows vs {len(tr_lines)} translation rows"
        )
    
    return src_lines, tr_lines, remerged_tr_lines

def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")
    step_started_at = time.time()
    
    df = read_excel_with_aliases(_4_2_TRANSLATION, required_columns=['Source', 'Translation'])
    src = [safe_text(item) for item in df['Source'].tolist()]
    trans = [safe_text(item) for item in df['Translation'].tolist()]
    
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    timing_attempts = []
    split_src, split_trans, remerged = src, trans, trans.copy()
    
    for attempt in range(3):  # 多次切割
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        attempt_started_at = time.time()
        line_timings = []
        source_copy = src.copy()
        translation_copy = trans.copy()
        to_split_count = sum(
            1
            for source_line, translation_line in zip(source_copy, translation_copy)
            if len(str(source_line)) > MAX_SUB_LENGTH
            or calc_len(str(translation_line)) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH
        )
        split_src, split_trans, remerged = split_align_subs(
            source_copy,
            translation_copy,
            timing_context=line_timings,
        )
        attempt_timing = {
            "attempt": attempt + 1,
            "input_lines": len(source_copy),
            "lines_needing_split": to_split_count,
            "duration": time.time() - attempt_started_at,
            "line_timings": sorted(line_timings, key=lambda item: item["line_index"]),
        }
        timing_attempts.append(attempt_timing)
        
        # 检查是否所有字幕都符合长度要求
        if all(len(safe_text(src)) <= MAX_SUB_LENGTH for src in split_src) and \
           all(calc_len(safe_text(tr)) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            attempt_timing["result"] = "length_limits_satisfied"
            break
        
        # 更新源数据继续下一轮分割
        attempt_timing["result"] = "needs_another_attempt"
        src, trans = split_src, split_trans

    # 确保二者有相同的长度，防止报错
    if len(src) > len(remerged):
        remerged += [None] * (len(src) - len(remerged))
    elif len(remerged) > len(src):
        src += [None] * (len(remerged) - len(src))
    
    if len(split_src) != len(split_trans):
        raise ValueError(
            f"Cannot write subtitle split output with mismatched lengths: "
            f"{len(split_src)} source rows vs {len(split_trans)} translation rows"
        )

    pd.DataFrame({
        'Source': [safe_text(item) for item in split_src],
        'Translation': [safe_text(item) for item in split_trans],
    }).to_excel(_5_SPLIT_SUB, index=False)
    pd.DataFrame({
        'Source': [safe_text(item) for item in src],
        'Translation': [safe_text(item) for item in remerged],
    }).to_excel(_5_REMERGED, index=False)

    total_duration = time.time() - step_started_at
    summary = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(step_started_at)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration": total_duration,
        "input_lines": len(df),
        "output_lines": len(split_src),
        "attempts": len(timing_attempts),
        "max_workers": load_key("max_workers"),
        "subtitle_max_length": MAX_SUB_LENGTH,
        "subtitle_target_multiplier": TARGET_SUB_MULTIPLIER,
        "total_lines_needing_split": sum(
            attempt["lines_needing_split"] for attempt in timing_attempts
        ),
        "total_validation_retries": sum(
            line.get("validation_retry_count", 0)
            for attempt in timing_attempts
            for line in attempt["line_timings"]
        ),
        "total_split_validation_retries": sum(
            line.get("split_validation_retries", 0)
            for attempt in timing_attempts
            for line in attempt["line_timings"]
        ),
        "total_align_validation_retries": sum(
            line.get("align_validation_retries", 0)
            for attempt in timing_attempts
            for line in attempt["line_timings"]
        ),
        "total_split_requests": sum(
            line.get("split_requests", 0)
            for attempt in timing_attempts
            for line in attempt["line_timings"]
        ),
        "total_align_requests": sum(
            line.get("align_requests", 0)
            for attempt in timing_attempts
            for line in attempt["line_timings"]
        ),
    }
    _write_split_timing(summary, timing_attempts)
    record_event(
        "Subtitle split timing summary",
        event_type="subtitle_split_summary",
        **{
            key: summary[key]
            for key in (
                "duration",
                "input_lines",
                "output_lines",
                "attempts",
                "max_workers",
                "total_lines_needing_split",
                "total_validation_retries",
                "total_split_validation_retries",
                "total_align_validation_retries",
                "total_split_requests",
                "total_align_requests",
            )
        },
    )

if __name__ == '__main__':
    split_for_sub_main()
