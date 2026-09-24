import pandas as pd
import json
import concurrent.futures
import os
from core.translate_lines import translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
from core._8_1_audio_task import check_len_then_trim
from core._6_gen_sub import align_timestamp
from core.utils import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from core.utils.proofread_context import proofreading_windows
from core.utils.translation_context import (
    split_translation_chunks, select_translation_context,
    sentence_split_point, assemble_translation_results,
    adjacent_duplicate_groups,
)
from core.utils.models import *
from core.utils.llm_stage_utils import (
    risky_row_indices,
    sentence_risk_score,
    stage_api_config,
    staged_llm_enabled,
)
console = Console()
SOURCE_SUBTITLE_PATH = "output/source_subtitle.srt"


def _translation_setting(key, default):
    try:
        return load_key(key)
    except KeyError:
        return default


def _valid_hard_translation(expected_ids):
    def validate(result):
        if not isinstance(result, dict):
            return {'status': 'error', 'message': 'Response must be a JSON object'}
        if set(result) != set(expected_ids):
            return {'status': 'error', 'message': 'Return exactly the editable ids, not context ids.'}
        for item_id in expected_ids:
            item = result.get(item_id)
            if not isinstance(item, dict) or not isinstance(item.get('translation'), str) or not item['translation'].strip():
                return {'status': 'error', 'message': f'Missing translation for id {item_id}'}
        return {'status': 'success', 'message': ''}
    return validate


def _refine_hard_translations(df):
    """Refine complete sentences containing risky rows, from an immutable draft."""
    if not staged_llm_enabled() or not load_key("llm_stages.hard_translation.enabled"):
        return df
    max_ratio = float(load_key("llm_stages.hard_translation.max_ratio"))
    min_score = float(load_key("llm_stages.hard_translation.min_score"))
    indices = risky_row_indices(df, max_ratio=max_ratio, min_score=min_score)
    if not indices:
        console.print("[green]No high-risk translation rows required a thinking pass.[/green]")
        return df

    output = df.copy()
    output["Initial Translation"] = output["Translation"]
    output["Translation Risk Score"] = output.apply(
        lambda row: sentence_risk_score(row.get("Source", ""), row.get("Translation", "")),
        axis=1,
    )
    output["Hard Translation Applied"] = False
    api_config = stage_api_config("hard_translation")
    rows = [{"id": str(position + 1), "source": str(row["Source"]),
             "translation": str(row["Translation"])}
            for position, (_, row) in enumerate(df.iterrows())]
    windows = list(proofreading_windows(
        rows, chunk_lines=5,
        context_lines=int(_translation_setting("translation_context_lines", 4)),
        selected_indices={df.index.get_loc(index) for index in indices},
        max_sentence_lines=int(_translation_setting("translation_max_sentence_lines", 28)),
    ))
    console.print(f"[cyan]Thinking pass: {len(indices)} risk seeds expanded to {sum(len(w['rows']) for w in windows)}/{len(df)} rows.[/cyan]")
    for window in windows:
        items = window["rows"]
        expected_ids = [item["id"] for item in items]
        template = {item_id: {"translation": "best concise translation"} for item_id in expected_ids}
        prompt = f"""
You are the difficult-line translation specialist in a subtitle pipeline.
Re-evaluate each continuous source sentence using its context and the current draft. Correct mistranslation, omission,
unsupported additions, ambiguous references, terminology, names, numbers, and units.
Keep documentary narration natural, concise, and suitable for spoken dubbing.
If the current translation is already correct, return it unchanged. Do not merge,
split, add, delete, or reorder rows. Output JSON only.
Judge the combined translation before modifying any row. If meaning was already
moved between adjacent rows, do NOT translate that meaning a second time.
Correct word senses rather than literal dictionary wording; a subtitle newline
does not end a sentence. Fix broken clause order and detached punctuation.
You may redistribute meaning between adjacent editable ids within the SAME
semantic group, preserving each proposition exactly once and every row nonempty.
Never redistribute meaning into/out of read-only context or other groups.

Read-only context before:
{json.dumps(window['context_before'], ensure_ascii=False)}

Read-only context after:
{json.dumps(window['context_after'], ensure_ascii=False)}

Semantic groups of editable ids:
{json.dumps(window['semantic_groups'])}

Input:
{json.dumps(items, ensure_ascii=False)}

Output shape:
{json.dumps(template, ensure_ascii=False, separators=(',', ':'))}
""".strip()
        result = ask_gpt(
            prompt,
            resp_type="json",
            valid_def=_valid_hard_translation(expected_ids),
            log_title="translate_hard",
            api_config=api_config,
        )
        for item in items:
            label = df.index[int(item["id"]) - 1]
            output.at[label, "Translation"] = result[item["id"]]["translation"].replace("\n", " ").strip()
            output.at[label, "Hard Translation Applied"] = True
    return output


def _repair_adjacent_duplicates(df):
    """Repair sentence-local rows flagged by the deterministic overlap screen."""
    output = df.copy()
    output["Adjacent Duplicate Repaired"] = False
    output["Adjacent Duplicate Signal"] = ""
    if not bool(_translation_setting("translation_duplicate_audit.enabled", True)) or output.empty:
        return output

    sources = output["Source"].fillna("").astype(str).tolist()
    translations = output["Translation"].fillna("").astype(str).tolist()
    groups = adjacent_duplicate_groups(
        sources,
        translations,
        min_phrase_chars=int(_translation_setting("translation_duplicate_audit.min_phrase_chars", 4)),
        similarity_threshold=float(_translation_setting("translation_duplicate_audit.similarity_threshold", 0.72)),
        max_sentence_lines=int(_translation_setting("translation_max_sentence_lines", 28)),
    )
    if not groups:
        console.print("[green]Adjacent translation audit: no suspicious duplication found.[/green]")
        return output

    console.print(f"[yellow]Adjacent translation audit found {len(groups)} suspicious group(s); repairing before save.[/yellow]")
    api_config = stage_api_config("hard_translation" if staged_llm_enabled() else "translate")
    for group in groups:
        start, end = group["start"], group["end"]
        items = [
            {"id": str(position + 1), "source": sources[position],
             "current_translation": str(output.iloc[position]["Translation"])}
            for position in range(start, end + 1)
        ]
        expected = [item["id"] for item in items]
        prompt = f'''You are repairing adjacent subtitle translations after an automatic duplicate audit.
Read the combined source passage, then rewrite the target rows so every source proposition appears exactly once.
Remove duplicated or paraphrased meaning and duplicated numbers. Restore any source fact displaced by duplication.
You may redistribute meaning only among these adjacent rows. Keep every id and every row non-empty.
Use concise natural spoken subtitles. Output JSON only.

Automatic signals: {json.dumps(group['reasons'], ensure_ascii=False)}
Rows: {json.dumps(items, ensure_ascii=False)}
Required shape: {json.dumps({item_id: {'translation': 'final text'} for item_id in expected}, ensure_ascii=False)}'''
        result = ask_gpt(
            prompt,
            resp_type="json",
            valid_def=_valid_hard_translation(expected),
            log_title="translate_duplicate_repair",
            api_config=api_config,
        )
        signal = "; ".join(group["reasons"])
        for position in range(start, end + 1):
            item_id = str(position + 1)
            label = output.index[position]
            output.at[label, "Translation"] = result[item_id]["translation"].replace("\n", " ").strip()
            output.at[label, "Adjacent Duplicate Repaired"] = True
            output.at[label, "Adjacent Duplicate Signal"] = signal
    return output

# Function to split text into chunks
def split_chunks_by_chars(chunk_size, max_i): 
    """Keep sentence spans together; character and line limits are soft targets."""
    with open(_3_2_SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
        sentences = file.read().strip().splitlines()
    return split_translation_chunks(
        sentences, chunk_chars=chunk_size, chunk_lines=max_i,
        max_sentence_lines=int(_translation_setting("translation_max_sentence_lines", 28)),
    )

# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index):
    before = [line for chunk in chunks[:chunk_index] for line in chunk.split('\n')]
    return select_translation_context(
        before, [], int(_translation_setting("translation_context_lines", 4)),
        int(_translation_setting("translation_max_sentence_lines", 28)),
    )[0]
def get_after_content(chunks, chunk_index):
    after = [line for chunk in chunks[chunk_index + 1:] for line in chunk.split('\n')]
    return select_translation_context(
        [], after, int(_translation_setting("translation_context_lines", 4)),
        int(_translation_setting("translation_max_sentence_lines", 28)),
    )[1]

# 🔍 Translate a single chunk
def _translate_chunk_with_fallback(chunk, previous_content_prompt, after_content_prompt, theme_prompt, index):
    try:
        things_to_note_prompt = search_things_to_note_in_prompt(chunk)
        return translate_lines(
            chunk,
            previous_content_prompt,
            after_content_prompt,
            things_to_note_prompt,
            theme_prompt,
            index,
        )
    except Exception:
        lines = chunk.split('\n')
        max_lines = int(_translation_setting("translation_max_sentence_lines", 28))
        midpoint = sentence_split_point(lines, max_lines)
        if midpoint is None:
            console.print(f"[red]Translation block {index} failed; preserving this sentence as one unit rather than splitting its meaning.[/red]")
            raise

        context_lines = int(_translation_setting("translation_context_lines", 4))
        left_before, left_after = select_translation_context(
            list(previous_content_prompt or []), lines[midpoint:] + list(after_content_prompt or []),
            context_lines, max_lines,
        )
        right_before, right_after = select_translation_context(
            list(previous_content_prompt or []) + lines[:midpoint], list(after_content_prompt or []),
            context_lines, max_lines,
        )
        console.print(
            f"[yellow]Translation block {index} failed with {len(lines)} lines. "
            f"Retrying as {midpoint}+{len(lines) - midpoint} smaller lines.[/yellow]"
        )
        left_translation, left_source = _translate_chunk_with_fallback(
            '\n'.join(lines[:midpoint]),
            left_before,
            left_after,
            theme_prompt,
            f"{index}.1",
        )
        right_translation, right_source = _translate_chunk_with_fallback(
            '\n'.join(lines[midpoint:]),
            right_before,
            right_after,
            theme_prompt,
            f"{index}.2",
        )
        return f"{left_translation}\n{right_translation}", f"{left_source}\n{right_source}"

def translate_chunk(chunk, chunks, theme_prompt, i):
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i)
    translation, english_result = _translate_chunk_with_fallback(
        chunk,
        previous_content_prompt,
        after_content_prompt,
        theme_prompt,
        i,
    )
    return i, english_result, translation

def _apply_uploaded_subtitle_timestamps(df_translate):
    if not (
        os.path.exists(SOURCE_SUBTITLE_PATH)
        and os.path.exists(_SOURCE_SUBTITLE_SEGMENTS)
    ):
        return None

    df_segments = pd.read_excel(_SOURCE_SUBTITLE_SEGMENTS)
    if len(df_segments) != len(df_translate):
        console.print(
            "[yellow]Uploaded subtitle timing exists, but row counts differ "
            f"({len(df_segments)} timings vs {len(df_translate)} translations). "
            "Falling back to timestamp alignment.[/yellow]"
        )
        return None

    df_time = df_translate.copy()
    df_time["timestamp"] = df_segments["timestamp"].tolist()
    df_time["duration"] = df_segments["duration"].astype(float).tolist()
    console.print(
        "[green]Using uploaded SRT blocks as translation rows and timestamps.[/green]"
    )
    return df_time

# 🚀 Main function to translate all chunks
@check_file_exists(_4_2_TRANSLATION)
def translate_all():
    console.print("[bold green]Start Translating All...[/bold green]")
    chunks = split_chunks_by_chars(
        chunk_size=load_key("translation_chunk_chars"),
        max_i=load_key("translation_chunk_lines")
    )
    with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
        theme_prompt = json.load(file).get('theme')

    # 🔄 Use concurrent execution for translation
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("[cyan]Translating chunks...", total=len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(translate_chunk, chunk, chunks, theme_prompt, i)
                futures.append(future)
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress.update(task, advance=1)

    src_text, trans_text = assemble_translation_results(chunks, results)
    
    df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
    subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
    df_time = _apply_uploaded_subtitle_timestamps(df_translate)
    if df_time is None:
        df_text = pd.read_excel(_2_CLEANED_CHUNKS)
        df_text['text'] = df_text['text'].str.strip('"').str.strip()
        df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
    df_time = _refine_hard_translations(df_time)
    df_time = _repair_adjacent_duplicates(df_time)
    df_time["Translation Workflow"] = "sentence_context_v1"
    console.print(df_time)
    # apply check_len_then_trim to df_time['Translation'], only when duration > MIN_TRIM_DURATION.
    df_time['Translation Before Timing Trim'] = df_time['Translation']
    df_time['Translation'] = df_time.apply(lambda x: check_len_then_trim(x['Translation'], x['duration']) if x['duration'] > load_key("min_trim_duration") else x['Translation'], axis=1)
    console.print(df_time)
    
    df_time.to_excel(_4_2_TRANSLATION, index=False)
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

if __name__ == '__main__':
    translate_all()
