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
from difflib import SequenceMatcher
from core.utils.models import *
from core.utils.llm_stage_utils import (
    risky_row_indices,
    sentence_risk_score,
    stage_api_config,
    staged_llm_enabled,
)
console = Console()
SOURCE_SUBTITLE_PATH = "output/source_subtitle.srt"


def _valid_hard_translation(expected_ids):
    def validate(result):
        if not isinstance(result, dict):
            return {'status': 'error', 'message': 'Response must be a JSON object'}
        for item_id in expected_ids:
            item = result.get(item_id)
            if not isinstance(item, dict) or not str(item.get('translation', '')).strip():
                return {'status': 'error', 'message': f'Missing translation for id {item_id}'}
        return {'status': 'success', 'message': ''}
    return validate


def _refine_hard_translations(df):
    """Re-translate only the highest-risk rows with the configured hard stage."""
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
    console.print(f"[cyan]Thinking pass selected {len(indices)}/{len(df)} high-risk rows.[/cyan]")
    for start in range(0, len(indices), 5):
        batch_indices = indices[start:start + 5]
        items = []
        for index in batch_indices:
            row = output.loc[index]
            position = output.index.get_loc(index)
            items.append({
                "id": str(index),
                "source": str(row["Source"]),
                "current_translation": str(row["Translation"]),
                "risk_score": sentence_risk_score(row["Source"], row["Translation"]),
                "previous_source": str(output.iloc[position - 1]["Source"]) if position > 0 else "",
                "next_source": str(output.iloc[position + 1]["Source"]) if position + 1 < len(output) else "",
            })
        expected_ids = [item["id"] for item in items]
        template = {item_id: {"translation": "best concise translation"} for item_id in expected_ids}
        prompt = f"""
You are the difficult-line translation specialist in a subtitle pipeline.
Re-evaluate each source line using its context. Correct mistranslation, omission,
unsupported additions, ambiguous references, terminology, names, numbers, and units.
Keep documentary narration natural, concise, and suitable for spoken dubbing.
If the current translation is already correct, return it unchanged. Do not merge,
split, add, delete, or reorder rows. Output JSON only.

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
        for index in batch_indices:
            output.at[index, "Translation"] = str(result[str(index)]["translation"]).replace("\n", " ").strip()
            output.at[index, "Hard Translation Applied"] = True
    return output

# Function to split text into chunks
def split_chunks_by_chars(chunk_size, max_i): 
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    with open(_3_2_SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
        sentences = file.read().strip().split('\n')

    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in sentences:
        if chunk and (len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i):
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks

# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index):
    return None if chunk_index == 0 else chunks[chunk_index - 1].split('\n')[-3:] # Get last 3 lines
def get_after_content(chunks, chunk_index):
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split('\n')[:2] # Get first 2 lines

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
        if len(lines) <= 1:
            raise

        midpoint = len(lines) // 2
        console.print(
            f"[yellow]Translation block {index} failed with {len(lines)} lines. "
            f"Retrying as {midpoint}+{len(lines) - midpoint} smaller lines.[/yellow]"
        )
        left_translation, left_source = _translate_chunk_with_fallback(
            '\n'.join(lines[:midpoint]),
            previous_content_prompt,
            after_content_prompt,
            theme_prompt,
            f"{index}.1",
        )
        right_translation, right_source = _translate_chunk_with_fallback(
            '\n'.join(lines[midpoint:]),
            previous_content_prompt,
            after_content_prompt,
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

# Add similarity calculation function
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


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

    results.sort(key=lambda x: x[0])  # Sort results based on original order
    
    # 💾 Save results to lists and Excel file
    src_text, trans_text = [], []
    for i, chunk in enumerate(chunks):
        chunk_lines = chunk.split('\n')
        src_text.extend(chunk_lines)
        
        # Calculate similarity between current chunk and translation results
        chunk_text = ''.join(chunk_lines).lower()
        matching_results = [(r, similar(''.join(r[1].split('\n')).lower(), chunk_text)) 
                          for r in results]
        best_match = max(matching_results, key=lambda x: x[1])
        
        # Check similarity and handle exceptions
        if best_match[1] < 0.9:
            console.print(f"[yellow]Warning: No matching translation found for chunk {i}[/yellow]")
            raise ValueError(f"Translation matching failed (chunk {i})")
        elif best_match[1] < 1.0:
            console.print(f"[yellow]Warning: Similar match found (chunk {i}, similarity: {best_match[1]:.3f})[/yellow]")
            
        trans_text.extend(best_match[0][2].split('\n'))
    
    df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
    subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
    df_time = _apply_uploaded_subtitle_timestamps(df_translate)
    if df_time is None:
        df_text = pd.read_excel(_2_CLEANED_CHUNKS)
        df_text['text'] = df_text['text'].str.strip('"').str.strip()
        df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
    df_time = _refine_hard_translations(df_time)
    console.print(df_time)
    # apply check_len_then_trim to df_time['Translation'], only when duration > MIN_TRIM_DURATION.
    df_time['Translation'] = df_time.apply(lambda x: check_len_then_trim(x['Translation'], x['duration']) if x['duration'] > load_key("min_trim_duration") else x['Translation'], axis=1)
    console.print(df_time)
    
    df_time.to_excel(_4_2_TRANSLATION, index=False)
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

if __name__ == '__main__':
    translate_all()
