import json
import os
import time
from contextlib import contextmanager
from datetime import datetime

from rich.console import Console

from core.utils import ask_gpt, load_key
from core.utils.excel_utils import read_excel_with_aliases
from core.utils.models import _4_2_TRANSLATION, _4_3_PROOFREAD_TRANSLATION
from core._8_1_audio_task import check_len_then_trim
from core.utils.llm_stage_utils import stage_api_config, staged_llm_enabled, risky_row_indices

console = Console()

PROOFREAD_LOCK = "output/log/translation_proofread.lock"


def is_proofread_running():
    return os.path.exists(PROOFREAD_LOCK)


@contextmanager
def _proofread_lock():
    os.makedirs(os.path.dirname(PROOFREAD_LOCK), exist_ok=True)
    try:
        fd = os.open(PROOFREAD_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"LLM proofreading is already running. If this is stale, delete {PROOFREAD_LOCK}."
        )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(datetime.now().isoformat(timespec="seconds"))
        yield
    finally:
        try:
            os.remove(PROOFREAD_LOCK)
        except FileNotFoundError:
            pass


def _safe_to_excel(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp.{os.getpid()}.xlsx"
    df.to_excel(temp_path, index=False)
    try:
        os.replace(temp_path, path)
        return path
    except PermissionError:
        fallback = f"{os.path.splitext(path)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        os.replace(temp_path, fallback)
        console.print(
            f"[yellow]Could not overwrite {path}. It may be open or another run may be writing it. "
            f"Saved proofread workbook to {fallback} instead.[/yellow]"
        )
        return fallback
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield start, items[start:start + chunk_size]


def _normalized_comparison_text(value):
    return " ".join(str(value or "").split())


def _set_proofread_changed_column(df):
    if "LLM Proofread" not in df.columns:
        return df
    originals = df["Original Translation"] if "Original Translation" in df.columns else df["Translation"]
    df = df.copy()
    df["Proofread Changed"] = [
        _normalized_comparison_text(original) != _normalized_comparison_text(proofread)
        for original, proofread in zip(originals, df["LLM Proofread"])
    ]
    return df


def _proofread_api_config():
    if staged_llm_enabled():
        return stage_api_config("proofread")
    if not load_key("llm_proofread.override_api"):
        return None
    return {
        "key": load_key("llm_proofread.api.key"),
        "base_url": load_key("llm_proofread.api.base_url"),
        "model": load_key("llm_proofread.api.model"),
        "llm_support_json": load_key("llm_proofread.api.llm_support_json"),
    }


def _build_prompt(rows):
    target_language = load_key("target_language")
    src_language = load_key("whisper.detected_language")
    input_rows = [
        {
            "id": str(row["id"]),
            "source": row["source"],
            "translation": row["translation"],
        }
        for row in rows
    ]
    output_template = {
        str(row["id"]): {
            "proofread": f"corrected {target_language} subtitle"
        }
        for row in rows
    }

    return f"""
Conservatively proofread subtitle translations from {src_language} to {target_language}.

Rules:
1. Only change a translation when there is a clear error.
2. Check for mistranslations, omissions, and information added without support from the source.
3. Do not rewrite for style, elegance, or personal preference.
4. Prefer the original translation when it is acceptable.
5. Verify pronoun references, people, places, organizations, technical/military terms, vehicle/weapon names, unit names, and operation names.
6. Preserve and verify all dates, times, quantities, calibers, model numbers, and measurement units.
7. Keep natural documentary narration style and concise wording suitable for speaking aloud.
8. Preserve the exact ids and row count; do not merge, split, add, delete, or reorder lines.
9. Do not add explanations or comments. If no clear issue exists, return the input translation unchanged.

Input rows:
{json.dumps(input_rows, ensure_ascii=False)}

Output only JSON in this shape:
{json.dumps(output_template, ensure_ascii=False, separators=(',', ':'))}
""".strip()


def _valid_proofread(rows):
    original_by_id = {
        str(row["id"]): str(row["translation"])
        for row in rows
    }
    expected_ids = list(original_by_id)

    def validate(response):
        if not isinstance(response, dict):
            return {"status": "error", "message": "Response must be a JSON object"}

        fallback_ids = []
        for item_id in expected_ids:
            item = response.get(item_id)
            if item is None:
                response[item_id] = {"proofread": original_by_id[item_id]}
                fallback_ids.append(item_id)
                continue
            if not isinstance(item, dict):
                return {"status": "error", "message": f"Invalid proofread entry for id {item_id}"}
            if not str(item.get("proofread", "")).strip():
                item["proofread"] = original_by_id[item_id]
                fallback_ids.append(item_id)

        if fallback_ids:
            console.print(
                "[yellow]Proofread response omitted text for ids "
                f"{fallback_ids[:5]}; kept the original translation for those rows.[/yellow]"
            )
        return {"status": "success", "message": "Proofread completed"}

    return validate


def _trim_for_subtitle_timing(df):
    if "duration" not in df.columns:
        return df

    min_trim_duration = load_key("min_trim_duration")

    def trim_row(row):
        text = str(row["Translation"])
        duration = row["duration"]
        if duration > min_trim_duration:
            return check_len_then_trim(text, duration)
        return text

    df = df.copy()
    df["Translation"] = df.apply(trim_row, axis=1)
    return df


def proofread_translation_if_enabled():
    if not load_key("llm_proofread.enabled"):
        return
    proofread_translation()


def proofread_translation(force=False):
    with _proofread_lock():
        _proofread_translation_unlocked(force=force)


def _proofread_translation_unlocked(force=False):
    if os.path.exists(_4_3_PROOFREAD_TRANSLATION) and not force:
        existing = read_excel_with_aliases(
            _4_3_PROOFREAD_TRANSLATION, required_columns=["Source", "Translation"]
        )
        if "LLM Proofread" in existing.columns and "Proofread Changed" not in existing.columns:
            existing = _set_proofread_changed_column(existing)
            _safe_to_excel(existing, _4_3_PROOFREAD_TRANSLATION)
            console.print("[green]Added Proofread Changed column to the existing proofread workbook.[/green]")
        console.print(f"[yellow]Proofread translation already exists: {_4_3_PROOFREAD_TRANSLATION}[/yellow]")
        return

    df = read_excel_with_aliases(_4_2_TRANSLATION, required_columns=["Source", "Translation"])
    rows = [
        {
            "id": index + 1,
            "source": str(row["Source"]),
            "translation": str(row["Translation"]),
        }
        for index, row in df.iterrows()
    ]

    proofread_text = [None] * len(rows)
    api_config = _proofread_api_config()
    chunk_lines = int(load_key("llm_proofread.chunk_lines"))

    if staged_llm_enabled() and load_key("llm_stages.proofread.only_risky"):
        risky_labels = risky_row_indices(df, max_ratio=0.15, min_score=2.5)
        risky = {df.index.get_loc(label) for label in risky_labels}
        selected_rows = [row for index, row in enumerate(rows) if index in risky]
        for index, row in enumerate(rows):
            if index not in risky:
                proofread_text[index] = row["translation"]
        console.print(f"[cyan]Risk-based proofreading selected {len(selected_rows)}/{len(rows)} rows.[/cyan]")
    else:
        selected_rows = rows

    for start, chunk in _chunks(selected_rows, chunk_lines):
        prompt = _build_prompt(chunk)
        row_ids = f"{chunk[0]['id']}-{chunk[-1]['id']}"
        console.print(f"[cyan]Proofreading row ids {row_ids}...[/cyan]")
        result = ask_gpt(
            prompt,
            resp_type="json",
            valid_def=_valid_proofread(chunk),
            log_title="translation_proofread",
            api_config=api_config,
            use_cache=not force,
        )
        for row in chunk:
            proofread_text[row["id"] - 1] = str(result[str(row["id"])]["proofread"]).replace("\n", " ").strip()
        console.print(f"[green]Proofread row ids {row_ids}[/green]")

    if any(not item for item in proofread_text):
        raise ValueError("LLM proofreading produced empty rows")

    output_df = df.copy()
    output_df["LLM Proofread"] = proofread_text
    output_df["Original Translation"] = df["Translation"]
    output_df = _set_proofread_changed_column(output_df)

    mode = load_key("llm_proofread.mode")
    if mode == "apply":
        output_df["Translation"] = output_df["LLM Proofread"]
        output_df = _trim_for_subtitle_timing(output_df)
        output_df["LLM Proofread"] = output_df["Translation"]
        output_df = _set_proofread_changed_column(output_df)
        _safe_to_excel(output_df, _4_2_TRANSLATION)

    saved_path = _safe_to_excel(output_df, _4_3_PROOFREAD_TRANSLATION)
    console.print(f"[green]Proofread translation saved to {saved_path}[/green]")


def apply_proofread_to_translation():
    df = read_excel_with_aliases(_4_3_PROOFREAD_TRANSLATION, required_columns=["Source", "Translation"])
    if "LLM Proofread" not in df.columns:
        raise KeyError("Missing 'LLM Proofread' column in proofread workbook")
    df["Original Translation"] = df.get("Original Translation", df["Translation"])
    df["Translation"] = df["LLM Proofread"]
    df = _trim_for_subtitle_timing(df)
    df["LLM Proofread"] = df["Translation"]
    df = _set_proofread_changed_column(df)
    _safe_to_excel(df, _4_2_TRANSLATION)
    _safe_to_excel(df, _4_3_PROOFREAD_TRANSLATION)
    console.print(f"[green]Applied LLM Proofread column to {_4_2_TRANSLATION}[/green]")
