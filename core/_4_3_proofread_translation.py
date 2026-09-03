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
from core.utils.proofread_context import PROOFREAD_ISSUE_TYPES, proofreading_windows, validate_proofread_response

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


def _proofread_setting(name, default):
    try:
        return load_key(f"llm_proofread.{name}")
    except KeyError:
        return default


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


def _build_prompt(rows, context_before=(), context_after=(), semantic_groups=None):
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
            "status": "ok | corrected | needs_review (choose one)",
            "issue_types": [],
            "reason": f"brief concrete finding in {target_language}",
            "proofread": f"corrected or unchanged {target_language} subtitle",
        }
        for row in rows
    }

    def context_payload(items):
        return [{"id": str(row["id"]), "source": row["source"],
                 "translation": row["translation"]} for row in items]

    return f"""
Audit and correct subtitle translations from {src_language} to {target_language}.
You are an independent bilingual editor, not an approver of the existing draft.
The draft may contain fluent-looking mistranslations, literal wording, broken
Chinese sentences, and incorrect sentence boundaries. Neither copying it by
default nor inventing edits to achieve a change quota is acceptable.

For each editable row, FIRST record a brief concrete finding and decision,
THEN supply its final subtitle. This is a concise editorial report, not a
request for private step-by-step reasoning. Perform both tasks in this one response.

Rules:
1. Actively check meaning, idioms/word senses, omissions/additions, terminology, pronouns, numbers/units, grammatical Chinese, punctuation, and cross-row continuity. Correct identifiable defects, including literal wording that obscures the intended meaning and sentence breaks that detach a clause from what it modifies.
2. Read the full continuous source and translation, including context, BEFORE judging individual rows. Check mistranslations, omissions, and unsupported additions across each semantic group as a whole, not by demanding literal one-to-one correspondence per row.
3. Necessary repairs to awkward literal wording, broken syntax, and misplaced punctuation ARE within scope; do not dismiss these as mere style. Do not replace equally good synonyms or embellish an already clear, faithful translation.
4. Judge the intended sense from the source and context, not from the draft. For example, an engine described as 'forgiving' under poor maintenance is tolerant of adverse conditions, not a person being '宽容'; 'fully equipped' does not itself mean 'heavily armed'. These illustrate distinctions, not mandatory word replacements. Fragmentary subtitle rows are valid when the combined sentence reads correctly; never force every row to end with a full stop.
5. Verify pronoun references, people, places, organizations, technical/military terms, vehicle/weapon names, unit names, and operation names.
6. Preserve and verify all dates, times, quantities, calibers, model numbers, and measurement units.
7. Keep natural documentary narration style and concise wording suitable for speaking aloud.
8. Preserve editable ids, their order and row count; do not merge/delete rows or change source text or timestamps. Every editable row must have nonempty translated text.
9. Within ONE listed semantic group, you MAY redistribute meaning between adjacent editable rows when required by Chinese word order (e.g. moving a following condition or modifier earlier). Preserve the group's entire meaning exactly once, without omissions or duplication. Do not move meaning across semantic groups or into/out of read-only context. Keep each row's speaking length reasonably close to its original; avoid timing drift and moving content far from the corresponding speech.
10. Context before/after is read-only reference for pronouns, terminology and sentence continuity. Never output or modify context ids. If a correction requires changing read-only context, retain the affected row and mark needs_review with the relevant ids and issue. Do not guess facts to repair unclear ASR or uncertain technical names; mark needs_review instead.
11. Return the COMPLETE editable batch, not only changed rows. Do not force a minimum number of changes. Keep diagnostic fields OUT of the subtitle text.

Required report fields for EVERY editable id:
- status: 'corrected' for a supported text change; 'ok' if no specific defect was found; 'needs_review' if a suspected issue cannot safely be corrected with this source/window.
- issue_types: array drawn only from {json.dumps(PROOFREAD_ISSUE_TYPES)}. Empty only for ok; nonempty for corrected and needs_review.
- reason: one short sentence in {target_language}. For corrected, identify the source phrase or adjacent ids and explain the specific defect and repair. For needs_review, identify the uncertainty or boundary blocking a safe edit. For ok, briefly state a concrete checked meaning or relationship; do not invent an issue. Avoid generic 'looks good' repeated for every row.
- proofread: final subtitle text only, nonempty. For ok and needs_review, copy the original translation exactly. For corrected, actually revise it; a change to an adjacent row must be reflected in that row's own report too.

Read-only context before:
{json.dumps(context_payload(context_before), ensure_ascii=False)}

Editable input rows (continuous subtitles):
{json.dumps(input_rows, ensure_ascii=False)}

Read-only context after:
{json.dumps(context_payload(context_after), ensure_ascii=False)}

Semantic groups (meaning may move only between adjacent editable ids within the same group):
{json.dumps(semantic_groups if semantic_groups is not None else [[str(row['id']) for row in rows]], ensure_ascii=False)}

Output only JSON in this shape:
{json.dumps(output_template, ensure_ascii=False, separators=(',', ':'))}
""".strip()


def _valid_proofread(rows):
    def validate(response):
        return validate_proofread_response(rows, response)

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
        for index, (_, row) in enumerate(df.iterrows())
    ]

    proofread_text = [row["translation"] for row in rows]
    proofread_status = ["not_reviewed"] * len(rows)
    proofread_issues = [""] * len(rows)
    proofread_reasons = ["未送审：未被风险筛选选中。"] * len(rows)
    api_config = _proofread_api_config()
    chunk_lines = int(load_key("llm_proofread.chunk_lines"))

    risky = None
    if staged_llm_enabled() and load_key("llm_stages.proofread.only_risky"):
        risky_labels = risky_row_indices(df, max_ratio=0.15, min_score=2.5)
        risky = {df.index.get_loc(label) for label in risky_labels}
        console.print(f"[cyan]Risk-based proofreading selected {len(risky)}/{len(rows)} seed rows; expanding to sentence spans.[/cyan]")

    windows = list(proofreading_windows(
        rows, chunk_lines=chunk_lines,
        context_lines=int(_proofread_setting("context_lines", 4)),
        selected_indices=risky,
        max_sentence_lines=int(_proofread_setting("max_sentence_lines", 28)),
    ))
    reviewed_ids = {row["id"] for window in windows for row in window["rows"]}
    console.print(f"[cyan]Proofreading {len(reviewed_ids)}/{len(rows)} editable rows in {len(windows)} continuous batches with read-only context.[/cyan]")
    if rows and not reviewed_ids:
        console.print("[yellow]No rows selected: no LLM proofreading will run. Disable 'Proofread only risky lines' for a full audit.[/yellow]")

    for window in windows:
        chunk = window["rows"]
        prompt = _build_prompt(**window)
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
            index = row["id"] - 1
            item = result[str(row["id"])]
            proofread_text[index] = item["proofread"].replace("\n", " ").strip()
            proofread_status[index] = item["status"]
            proofread_issues[index] = ", ".join(item["issue_types"])
            proofread_reasons[index] = item["reason"].replace("\n", " ").strip()
        console.print(f"[green]Proofread row ids {row_ids}[/green]")

    if any(not item for item in proofread_text):
        raise ValueError("LLM proofreading produced empty rows")

    output_df = df.copy()
    output_df["LLM Proofread"] = proofread_text
    output_df["Original Translation"] = df["Translation"]
    output_df["Proofread Reviewed"] = [row["id"] in reviewed_ids for row in rows]
    output_df = _set_proofread_changed_column(output_df)
    # These describe the model's audit before optional timing trimming.
    output_df["Proofread Status"] = proofread_status
    output_df["Proofread Issues"] = proofread_issues
    output_df["Proofread Reason"] = proofread_reasons
    console.print(
        f"[cyan]Model audit: {proofread_status.count('corrected')} corrected, "
        f"{proofread_status.count('ok')} unchanged, "
        f"{proofread_status.count('needs_review')} need manual review, "
        f"{proofread_status.count('not_reviewed')} not reviewed.[/cyan]"
    )

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
