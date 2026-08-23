from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd

from core.utils.models import (
    _2_CLEANED_CHUNKS,
    _3_1_SPLIT_BY_NLP,
    _3_2_SPLIT_BY_MEANING,
    _SOURCE_SUBTITLE_SEGMENTS,
)


@dataclass
class SubtitleBlock:
    start: float
    end: float
    text: str


def _srt_time_to_seconds(value: str) -> float:
    match = re.match(
        r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*$",
        value,
    )
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")

    hours, minutes, seconds, millis = match.groups()
    millis = millis.ljust(3, "0")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000
    )


def _seconds_to_srt_time(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_value, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d},{milliseconds:03d}"


def _format_srt_timestamp(start: float, end: float) -> str:
    return f"{_seconds_to_srt_time(start)} --> {_seconds_to_srt_time(end)}"


def parse_srt(content: str) -> list[SubtitleBlock]:
    blocks: list[SubtitleBlock] = []
    normalized = content.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")

    for raw_block in re.split(r"\n\s*\n", normalized.strip()):
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        if not lines:
            continue

        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue

        start_raw, end_raw = [part.strip() for part in lines[time_index].split("-->", 1)]
        text = " ".join(lines[time_index + 1 :]).strip()
        if not text:
            continue

        start = _srt_time_to_seconds(start_raw)
        end = _srt_time_to_seconds(end_raw.split()[0])
        if end <= start:
            continue

        blocks.append(SubtitleBlock(start=start, end=end, text=text))

    if not blocks:
        raise ValueError("No valid subtitle blocks found in the SRT file.")

    return blocks


def _split_text_to_units(text: str) -> list[str]:
    units = [unit.strip() for unit in re.split(r"\s+", text) if unit.strip()]
    if units:
        return units

    return [char for char in text if not char.isspace()]


def srt_blocks_to_cleaned_chunks(blocks: list[SubtitleBlock]) -> pd.DataFrame:
    rows = []

    for block in blocks:
        units = _split_text_to_units(block.text)
        if not units:
            continue

        duration = block.end - block.start
        step = duration / len(units)

        for index, unit in enumerate(units):
            rows.append(
                {
                    "text": f'"{unit}"',
                    "start": block.start + index * step,
                    "end": block.start + (index + 1) * step,
                    "speaker_id": None,
                }
            )

    if not rows:
        raise ValueError("No subtitle text found in the SRT file.")

    return pd.DataFrame(rows)


def srt_blocks_to_segments(blocks: list[SubtitleBlock]) -> pd.DataFrame:
    rows = [
        {
            "Source": block.text,
            "start": block.start,
            "end": block.end,
            "timestamp": _format_srt_timestamp(block.start, block.end),
            "duration": block.end - block.start,
        }
        for block in blocks
    ]

    if not rows:
        raise ValueError("No subtitle text found in the SRT file.")

    return pd.DataFrame(rows)


def _write_split_inputs(blocks: list[SubtitleBlock], *paths: str) -> None:
    lines = [block.text.strip() for block in blocks if block.text.strip()]
    if not lines:
        raise ValueError("No subtitle text found in the SRT file.")

    content = "\n".join(lines)
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)


def import_srt_to_cleaned_chunks(
    content: str,
    output_path: str = _2_CLEANED_CHUNKS,
    segment_output_path: str = _SOURCE_SUBTITLE_SEGMENTS,
    nlp_split_output_path: str = _3_1_SPLIT_BY_NLP,
    meaning_split_output_path: str = _3_2_SPLIT_BY_MEANING,
) -> tuple[str, int, int]:
    blocks = parse_srt(content)
    df = srt_blocks_to_cleaned_chunks(blocks)
    df_segments = srt_blocks_to_segments(blocks)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    os.makedirs(os.path.dirname(segment_output_path), exist_ok=True)
    df_segments.to_excel(segment_output_path, index=False)
    _write_split_inputs(blocks, nlp_split_output_path, meaning_split_output_path)

    return output_path, len(blocks), len(df)
