from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd

from core.utils.models import _2_CLEANED_CHUNKS


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


def import_srt_to_cleaned_chunks(
    content: str,
    output_path: str = _2_CLEANED_CHUNKS,
) -> tuple[str, int, int]:
    blocks = parse_srt(content)
    df = srt_blocks_to_cleaned_chunks(blocks)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    return output_path, len(blocks), len(df)
