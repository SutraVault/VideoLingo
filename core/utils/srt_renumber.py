from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_blocks(content: str) -> list[list[str]]:
    return [
        [line.rstrip() for line in block.splitlines() if line.strip()]
        for block in re.split(r"\r?\n\r?\n", content)
        if block.strip()
    ]


def normalize_blocks(blocks: list[list[str]]) -> tuple[list[tuple[str, list[str]]], list[int]]:
    normalized = []
    skipped = []

    for raw_index, lines in enumerate(blocks, 1):
        if len(lines) < 3:
            skipped.append(raw_index)
            continue

        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            skipped.append(raw_index)
            continue

        text_lines = lines[time_index + 1 :]
        if not text_lines:
            skipped.append(raw_index)
            continue

        normalized.append((lines[time_index], text_lines))

    return normalized, skipped


def build_srt(items: list[tuple[str, list[str]]]) -> str:
    parts = []
    for index, (time_line, text_lines) in enumerate(items, 1):
        parts.append(f"{index}\n{time_line}\n" + "\n".join(text_lines))
    return "\n\n".join(parts) + "\n"


def renumber_srt(path: Path, output: Path | None = None) -> tuple[Path, int, list[int]]:
    content = path.read_text(encoding="utf-8")
    blocks = parse_blocks(content)
    normalized, skipped = normalize_blocks(blocks)
    destination = output or path
    destination.write_text(build_srt(normalized), encoding="utf-8")
    return destination, len(normalized), skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Renumber SRT blocks, skipping malformed or empty blocks."
    )
    parser.add_argument("path", help="Input SRT file path")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output SRT path. Defaults to in-place overwrite.",
    )
    args = parser.parse_args()

    input_path = Path(args.path)
    output_path = Path(args.output) if args.output else None

    written_path, count, skipped = renumber_srt(input_path, output_path)
    print(f"Written: {written_path}")
    print(f"Blocks kept: {count}")
    print(f"Blocks skipped: {len(skipped)}")
    if skipped:
        print("Skipped raw block numbers:", ", ".join(map(str, skipped)))


if __name__ == "__main__":
    main()
