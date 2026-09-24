"""Sentence-aware translation batching, context and deterministic assembly."""

import re
from difflib import SequenceMatcher

from core.utils.proofread_context import sentence_spans


def split_translation_chunks(lines, chunk_chars=1600, chunk_lines=12,
                             max_sentence_lines=28):
    """Budgets are soft: keep a sentence together, bounded by row count.

    An individual source row is never split, including an oversized uploaded
    subtitle. Without punctuation, max_sentence_lines is a safety boundary.
    """
    if chunk_chars < 1 or chunk_lines < 1:
        raise ValueError("Translation chunk budgets must be positive")
    if not lines:
        return []
    spans = sentence_spans([{"source": line} for line in lines], max_sentence_lines)
    chunks, current = [], []
    for start, end in spans:
        unit = lines[start:end]
        combined = current + unit
        if current and (len(combined) > chunk_lines or len("\n".join(combined)) > chunk_chars):
            chunks.append("\n".join(current))
            current = []
        current.extend(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


def select_translation_context(before, after, context_lines=4, max_sentence_lines=28):
    """Read-only context on each side, expanded to the nearest sentence edge."""
    if context_lines < 0:
        raise ValueError("Translation context_lines must be nonnegative")
    if not context_lines:
        return [], []
    left, right = max(0, len(before) - context_lines), min(len(after), context_lines)
    if before:
        spans = sentence_spans([{"source": line} for line in before], max_sentence_lines)
        left = next(a for a, b in spans if a <= left < b)
    if after:
        spans = sentence_spans([{"source": line} for line in after], max_sentence_lines)
        right = next(b for a, b in spans if a < right <= b)
    return before[left:], after[:right]


def sentence_split_point(lines, max_sentence_lines=28):
    """Fallback at a sentence boundary only; a single unit must fail atomically."""
    spans = sentence_spans([{"source": line} for line in lines], max_sentence_lines)
    boundaries = [end for _, end in spans[:-1]]
    return min(boundaries, key=lambda end: abs(end - len(lines) / 2)) if boundaries else None


def assemble_translation_results(chunks, results):
    """Never use text similarity to match results: repeated source is valid."""
    by_index = {}
    for index, source, translation in results:
        if index in by_index or not isinstance(index, int) or not 0 <= index < len(chunks):
            raise ValueError(f"Duplicate or invalid translation block id: {index}")
        by_index[index] = (source, translation)
    if set(by_index) != set(range(len(chunks))):
        raise ValueError("Missing translation block results")
    source_rows, translated_rows = [], []
    for index, chunk in enumerate(chunks):
        source, translation = by_index[index]
        if source != chunk:
            raise ValueError(f"Translation block {index} source mismatch")
        translated = translation.split("\n")
        if len(translated) != len(chunk.split("\n")) or any(not line.strip() for line in translated):
            raise ValueError(f"Translation block {index} has missing/empty rows")
        source_rows.extend(chunk.split("\n"))
        translated_rows.extend(translated)
    return source_rows, translated_rows


def _compact_text(value):
    return re.sub(r"[^0-9A-Za-z\u3400-\u9fff%]+", "", str(value or "")).lower()


def _longest_common_substring(left, right):
    match = SequenceMatcher(None, left, right, autojunk=False).find_longest_match()
    return left[match.a:match.a + match.size]


def adjacent_duplicate_groups(sources, translations, min_phrase_chars=4,
                              similarity_threshold=0.72,
                              max_sentence_lines=28):
    """Find connected, sentence-local rows with suspicious target duplication."""
    if len(sources) != len(translations):
        raise ValueError("Source and translation row counts must match")
    spans = sentence_spans(
        [{"source": str(source or "")} for source in sources], max_sentence_lines
    )
    suspicious_pairs = []
    number_re = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?%?(?![A-Za-z0-9])")
    for start, end in spans:
        for index in range(start, end - 1):
            left = _compact_text(translations[index])
            right = _compact_text(translations[index + 1])
            if not left or not right:
                continue
            common = _longest_common_substring(left, right)
            target_numbers = set(number_re.findall(left)) & set(number_re.findall(right))
            source_pair = " ".join(str(item or "") for item in sources[index:index + 2])
            duplicated_numbers = {
                number for number in target_numbers
                if len(re.findall(re.escape(number), source_pair, flags=re.IGNORECASE)) <= 1
            }
            similarity = SequenceMatcher(None, left, right, autojunk=False).ratio()
            reasons = []
            if len(common) >= int(min_phrase_chars):
                reasons.append(f'repeated phrase "{common}"')
            if duplicated_numbers:
                reasons.append("repeated number " + ", ".join(sorted(duplicated_numbers)))
            if min(len(left), len(right)) >= 8 and similarity >= float(similarity_threshold):
                reasons.append(f"high adjacent similarity {similarity:.2f}")
            if reasons:
                suspicious_pairs.append((index, index + 1, reasons))

    groups = []
    for left, right, reasons in suspicious_pairs:
        if groups and left <= groups[-1]["end"]:
            groups[-1]["end"] = max(groups[-1]["end"], right)
            groups[-1]["reasons"].extend(reasons)
        else:
            groups.append({"start": left, "end": right, "reasons": list(reasons)})
    for group in groups:
        group["reasons"] = sorted(set(group["reasons"]))
    return groups
