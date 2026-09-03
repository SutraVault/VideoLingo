"""Bounded, contiguous proofreading windows (no model or filesystem access)."""

import re


PROOFREAD_ISSUE_TYPES = (
    "meaning", "omission", "addition", "terminology", "grammar",
    "punctuation", "word_order", "reference", "numbers_units", "source_unclear",
)


def _sentence_end(source):
    text = str(source).strip().rstrip('\"\'”’)]）')
    if not re.search(r"[.!?。！？]$", text):
        return False
    # Common titles/initials must not cut off their following name.
    if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Gen|Col|Lt|Capt|Sgt|St|No|vs|e\.g|i\.e)\.$", text, re.I):
        return False
    return not bool(re.search(r"\b(?:[A-Za-z]\.)+$", text))


def sentence_spans(rows, max_sentence_lines=28):
    """Half-open spans; punctuation is a heuristic, not semantic parsing.

    Missing punctuation is bounded explicitly. A forced boundary is also a
    redistribution boundary: adjacent read-only context cannot be edited.
    """
    if max_sentence_lines < 1:
        raise ValueError("max_sentence_lines must be positive")
    spans = []
    start = 0
    for index, row in enumerate(rows):
        if _sentence_end(row["source"]) or index + 1 - start >= max_sentence_lines:
            spans.append((start, index + 1))
            start = index + 1
    if start < len(rows):
        spans.append((start, len(rows)))
    return spans


def proofreading_windows(rows, chunk_lines=7, context_lines=4,
                         selected_indices=None, max_sentence_lines=28):
    """Select whole sentences; batch only adjoining spans, never sparse rows.

    chunk_lines is a soft target. A complete sentence can exceed it, up to
    max_sentence_lines. Every editable row belongs to exactly one window.
    Context is taken from the original snapshot and expanded to span edges.
    """
    if chunk_lines < 1 or context_lines < 0:
        raise ValueError("chunk_lines must be positive and context_lines nonnegative")
    spans = sentence_spans(rows, max_sentence_lines)
    selected = None if selected_indices is None else set(selected_indices)
    groups = []
    for start, end in spans:
        if selected is not None and not any(i in selected for i in range(start, end)):
            continue
        if groups and groups[-1][-1][1] == start and end - groups[-1][0][0] <= chunk_lines:
            groups[-1].append((start, end))
        else:
            groups.append([(start, end)])

    for group in groups:
        start, end = group[0][0], group[-1][1]
        left, right = max(0, start - context_lines), min(len(rows), end + context_lines)
        if context_lines:
            left = next(a for a, b in spans if a <= left < b)
            right = next(b for a, b in spans if a < right <= b)
        yield {
            "rows": rows[start:end],
            "context_before": rows[left:start],
            "context_after": rows[end:right],
            "semantic_groups": [[str(row["id"]) for row in rows[a:b]] for a, b in group],
        }


def validate_proofread_response(rows, response):
    """Reject partial batches: per-row fallback is unsafe after redistribution."""
    expected = {str(row["id"]) for row in rows}
    if not isinstance(response, dict):
        return {"status": "error", "message": "Response must be a JSON object"}
    if set(response) != expected:
        return {"status": "error", "message": "Return exactly all editable ids, no context ids; resubmit the entire batch."}
    originals = {str(row["id"]): row["translation"] for row in rows}
    for item_id in expected:
        item = response[item_id]
        if not isinstance(item, dict) or not isinstance(item.get("proofread"), str) or not item["proofread"].strip():
            return {"status": "error", "message": f"Nonempty proofread string required for id {item_id}; resubmit the entire batch."}
        status = item.get("status")
        if status not in ("ok", "corrected", "needs_review"):
            return {"status": "error", "message": f"id {item_id}: status must be ok, corrected, or needs_review."}
        issues = item.get("issue_types")
        if not isinstance(issues, list) or any(not isinstance(issue, str) or issue not in PROOFREAD_ISSUE_TYPES for issue in issues):
            return {"status": "error", "message": f"id {item_id}: issue_types must be an array drawn from {PROOFREAD_ISSUE_TYPES}."}
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            return {"status": "error", "message": f"id {item_id}: provide a concise evidence-based reason, including for unchanged rows."}
        changed = " ".join(str(originals[item_id]).split()) != " ".join(item["proofread"].split())
        if status == "corrected" and (not changed or not issues):
            return {"status": "error", "message": f"id {item_id}: corrected requires a changed translation and nonempty issue_types."}
        if status in ("ok", "needs_review") and changed:
            return {"status": "error", "message": f"id {item_id}: {status} must preserve the original translation; use corrected for supported edits."}
        if (status == "ok" and issues) or (status == "needs_review" and not issues):
            return {"status": "error", "message": f"id {item_id}: ok requires empty issue_types; needs_review requires an identified issue."}
    return {"status": "success", "message": "Proofread completed"}
