"""Human-labeled proofreading outcomes. No LLM-based quality claims."""

import math
from collections import Counter


VERDICTS = {"fixed", "regression", "mixed", "style_only", "no_issue", "missed", "uncertain"}
CHANGED_VERDICTS = {"fixed", "regression", "mixed", "style_only"}
UNCHANGED_VERDICTS = {"no_issue", "missed"}


def text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def summarize_records(records):
    counts = Counter(rows=0, changed_rows=0, reviewed_rows=0, unknown_reviewed_rows=0,
                     labeled_rows=0, labeled_changed_rows=0, pending_rows=0,
                     pending_changed_rows=0, invalid_labels=0, unreviewed_labels=0)
    human = Counter({key: 0 for key in sorted(VERDICTS)})
    model = Counter()
    workflows = Counter()
    for row in records:
        counts["rows"] += 1
        # Compare actual texts rather than trusting a possibly stale boolean.
        original = row.get("Original Translation", row.get("Translation", ""))
        changed = " ".join(text(original).split()) != " ".join(text(row.get("LLM Proofread", original)).split())
        counts["changed_rows"] += int(changed)
        reviewed = text(row.get("Proofread Reviewed")).lower()
        known_reviewed = reviewed in ("true", "1", "1.0")
        known_skipped = reviewed in ("false", "0", "0.0")
        counts["reviewed_rows"] += int(known_reviewed)
        counts["unknown_reviewed_rows"] += int(not known_reviewed and not known_skipped)
        model[text(row.get("Proofread Status")) or "unknown"] += 1
        workflows[text(row.get("Translation Workflow")) or "legacy/unknown"] += 1
        verdict = text(row.get("Human Verdict")).lower()
        if not verdict:
            counts["pending_rows"] += 1
            counts["pending_changed_rows"] += int(changed)
            continue
        if (verdict not in VERDICTS or
                (verdict in CHANGED_VERDICTS and not changed) or
                (verdict in UNCHANGED_VERDICTS and changed)):
            counts["invalid_labels"] += 1
            continue
        # A skipped row must not count as a model miss or a successful review.
        if known_skipped:
            counts["unreviewed_labels"] += 1
            continue
        human[verdict] += 1
        counts["labeled_rows"] += 1
        counts["labeled_changed_rows"] += int(changed)
    return {"counts": dict(counts), "human_verdicts": dict(human),
            "model_statuses": dict(model), "translation_workflows": dict(workflows)}


def combine_summaries(summaries):
    total = {key: Counter() for key in ("counts", "human_verdicts", "model_statuses", "translation_workflows")}
    for summary in summaries:
        for key in total:
            total[key].update(summary[key])
    return {key: dict(value) for key, value in total.items()}
