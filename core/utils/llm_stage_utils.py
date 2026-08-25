"""Model-independent per-stage LLM routing and sentence risk scoring."""

import math
import re

from core.utils.config_utils import load_key


def _optional(key, default=None):
    try:
        return load_key(key)
    except KeyError:
        return default


def staged_llm_enabled():
    return bool(_optional("llm_stages.enabled", False))


def stage_api_config(stage):
    """Build ask_gpt config for a stage, or None for legacy global routing."""
    if not staged_llm_enabled():
        return None
    prefix = f"llm_stages.{stage}"
    model = str(_optional(f"{prefix}.model", "") or load_key("api.model"))
    reasoning_mode = str(_optional(f"{prefix}.reasoning", "auto") or "auto").lower()
    config = {
        "key": load_key("api.key"),
        "base_url": load_key("api.base_url"),
        "model": model,
        "llm_support_json": load_key("api.llm_support_json"),
        # Stage routing must not accidentally inherit api.reasoning.
        "inherit_reasoning": False,
    }
    if reasoning_mode == "off":
        config["reasoning"] = {"enabled": False}
    elif reasoning_mode not in ("auto", ""):
        config["reasoning"] = {"effort": reasoning_mode}
    return config


_PRONOUNS = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|which|who|whose|he|she|his|her)\b",
    re.IGNORECASE,
)
_CLAUSES = re.compile(
    r"\b(although|because|while|whereas|unless|which|who|that|when|if|but|however)\b",
    re.IGNORECASE,
)
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?(?:%|mm|cm|m|km|kg|lb|mph|km/h)?\b", re.IGNORECASE)


def sentence_risk_score(source, translation=""):
    """Deterministic risk score used to select a small hard-sentence subset."""
    source = str(source or "")
    translation = str(translation or "")
    words = re.findall(r"\b[\w'-]+\b", source)
    score = max(0.0, (len(words) - 14) / 6)
    score += min(2.0, len(_CLAUSES.findall(source)) * 0.55)
    score += min(1.5, len(_PRONOUNS.findall(source)) * 0.3)
    score += min(1.5, len(_ACRONYM.findall(source)) * 0.5)
    score += min(1.5, len(_NUMBER.findall(source)) * 0.45)
    score += min(1.0, source.count(",") * 0.2 + source.count(";") * 0.4)
    if translation and len(words) >= 8 and len(translation.strip()) < 4:
        score += 2.0
    return round(score, 3)


def risky_row_indices(df, max_ratio=0.1, min_score=3.0):
    if df is None or df.empty:
        return []
    scored = []
    for index, row in df.iterrows():
        score = sentence_risk_score(row.get("Source", ""), row.get("Translation", ""))
        if score >= float(min_score):
            scored.append((index, score))
    limit = max(1, math.ceil(len(df) * float(max_ratio))) if scored else 0
    return [index for index, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:limit]]
