"""Answer normalization — deterministic, no LLM judge."""

from __future__ import annotations

import re
from typing import Any, Optional


SPECIAL_TOKENS = {"UNKNOWN", "CONFLICT", "REJECTED_INVALID_CANDIDATE"}


def normalize_answer(text: str) -> tuple[str, bool]:
    """Return (normalized_text, format_violation).

    format_violation=True if a special token was buried in extra text.
    """
    if text is None:
        return "", False
    s = str(text).strip()
    # Strip markdown fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:\w+)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    # Strip common prefixes
    s = re.sub(r"^(Answer|答案|回答)\s*[:：]\s*", "", s, flags=re.IGNORECASE).strip()

    upper = s.upper()
    format_violation = False
    for tok in SPECIAL_TOKENS:
        if upper == tok:
            return tok, False
        # First-token extraction for CONFLICT / UNKNOWN with extra text
        first = re.split(r"[\s,，。；;:!！？\n]+", s, maxsplit=1)[0].strip().upper()
        if first == tok:
            format_violation = len(s.strip()) > len(tok)
            return tok, format_violation

    # Numeric normalization later at compare time; keep string here.
    return s, False


def answers_equal(pred: str, gold: Any) -> bool:
    """Strict semantic compare after light normalization."""
    if pred in SPECIAL_TOKENS:
        return False
    p = pred.strip()
    g = str(gold).strip()

    # Case-insensitive string compare
    if p.lower() == g.lower():
        return True

    # Numeric compare if gold is int-like
    try:
        g_num = float(g)
        p_num = float(p)
        if float(int(g_num)) == g_num:  # gold is integer-valued
            return abs(p_num - g_num) < 1e-9
        return abs(p_num - g_num) < 1e-9
    except (TypeError, ValueError):
        return False


def classify_special(normalized: str) -> Optional[str]:
    if normalized in SPECIAL_TOKENS:
        return normalized
    return None
