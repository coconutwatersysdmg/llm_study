"""Evidence formatting and deterministic order shuffling."""

from __future__ import annotations

import random
from typing import Any


def format_evidence_block(records: list[dict[str, Any]]) -> str:
    """Stable textual serialization for prompts (field order preserved)."""
    lines: list[str] = []
    for i, rec in enumerate(records, start=1):
        parts = [f"{k}={rec[k]}" for k in rec.keys()]
        lines.append(f"[{i}] " + ", ".join(parts))
    return "\n".join(lines)


def shuffle_pair(
    a: dict[str, Any],
    b: dict[str, Any],
    question_id: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically shuffle a conflict pair by question_id + seed."""
    rng = random.Random(f"{seed}:{question_id}:pair")
    pair = [a, b]
    rng.shuffle(pair)
    return pair


def sample_distractors(
    pool: list[dict[str, Any]],
    exclude_keys: set[tuple[Any, ...]],
    key_attrs: tuple[str, ...],
    question_id: str,
    seed: int,
    k_min: int = 2,
    k_max: int = 4,
) -> list[dict[str, Any]]:
    """Sample 2–4 non-conflicting distractor tuples (deterministic)."""
    rng = random.Random(f"{seed}:{question_id}:dist")
    candidates = []
    for rec in pool:
        key = tuple(rec[a] for a in key_attrs)
        if key in exclude_keys:
            continue
        candidates.append(rec)
    if not candidates:
        return []
    k = min(len(candidates), rng.randint(k_min, k_max))
    return rng.sample(candidates, k)


def same_field_order(
    records: list[dict[str, Any]],
    field_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [{a: rec[a] for a in field_order} for rec in records]
