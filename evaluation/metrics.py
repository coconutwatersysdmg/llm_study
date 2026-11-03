"""Metric helpers for Exp1 / Exp2."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def accuracy(correct_flags: Sequence[bool]) -> float:
    if not correct_flags:
        return 0.0
    return sum(1 for x in correct_flags if x) / len(correct_flags)


def rate(flags: Sequence[bool]) -> float:
    return accuracy(flags)


def conflict_prf(
    y_true_conflict: Sequence[bool],
    y_pred_conflict: Sequence[bool],
) -> dict[str, float]:
    tp = fp = fn = 0
    for yt, yp in zip(y_true_conflict, y_pred_conflict):
        if yp and yt:
            tp += 1
        elif yp and not yt:
            fp += 1
        elif (not yp) and yt:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "conflict_precision": precision,
        "conflict_recall": recall,
        "conflict_f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def group_by(rows: Iterable[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out
