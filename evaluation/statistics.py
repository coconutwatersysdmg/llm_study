"""Statistical tests for paired condition comparisons."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def bootstrap_ci(
    flags: Sequence[bool],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(flags, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        samples.append(arr[idx].mean())
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


def paired_bootstrap_diff(
    a_flags: Sequence[bool],
    b_flags: Sequence[bool],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for accuracy(a) - accuracy(b) on paired items."""
    a = np.asarray(a_flags, dtype=float)
    b = np.asarray(b_flags, dtype=float)
    if a.size != b.size or a.size == 0:
        return {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, a.size, size=a.size)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs_arr = np.asarray(diffs)
    return {
        "diff": float(a.mean() - b.mean()),
        "ci_low": float(np.quantile(diffs_arr, alpha / 2)),
        "ci_high": float(np.quantile(diffs_arr, 1 - alpha / 2)),
    }


def mcnemar_test(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
) -> dict[str, float]:
    """McNemar exact/binomial test on paired correctness."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if a.size != b.size or a.size == 0:
        return {"b01": 0.0, "b10": 0.0, "p_value": 1.0}
    b01 = int(np.sum(~a & b))  # a wrong, b correct
    b10 = int(np.sum(a & ~b))  # a correct, b wrong
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        # two-sided exact binomial under p=0.5
        p = float(stats.binomtest(b10, n, 0.5, alternative="two-sided").pvalue)
    return {"b01": float(b01), "b10": float(b10), "p_value": p}
