"""Metric primitives, dependency-free (numpy only).

Lives in ``utils`` because both ``critic/gates.py`` (G3) and ``eval/stats.py``
need them, and ``eval`` sits *above* ``critic`` in the dependency order. Anything
aggregate/CI-heavy (IQM, performance profiles, stratified bootstrap) belongs in
``eval/stats.py`` on top of ``rliable``.

Every estimator here is written to be auditable rather than fast: AUROC by the
rank formula, ECE by equal-width bins, Wilson by the closed form.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "auprc",
    "auroc",
    "bootstrap_ci",
    "class_balance",
    "ece",
    "fisher_exact",
    "fnr_at_threshold",
    "holm_bonferroni",
    "iqm",
    "mcnemar_exact",
    "precision_recall_at_threshold",
    "reliability_curve",
    "two_proportion_ci_diff",
    "wilson_interval",
]


def _as_arrays(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    s = np.asarray(y_score, dtype=float).reshape(-1)
    if y.shape != s.shape:
        raise ValueError(f"shapes differ: y_true {y.shape} vs y_score {s.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    return y, s


def auroc(y_true: Any, y_score: Any) -> float:
    """Rank-based AUROC (Mann-Whitney U). Ties get average ranks."""
    y, s = _as_arrays(y_true, y_score)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average tied ranks
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def auprc(y_true: Any, y_score: Any) -> float:
    """Average precision (step-wise, the standard AUPRC estimator)."""
    y, s = _as_arrays(y_true, y_score)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / n_pos
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev_recall) * precision))


def reliability_curve(y_true: Any, y_prob: Any, n_bins: int = 15) -> dict[str, list[float]]:
    y, p = _as_arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        counts.append(float(mask.sum()))
        mean_pred.append(float(p[mask].mean()) if mask.any() else float("nan"))
        frac_pos.append(float(y[mask].mean()) if mask.any() else float("nan"))
    return {
        "bin_edges": [float(e) for e in edges],
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
        "counts": counts,
    }


def ece(y_true: Any, y_prob: Any, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins (Guo et al. 2017 [R])."""
    y, p = _as_arrays(y_true, y_prob)
    rc = reliability_curve(y, p, n_bins)
    counts = np.asarray(rc["counts"], dtype=float)
    mean_pred = np.asarray(rc["mean_pred"], dtype=float)
    frac_pos = np.asarray(rc["frac_pos"], dtype=float)
    valid = counts > 0
    if not valid.any():
        return float("nan")
    return float(
        np.sum(counts[valid] / counts[valid].sum() * np.abs(mean_pred[valid] - frac_pos[valid]))
    )


def precision_recall_at_threshold(y_true: Any, y_score: Any, threshold: float) -> dict[str, float]:
    y, s = _as_arrays(y_true, y_score)
    pred = s >= threshold
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall))
        if precision + recall > 0
        else float("nan"),
    }


def fnr_at_threshold(y_true: Any, y_score: Any, threshold: float) -> float:
    """False-negative rate: P(score < threshold | violation). G3 caps this at tau."""
    recall = precision_recall_at_threshold(y_true, y_score, threshold)["recall"]
    return float(1.0 - recall) if not math.isnan(recall) else float("nan")


def class_balance(y_true: Any) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    if y.size == 0:
        return {"n": 0.0, "pos_rate": float("nan")}
    return {"n": float(y.size), "pos_rate": float((y == 1).mean())}


def wilson_interval(n_success: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no continuity correction)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = _z(alpha)
    p = n_success / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_ci_diff(
    n_succ_a: int, n_a: int, n_succ_b: int, n_b: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Newcombe/Wilson-style interval for ``p_a - p_b`` (used for per-cell success deltas)."""
    lo_a, hi_a = wilson_interval(n_succ_a, n_a, alpha)
    lo_b, hi_b = wilson_interval(n_succ_b, n_b, alpha)
    p_a = n_succ_a / n_a if n_a else float("nan")
    p_b = n_succ_b / n_b if n_b else float("nan")
    diff = p_a - p_b
    lower = diff - math.sqrt((p_a - lo_a) ** 2 + (hi_b - p_b) ** 2)
    upper = diff + math.sqrt((hi_a - p_a) ** 2 + (p_b - lo_b) ** 2)
    return (lower, upper)


def _z(alpha: float) -> float:
    # two-sided normal quantile via the inverse error function
    return float(math.sqrt(2.0) * _erfinv(1.0 - alpha))


def _erfinv(x: float) -> float:
    # Winitzki's approximation is plenty accurate for CI work (|err| < 1e-4).
    a = 0.147
    ln = math.log(1.0 - x * x)
    t = 2.0 / (math.pi * a) + ln / 2.0
    return float(math.copysign(math.sqrt(max(math.sqrt(t * t - ln / a) - t, 0.0)), x))


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    statistic: str = "mean",
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI over a 1-D sample (e.g. per-episode paired deltas)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0.0}
    fn = np.mean if statistic == "mean" else np.median
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    samples = fn(arr[idx], axis=1)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(fn(arr)), "lo": float(lo), "hi": float(hi), "n": float(arr.size)}


def mcnemar_exact(b: int, c: int) -> dict[str, float]:
    """Exact McNemar test for paired binary outcomes.

    ``b`` = discordant pairs where A succeeded and B failed; ``c`` the reverse.
    """
    n = b + c
    if n == 0:
        return {"n_discordant": 0.0, "p_value": 1.0, "odds_ratio": float("nan")}
    k = min(b, c)
    p = min(1.0, 2.0 * sum(math.comb(n, i) * 0.5**n for i in range(k + 1)))
    return {
        "n_discordant": float(n),
        "p_value": float(p),
        "odds_ratio": (b / c) if c else float("inf"),
    }


def fisher_exact(a: int, b: int, c: int, d: int) -> dict[str, float]:
    """Two-sided Fisher exact test on the 2x2 table [[a, b], [c, d]] (hypergeometric)."""
    from scipy.stats import fisher_exact as _scipy_fisher

    res = _scipy_fisher([[a, b], [c, d]], alternative="two-sided")
    return {"odds_ratio": float(res.statistic), "p_value": float(res.pvalue)}


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float]]:
    """Holm-Bonferroni step-down over the *pre-registered* primary comparisons only."""
    items = sorted(
        p_values.items(), key=lambda kv: (math.inf if math.isnan(kv[1]) else kv[1], kv[0])
    )
    m = len(items)
    out: dict[str, dict[str, float]] = {}
    rejected_so_far = True
    for i, (name, p) in enumerate(items):
        threshold = alpha / (m - i)
        rejected = bool(rejected_so_far and p <= threshold)
        rejected_so_far = rejected
        out[name] = {"p": float(p), "threshold": float(threshold), "reject": float(rejected)}
    return out


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: the aggregate rliable recommends over the plain mean."""
    arr = np.sort(np.asarray(list(values), dtype=float))
    if arr.size == 0:
        return float("nan")
    if arr.size < 4:
        return float(arr.mean())
    lo, hi = int(np.floor(0.25 * arr.size)), int(np.ceil(0.75 * arr.size))
    return float(arr[lo:hi].mean())
