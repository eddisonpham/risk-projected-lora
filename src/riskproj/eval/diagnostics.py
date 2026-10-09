"""Projection diagnostics, computed from archived decision logs only (``09`` §1).

Nothing here touches a live model: every function takes the JSONL rows (or the
:class:`~riskproj.types.DecisionRecord` objects that produced them), so a table can
be rebuilt from a run directory months later. That is also why the numbers are
allowed to be unflattering — the correction rate, the unusable-gradient rate and
the linearization error are diagnostics of the *critic*, and a large
unusable-gradient rate is a finding about the critic rather than a bug to hide.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from riskproj.data.splits import ROLE_TEST_COMBO, ROLE_TEST_RANGE, role_of_cell
from riskproj.types import FallbackCode, VIOLATION_TYPES

__all__ = [
    "decision_frame",
    "depth_sweep_table",
    "linearization_error_table",
    "per_cell_heatmap",
    "split_of_cell",
]

#: Map a fallback code to its ladder label (``01`` §3). Explicit, because
#: ``vars(FallbackCode)`` also holds ``NAMES`` and would silently match nothing.
_FALLBACK_LABEL = {
    FallbackCode.NONE: "none",
    FallbackCode.F1_BASE_ACTION: "F1",
    FallbackCode.F2_HOLD: "F2",
    FallbackCode.F3_TERMINATE: "F3",
}


def _as_dict(record: Any) -> dict[str, Any]:
    """Accept either a dataclass record or the JSON row it was written as."""
    if record is None:
        raise TypeError("record is None")
    if isinstance(record, Mapping):
        return dict(record)
    to_json = getattr(record, "to_json", None)
    if callable(to_json):
        return dict(to_json())
    raise TypeError(f"unsupported record type {type(record).__name__}; expected a mapping or record")


def _finite(values: Sequence[Any]) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _mean_or_nan(values: Sequence[Any]) -> float:
    arr = _finite(values)
    return float(arr.mean()) if arr.size else float("nan")


# --------------------------------------------------------------------------
# decision-level aggregation
# --------------------------------------------------------------------------


def decision_frame(records: Sequence[Any]) -> dict[str, Any]:
    """Aggregate ``agents/09`` §1 diagnostics over decision records.

    Returns the correction rate, the relative correction norm, the
    unusable-gradient rate, the K distribution, the nonlinear-vs-linearized risk
    pair (the E2 quantity) and the F1/F2/F3 fallback rates.
    """
    rows = [_as_dict(r) for r in records]
    n = len(rows)
    if n == 0:
        return {
            "n_decisions": 0,
            "correction_rate": float("nan"),
            "unusable_rate": float("nan"),
            "mean_corr_rel": float("nan"),
            "median_corr_rel": float("nan"),
            "k_histogram": {},
            "mean_r_before": float("nan"),
            "mean_r_after": float("nan"),
            "mean_r_pred": float("nan"),
            "mean_linearization_error": float("nan"),
            "fallback_rates": {"F1": float("nan"), "F2": float("nan"), "F3": float("nan")},
            "n_active": 0,
            "n_unusable": 0,
        }

    active = [bool(r.get("active", False)) for r in rows]
    usable = [bool(r.get("usable", False)) for r in rows]
    n_active = int(sum(active))
    n_unusable = int(sum(1 for a, u in zip(active, usable, strict=True) if a and not u))

    k_values = [int(r.get("k_used") or 0) for r in rows]
    k_hist: dict[str, int] = {}
    for k in k_values:
        k_hist[str(k)] = k_hist.get(str(k), 0) + 1

    # Nonzero correction among active rows: a safe candidate is correctly untouched,
    # so counting it as "not corrected" would understate the correction rate.
    corrected = [bool(u) for a, u in zip(active, usable, strict=True)]
    corr_rel_active = [r.get("corr_rel") for r, c in zip(rows, corrected, strict=True) if c]

    r_before = [r.get("r_cand") for r in rows]
    r_after = [r.get("r_after", [None])[-1] if r.get("r_after") else None for r in rows]
    r_pred = [r.get("r_pred", [None])[-1] if r.get("r_pred") else None for r in rows]
    # Only corrected rows carry a linearization error: where no correction was applied
    # the step is exactly zero, so r(h + 0) = r(h) identically and including those rows
    # would drag the diagnostic toward zero in proportion to the fallback/gating rate.
    # Same reason ``mean_corr_rel`` above is restricted to corrected rows.
    lin_err = [
        (a - p) if c and (a is not None and p is not None) else None
        for a, p, c in zip(r_after, r_pred, corrected, strict=True)
    ]

    fallback_counts = {"F1": 0, "F2": 0, "F3": 0}
    for r in rows:
        label = _FALLBACK_LABEL.get(int(r.get("fallback_code") or 0))
        if label in fallback_counts:
            fallback_counts[label] += 1

    corr = _finite(corr_rel_active)
    return {
        "n_decisions": n,
        "n_active": n_active,
        "n_unusable": n_unusable,
        "correction_rate": float(n_active / n),
        "usable_rate": float(sum(corrected) / n),
        "unusable_rate": float(n_unusable / n) if n_active else 0.0,
        "unusable_rate_of_active": float(n_unusable / n_active) if n_active else float("nan"),
        "mean_corr_rel": float(corr.mean()) if corr.size else float("nan"),
        "median_corr_rel": float(np.median(corr)) if corr.size else float("nan"),
        "max_corr_rel": float(corr.max()) if corr.size else float("nan"),
        "k_histogram": k_hist,
        "mean_k_used": _mean_or_nan(k_values),
        "mean_r_before": _mean_or_nan(r_before),
        "mean_r_after": _mean_or_nan(r_after),
        "mean_r_pred": _mean_or_nan(r_pred),
        "mean_linearization_error": _mean_or_nan(lin_err),
        "fallback_rates": {k: v / n for k, v in fallback_counts.items()},
        "fallback_counts": fallback_counts,
    }


def linearization_error_table(records: Sequence[Any], *, bins: int = 8) -> dict[str, Any]:
    """Figure F1: signed/absolute linearization error against the correction size.

    Binned by ``corr_rel = ||Δh|| / ||h||``, because the first-order model is only
    claimed to be local (`01` §4): the interesting question is where it stops being
    usable, and that is a property of the *step*, not of the batch.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    samples: list[tuple[float, float]] = []
    for raw in records:
        row = _as_dict(raw)
        afters = row.get("r_after") or []
        preds = row.get("r_pred") or []
        if not afters or not preds:
            continue
        corr_rel = row.get("corr_rel")
        if corr_rel is None or not np.isfinite(float(corr_rel)):
            continue
        samples.append((float(corr_rel), float(afters[-1]) - float(preds[-1])))
    if not samples:
        return {"n_samples": 0, "bins": [], "note": "no decision carried both r_after and r_pred"}

    sizes = np.asarray([s[0] for s in samples], dtype=float)
    errors = np.asarray([s[1] for s in samples], dtype=float)
    edges = np.unique(np.quantile(sizes, np.linspace(0.0, 1.0, bins + 1)))
    out: list[dict[str, float]] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (sizes >= lo) & (sizes <= hi if i == len(edges) - 2 else sizes < hi)
        if not mask.any():
            continue
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n": float(mask.sum()),
                "median_step": float(np.median(sizes[mask])),
                "median_signed_error": float(np.median(errors[mask])),
                "median_abs_error": float(np.median(np.abs(errors[mask]))),
                # Positive means the nonlinear risk ended up *below* the linear
                # prediction: the first-order step over-performed.
                "frac_improved_beyond_prediction": float(
                    np.mean(errors[mask] < 0.0)
                ),
            }
        )
    return {
        "n_samples": len(samples),
        "bins": out,
        "sign_convention": (
            "signed error = r_after - r_pred (nonlinear risk minus its first-order "
            "prediction); negative means the nonlinear risk came in below the "
            "prediction, positive means it exceeded it (the first-order model "
            "under-predicted the risk)"
        ),
        "note": "E2 — a first-order region is 'usable' only if the absolute error is small where decisions are made",
    }


# --------------------------------------------------------------------------
# episode-level views
# --------------------------------------------------------------------------


def split_of_cell(cell: str) -> str:
    """Role of a cell, or ``"undesignated"`` — never a guess."""
    try:
        return role_of_cell(cell)
    except ValueError:
        return "undesignated"


def per_cell_heatmap(records: Sequence[Any], *, metric: str = "success") -> dict[str, Any]:
    """Figure F4 data: ``{method: {cell: value}}`` plus each cell's split.

    The split travels with the value so that a caller cannot pool TEST-combo with
    TEST-range without noticing (``09`` §2).
    """
    if metric not in ("success", "violation", "steps", "fallback"):
        raise ValueError(f"unknown metric {metric!r}")
    accum: dict[str, dict[str, list[float]]] = {}
    splits: dict[str, str] = {}
    for raw in records:
        row = _as_dict(raw)
        cell = str(row["cell_id"])
        splits[cell] = str(row.get("split") or split_of_cell(cell))
        value = _episode_metric(row, metric)
        accum.setdefault(str(row["method"]), {}).setdefault(cell, []).append(value)
    table = {
        method: {cell: float(np.mean(vals)) for cell, vals in cells.items()}
        for method, cells in accum.items()
    }
    return {
        "metric": metric,
        "table": table,
        "cell_split": splits,
        "nested": {
            "test_combo": [
                c for c, s in splits.items() if s == ROLE_TEST_COMBO
            ],
            "test_range": [c for c, s in splits.items() if s == ROLE_TEST_RANGE],
        },
        "guard": "TEST-combo and TEST-range are reported separately and never pooled (09 §2)",
    }


def _episode_metric(row: Mapping[str, Any], metric: str) -> float:
    if metric == "success":
        return 1.0 if row.get("success") else 0.0
    if metric == "violation":
        violations = row.get("violations") or {}
        return 1.0 if any(float(v["count"]) > 0 for v in violations.values()) else 0.0
    if metric == "steps":
        return float(row.get("steps", 0))
    fallback = row.get("fallback") or {}
    return 1.0 if sum(int(v) for v in fallback.values()) > 0 else 0.0


def depth_sweep_table(records: Sequence[Any]) -> dict[str, Any]:
    """Figure F5 (ablation A9): per-site success and violation rates.

    Requires the site id to have been recorded with the episode. Raising is
    deliberate: an empty table would look like "no effect" instead of "the records
    did not carry the information".
    """
    rows = [_as_dict(r) for r in records]
    missing = [r for r in rows if "site" not in (r.get("extra") or {})]
    if rows and len(missing) == len(rows):
        raise KeyError(
            "no episode record carries extra['site']; the depth sweep (A9) needs the site "
            "id in the record, not inferred from the method name"
        )
    per_site: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        site = str((r.get("extra") or {}).get("site", "unrecorded"))
        bucket = per_site.setdefault(site, {"success": [], "violation": [], "steps": []})
        bucket["success"].append(1.0 if r.get("success") else 0.0)
        violations = r.get("violations") or {}
        bucket["violation"].append(
            1.0 if any(float(v["count"]) > 0 for v in violations.values()) else 0.0
        )
        bucket["steps"].append(float(r.get("steps", 0)))
    return {
        "per_site": {
            site: {
                "n": len(vals["success"]),
                "success_rate": float(np.mean(vals["success"])) if vals["success"] else float("nan"),
                "violation_rate": float(np.mean(vals["violation"])) if vals["violation"] else float("nan"),
                "mean_steps": float(np.mean(vals["steps"])) if vals["steps"] else float("nan"),
            }
            for site, vals in sorted(per_site.items())
        },
        "n_records": len(rows),
        "n_records_without_site": len(missing),
        "violation_types": list(VIOLATION_TYPES),
    }


def pooled_violation_rates(records: Sequence[Any]) -> dict[str, float]:
    """Per-type violation rates over archived episodes (offline recomputable)."""
    rows = [_as_dict(r) for r in records]
    if not rows:
        return dict.fromkeys(VIOLATION_TYPES, float("nan"))
    return {
        v: float(np.mean([float(r["violations"][v]["count"]) > 0.0 for r in rows]))
        for v in VIOLATION_TYPES
    }
