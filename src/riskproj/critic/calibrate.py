"""Critic calibration and the VAL-only choice of tau (``06`` §3-§6).

``tau`` is selected **only here**, on VAL cells, by a predeclared rule, and then
frozen before any TEST episode (``CLAUDE.md`` #2). Nothing in this module may be
called with TEST data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from riskproj.utils.metrics import (
    ece,
    fnr_at_threshold,
    precision_recall_at_threshold,
    reliability_curve,
)

#: Temperature bounds; outside this range calibration is a sign of a broken critic,
#: not something to paper over.
T_MIN, T_MAX = 1e-2, 1e2


def _logits(logits: Any) -> Tensor:
    t = logits if isinstance(logits, Tensor) else torch.as_tensor(np.asarray(logits))
    t = t.detach().float().reshape(-1)
    if t.numel() == 0:
        raise ValueError("empty logits")
    return t


def _targets(targets: Any, n: int) -> Tensor:
    y = targets if isinstance(targets, Tensor) else torch.as_tensor(np.asarray(targets))
    y = y.detach().float().reshape(-1)
    if y.numel() != n:
        raise ValueError(f"targets length {y.numel()} != logits length {n}")
    if not torch.all((y == 0) | (y == 1)):
        raise ValueError("targets must be binary (0/1)")
    return y


def apply_temperature(logits: Any, temperature: float) -> Tensor:
    """Rescale logits by ``1/T`` (``06`` §3)."""
    if not (0.0 < float(temperature) < float("inf")):
        raise ValueError(f"temperature must be > 0; got {temperature}")
    return _logits(logits) / float(temperature)


def fit_temperature(
    logits: Any, targets: Any, *, grid_points: int = 61, refine_iters: int = 60
) -> float:
    """Fit a single temperature by minimising VAL BCE (Guo et al. 2017 [R]).

    A coarse grid gives a safe initialisation (LBFGS alone can run to a saturated
    solution on small VAL sets), then LBFGS refines it in log space.
    """
    z = _logits(logits)
    y = _targets(targets, z.numel())
    if float(y.mean()) in (0.0, 1.0):
        # A single class carries no calibration signal; T=1 is the honest answer.
        return 1.0

    def nll(log_t: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(z / log_t.exp().clamp(T_MIN, T_MAX), y)

    with torch.no_grad():
        grid = torch.exp(torch.linspace(np.log(0.05), np.log(20.0), grid_points))
        losses = torch.stack([nll(t) for t in grid])
        best = grid[int(torch.argmin(losses))]

    log_t = torch.nn.Parameter(torch.log(best).clone().detach())
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=refine_iters, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        opt.zero_grad(set_to_none=True)
        loss = nll(log_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.detach().exp(), T_MIN, T_MAX))


def reliability(logits: Any, targets: Any, n_bins: int = 15) -> dict[str, list[float]]:
    """Reliability curve on sigmoid(logits) (``09`` §1, F8)."""
    z = _logits(logits)
    y = _targets(targets, z.numel())
    return reliability_curve(y.numpy(), torch.sigmoid(z).numpy(), n_bins=n_bins)


def ece_before_after(
    logits: Any, targets: Any, temperature: float, n_bins: int = 15
) -> dict[str, float]:
    """ECE/NLL/Brier before and after temperature scaling (``06`` §3)."""
    z = _logits(logits)
    y = _targets(targets, z.numel())
    z_after = apply_temperature(z, temperature)
    return {
        "temperature": float(temperature),
        "ece_before": float(ece(y.numpy(), torch.sigmoid(z).numpy(), n_bins)),
        "ece_after": float(ece(y.numpy(), torch.sigmoid(z_after).numpy(), n_bins)),
        "nll_before": float(F.binary_cross_entropy_with_logits(z, y)),
        "nll_after": float(F.binary_cross_entropy_with_logits(z_after, y)),
        "brier_before": float(((torch.sigmoid(z) - y) ** 2).mean()),
        "brier_after": float(((torch.sigmoid(z_after) - y) ** 2).mean()),
        "n": float(z.numel()),
    }


def choose_tau(probs: Any, targets: Any, spec: dict[str, Any] | None = None) -> dict[str, float]:
    """Pick ``tau`` on VAL by the predeclared rule (``06`` §6).

    Rule: among thresholds whose false-negative rate is within the cap, keep the
    one with the highest precision; break ties by the smaller predicted-positive
    rate, then by the smaller ``tau``. If nothing satisfies the cap the result is
    explicitly ``feasible=False`` with the best-effort threshold reported, because
    a silently relaxed cap would be a protocol change.

    ``spec`` keys: ``fnr_cap`` (default 0.15, matching G3), ``min_precision``
    (default 0.0), ``n_grid`` (default 199), ``tau_lo``/``tau_hi``.
    """
    spec = dict(spec or {})
    fnr_cap = float(spec.get("fnr_cap", 0.15))
    min_precision = float(spec.get("min_precision", 0.0))
    n_grid = int(spec.get("n_grid", 199))
    tau_lo = float(spec.get("tau_lo", 0.01))
    tau_hi = float(spec.get("tau_hi", 0.99))
    if not (0.0 <= tau_lo < tau_hi <= 1.0):
        raise ValueError(f"require 0 <= tau_lo < tau_hi <= 1; got {tau_lo}, {tau_hi}")

    p = np.asarray(probs, dtype=float).reshape(-1)
    y = np.asarray(targets, dtype=float).reshape(-1)
    if p.shape != y.shape:
        raise ValueError(f"probs {p.shape} and targets {y.shape} differ")
    if p.size == 0:
        raise ValueError("empty input")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("probs must lie in [0,1] (pass sigmoid(logits), not logits)")
    n_pos = int((y == 1).sum())

    if n_pos == 0:
        return {
            "tau": tau_lo,
            "feasible": 0.0,
            "precision": float("nan"),
            "recall": float("nan"),
            "fnr": float("nan"),
            "predicted_positive_rate": 0.0,
            "n_candidates": 0.0,
            "n_positives": 0.0,
            "n": float(p.size),
            "reason": 1.0,  # no positives: nothing to detect
        }

    candidates = np.unique(
        np.concatenate([np.quantile(p, np.linspace(0.0, 1.0, n_grid)), [tau_lo, tau_hi]])
    )
    candidates = candidates[(candidates >= tau_lo) & (candidates <= tau_hi)]
    rows = []
    for t in candidates:
        pr = precision_recall_at_threshold(y, p, float(t))
        recall = pr["recall"]
        rows.append(
            {
                "tau": float(t),
                "precision": float(pr["precision"]),
                "recall": float(recall),
                "fnr": float(1.0 - recall),
                "predicted_positive_rate": float((p >= t).mean()),
            }
        )

    feasible = [
        r
        for r in rows
        if np.isfinite(r["fnr"])
        and r["fnr"] <= fnr_cap
        and (r["precision"] >= min_precision if np.isfinite(r["precision"]) else False)
    ]
    if not feasible:
        best = min(
            rows, key=lambda r: (r["fnr"] if np.isfinite(r["fnr"]) else np.inf, -r["precision"])
        )
        return {
            **best,
            "feasible": 0.0,
            "n_candidates": float(len(rows)),
            "n_positives": float(n_pos),
            "n": float(p.size),
            "reason": 2.0,  # FNR cap unreachable within the tau range
        }

    chosen = max(
        feasible,
        key=lambda r: (r["precision"], -r["predicted_positive_rate"], -r["tau"]),
    )
    return {
        **chosen,
        "feasible": 1.0,
        "n_candidates": float(len(rows)),
        "n_positives": float(n_pos),
        "n": float(p.size),
        "reason": 0.0,
    }


def fnr_curve(probs: Any, targets: Any, taus: Any) -> dict[str, list[float]]:
    """FNR and precision for an explicit grid of thresholds (used by the tau report)."""
    p = np.asarray(probs, dtype=float).reshape(-1)
    y = np.asarray(targets, dtype=float).reshape(-1)
    grid = np.asarray(taus, dtype=float).reshape(-1)
    return {
        "tau": [float(t) for t in grid],
        "fnr": [float(fnr_at_threshold(y, p, float(t))) for t in grid],
        "precision": [
            float(precision_recall_at_threshold(y, p, float(t))["precision"]) for t in grid
        ],
    }


__all__ = [
    "T_MAX",
    "T_MIN",
    "apply_temperature",
    "choose_tau",
    "ece_before_after",
    "fit_temperature",
    "fnr_curve",
    "reliability",
]
