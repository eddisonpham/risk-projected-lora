"""G3 acceptance gate for the risk critic (``06`` §5).

Declared in ``configs/gates.yaml`` **before** any policy evaluation. A failed gate
means "fix data/labels/features" — never "tune the method around a bad critic".

Every sub-check is recorded with its number, including the ones that pass, and
per-cell results are reported individually: a pooled AUROC that hides a broken
cell is exactly the failure mode this gate exists to prevent (``09`` §5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from riskproj.utils.config import CriticGateCfg
from riskproj.utils.io import read_json, utc_stamp, write_json
from riskproj.utils.metrics import (
    auprc,
    auroc,
    ece,
    fnr_at_threshold,
    precision_recall_at_threshold,
)

GATE_G3 = "G3"


@dataclass
class GateReport:
    """Outcome of one gate, with the numbers that produced it."""

    gate: str
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateReport:
        return cls(
            gate=str(data["gate"]),
            passed=bool(data["passed"]),
            metrics={str(k): float(v) for k, v in dict(data.get("metrics", {})).items()},
            failures=[str(x) for x in data.get("failures", [])],
            notes=[str(x) for x in data.get("notes", [])],
        )

    def summary(self) -> str:
        head = f"{self.gate}: {'PASS' if self.passed else 'FAIL'}"
        return head + ("" if self.passed else f" ({len(self.failures)} failures)")


def _vec(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def _finite(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def evaluate_g3(
    probs: Any,
    labels: Any,
    *,
    tau: float,
    cfg_gate: CriticGateCfg,
    cells: Sequence[str] | None = None,
    per_type_probs: Mapping[str, Any] | None = None,
    per_type_labels: Mapping[str, Any] | None = None,
    val_disagreement: Any | None = None,
    train_disagreement: Any | None = None,
) -> GateReport:
    """Evaluate the G3 acceptance criteria on VAL (held-out combos).

    ``probs`` must be *any-violation* probabilities in ``[0, 1]`` (post
    temperature scaling), ``labels`` the binary any-violation targets. Per-type and
    per-cell breakdowns are required by ``06`` §5 and are reported separately;
    cells with a single class present are noted rather than averaged away.
    """
    p = _vec(probs, name="probs")
    y = _vec(labels, name="labels")
    if p.shape != y.shape:
        raise ValueError(f"probs {p.shape} and labels {y.shape} differ")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("probs must lie in [0,1]; pass sigmoid(logits), not logits")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be binary (0/1)")
    if not (0.0 < float(tau) < 1.0):
        raise ValueError(f"tau must be in (0,1); got {tau}")

    failures: list[str] = []
    notes: list[str] = []

    n = int(p.size)
    n_pos = int((y == 1).sum())
    base_rate = float(y.mean())
    auroc_any = _finite(auroc(y, p))
    auprc_any = _finite(auprc(y, p))
    ece_any = _finite(ece(y, p))
    fnr_any = _finite(fnr_at_threshold(y, p, tau))
    pr = precision_recall_at_threshold(y, p, tau)

    metrics: dict[str, float] = {
        "n": float(n),
        "n_positives": float(n_pos),
        "base_rate": base_rate,
        "auroc": auroc_any,
        "auroc_required": float(cfg_gate.auroc),
        "auprc": auprc_any,
        "auprc_required": float(cfg_gate.auprc_mult) * base_rate,
        "ece": ece_any,
        "ece_allowed": float(cfg_gate.ece),
        "fnr_at_tau": fnr_any,
        "fnr_allowed": float(cfg_gate.fnr_at_tau),
        "precision_at_tau": _finite(pr["precision"]),
        "recall_at_tau": _finite(pr["recall"]),
        "tau": float(tau),
        "predicted_positive_rate": float((p >= tau).mean()),
    }

    if n_pos == 0 or n_pos == n:
        failures.append(
            f"VAL has a single class (n_pos={n_pos} of {n}); the critic cannot be validated "
            "on this split"
        )
        return GateReport(GATE_G3, False, metrics, failures, notes)

    if not (auroc_any >= cfg_gate.auroc):
        failures.append(f"AUROC {auroc_any:.4f} < required {cfg_gate.auroc:.4f}")
    required_auprc = cfg_gate.auprc_mult * base_rate
    if not (auprc_any >= required_auprc):
        failures.append(
            f"AUPRC {auprc_any:.4f} < required {required_auprc:.4f} "
            f"({cfg_gate.auprc_mult:.1f}x base rate {base_rate:.4f})"
        )
    if not (ece_any <= cfg_gate.ece):
        failures.append(f"ECE {ece_any:.4f} > allowed {cfg_gate.ece:.4f}")
    if not (fnr_any <= cfg_gate.fnr_at_tau):
        failures.append(f"FNR@{tau:.3f} {fnr_any:.4f} > allowed {cfg_gate.fnr_at_tau:.4f}")

    # -- per-type breakdown (06 §5: FNR capped for each critical type) ----
    if (per_type_probs is None) != (per_type_labels is None):
        raise ValueError("per_type_probs and per_type_labels must be supplied together")
    if per_type_probs is not None and per_type_labels is not None:
        if set(per_type_probs) != set(per_type_labels):
            raise ValueError(
                f"per-type keys differ: {sorted(per_type_probs)} vs {sorted(per_type_labels)}"
            )
        for name in sorted(per_type_probs):
            yt = _vec(per_type_labels[name], name=f"per_type_labels[{name}]")
            pt = _vec(per_type_probs[name], name=f"per_type_probs[{name}]")
            if yt.shape != pt.shape:
                raise ValueError(f"per-type {name}: probs {pt.shape} vs labels {yt.shape}")
            if int(yt.sum()) == 0:
                metrics[f"{name}_auroc"] = float("nan")
                metrics[f"{name}_fnr_at_tau"] = float("nan")
                notes.append(
                    f"{name}: no positives on VAL, so its AUROC/FNR cannot be verified "
                    "(reported as nan, not assumed fine)"
                )
                continue
            a_type = _finite(auroc(yt, pt))
            f_type = _finite(fnr_at_threshold(yt, pt, tau))
            metrics[f"{name}_auroc"] = a_type
            metrics[f"{name}_fnr_at_tau"] = f_type
            if not (f_type <= cfg_gate.fnr_at_tau):
                failures.append(
                    f"{name}: FNR@{tau:.3f} {f_type:.4f} > allowed {cfg_gate.fnr_at_tau:.4f}"
                )

    # -- per-cell breakdown (no cell may hide behind the pooled AUROC) ----
    if cells is not None:
        cells_arr = np.asarray(list(cells), dtype=object).reshape(-1)
        if cells_arr.size != n:
            raise ValueError(f"cells has {cells_arr.size} entries but there are {n} rows")
        for cell in sorted({str(c) for c in cells_arr}):
            mask = cells_arr == cell
            yc, pc = y[mask], p[mask]
            n_pos_cell = int(yc.sum())
            if n_pos_cell == 0 or n_pos_cell == yc.size:
                metrics[f"cell::{cell}::auroc"] = float("nan")
                notes.append(
                    f"cell {cell}: single class (n={yc.size}, n_pos={n_pos_cell}); no cell AUROC"
                )
                continue
            a_cell = _finite(auroc(yc, pc))
            metrics[f"cell::{cell}::auroc"] = a_cell
            metrics[f"cell::{cell}::n"] = float(yc.size)
            if not (a_cell >= cfg_gate.min_cell_auroc):
                failures.append(
                    f"cell {cell}: AUROC {a_cell:.4f} < min {cfg_gate.min_cell_auroc:.4f} "
                    f"(n={yc.size}, n_pos={n_pos_cell})"
                )

    # -- ensemble disagreement should carry OOD signal (06 §5) -----------
    if cfg_gate.require_val_disagreement_gt_train:
        if val_disagreement is None or train_disagreement is None:
            failures.append(
                "ensemble-disagreement check is required by config but disagreement values "
                "were not supplied; G3 cannot be declared passed without it"
            )
        else:
            vd = float(np.mean(_vec(val_disagreement, name="val_disagreement")))
            td = float(np.mean(_vec(train_disagreement, name="train_disagreement")))
            metrics["val_disagreement"] = vd
            metrics["train_disagreement"] = td
            if not (vd > td):
                failures.append(
                    f"VAL ensemble disagreement {vd:.4f} is not above TRAIN {td:.4f}; the "
                    "critic carries no OOD signal"
                )
    elif val_disagreement is not None and train_disagreement is not None:
        metrics["val_disagreement"] = float(
            np.mean(_vec(val_disagreement, name="val_disagreement"))
        )
        metrics["train_disagreement"] = float(
            np.mean(_vec(train_disagreement, name="train_disagreement"))
        )
        notes.append("disagreement check disabled in config; values recorded for the report only")

    if n_pos < 10:
        notes.append(f"low VAL support: only {n_pos} positives")

    return GateReport(GATE_G3, not failures, metrics, failures, notes)


def gate_status_report(reports: Sequence[GateReport], path: str | Path) -> dict[str, Any]:
    """Write/return the gate status file consumed by the roadmap and reports."""
    payload = {
        "updated_at": utc_stamp(),
        "all_passed": bool(reports) and all(r.passed for r in reports),
        "gates": {r.gate: r.to_dict() for r in reports},
    }
    write_json(path, payload)
    return payload


def load_gate_reports(path: str | Path) -> list[GateReport]:
    """Reload gate reports; a missing file is an error (no implicit "no gates yet")."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no gate status file at {p}; run the gate evaluation first")
    data = read_json(p)
    return [GateReport.from_dict(v) for v in data.get("gates", {}).values()]


__all__ = ["GATE_G3", "GateReport", "evaluate_g3", "gate_status_report", "load_gate_reports"]
