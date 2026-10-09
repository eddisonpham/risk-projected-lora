"""Critic metrics table T1, the proxy-gold curve and the claim guard (``09`` §1, §4).

The critic is a **proxy**. Everything here either measures how good the proxy is on
data it did not train on, or refuses a claim the table does not support:

* :func:`critic_table` builds T1: discrimination, calibration, FNR at the operating
  point, per-type and per-cell breakdowns, ensemble disagreement against error, and
  the input-gradient diagnostics that decide whether the projection has anything to
  descend (``06`` §4).
* :func:`proxy_gold_curve` builds F3 / ablation A8 from **archived decision logs**
  only: critic-risk reduction against measured-violation reduction as the correction
  norm grows, including the point where the two diverge (critic exploitation).
* :func:`critic_claim_guard` lists the statements the table does *not* license
  (``12``), so a report cannot quietly turn "AUROC on violation labels" into "the
  critic predicts success".

Calibration is a VAL-only operation: the temperature is fitted on the split it is
reported for, and running this table on a sealed split without a frozen temperature
raises, because fitting a temperature on TEST is tuning on TEST (``CLAUDE.md`` #2).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from riskproj.critic.calibrate import fit_temperature
from riskproj.critic.train import gradient_diagnostics
from riskproj.data.splits import SEALED_ROLES
from riskproj.types import VIOLATION_TYPES
from riskproj.utils.io import write_json
from riskproj.utils.metrics import (
    auprc,
    auroc,
    ece,
    fnr_at_threshold,
    precision_recall_at_threshold,
    reliability_curve,
)

#: Keys for "did the *corrected* chunk violate" in a decision record.
MEASURED_VIOLATION_KEYS: tuple[str, ...] = (
    "measured_violation",
    "corrected_violation",
    "violation_after",
)
#: Keys for "did the *original* chunk violate", needed for a paired gold reduction.
#: ``true_violation_next_H`` is the label of the executed chunk from ``09`` §3, which
#: for a counterfactual pair is the original one.
ORIGINAL_VIOLATION_KEYS: tuple[str, ...] = (
    "original_violation",
    "violation_before",
    "true_violation_next_H",
)
#: Keys accepted for the critic-risk reduction achieved by the correction.
CRITIC_REDUCTION_KEYS: tuple[str, ...] = ("critic_risk_reduction", "r_reduction")


# --------------------------------------------------------------------------
# dataset / critic plumbing
# --------------------------------------------------------------------------


@dataclass
class CriticBatch:
    """Feature matrix and labels pulled out of whatever dataset object was passed."""

    features: Tensor  # [N, D]
    y_any: Tensor  # [N]
    y_type: Tensor  # [N, 5]
    cell_ids: list[str] | None
    n_skipped: int = 0


def _as_features(item: Mapping[str, Any] | Sequence[Any], critic: Any) -> tuple[Tensor, Tensor, Tensor]:
    """Normalise one dataset item to ``(features [D], any, per_type [5])``.

    Three shapes are accepted because D-crit storage and the training loader differ:
    a mapping with ``features``, a 3-tuple ``(features, any, per_type)`` (what
    ``critic.train`` consumes) and a 4-tuple ``(ctx, chunk, any, per_type)`` (what
    ``data.schema.CriticDataset`` yields, where features are built on the fly).
    """
    if isinstance(item, Mapping):
        if "features" in item:
            feats = item["features"]
        elif "ctx" in item and "chunk" in item:
            feats = critic.builder(item["ctx"], _batched_chunk(item["chunk"]))
        else:
            raise KeyError(
                "critic sample mapping must carry 'features' or ('ctx' and 'chunk'); "
                f"got keys {sorted(item)}"
            )
        y_any = item.get("any_violation", item.get("y_any"))
        y_type = item.get("per_type", item.get("y_type", item.get("violations")))
    elif isinstance(item, (tuple, list)):
        if len(item) == 3:
            feats, y_any, y_type = item
        elif len(item) == 4:
            ctx, chunk, y_any, y_type = item
            feats = critic.builder(ctx, _batched_chunk(chunk))
        else:
            raise ValueError(
                f"critic sample tuple must have 3 or 4 entries; got {len(item)}"
            )
    else:
        raise TypeError(f"unsupported critic sample type {type(item).__name__}")

    if y_any is None or y_type is None:
        raise KeyError("critic sample must carry an any-violation label and per-type labels")
    feats_t = torch.as_tensor(feats, dtype=torch.float32).reshape(-1)
    y_type_t = torch.as_tensor(y_type, dtype=torch.float32).reshape(-1)
    if y_type_t.numel() != len(VIOLATION_TYPES):
        raise ValueError(
            f"per-type labels must have {len(VIOLATION_TYPES)} entries; got {y_type_t.numel()}"
        )
    return feats_t, torch.tensor(float(y_any), dtype=torch.float32), y_type_t


def _batched_chunk(chunk: Any) -> Any:
    """Promote a single sample's ``[H, A]`` chunk to the ``[B, H, A]`` the builder wants.

    ``CriticDataset`` yields samples, not batches, so the chunk arrives unbatched; the
    feature builder only speaks batch-first (``critic/features.py``). Reshaping here
    keeps that precondition in one place instead of at every call site.
    """
    ndim = getattr(chunk, "ndim", None)
    if ndim == 2:
        return chunk[None, ...]
    if ndim == 3:
        return chunk
    raise ValueError(
        f"a critic chunk must be [H, A] or [B, H, A]; got shape {getattr(chunk, 'shape', None)}"
    )


def critic_features_from_dataset(
    dataset: Sequence[Any], critic: Any, *, max_samples: int | None = None
) -> CriticBatch:
    """Materialise ``(features, labels, cell ids)`` for a whole dataset.

    The feature matrix is built once and reused for every metric, so the critic's
    forward pass is not repeated per metric. ``max_samples`` subsamples the *head* of
    the dataset — use it for a quick look, never for a reported number, since D-crit
    is not shuffled.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("critic dataset is empty")
    if max_samples is not None:
        n = min(n, int(max_samples))
    feats: list[Tensor] = []
    anys: list[Tensor] = []
    types: list[Tensor] = []
    for i in range(n):
        f, a, t = _as_features(dataset[i], critic)
        feats.append(f)
        anys.append(a)
        types.append(t)
    width = int(critic.feature_spec.dim)
    stacked = torch.stack(feats)
    if stacked.shape[1] != width:
        raise ValueError(
            f"dataset features have width {stacked.shape[1]} but the critic was trained on "
            f"{width}; regenerate D-crit or load the matching checkpoint"
        )
    cell_ids: list[str] | None = None
    tuples = getattr(dataset, "tuples", None)
    if tuples is not None:
        ids = [getattr(t, "cell_id", None) for t in tuples[:n]]
        cell_ids = [str(c) if c is not None else "unknown" for c in ids]
    return CriticBatch(
        features=stacked,
        y_any=torch.stack(anys),
        y_type=torch.stack(types),
        cell_ids=cell_ids,
        n_skipped=len(dataset) - n,
    )


def _probs(critic: Any, features: Tensor) -> np.ndarray:
    with torch.no_grad():
        return critic.probs_from_features(features).cpu().numpy().astype(float)


def ensemble_disagreement(critic: Any, ctx: dict[str, Any], chunk: Tensor) -> Tensor:
    """``[B]`` ensemble disagreement (``06`` §5): the only OOD signal the critic has."""
    return critic.ensemble_std(ctx, chunk)


# --------------------------------------------------------------------------
# T1: the critic table
# --------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def critic_table(
    critic: Any,
    dataset: Sequence[Any],
    *,
    tau: float,
    split: str,
    cells: Sequence[str] | None = None,
    out_dir: str | Path | None = None,
    n_bins: int = 15,
    device: str = "cpu",
    temperature: float | None = None,
    train_disagreement: Sequence[float] | None = None,
    gate_cfg: Any | None = None,
    gradient_risk_fn: Callable[[Tensor], Tensor] | None = None,
    gradient_h: Tensor | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """T1 (``09`` §4) on one split, with calibration and the gradient diagnostics.

    ``temperature`` is fitted on VAL when omitted; passing a frozen value is
    mandatory on a sealed split, and omitting it there raises rather than silently
    tuning a calibration parameter on TEST.

    ``gradient_risk_fn``/``gradient_h`` (an activation and its risk functional) add
    the *site-level* gradient diagnostics E2 reports; the feature-input diagnostics
    are always computed, and the two are labelled separately because they are
    different quantities.
    """
    if not 0.0 < float(tau) < 1.0:
        raise ValueError(f"tau must be in (0,1); got {tau}")
    if split in SEALED_ROLES and temperature is None:
        raise ValueError(
            f"split={split!r} is sealed: pass the frozen `temperature` fitted on VAL. "
            "Fitting a calibration parameter on a TEST split is tuning on TEST "
            "(CLAUDE.md #2)"
        )

    batch = critic_features_from_dataset(dataset, critic, max_samples=max_samples)
    feats = batch.features.to(device)
    y_any = batch.y_any.cpu().numpy().astype(float)
    y_type = batch.y_type.cpu().numpy().astype(float)
    logits = critic.logits_from_features(feats).detach().cpu().numpy().astype(float)
    probs = _probs(critic, feats)
    if probs.ndim != 2 or probs.shape[1] != 1 + len(VIOLATION_TYPES):
        raise ValueError(f"critic probabilities must be [N, 1+5]; got {probs.shape}")

    logits_any = logits[:, 0]
    p_before = probs[:, 0]

    notes: list[str] = []
    if temperature is None:
        t = float(fit_temperature(torch.as_tensor(logits_any), torch.as_tensor(y_any)))
        if not np.isfinite(t) or t <= 0.0:
            notes.append(
                f"temperature fit returned {t}; falling back to 1.0 (uncalibrated) rather "
                "than reporting a meaningless calibration"
            )
            t = 1.0
        calibration_source = "fitted-on-this-split"
    else:
        t = float(temperature)
        calibration_source = "frozen (supplied)"

    p_after = 1.0 / (1.0 + np.exp(-logits_any / t))
    base_rate = float(y_any.mean())
    pr = precision_recall_at_threshold(y_any, p_after, tau)
    ap = float(auprc(y_any, p_after))

    per_type: dict[str, dict[str, float]] = {}
    for k, name in enumerate(VIOLATION_TYPES):
        yk = y_type[:, k]
        pk = probs[:, k + 1]
        if yk.sum() == 0:
            per_type[name] = {
                "n_pos": 0.0,
                "auroc": float("nan"),
                "fnr_at_tau": float("nan"),
                "note": 1.0,  # marker: no positives, so AUROC/FNR are unverifiable
            }
            notes.append(f"{name}: no positives on {split}, so its AUROC/FNR cannot be verified")
            continue
        per_type[name] = {
            "n_pos": float(yk.sum()),
            "auroc": float(auroc(yk, pk)),
            "fnr_at_tau": float(fnr_at_threshold(yk, pk, tau)),
            "note": 0.0,
        }

    # -- per cell: a pooled AUROC must not hide a broken cell (06 §5) ------
    cell_ids = list(cells) if cells is not None else batch.cell_ids
    if cells is not None and len(cells) != y_any.size:
        raise ValueError(f"cells has {len(cells)} entries but the split has {y_any.size} rows")
    per_cell: dict[str, dict[str, float]] = {}
    if cell_ids is not None:
        if len(cell_ids) != y_any.size:
            raise ValueError(f"cell ids {len(cell_ids)} != rows {y_any.size}")
        for cell in sorted(set(cell_ids)):
            mask = np.asarray([c == cell for c in cell_ids])
            yc, pc = y_any[mask], p_after[mask]
            if yc.sum() in (0.0, float(yc.size)):
                per_cell[cell] = {"n": float(yc.size), "auroc": float("nan"), "n_pos": float(yc.sum())}
                notes.append(f"cell {cell}: single class (n={yc.size}); no cell AUROC")
                continue
            per_cell[cell] = {
                "n": float(yc.size),
                "n_pos": float(yc.sum()),
                "auroc": float(auroc(yc, pc)),
                "fnr_at_tau": float(fnr_at_threshold(yc, pc, tau)),
            }
    else:
        notes.append(
            "no cell ids available (pass `cells=` or use a dataset exposing .tuples[].cell_id); "
            "the per-cell breakdown required by 06 §5 is missing"
        )

    # -- ensemble disagreement vs error -----------------------------------
    with torch.no_grad():
        member_probs = torch.sigmoid(critic.member_logits_from_features(feats))
        disagreement = member_probs.std(dim=0)[:, 0].cpu().numpy().astype(float)
    error = np.abs(p_after - y_any)
    median_err = float(np.median(error))
    high = disagreement[error > median_err]
    low = disagreement[error <= median_err]

    # -- input-gradient diagnostics (06 §4) -------------------------------
    def _feature_risk(x: Tensor) -> Tensor:
        return torch.sigmoid(critic.logits_from_features(x)[:, 0])

    feature_grad = gradient_diagnostics(
        _feature_risk, feats.detach().requires_grad_(True), eta=1e-8
    )
    site_grad = {
        "mean_gn2": float("nan"),
        "frac_below_eta": float("nan"),
        "saturation_rate": float("nan"),
        "direction_agreement": float("nan"),
    }
    if gradient_risk_fn is not None and gradient_h is not None:
        site_grad = gradient_diagnostics(gradient_risk_fn, gradient_h, eta=1e-8)
    else:
        notes.append(
            "site-level gradient diagnostics not computed (no activation probe supplied): "
            "they are E2's quantity and need the site activation, not the feature input"
        )

    table: dict[str, Any] = {
        "table": "T1",
        "split": split,
        "tau": float(tau),
        "n": float(y_any.size),
        "n_positives": float(y_any.sum()),
        "base_rate": base_rate,
        "n_skipped": float(batch.n_skipped),
        "auroc": float(auroc(y_any, p_after)),
        "auprc": ap,
        "auprc_over_base_rate": float(ap / base_rate) if base_rate > 0 else float("nan"),
        "brier": float(np.mean((p_after - y_any) ** 2)),
        "ece_before": float(ece(y_any, p_before, n_bins)),
        "ece_after": float(ece(y_any, p_after, n_bins)),
        "temperature": t,
        "calibration_source": calibration_source,
        "fnr_at_tau": float(fnr_at_threshold(y_any, p_after, tau)),
        "precision_at_tau": float(pr["precision"]),
        "recall_at_tau": float(pr["recall"]),
        "predicted_positive_rate": float((p_after >= tau).mean()),
        "saturation_rate": float(np.mean(np.abs(logits_any) > 6.0)),
        "mean_disagreement": float(disagreement.mean()),
        "disagreement_error_corr": _pearson(disagreement, error),
        "mean_disagreement_high_error": float(high.mean()) if high.size else float("nan"),
        "mean_disagreement_low_error": float(low.mean()) if low.size else float("nan"),
        "per_type": per_type,
        "per_cell": per_cell,
        # Only numeric diagnostics are folded in: an implementation that adds a
        # non-numeric field must not break the table (a nan is returned for anything
        # that could not be computed, 06 §4).
        "gradient_diagnostics": {
            "scope": "critic_feature_input",
            **{k: float(v) for k, v in feature_grad.items() if isinstance(v, (int, float))},
        },
        "gradient_diagnostics_site": {
            "scope": "site_activation",
            **{k: float(v) for k, v in site_grad.items() if isinstance(v, (int, float))},
        },
        "notes": notes,
    }

    if gate_cfg is not None:
        from riskproj.critic.gates import evaluate_g3

        report = evaluate_g3(
            p_after,
            y_any,
            tau=float(tau),
            cfg_gate=gate_cfg,
            cells=cell_ids,
            per_type_probs={n: probs[:, k + 1] for k, n in enumerate(VIOLATION_TYPES)},
            per_type_labels={n: y_type[:, k] for k, n in enumerate(VIOLATION_TYPES)},
            val_disagreement=disagreement if split.startswith("val") else None,
            train_disagreement=train_disagreement,
        )
        table["gate"] = report.to_dict()
    else:
        table["gate"] = None
        notes.append(
            "no gate_cfg supplied, so G3 was not evaluated here; call "
            "critic.gates.evaluate_g3 (or pass gate_cfg) for the acceptance decision"
        )

    if out_dir is not None:
        write_json(Path(out_dir) / "critic_table.json", table)
    return table


def reliability_diagram_data(
    probs: Sequence[float], labels: Sequence[float], *, n_bins: int = 15
) -> dict[str, list[float]]:
    """Binned reliability data for figure F8 (delegates to ``utils.metrics``)."""
    return reliability_curve(labels, probs, n_bins=n_bins)


# --------------------------------------------------------------------------
# proxy vs gold (A8 / F3)
# --------------------------------------------------------------------------


def _optional_lookup(rec: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    """First present value among ``keys`` (also searching ``rec['extra']``), else None."""
    extra = rec.get("extra") if isinstance(rec.get("extra"), Mapping) else {}
    for key in keys:
        for source in (rec, extra):
            if isinstance(source, Mapping) and source.get(key) is not None:
                return float(source[key])
    return None


def _lookup(rec: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, float]:
    for key in keys:
        value = _optional_lookup(rec, (key,))
        if value is not None:
            return key, value
    raise KeyError(
        f"decision record has none of {list(keys)}; got {sorted(rec)} "
        "(extra keys are also searched)"
    )


def _critic_reduction(rec: Mapping[str, Any]) -> float:
    """Critic-risk reduction of a decision: explicit key, else ``r_cand - r_after[-1]``."""
    try:
        return _lookup(rec, CRITIC_REDUCTION_KEYS)[1]
    except KeyError:
        pass
    r_cand = rec.get("r_cand")
    r_after = rec.get("r_after")
    if r_cand is None or not r_after:
        raise KeyError(
            "decision record carries neither a critic-reduction key "
            f"({list(CRITIC_REDUCTION_KEYS)}) nor ('r_cand', 'r_after'); got {sorted(rec)}"
        )
    return float(r_cand) - float(np.asarray(r_after, dtype=float).reshape(-1)[-1])


def proxy_gold_curve(
    records: Sequence[Mapping[str, Any]],
    *,
    x: str = "corr_rel",
    y: str = "measured_violation",
    n_bins: int = 5,
) -> dict[str, Any]:
    """Critic-risk reduction versus measured-violation reduction (F3, ablation A8).

    Reads **archived decision records only**. Bins by correction norm ``corr_rel``
    (or by ``k_used`` when ``x="k_used"``) and reports, per bin: how much critic risk
    the correction removed, and how much *measured* violation it removed.

    The measured quantity is a paired reduction — the original chunk's outcome minus
    the corrected chunk's outcome on the same state (``original_violation`` /
    ``true_violation_next_H`` against ``measured_violation`` / ``corrected_violation``).
    Records without the original label still contribute their violation *rate*, and
    ``n_unpaired`` records how many did.

    ``divergence_point`` is the first bin where the measured reduction falls while the
    critic reduction keeps rising — the Goodhart point: past it the critic is being
    optimized, not the physics, and the correction norm at that bin is a principled
    trust-region bound (``10`` A8).
    """
    if not records:
        raise ValueError("no decision records supplied")
    if x not in ("corr_rel", "k_used", "correction_norm"):
        raise ValueError(f"x must be 'corr_rel', 'correction_norm' or 'k_used'; got {x!r}")

    xs: list[float] = []
    critic_red: list[float] = []
    corrected: list[float] = []
    original: list[float | None] = []
    for i, rec in enumerate(records):
        if x == "k_used":
            if rec.get("k_used") is None:
                raise KeyError(f"record {i} has no 'k_used' for the x axis")
            xs.append(float(rec["k_used"]))
        else:
            xs.append(_lookup(rec, ("corr_rel", "correction_norm"))[1])
        critic_red.append(_critic_reduction(rec))
        corrected.append(_lookup(rec, (y, *MEASURED_VIOLATION_KEYS))[1])
        original.append(_optional_lookup(rec, ORIGINAL_VIOLATION_KEYS))

    x_arr = np.asarray(xs, dtype=float)
    c_arr = np.asarray(critic_red, dtype=float)
    after = np.asarray(corrected, dtype=float)
    before = np.asarray([np.nan if o is None else o for o in original], dtype=float)
    if np.any(~np.isfinite(x_arr)):
        raise ValueError("x axis contains non-finite values (an inactive decision should be excluded)")

    # The gold quantity is a *paired* reduction: the same state's original chunk
    # versus its corrected chunk. Without the original label there is no reduction to
    # report, only a rate -- which is why it is reported separately and marked.
    reduction = before - after
    n_unpaired = int(np.sum(~np.isfinite(reduction)))
    notes: list[str] = []
    if n_unpaired:
        notes.append(
            f"{n_unpaired} of {len(records)} records carry no original-chunk violation "
            f"(looked for {list(ORIGINAL_VIOLATION_KEYS)}), so no paired reduction is available "
            "for them; only their violation rate is used"
        )

    if x == "k_used":
        ks = sorted({float(v) for v in x_arr.tolist()})
        edges = np.asarray([k - 0.5 for k in ks] + [ks[-1] + 0.5], dtype=float)
        labels = [f"K={int(k)}" for k in ks]
    else:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(x_arr, qs))
        if edges.size < 2:
            edges = np.asarray([x_arr.min(), x_arr.max() + 1e-12])
        labels = [f"bin{i}" for i in range(edges.size - 1)]

    bins: list[dict[str, Any]] = []
    for lo, hi, label in zip(edges[:-1], edges[1:], labels, strict=True):
        mask = (x_arr >= lo) & (x_arr < hi) if hi < edges[-1] else (x_arr >= lo) & (x_arr <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        paired = mask & np.isfinite(reduction)
        bins.append(
            {
                "label": label,
                "x_lo": float(lo),
                "x_hi": float(hi),
                "n": float(n),
                "n_paired": float(paired.sum()),
                "mean_x": float(x_arr[mask].mean()),
                "critic_risk_reduction": float(c_arr[mask].mean()),
                "measured_violation_rate": float(after[mask].mean()),
                "measured_violation_reduction": (
                    float(reduction[paired].mean()) if paired.any() else float("nan")
                ),
            }
        )
    if len(bins) < 2:
        return {
            "x": x,
            "y": y,
            "bins": bins,
            "n_records": float(len(records)),
            "n_unpaired": float(n_unpaired),
            "divergence_point": None,
            "notes": [
                *notes,
                "fewer than two populated bins; a proxy-gold curve needs a range of norms",
            ],
        }

    # Divergence: critic risk keeps falling while the gold quantity stops improving.
    # Prefer the paired reduction; when no record is pairable, a *rising* violation
    # rate is the same signal, and the note above records which was used.
    use_reduction = any(np.isfinite(b["measured_violation_reduction"]) for b in bins)
    divergence: float | None = None
    for prev, cur in zip(bins[:-1], bins[1:], strict=True):
        if not (cur["critic_risk_reduction"] > prev["critic_risk_reduction"]):
            continue
        if use_reduction:
            a, b = prev["measured_violation_reduction"], cur["measured_violation_reduction"]
            if np.isfinite(a) and np.isfinite(b) and b < a:
                divergence = float(cur["mean_x"])
                break
        elif cur["measured_violation_rate"] > prev["measured_violation_rate"]:
            divergence = float(cur["mean_x"])
            break

    if divergence is None:
        notes.append(
            "no bin shows critic risk falling while measured violations stop improving in this "
            "range: the proxy-gold divergence is not yet visible"
        )
    return {
        "x": x,
        "y": y,
        "bins": bins,
        "n_records": float(len(records)),
        "n_unpaired": float(n_unpaired),
        "gold_quantity": "paired_reduction" if use_reduction else "violation_rate",
        "divergence_point": divergence,
        "notes": notes,
    }


# --------------------------------------------------------------------------
# claim guard
# --------------------------------------------------------------------------

#: Statements ``12`` forbids, with the evidence each one would need.
_FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "that the critic predicts task success, reward or return (only violation labels were measured)",
    "that the policy is safe, that violations are bounded, or that any guarantee holds",
)


def critic_table_forbids(table: Mapping[str, Any]) -> list[str]:
    """Statements the T1 table does **not** license (``12`` claim-language rules).

    Returned as plain sentences for a report's limitations paragraph. A table that is
    missing a quantity licenses strictly *fewer* claims, so a missing key adds a
    line rather than being skipped.
    """

    def _num(key: str) -> float:
        value = table.get(key)
        return float(value) if isinstance(value, (int, float)) and np.isfinite(float(value)) else float("nan")

    forbidden = list(_FORBIDDEN_CLAIMS)

    ece_after = _num("ece_after")
    if not np.isfinite(ece_after):
        forbidden.append(
            "that the critic's risk values are calibrated probabilities (no post-scaling ECE "
            "in this table)"
        )
    elif ece_after > 0.05:
        forbidden.append(
            f"that the critic's risk values are calibrated probabilities (ECE {ece_after:.3f} > 0.05)"
        )
    elif table.get("calibration_source") == "fitted-on-this-split":
        forbidden.append(
            "that the calibration transfers to unseen data (the temperature was fitted on the "
            "same split it is reported for)"
        )

    fnr = _num("fnr_at_tau")
    if not np.isfinite(fnr):
        forbidden.append("that the critic does not miss violations at tau (FNR not reported)")
    elif fnr > 0.15:
        forbidden.append(f"that the critic does not miss violations at tau (FNR {fnr:.3f} > 0.15)")

    auroc_any = _num("auroc")
    if not np.isfinite(auroc_any):
        forbidden.append("that the critic is a valid risk proxy (AUROC not reported)")
    elif auroc_any < 0.85:
        forbidden.append(f"that the critic is a valid risk proxy (AUROC {auroc_any:.3f} < 0.85)")

    per_cell = table.get("per_cell") or {}
    if not per_cell:
        forbidden.append("any per-cell claim (this table has no per-cell breakdown)")
    else:
        broken = [
            str(name)
            for name, entry in per_cell.items()
            if not np.isfinite(float((entry or {}).get("auroc", float("nan"))))
        ]
        if broken:
            forbidden.append(
                f"that the critic is uniformly reliable across cells (no cell AUROC for {broken})"
            )

    grad = table.get("gradient_diagnostics_site") or {}
    if not np.isfinite(float((grad or {}).get("mean_gn2", float("nan")))):
        forbidden.append(
            "that the critic gradient is informative at the projection site (site-level gradient "
            "diagnostics were not computed; that is E2's measurement)"
        )
    return forbidden


# ``critic_claim_guard`` is the name the analysis scripts import.
critic_claim_guard = critic_table_forbids


__all__ = [
    "CRITIC_REDUCTION_KEYS",
    "MEASURED_VIOLATION_KEYS",
    "ORIGINAL_VIOLATION_KEYS",
    "CriticBatch",
    "critic_claim_guard",
    "critic_features_from_dataset",
    "critic_table",
    "critic_table_forbids",
    "ensemble_disagreement",
    "proxy_gold_curve",
    "reliability_diagram_data",
]
