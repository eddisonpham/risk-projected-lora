"""Critic training and gradient-quality diagnostics (``06`` §2-§4).

The critic is trained **once** on TRAIN cells, calibrated and selected on VAL
cells, and frozen before any Stage-B projection experiment. Its parameters never
receive task gradients (``06`` §3) — :func:`freeze` / :func:`assert_frozen` make
that checkable rather than aspirational.

Samples are consumed as *precomputed features*: the ``(ctx, chunk) -> dim`` map is
applied once when D-crit is generated (``04`` §7, and G2's budget for critic
training assumes cached features), so training never re-runs the VLM.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from riskproj.critic.features import FeatureSpec
from riskproj.critic.model import ANY_VIOLATION_COL, RiskCritic
from riskproj.protocols import RiskFn
from riskproj.types import VIOLATION_TYPES
from riskproj.utils.config import CriticCfg
from riskproj.utils.io import write_json
from riskproj.utils.metrics import auroc

#: ``(features [D], any_violation, per_type [5])`` for one tuple.
CriticSample = tuple[Tensor, Tensor, Tensor]


@runtime_checkable
class CriticSampleSource(Protocol):
    """Anything indexable returning :data:`CriticSample` (or a dict of the same)."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> CriticSample | Mapping[str, Any]: ...


@dataclass
class CriticTrainConfig:
    """Optimisation knobs, mirroring the relevant :class:`CriticCfg` fields."""

    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 200
    weight_decay: float = 1e-4
    class_weight: str = "balanced"
    bootstrap: bool = True
    seed: int = 0
    grad_norm_penalty: float = 0.0
    early_stop_patience: int | None = None
    min_delta: float = 1e-5
    grad_clip: float | None = 5.0
    val_fraction: float = 0.2

    @classmethod
    def from_cfg(cls, cfg: CriticCfg) -> CriticTrainConfig:
        return cls(
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            weight_decay=cfg.weight_decay,
            class_weight=cfg.class_weight,
            bootstrap=cfg.bootstrap,
            seed=cfg.seed,
            grad_norm_penalty=cfg.grad_norm_penalty,
        )

    def apply_to(self, base: CriticCfg) -> CriticCfg:
        """Round-trip back to :class:`CriticCfg` (used when a VAL search edits knobs)."""
        return base.model_copy(
            update={
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "weight_decay": self.weight_decay,
                "class_weight": self.class_weight,
                "bootstrap": self.bootstrap,
                "seed": self.seed,
                "grad_norm_penalty": self.grad_norm_penalty,
            }
        )


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def _violation_count(value: Any) -> float:
    if isinstance(value, Mapping):
        if "count" not in value:
            raise KeyError(f"violation entry {value!r} has no 'count' key")
        return float(value["count"])
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def labels_from_records(records: Iterable[Mapping[str, Any]]) -> tuple[Tensor, Tensor]:
    """``[N]`` any-violation and ``[N, 5]`` per-type targets from D-crit records.

    The scalar label is ``any(V1..V5)`` and per-type labels are always kept
    (``04`` §4); a missing key is an error, never a silent zero.
    """
    any_list: list[float] = []
    type_rows: list[list[float]] = []
    for i, rec in enumerate(records):
        if "violations" not in rec:
            raise KeyError(f"record {i} has no 'violations' mapping")
        viol = rec["violations"]
        if not isinstance(viol, Mapping):
            raise TypeError(
                f"record {i}: 'violations' must be a mapping; got {type(viol).__name__}"
            )
        missing = [v for v in VIOLATION_TYPES if v not in viol]
        if missing:
            raise KeyError(f"record {i} is missing per-type labels {missing}")
        row = [1.0 if _violation_count(viol[v]) > 0 else 0.0 for v in VIOLATION_TYPES]
        any_value = (
            float(bool(rec["any_violation"]))
            if "any_violation" in rec
            else (1.0 if max(row) > 0 else 0.0)
        )
        any_list.append(any_value)
        type_rows.append(row)
    if not any_list:
        raise ValueError("no records to build labels from")
    return (
        torch.tensor(any_list, dtype=torch.float32),
        torch.tensor(type_rows, dtype=torch.float32),
    )


# --------------------------------------------------------------------------
# dataset plumbing
# --------------------------------------------------------------------------


def _split_sample(sample: CriticSample | Mapping[str, Any]) -> CriticSample:
    """Normalise a dataset item to ``(features [D], any, per_type [5])``."""
    if isinstance(sample, Mapping):
        if "features" not in sample:
            raise KeyError(f"critic sample dict has no 'features' key; got {sorted(sample)}")
        feats = sample["features"]
        y_any = sample.get("any_violation", sample.get("y_any"))
        y_type = sample.get("per_type", sample.get("violations", sample.get("y_type")))
        if y_any is None or y_type is None:
            raise KeyError(
                "critic sample dict must carry 'any_violation' and 'per_type' (or their "
                f"aliases); got keys {sorted(sample)}"
            )
        return (
            torch.as_tensor(feats, dtype=torch.float32),
            torch.as_tensor(y_any, dtype=torch.float32).reshape(()),
            torch.as_tensor(y_type, dtype=torch.float32).reshape(-1),
        )
    if len(sample) != 3:
        raise ValueError(
            f"critic sample must be (features, any, per_type); got {len(sample)} items"
        )
    feats, y_any, y_type = sample
    return (
        torch.as_tensor(feats, dtype=torch.float32),
        torch.as_tensor(y_any, dtype=torch.float32).reshape(()),
        torch.as_tensor(y_type, dtype=torch.float32).reshape(-1),
    )


def _collate(batch: Sequence[CriticSample | Mapping[str, Any]]) -> tuple[Tensor, Tensor, Tensor]:
    feats, y_any, y_type = [], [], []
    for sample in batch:
        f, a, t = _split_sample(sample)
        feats.append(f)
        y_any.append(a)
        y_type.append(t)
    return torch.stack(feats), torch.stack(y_any), torch.stack(y_type)


def _targets(y_any: Tensor, y_type: Tensor) -> Tensor:
    return torch.cat([y_any.reshape(-1, 1), y_type.reshape(-1, len(VIOLATION_TYPES))], dim=1)


def _collect_labels(dataset: CriticSampleSource, cap: int | None) -> tuple[np.ndarray, np.ndarray]:
    n = len(dataset)
    if n == 0:
        raise ValueError("critic dataset is empty")
    if cap is not None and cap < n:
        n = int(cap)
    any_list, type_rows = [], []
    for i in range(n):
        _, a, t = _split_sample(dataset[i])
        any_list.append(float(a))
        type_rows.append(np.asarray(t, dtype=float).reshape(-1))
    return np.asarray(any_list, dtype=float), np.asarray(type_rows, dtype=float)


def _pos_weight(y_any: np.ndarray, y_type: np.ndarray, class_weight: str) -> Tensor:
    """Balance the 6 heads by ``neg/pos`` (``06`` §3)."""
    counts = np.concatenate([y_any.reshape(-1, 1), y_type], axis=1)
    n = counts.shape[0]
    if class_weight == "none":
        return torch.ones(counts.shape[1], dtype=torch.float32)
    if class_weight != "balanced":
        raise ValueError(f"class_weight must be 'balanced' or 'none'; got {class_weight!r}")
    pos = counts.sum(axis=0)
    neg = n - pos
    # A head with no positives can never be learned here; weight 1.0 so it contributes
    # a constant and the fact shows up in the recorded per-type balance instead.
    weights = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.clip(weights, 1e-6, 1e6), dtype=torch.float32)


def _head_grad_penalty(head: nn.Module, features: Tensor, *, create_graph: bool = True) -> Tensor:
    x = features if features.requires_grad else features.detach().requires_grad_(True)
    z = head(x)[:, ANY_VIOLATION_COL]
    (grad,) = torch.autograd.grad(z.sum(), x, create_graph=create_graph)
    return grad.flatten(1).square().sum(dim=1).mean()


def _eval_features(
    model: RiskCritic, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ``[N, 1+5]`` ensemble probabilities and labels over a loader."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for feats, y_any, y_type in loader:
            probs.append(model.probs_from_features(feats.to(device)).cpu().numpy())
            labels.append(_targets(y_any, y_type).numpy())
    if not probs:
        raise ValueError("empty loader: no samples to evaluate")
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def train_critic(
    dataset: CriticSampleSource,
    cfg: CriticCfg,
    *,
    val_dataset: CriticSampleSource | None = None,
    feature_spec: FeatureSpec,
    device: str = "cpu",
    out_dir: Path | None = None,
    logger: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[RiskCritic, dict[str, Any]]:
    """Train the ensemble on precomputed features; return ``(frozen model, metrics)``.

    Each ensemble member is trained on its own bootstrap resample of TRAIN (when
    ``cfg.bootstrap``) and selected on VAL, which is what makes ensemble
    disagreement a usable OOD signal rather than an artefact of one seed.
    """
    tcfg = CriticTrainConfig.from_cfg(cfg)
    n = len(dataset)
    if n == 0:
        raise ValueError("critic dataset is empty")
    probe = _split_sample(dataset[0])[0]
    if probe.numel() != feature_spec.dim:
        raise ValueError(
            f"dataset features have width {probe.numel()} but FeatureSpec.dim is "
            f"{feature_spec.dim}; regenerate D-crit or fix the spec"
        )

    y_any, y_type = _collect_labels(dataset, cap=cfg.max_tuples)
    pos_weight = _pos_weight(y_any, y_type, cfg.class_weight).to(device)

    rng = np.random.default_rng(cfg.seed)
    if val_dataset is None:
        n_val = max(1, int(tcfg.val_fraction * n + 0.5)) if n >= 5 else 0
        perm = rng.permutation(n)
        val_idx = [int(i) for i in perm[:n_val]]
        train_idx = [int(i) for i in perm[n_val:]]
        if not train_idx:  # tiny dataset: train and validate on the same rows, and say so
            train_idx, val_idx = list(range(n)), list(range(n))
        if not val_idx:  # too few rows to hold any out: validate on TRAIN, flagged above
            val_idx = list(train_idx)
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=tcfg.batch_size, shuffle=False, collate_fn=_collate
        )
        val_source = "self-split" if len(val_idx) < n else "train-equals-val (too few rows)"
    else:
        train_idx = list(range(n))
        val_idx = list(range(len(val_dataset)))
        val_loader = DataLoader(
            val_dataset, batch_size=tcfg.batch_size, shuffle=False, collate_fn=_collate
        )
        val_source = "provided"

    model = RiskCritic(cfg, feature_spec).to(device)
    patience = tcfg.early_stop_patience or max(5, tcfg.epochs // 10)
    history: list[dict[str, float]] = []

    for member_idx, head in enumerate(model.members):
        member_rng = np.random.default_rng(cfg.seed + 1000 + member_idx)
        if cfg.bootstrap:
            sub_idx = member_rng.choice(np.asarray(train_idx), size=len(train_idx), replace=True)
        else:
            sub_idx = np.asarray(train_idx)
        train_loader = DataLoader(
            Subset(dataset, [int(i) for i in sub_idx]),
            batch_size=tcfg.batch_size,
            shuffle=not cfg.bootstrap,
            collate_fn=_collate,
        )
        opt = torch.optim.AdamW(head.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        bad_epochs = 0

        for epoch in range(tcfg.epochs):
            head.train()
            running, batches = 0.0, 0
            for feats, y_any_b, y_type_b in train_loader:
                feats = feats.to(device)
                target = _targets(y_any_b, y_type_b).to(device)
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(
                    head(feats), target, pos_weight=pos_weight
                )
                if tcfg.grad_norm_penalty > 0:
                    loss = loss + tcfg.grad_norm_penalty * _head_grad_penalty(head, feats)
                loss.backward()
                if tcfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), tcfg.grad_clip)
                opt.step()
                running += float(loss.detach())
                batches += 1
            train_loss = running / max(batches, 1)

            head.eval()
            val_total, val_batches = 0.0, 0
            with torch.no_grad():
                for feats, y_any_b, y_type_b in val_loader:
                    target = _targets(y_any_b, y_type_b).to(device)
                    val_total += float(
                        F.binary_cross_entropy_with_logits(
                            head(feats.to(device)), target, pos_weight=pos_weight
                        )
                    )
                    val_batches += 1
            val_loss = val_total / max(val_batches, 1)
            history.append(
                {
                    "member": float(member_idx),
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )
            if logger is not None:
                logger(history[-1])
            if val_loss < best_val - tcfg.min_delta:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        head.load_state_dict(best_state)

    model.eval()
    val_probs, val_labels = _eval_features(model, val_loader, device)
    per_head = {"any": float(auroc(val_labels[:, 0], val_probs[:, 0]))}
    for k, name in enumerate(VIOLATION_TYPES):
        per_head[name] = float(auroc(val_labels[:, k + 1], val_probs[:, k + 1]))

    metrics: dict[str, Any] = {
        "val_source": val_source,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_members": model.n_members,
        "bootstrap": bool(cfg.bootstrap),
        "pos_weight": [float(x) for x in pos_weight.detach().cpu()],
        "base_rate_any": float(y_any.mean()),
        "per_head_auroc": per_head,
        "final_train_loss": float(history[-1]["train_loss"]) if history else float("nan"),
        "final_val_loss": float(history[-1]["val_loss"]) if history else float("nan"),
        "history": history,
    }
    frozen = freeze(model)
    if out_dir is not None:
        out = Path(out_dir)
        frozen.save(out / "critic.pt")
        write_json(out / "metrics.json", metrics)
    return frozen, metrics


# --------------------------------------------------------------------------
# freezing
# --------------------------------------------------------------------------


def freeze(model: RiskCritic) -> RiskCritic:
    """Put the critic in eval mode with no trainable parameters (``06`` §3)."""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def assert_frozen(model: RiskCritic) -> None:
    """Raise unless every parameter is frozen and the module is in eval mode."""
    if model.training:
        raise AssertionError("critic is in train() mode; call freeze(model) before evaluation")
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise AssertionError(
            f"critic has {len(trainable)} trainable parameter tensors (e.g. {trainable[:3]}); "
            "its parameters must never receive task gradients (06 §3)"
        )


# --------------------------------------------------------------------------
# gradient-quality diagnostics (06 §4)
# --------------------------------------------------------------------------


def gradient_diagnostics(
    risk_fn: RiskFn,
    h: Tensor,
    *,
    eta: float = 1e-8,
    logit_fn: Callable[[Tensor], Tensor] | None = None,
    per_member_risk_fns: Sequence[RiskFn] | None = None,
    saturation_threshold: float = 6.0,
) -> dict[str, float]:
    """Input-gradient quality at a site activation: this is E2's raw material.

    Poor gradient quality is a *finding*, not something to patch silently (``06``
    §4), so every diagnostic is reported — as ``nan`` when it cannot be computed,
    never omitted.
    """
    if not isinstance(h, Tensor):
        raise TypeError(f"h must be a Tensor; got {type(h).__name__}")
    hh = h if h.requires_grad else h.detach().requires_grad_(True)
    r = risk_fn(hh)
    if not isinstance(r, Tensor) or r.dim() != 1:
        shape = tuple(r.shape) if isinstance(r, Tensor) else type(r).__name__
        raise ValueError(f"risk_fn must return a [B] tensor; got {shape}")
    (grad,) = torch.autograd.grad(r.sum(), hh, retain_graph=True)
    gn2 = grad.detach().flatten(1).float().square().sum(dim=1)

    out = {
        "mean_gn2": float(gn2.mean()),
        "median_gn2": float(gn2.median()),
        "min_gn2": float(gn2.min()),
        "max_gn2": float(gn2.max()),
        "frac_below_eta": float((gn2 < eta).float().mean()),
        "n_examples": float(gn2.numel()),
    }

    if logit_fn is not None:
        z = logit_fn(hh).detach()
        out["saturation_rate"] = float((z.abs() > saturation_threshold).float().mean())
        out["mean_abs_logit"] = float(z.abs().mean())
    else:
        out["saturation_rate"] = float("nan")
        out["mean_abs_logit"] = float("nan")

    if per_member_risk_fns:
        grads = []
        for fn in per_member_risk_fns:
            (gm,) = torch.autograd.grad(fn(hh).sum(), hh, retain_graph=True)
            grads.append(gm.detach().flatten(1).float())
        stacked = torch.stack(grads, dim=0)  # [M, B, D]
        norms = stacked.norm(dim=2).clamp_min(1e-12)
        cosines = []
        m = stacked.shape[0]
        for i in range(m):
            for j in range(i + 1, m):
                cosines.append((stacked[i] * stacked[j]).sum(dim=1) / (norms[i] * norms[j]))
        out["direction_agreement"] = (
            float(torch.stack(cosines, dim=0).mean()) if cosines else float("nan")
        )
    else:
        out["direction_agreement"] = float("nan")
    return out


__all__ = [
    "CriticSample",
    "CriticSampleSource",
    "CriticTrainConfig",
    "assert_frozen",
    "freeze",
    "gradient_diagnostics",
    "labels_from_records",
    "train_critic",
]
