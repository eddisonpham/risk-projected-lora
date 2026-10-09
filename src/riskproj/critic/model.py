"""Deep-ensemble risk critic ``V_phi(c, a_hat) -> logits in R^{1+5}`` (``06`` §3).

Design choices that are load-bearing elsewhere:

* **LayerNorm only** — no BatchNorm. Per-example independence is required by the
  projection operator (``07`` §1): a batch-normalised critic would make one
  example's risk depend on its batch-mates.
* The ensemble is the uncertainty signal (``06`` §6 aggregators) and the
  ``direction_agreement`` diagnostic (``06`` §4). Ensemble disagreement on VAL
  should exceed TRAIN disagreement (G3) — it is the only OOD signal we have.
* The critic is **frozen** before Stage-B projection experiments: its parameters
  must never receive task gradients (``06`` §3). :func:`make_risk_fn` therefore
  refuses to build a risk functional from a model in ``train()`` mode.
* Weights are held in fp32 and all risk math is fp32 (``05`` §6).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from riskproj.critic.features import (
    FEATURE_LAYOUT_VERSION,
    CriticFeatureBuilder,
    FeatureSpec,
)
from riskproj.protocols import RiskFn
from riskproj.utils.config import CriticCfg

#: Column layout of the critic head: index 0 is any-violation, 1..5 are V1..V5.
ANY_VIOLATION_COL = 0
N_HEADS = 6
CHECKPOINT_FORMAT = "riskproj.critic.v1"

Aggregator = Literal["mean", "mean_plus_k_std", "max"]
AGGREGATORS: tuple[str, ...] = ("mean", "mean_plus_k_std", "max")


class _CriticHead(nn.Module):
    """One ensemble member: MLP -> LayerNorm stack -> ``1 + 5`` logits."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        depth: int,
        out_dim: int = N_HEADS,
        dropout: float = 0.0,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            lin: nn.Module = nn.Linear(prev, hidden)
            if spectral_norm:
                lin = nn.utils.parametrizations.spectral_norm(lin)
            layers += [lin, nn.LayerNorm(hidden), nn.GELU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(prev, out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Small output init: an untrained ensemble must not saturate (|logit| > 6),
        # otherwise the small-gradient failure mode would be an artefact of init.
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.out(self.trunk(features))


class RiskCritic(nn.Module):
    """Ensemble risk critic over deployable inputs (``06`` §1-§3)."""

    def __init__(self, cfg: CriticCfg, spec: FeatureSpec) -> None:
        super().__init__()
        if cfg.ensemble < 1:
            raise ValueError("CriticCfg.ensemble must be >= 1")
        self.cfg = cfg
        self._spec = spec
        self.feature_spec = spec  # public, read by train/eval code
        self.builder = CriticFeatureBuilder.from_spec(cfg, spec)
        self.members = nn.ModuleList(
            [
                _CriticHead(
                    in_dim=spec.dim,
                    hidden=cfg.hidden,
                    depth=cfg.depth,
                    dropout=cfg.dropout,
                    spectral_norm=cfg.spectral_norm,
                )
                for _ in range(cfg.ensemble)
            ]
        )
        self.horizon = spec.chunk_horizon
        self.to(torch.float32)

    # -- construction helpers -------------------------------------------
    @classmethod
    def from_cfg(
        cls,
        cfg: CriticCfg,
        *,
        state_dim: int,
        action_dim: int,
        chunk_horizon: int | None = None,
        vision_dim: int = 0,
        force_dim: int = 0,
    ) -> RiskCritic:
        spec = FeatureSpec(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_horizon=chunk_horizon if chunk_horizon is not None else cfg.horizon,
            hist_len=cfg.use_history,
            vision_dim=vision_dim if cfg.use_vision else 0,
            force_dim=force_dim if cfg.privileged_force else 0,
            use_vision=cfg.use_vision,
            privileged_force=cfg.privileged_force,
        )
        return cls(cfg, spec)

    @property
    def n_members(self) -> int:
        return len(self.members)

    # -- forward ---------------------------------------------------------
    def features(self, ctx: dict[str, Any], chunk: Tensor) -> Tensor:
        """``[B, spec.dim]`` deployable features (guards applied here)."""
        return self.builder(ctx, chunk)

    # -- precomputed-feature path (D-crit stores features, not VLM activations) --
    def member_logits_from_features(self, features: Tensor) -> Tensor:
        """``[M, B, 1 + 5]`` raw logits per ensemble member from a feature matrix."""
        if features.dim() != 2 or features.shape[1] != self._spec.dim:
            raise ValueError(
                f"features must be [B, {self._spec.dim}]; got {tuple(features.shape)} "
                "(the checkpoint records this layout)"
            )
        return torch.stack([member(features) for member in self.members], dim=0)

    def logits_from_features(self, features: Tensor) -> Tensor:
        """``[B, 1 + 5]`` ensemble-mean logits from a feature matrix."""
        return self.member_logits_from_features(features).mean(dim=0)

    def probs_from_features(self, features: Tensor) -> Tensor:
        """``[B, 1 + 5]`` sigmoid probabilities from a feature matrix."""
        return torch.sigmoid(self.logits_from_features(features))

    def per_member_risk_from_features(self, features: Tensor, *, space: str = "prob") -> Tensor:
        """``[M, B]`` any-violation risk per member from a feature matrix."""
        z = self.member_logits_from_features(features)[..., ANY_VIOLATION_COL]
        return torch.sigmoid(z) if space == "prob" else z

    # -- context + chunk path -------------------------------------------
    def per_member_logits(self, ctx: dict[str, Any], chunk: Tensor) -> Tensor:
        """``[M, B, 1 + 5]`` raw logits per ensemble member."""
        return self.member_logits_from_features(self.features(ctx, chunk))

    def logits(self, ctx: dict[str, Any], chunk: Tensor) -> Tensor:
        """``[B, 1 + 5]`` ensemble-mean logits (the ``Critic`` protocol method)."""
        return self.per_member_logits(ctx, chunk).mean(dim=0)

    def per_member_risk(self, ctx: dict[str, Any], chunk: Tensor, *, space: str = "prob") -> Tensor:
        """``[M, B]`` any-violation risk per member."""
        z = self.per_member_logits(ctx, chunk)[..., ANY_VIOLATION_COL]
        return torch.sigmoid(z) if space == "prob" else z

    def risk(
        self,
        ctx: dict[str, Any],
        chunk: Tensor,
        *,
        space: str = "prob",
        aggregator: str = "mean",
        k: float = 1.0,
    ) -> Tensor:
        """Aggregated any-violation risk ``[B]`` (``06`` §6).

        ``space='prob'`` aggregates sigmoid probabilities; ``space='logit'``
        aggregates raw logits (an ablation, ``10`` A13). ``aggregator='max'`` is
        the pessimistic choice that makes critic exploitation harder.
        """
        if space not in ("prob", "logit"):
            raise ValueError(f"space must be 'prob' or 'logit'; got {space!r}")
        per_member = self.per_member_risk(ctx, chunk, space=space)  # [M, B]
        return aggregate_members(per_member, aggregator=aggregator, k=k)

    def ensemble_std(self, ctx: dict[str, Any], chunk: Tensor) -> Tensor:
        """``[B]`` std of member probabilities — disagreement, higher on OOD data."""
        return self.per_member_risk(ctx, chunk, space="prob").std(dim=0, unbiased=False)

    def forward(self, ctx: dict[str, Any], chunk: Tensor) -> Tensor:
        return self.logits(ctx, chunk)

    # -- diagnostics -----------------------------------------------------
    def grad_norm_penalty(
        self, ctx: dict[str, Any], chunk: Tensor, *, create_graph: bool = True
    ) -> Tensor:
        """Mean squared input-gradient norm (``06`` §3, optional regulariser).

        Sigmoid saturation starves gradients; penalising (or not) the input-gradient
        norm changes whether the projection has anything to descend. Reported in
        ``06`` §4 diagnostics either way.
        """
        return self.grad_norm_penalty_from_features(
            self.features(ctx, chunk), create_graph=create_graph
        )

    def grad_norm_penalty_from_features(
        self, features: Tensor, *, create_graph: bool = True
    ) -> Tensor:
        x = features if features.requires_grad else features.detach().requires_grad_(True)
        z = torch.stack([m(x)[:, ANY_VIOLATION_COL] for m in self.members], dim=0).mean(dim=0)
        (grad,) = torch.autograd.grad(z.sum(), x, create_graph=create_graph)
        return grad.flatten(1).square().sum(dim=1).mean()

    # -- io --------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "feature_layout_version": FEATURE_LAYOUT_VERSION,
                "critic_cfg": self.cfg.model_dump(),
                "feature_spec": self._spec.to_dict(),
                "state_dict": self.state_dict(),
            },
            p,
        )
        return p

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, device: str | torch.device = "cpu", strict: bool = True
    ) -> RiskCritic:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"{path}: not a {CHECKPOINT_FORMAT} checkpoint (got {payload.get('format')!r})"
            )
        if payload.get("feature_layout_version") != FEATURE_LAYOUT_VERSION:
            raise ValueError(
                f"{path}: feature layout version {payload.get('feature_layout_version')} != "
                f"{FEATURE_LAYOUT_VERSION}; retrain rather than reshape a stale layout"
            )
        cfg = CriticCfg.model_validate(payload["critic_cfg"])
        spec = FeatureSpec.from_dict(payload["feature_spec"])
        model = cls(cfg, spec)
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model.to(device)


def aggregate_members(per_member: Tensor, *, aggregator: str = "mean", k: float = 1.0) -> Tensor:
    """Aggregate ``[M, B]`` member risks into ``[B]`` (``06`` §6).

    ``max`` is >= ``mean`` elementwise, which the unit tests rely on.
    """
    if per_member.dim() != 2:
        raise ValueError(f"per_member must be [M, B]; got {tuple(per_member.shape)}")
    if aggregator == "mean":
        return per_member.mean(dim=0)
    if aggregator == "mean_plus_k_std":
        return per_member.mean(dim=0) + k * per_member.std(dim=0, unbiased=False)
    if aggregator == "max":
        return per_member.max(dim=0).values
    raise ValueError(f"unknown aggregator {aggregator!r}; expected one of {AGGREGATORS}")


def make_risk_fn(
    critic: RiskCritic,
    ctx: dict[str, Any],
    f_act: Callable[[Tensor], Tensor],
    *,
    space: str = "prob",
    aggregator: str = "mean",
    k: float = 1.0,
    chunk_horizon: int | None = None,
    require_eval: bool = True,
) -> RiskFn:
    """Build ``h -> r`` for the projection operator (``07`` §1).

    ``f_act`` maps a site activation to the terminal action estimate; the returned
    callable keeps the autograd graph intact (no ``no_grad``/``detach``) because
    the projection differentiates the risk with respect to the activation.

    The critic is required to be in ``eval()`` mode: dropout or any stochastic
    layer inside the critic would make ``r(h)`` non-deterministic, so a projection
    could reduce risk by resampling rather than by correcting.
    """
    if require_eval and critic.training:
        raise RuntimeError(
            "make_risk_fn requires critic.eval(): a stochastic critic would let the "
            "projection reduce risk by resampling instead of correcting (07 §1)"
        )
    horizon = chunk_horizon if chunk_horizon is not None else critic.horizon

    def risk_fn(h: Tensor) -> Tensor:
        chunk = f_act(h)
        if not isinstance(chunk, Tensor) or chunk.dim() != 3:
            shape = tuple(chunk.shape) if isinstance(chunk, Tensor) else type(chunk).__name__
            raise ValueError(f"f_act must return a [B, H, A] tensor; got {shape}")
        if chunk.shape[1] != horizon:
            raise ValueError(
                f"f_act returned horizon {chunk.shape[1]} but the critic expects {horizon} "
                "(exec_horizon == critic_horizon, 04 §1)"
            )
        return critic.risk(ctx, chunk, space=space, aggregator=aggregator, k=k)

    return risk_fn


__all__ = [
    "AGGREGATORS",
    "ANY_VIOLATION_COL",
    "CHECKPOINT_FORMAT",
    "N_HEADS",
    "Aggregator",
    "RiskCritic",
    "aggregate_members",
    "make_risk_fn",
]
