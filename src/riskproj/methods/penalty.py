"""B2 / B2-L — use the critic as a *training-time penalty*, not as a projection.

Tests the claim: the simplest sensible way to use a risk critic is to add
``lam * relu(r(a_hat) - tau)`` to the training loss (``08`` §2, B2). If B2 matches or
beats B3, then the latent projection adds nothing that a scalar penalty on the
action does not already give — and the interesting question becomes why, not
whether. ``B2-L`` replaces the fixed weight with a dual ascent toward a violation
budget (CMDP-style, SafeVLA-adjacent [Va]).

Two things this module deliberately does *not* do:

* It never lets the critic receive task gradients. The critic is frozen
  (:func:`riskproj.critic.train.freeze` + ``assert_frozen``) and its parameters have
  ``requires_grad=False``, so no gradient can accumulate even by accident; the loss
  still flows *through* ``r`` into the policy, which is the point.
* It does not claim the penalty is a constraint. A penalty is a soft trade-off; the
  constraint language belongs to the projection arms and even there it is bounded
  (``01`` §4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from riskproj.critic.model import RiskCritic
from riskproj.critic.train import assert_frozen, freeze
from riskproj.methods.base import (
    MethodBase,
    MethodResult,
    critic_context_from_batch,
    flow_matching_loss,
)
from riskproj.projection import threshold_in_space
from riskproj.utils.config import ExperimentCfg, MethodCfg
from riskproj.utils.registry import register

__all__ = ["CriticPenalty", "LagrangianPenalty"]


class CriticPenalty(MethodBase):
    """B2 — LoRA SFT plus a hinge penalty on the critic's risk of the terminal estimate.

    ``01`` §6 item 2: on a flow head the risk must be evaluated on the *terminal
    estimate* ``a0_hat``, not on the noisy ``x_tau`` — at ``tau`` near 1 the estimate
    is noise and the penalty would regularise the wrong object.
    """

    ids = ("B2",)

    def __init__(self, method_cfg: MethodCfg) -> None:
        super().__init__(method_cfg)
        if self.name not in self.ids:
            raise ValueError(f"CriticPenalty serves {self.ids}; got {self.name!r}")
        self.lam = float(method_cfg.lam)
        self._critic_checked = False

    # -- training ---------------------------------------------------------
    def train(
        self,
        cfg: ExperimentCfg,
        policy: Any,
        *,
        train_seed: int,
        data: Any,
        critic: RiskCritic | None = None,
        out_dir: Path | str | None = None,
        logger: Any | None = None,
    ) -> MethodResult:
        if critic is None:
            raise ValueError(
                f"{self.name} needs a trained critic: the penalty is defined by it. "
                "Train the critic and pass it, or run B1 instead (06 §3)."
            )
        freeze(critic)
        assert_frozen(critic)
        result = super().train(
            cfg,
            policy,
            train_seed=train_seed,
            data=data,
            critic=critic,
            out_dir=out_dir,
            logger=logger,
        )
        result.metrics["lam"] = float(self.lam)
        return result

    def penalty(self, critic: RiskCritic, batch: dict[str, Any], chunk: Tensor) -> Tensor:
        """``relu(r(chunk) - thr)`` in the configured space (``06`` §6, ``07`` §2)."""
        if critic.training:
            raise RuntimeError(
                "the critic must be in eval() mode during Stage B: a stochastic critic would "
                "make the penalty a moving target (06 §3)"
            )
        ctx = critic_context_from_batch(batch)
        risk = critic.risk(ctx, chunk, space=self.method_cfg.projection.space)
        threshold = threshold_in_space(self.method_cfg.projection)
        return torch.relu(risk - threshold)

    def training_loss(
        self,
        policy: Any,
        batch: dict[str, Any],
        cfg: ExperimentCfg,
        critic: RiskCritic | None,
        step: int,
    ) -> tuple[Tensor, dict[str, float]]:
        if critic is None:
            raise ValueError(f"{self.name} requires a critic")
        site = self.site_or_default(policy)
        out = policy.forward(dict(batch), site=site)
        base_loss, info = flow_matching_loss(policy, batch, site=site)
        penalty = self.penalty(critic, batch, out.chunk)
        weighted = float(self.lam) * penalty.mean()
        total = base_loss + weighted
        info.update(
            {
                "loss_flow": float(base_loss.detach()),
                # ``penalty`` is the term that actually entered the loss
                # (lambda * relu(r - tau)): with lambda = 0 it is identically zero,
                # which is what makes "B2 with lambda=0 equals B1" checkable. The
                # unweighted excess is recorded separately so that a zero penalty
                # can never hide an active critic behind a zero weight.
                "penalty": float(weighted.detach()),
                "risk_excess": float(penalty.mean().detach()),
                "lam": float(self.lam),
                "risk_mean": float(
                    critic.risk(
                        critic_context_from_batch(batch),
                        out.chunk,
                        space=self.method_cfg.projection.space,
                    )
                    .mean()
                    .detach()
                ),
                "lam": float(self.lam),
            }
        )
        return total, info


class LagrangianPenalty(CriticPenalty):
    """B2-L — the same penalty with ``lam`` moved by dual ascent toward a budget.

    The dual step is ``lam <- clip(lam + alpha * (mean_risk - budget), 0, inf)`` on
    the *batch* critic risk. That is a proxy for the VAL violation budget the spec
    asks for (``08`` §2): the true measured violation rate is only available from
    rollouts, which do not exist mid-training. Recorded as such in the metrics, with
    the whole ``lam`` trajectory, because the size of that proxy gap is itself
    reportable (``09`` §1, proxy-gold).
    """

    ids = ("B2-L",)

    def __init__(self, method_cfg: MethodCfg) -> None:
        super().__init__(method_cfg)
        self.alpha = float(method_cfg.lambda_dual_alpha)
        self.budget = (
            float(method_cfg.violation_budget)
            if method_cfg.violation_budget is not None
            else float(method_cfg.projection.tau)
        )
        self.lam = float(method_cfg.lam)
        self.lam_trajectory: list[float] = []

    def training_loss(
        self,
        policy: Any,
        batch: dict[str, Any],
        cfg: ExperimentCfg,
        critic: RiskCritic | None,
        step: int,
    ) -> tuple[Tensor, dict[str, float]]:
        if critic is None:
            raise ValueError(f"{self.name} requires a critic")
        total, info = super().training_loss(policy, batch, cfg, critic, step)
        # Dual ascent on the batch-mean risk (documented proxy for the VAL budget).
        excess = float(info["risk_mean"]) - self.budget
        self.lam = max(0.0, self.lam + self.alpha * excess)
        self.lam_trajectory.append(self.lam)
        info["lam"] = float(self.lam)
        info["lam_excess"] = float(excess)
        return total, info

    def train(
        self,
        cfg: ExperimentCfg,
        policy: Any,
        *,
        train_seed: int,
        data: Any,
        critic: RiskCritic | None = None,
        out_dir: Path | str | None = None,
        logger: Any | None = None,
    ) -> MethodResult:
        result = super().train(
            cfg,
            policy,
            train_seed=train_seed,
            data=data,
            critic=critic,
            out_dir=out_dir,
            logger=logger,
        )
        if self.lam_trajectory:
            result.metrics["lam_final"] = float(self.lam_trajectory[-1])
            result.metrics["lam_max"] = float(max(self.lam_trajectory))
            half = len(self.lam_trajectory) // 2
            if half:
                result.metrics["lam_growth_second_half"] = float(
                    self.lam_trajectory[-1] - self.lam_trajectory[half]
                )
        result.metrics["violation_budget"] = float(self.budget)
        return result


@register("method", "B2")
def _make_b2(method_cfg: MethodCfg) -> CriticPenalty:
    return CriticPenalty(method_cfg)


@register("method", "B2-L")
def _make_b2l(method_cfg: MethodCfg) -> LagrangianPenalty:
    return LagrangianPenalty(method_cfg)
