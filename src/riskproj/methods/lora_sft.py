"""B0 / B1 / B1-LN — the frozen baseline and standard LoRA SFT.

Tests the claim: *standard PEFT on dynamics-diverse demos* (``08`` §2, B1). Every
other arm is measured against it, so it is the one arm whose training path must be
boring and reproducible.

``B1-LN`` is **not** a fix and not a variant of B3: it is the FACT noise-schedule
control (``02`` §A). FACT measured that the default Beta sampler gives only 8.9% of
the gradient mass to ``tau < 0.2`` — exactly the low-noise regime a contact-rich
correction lives in — and that switching to a Logit-Normal schedule recovers large
amounts of performance on precision tasks *without any risk critic*. If B1-LN
matches B3, the honest conclusion is that we re-discovered a schedule effect
(ablation A5), and the method's claim dies. So the only difference between B1 and
B1-LN is the sampler, asserted in :meth:`LoraSFT.__init__` and visible in the
training metrics as the low-``tau`` mass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import Tensor

from riskproj.methods.base import MethodBase, PolicyWrapper, flow_matching_loss
from riskproj.policies.flow import tau_schedule
from riskproj.utils.config import ExperimentCfg, MethodCfg
from riskproj.utils.registry import register

__all__ = ["LoraSFT", "low_noise_mass"]

#: ``tau`` below which the terminal estimate is trustworthy enough to constrain
#: (``05`` §4). FACT's precision-failure regime is ``tau < 0.2``.
LOW_TAU = 0.2


def low_noise_mass(
    schedule: str, *, ln_m: float = 1.5, ln_s: float = 1.0, n: int = 8192, seed: int = 0
) -> float:
    """Fraction of sampled ``tau`` below :data:`LOW_TAU` — the FACT quantity.

    Measured by sampling rather than by an analytic formula for a reason: the whole
    point of B1-LN is that the *implemented* sampler puts more mass at low noise,
    and an analytic estimate of a sampler that is not the one in the loop would not
    show that.
    """
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    tau = tau_schedule(schedule, (int(n),), generator=generator, ln_m=ln_m, ln_s=ln_s)
    return float((tau < LOW_TAU).float().mean())


class LoraSFT(MethodBase):
    """Standard LoRA SFT; also the frozen baseline (B0) and the LN control (B1-LN).

    ``B0`` trains nothing: :meth:`MethodBase.train` refuses, so a "B0 run" that
    quietly did nothing cannot be mistaken for a successful baseline. Its
    ``wrap`` returns the policy untouched — that is the whole point of the arm.
    """

    ids = ("B0", "B1", "B1-LN")

    def __init__(self, method_cfg: MethodCfg) -> None:
        super().__init__(method_cfg)
        if self.name not in self.ids:
            raise ValueError(f"LoraSFT serves {self.ids}; got {self.name!r}")
        if self.name == "B1-LN" and method_cfg.noise_schedule != "logit_normal":
            raise ValueError(
                "B1-LN is the FACT Logit-Normal control: noise_schedule must be "
                f"'logit_normal'; got {method_cfg.noise_schedule!r} (08 §2)"
            )
        if self.name in ("B0", "B1") and method_cfg.control != "none":
            raise ValueError(f"{self.name} is a baseline; controls belong to the B3 pipeline")

    # -- training ---------------------------------------------------------
    def training_loss(
        self,
        policy: Any,
        batch: dict[str, Any],
        cfg: ExperimentCfg,
        critic: Any | None,
        step: int,
    ) -> tuple[Tensor, dict[str, float]]:
        """Plain flow-matching loss, plus the schedule evidence for B1 vs B1-LN."""
        loss, info = flow_matching_loss(policy, batch, site=self.site_or_default(policy))
        tau = batch["tau"]
        info["tau_mean"] = float(tau.mean())
        info["tau_low_frac"] = float((tau <= LOW_TAU).float().mean())
        info["loss"] = float(loss.detach())
        return loss, info

    def train(
        self,
        cfg: ExperimentCfg,
        policy: Any,
        *,
        train_seed: int,
        data: Any,
        critic: Any | None = None,
        out_dir: Path | str | None = None,
        logger: Any | None = None,
    ) -> Any:
        result = super().train(
            cfg,
            policy,
            train_seed=train_seed,
            data=data,
            critic=critic,
            out_dir=out_dir,
            logger=logger,
        )
        # Recorded so the report can state which sampler produced the checkpoint.
        result.metrics["p_tau_lt_0p2"] = low_noise_mass(
            self.method_cfg.noise_schedule, ln_m=self.method_cfg.ln_m, ln_s=self.method_cfg.ln_s
        )
        result.metrics["noise_schedule_logit_normal"] = float(
            self.method_cfg.noise_schedule == "logit_normal"
        )
        return result

    # -- inference --------------------------------------------------------
    def wrap(self, policy: Any, *, critic: Any | None = None, base_seed: int = 0) -> PolicyWrapper:
        """No intervention at inference: the chunk is whatever the head produced.

        A critic may still be attached — the eval layer uses it to *measure* risk for
        every arm — but it never changes the action here, which is exactly the
        baseline's definition (``08`` §2).
        """
        if self.control != "none":
            raise ValueError(f"{self.name} is a baseline and cannot carry control {self.control!r}")
        return PolicyWrapper(
            self.name,
            policy,
            intervention=None,
            critic=critic,
            proj_cfg=self.method_cfg.projection,
            site=self.site_or_default(policy),
            control="none",
            mode="0",
            base_seed=base_seed,
        )


@register("method", "B0")
def _make_b0(method_cfg: MethodCfg) -> LoraSFT:
    return LoraSFT(method_cfg)


@register("method", "B1")
def _make_b1(method_cfg: MethodCfg) -> LoraSFT:
    return LoraSFT(method_cfg)


@register("method", "B1-LN")
def _make_b1_ln(method_cfg: MethodCfg) -> LoraSFT:
    return LoraSFT(method_cfg)
