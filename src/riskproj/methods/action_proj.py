"""B4a / B4b — do the same correction in *action space* instead of at a latent site.

These are the comparison that decides whether the latent depth in ``05`` §3 is
special or a continuum (``10`` A6, A9): Dalal et al.'s linearised safety layer is
B4a (post-hoc correction of the final chunk), and classifier-guidance-style in-flow
correction of the noisy chunk ``x_tau`` is B4b. If either matches B3, the honest
conclusion is that the latent site adds nothing, and that is a published-worthy
negative result rather than a failure to hide.

Both are **inference-only over a B1 checkpoint** (``08`` §2). Training-time
action-space projection is deliberately *not* implemented: the flow-matching loss
constrains the velocity field, and projecting the decoded chunk does not enter that
target in any well-defined way, so a "train with action projection" arm would be an
unstated research choice rather than a port of the same idea. The training-time
counterpart that *is* well defined is the critic penalty (B2), and ablation A16
compares it with B3.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from riskproj.critic.model import RiskCritic
from riskproj.methods.base import MethodResult, PolicyWrapper
from riskproj.methods.lora_sft import LoraSFT
from riskproj.utils.config import ExperimentCfg, MethodCfg
from riskproj.utils.registry import register

__all__ = ["ActionProjection"]


class ActionProjection(LoraSFT):
    """B4a (``inference_mode='act-post'``) and B4b (``'act-flow'``).

    Training is plain LoRA SFT (inherited), so the *only* difference from B1 is the
    inference-time intervention — which is exactly what makes these clean controls
    for "where does the correction act".
    """

    ids = ("B4a", "B4b")

    def __init__(self, method_cfg: MethodCfg) -> None:
        super().__init__(method_cfg)
        if self.name not in self.ids:
            raise ValueError(f"ActionProjection serves {self.ids}; got {self.name!r}")
        expected = "act-post" if self.name == "B4a" else "act-flow"
        if method_cfg.inference_mode != expected:
            raise ValueError(
                f"{self.name} requires inference_mode {expected!r}; got "
                f"{method_cfg.inference_mode!r} (08 §2)"
            )
        if method_cfg.train_mode not in ("0", "act"):
            raise ValueError(
                f"{self.name} trains plain LoRA (train_mode '0'); got {method_cfg.train_mode!r}"
            )
        if method_cfg.train_mode == "act":
            raise NotImplementedError(
                f"{self.name} with train_mode='act' is not implemented: for a flow-matching head "
                "the training loss constrains the velocity field, and no well-defined way exists "
                "to fold a post-hoc chunk projection into it. Use train_mode='0' for the "
                "inference-time comparison (08 §2) or B2 for the training-time penalty (A16)."
            )
        if method_cfg.control != "none":
            raise ValueError(f"{self.name} cannot carry control {method_cfg.control!r}")

    def wrap(
        self, policy: Any, *, critic: RiskCritic | None = None, base_seed: int = 0
    ) -> PolicyWrapper:
        """Delegate the intervention to the projected-method machinery.

        The action-space arms share B3's code path (same operator, same fallback
        ladder, same decision log) and differ only in *where* the correction is
        applied — otherwise a difference between B3 and B4 could come from the
        logging or the ladder rather than from the site.
        """
        from riskproj.methods.controls import ProjectedMethod

        projected = ProjectedMethod(self.method_cfg)
        projected.tuning_budget = self.tuning_budget
        return projected.wrap(policy, critic=critic, base_seed=base_seed)

    def train(
        self,
        cfg: ExperimentCfg,
        policy: Any,
        *,
        train_seed: int,
        data: Any,
        critic: RiskCritic | None = None,
        out_dir: Path | str | None = None,
        logger: Callable[[dict[str, Any]], None] | None = None,
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
        result.metrics["inference_mode_act_post"] = float(
            self.method_cfg.inference_mode == "act-post"
        )
        result.metrics["inference_mode_act_flow"] = float(
            self.method_cfg.inference_mode == "act-flow"
        )
        return result


@register("method", "B4a")
def _make_b4a(method_cfg: MethodCfg) -> ActionProjection:
    return ActionProjection(method_cfg)


@register("method", "B4b")
def _make_b4b(method_cfg: MethodCfg) -> ActionProjection:
    return ActionProjection(method_cfg)
