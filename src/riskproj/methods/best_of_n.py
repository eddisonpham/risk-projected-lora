"""B5 / B6 — use the critic *without* a gradient: selection and gating.

These two arms exist to destroy the most likely alternative explanations of any B3
gain (``08`` §2, ablations A3-A4):

* **B5 (best-of-N)** samples ``N`` chunks with different flow noise and keeps the
  lowest-risk one (V-GPS-style re-ranking [Va]). If selection matches projection,
  the gradient was never the useful part of the critic — the *ranking* was.
* **B6 (gate-only)** lets the critic veto a chunk and hand it to the fallback ladder
  with no correction at all. If gating matches projection, then what helps is
  refusing to act, not correcting.

B5 needs a stochastic head: it is defined for flow matching and meaningless for a
deterministic head, so it raises rather than silently returning the same chunk ``N``
times (``08`` §2). Both are inference-only over B1.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from riskproj.critic.model import RiskCritic
from riskproj.methods.base import (
    MethodResult,
    PolicyWrapper,
    critic_context_from_batch,
    policy_horizon,
    run_inference,
)
from riskproj.methods.lora_sft import LoraSFT
from riskproj.projection import FallbackLadder, threshold_in_space
from riskproj.types import PolicyOut
from riskproj.utils.config import ExperimentCfg, MethodCfg
from riskproj.utils.registry import register
from riskproj.utils.seed import common_random_noise

__all__ = ["BestOfN", "CriticGate"]


def _require_stochastic_head(policy: Any, method: str) -> None:
    head = str(getattr(policy, "action_head", "flow_matching"))
    if head != "flow_matching":
        raise ValueError(
            f"{method} needs a stochastic head (flow matching): with a deterministic head every "
            "candidate is identical, so 'best of N' would be the same chunk N times and the arm "
            "would look harmless while testing nothing (08 §2)"
        )


class BestOfN(LoraSFT):
    """B5 — sample ``N`` chunks with different noise, keep the critic's favourite."""

    ids = ("B5",)

    def __init__(self, method_cfg: MethodCfg) -> None:
        super().__init__(method_cfg)
        if method_cfg.best_of_n < 1:
            raise ValueError("best_of_n must be >= 1")
        if method_cfg.inference_mode != "bon":
            raise ValueError(
                f"B5 requires inference_mode 'bon'; got {method_cfg.inference_mode!r} (08 §2)"
            )
        if method_cfg.control != "none":
            raise ValueError(f"B5 cannot carry control {method_cfg.control!r}")
        self.n_candidates = int(method_cfg.best_of_n)

    def wrap(
        self, policy: Any, *, critic: RiskCritic | None = None, base_seed: int = 0
    ) -> PolicyWrapper:
        if critic is None:
            raise ValueError("B5 ranks candidates with the critic; pass the trained critic (06 §3)")
        if critic.training:
            raise RuntimeError("wrap() requires critic.eval()")
        _require_stochastic_head(policy, "B5")
        wrapper = PolicyWrapper(
            self.name,
            policy,
            critic=critic,
            proj_cfg=self.method_cfg.projection,
            site=self.site_or_default(policy),
            mode="bon",
            ladder=FallbackLadder(window=10, enabled=False),
            base_seed=base_seed,
        )

        def intervention(batch: dict[str, Any]) -> PolicyOut:
            data = dict(batch)
            horizon = policy_horizon(policy)
            action_dim = int(getattr(policy, "action_dim", getattr(policy, "_action_dim", 4)))
            source = data.get("state", data.get("a0", data.get("chunk")))
            if source is None:
                raise KeyError(
                    "B5 needs 'state' (or 'a0') to know the batch size of the candidate chunks"
                )
            batch_size = int(torch.as_tensor(source).shape[0])
            # Candidate chunks are [B, H, A]; every candidate differs only in its noise.
            shape = (batch_size, horizon, action_dim)
            critic_ctx = critic_context_from_batch(data)

            risks: list[Tensor] = []
            chunks: list[Tensor] = []
            infos: list[dict[str, float]] = []
            for i in range(self.n_candidates):
                eps = common_random_noise(
                    base_seed, wrapper.episode_id, wrapper.step, shape, flow_step=i
                )
                out = run_inference(policy, {**data, "eps": eps}, site=wrapper.site)
                chunk = out.chunk
                if chunk.dim() != 3 or chunk.shape[1] != horizon:
                    raise ValueError(
                        f"the policy returned {tuple(chunk.shape)}; expected [B, {horizon}, A]"
                    )
                chunks.append(chunk)
                risk = critic.risk(critic_ctx, chunk, space=self.method_cfg.projection.space)
                risks.append(risk)
                infos.append({"r": float(risk.mean())})

            stacked = torch.stack(risks, dim=0)  # [N, B]
            best = stacked.argmin(dim=0)  # [B] per-example winner
            chosen = torch.stack(
                [chunks[int(best[b])][b] for b in range(stacked.shape[1])], dim=0
            )
            spread = float(stacked.max(dim=0).values.mean() - stacked.min(dim=0).values.mean())
            wrapper.record_extra(
                {
                    "bon_n": float(self.n_candidates),
                    "bon_risk_spread": spread,
                    "bon_mean_risk": float(stacked.mean()),
                    "bon_best_risk": float(stacked.min(dim=0).values.mean()),
                    "bon_worst_risk": float(stacked.max(dim=0).values.mean()),
                    "bon_index_0_wins": float((best == 0).float().mean()),
                }
            )
            if wrapper.ladder is not None:
                wrapper.ladder.note_action(chosen.detach())
            return PolicyOut(
                chunk=chosen,
                site=wrapper.site,
                h=None,
                extras={"bon_risks": [info["r"] for info in infos]},
            )

        wrapper.intervention = intervention
        return wrapper

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
        # B5 trains plain LoRA (the selection happens only at inference).
        return super().train(
            cfg,
            policy,
            train_seed=train_seed,
            data=data,
            critic=critic,
            out_dir=out_dir,
            logger=logger,
        )


class CriticGate(LoraSFT):
    """B6 — the critic only vetoes: risky chunk -> fallback ladder, no correction."""

    ids = ("B6",)

    def __init__(self, method_cfg: MethodCfg) -> None:
        super().__init__(method_cfg)
        if method_cfg.inference_mode != "gate":
            raise ValueError(
                f"B6 requires inference_mode 'gate'; got {method_cfg.inference_mode!r} (08 §2)"
            )
        if method_cfg.control != "none":
            raise ValueError(f"B6 cannot carry control {method_cfg.control!r}")

    def wrap(
        self, policy: Any, *, critic: RiskCritic | None = None, base_seed: int = 0
    ) -> PolicyWrapper:
        if critic is None:
            raise ValueError("B6 is defined by the critic; pass the trained critic (06 §3)")
        if critic.training:
            raise RuntimeError("wrap() requires critic.eval()")
        proj_cfg = self.method_cfg.projection
        wrapper = PolicyWrapper(
            self.name,
            policy,
            critic=critic,
            proj_cfg=proj_cfg,
            site=self.site_or_default(policy),
            mode="gate",
            ladder=FallbackLadder(window=10, enabled=bool(proj_cfg.fallback)),
            base_seed=base_seed,
        )

        def intervention(batch: dict[str, Any]) -> PolicyOut:
            from riskproj.methods.controls import _ensure_eps

            data = _ensure_eps(batch, base_seed, wrapper.episode_id, wrapper.step, policy)
            out = run_inference(policy, data, site=wrapper.site)
            critic_ctx = critic_context_from_batch(data)
            risk = critic.risk(critic_ctx, out.chunk, space=proj_cfg.space)
            threshold = threshold_in_space(proj_cfg)
            wrapper.record_extra(
                {
                    "gate_mean_risk": float(risk.mean()),
                    "gate_threshold": float(threshold),
                    "gate_blocked_frac": float((risk > threshold).float().mean()),
                }
            )
            blocked = bool((risk > threshold).any())
            if not blocked:
                if wrapper.ladder is not None:
                    wrapper.ladder.note_action(out.chunk.detach())
                return out

            ladder = wrapper.ladder
            if ladder is None:  # pragma: no cover - constructed above
                return out
            base = wrapper.lora_scale_zeroed_action(data)
            base_risk = critic.risk(critic_ctx, base, space=proj_cfg.space)
            needs = risk > threshold
            decision = ladder.update(
                usable=~needs,
                action_candidate=out.chunk,
                base_action=base,
                base_action_risk=base_risk,
                threshold=threshold,
                evidence={"method": "B6", "mode": "gate"},
            )
            wrapper.last_fallback = decision
            # ``active=True, usable=False`` is what distinguishes gating from
            # projection in the decision log (there is no correction to log).
            wrapper.log_fallback(decision, flow_step=-1)
            ladder.note_action(decision.action.detach() if decision.action is not None else out.chunk.detach())
            chunk = decision.action if decision.action is not None else out.chunk
            return PolicyOut(
                chunk=chunk, site=wrapper.site, h=None, extras={"gate_blocked": 1.0}
            )

        wrapper.intervention = intervention
        return wrapper


@register("method", "B5")
def _make_b5(method_cfg: MethodCfg) -> BestOfN:
    return BestOfN(method_cfg)


@register("method", "B6")
def _make_b6(method_cfg: MethodCfg) -> CriticGate:
    return CriticGate(method_cfg)
