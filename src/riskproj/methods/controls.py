"""B3 / B1+TTP / B3-trainonly and the negative controls C1-C3.

This is the method under test. ``01`` §3 defines it: at one site, the LoRA-induced
activation ``h_c`` is corrected against a simulator-supervised risk critic by a
first-order half-space step, iterated at most ``K`` times, with an explicit fallback
ladder whenever the gradient is unusable. The correction is a *first-order local*
step, never an exact nonlinear projection, and the nonlinear risk after the step is
logged next to the linearised prediction so the gap is a measured number (E2)
rather than an assumption.

The controls are the reason this module exists in the shape it does (``08`` §2,
ablations A1-A3):

* **C1** replaces the gradient direction with a *random* direction of the same norm
  — so "the correction is perturbation, not information" is testable.
* **C2** shrinks the LoRA delta ``Δh`` toward zero with a matched mean norm — so
  "it just preserves the base behaviour" is testable.
* **C3** projects against a critic whose per-example risk outputs are permuted — a
  stand-in for a critic trained on shuffled labels, i.e. a critic with no usable
  ranking. It is an approximation, recorded as such, and its job is to show what a
  *uninformative* gradient buys.

If any control matches B3, the critic-gradient story is dead and the write-up says
so. That is the falsification-first contract of the project, not an afterthought.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from riskproj.critic.model import RiskCritic, make_risk_fn
from riskproj.methods.base import (
    MethodBase,
    MethodResult,
    PolicyWrapper,
    critic_context_from_batch,
    ensure_eps,
    lora_disabled,
    policy_horizon,
    run_inference,
)
from riskproj.policies.flow import flow_target, terminal_estimate
from riskproj.projection import FallbackLadder, ProjectionInfo, project, threshold_in_space
from riskproj.types import PolicyOut
from riskproj.utils.config import ExperimentCfg, MethodCfg, ProjCfg
from riskproj.utils.registry import register
from riskproj.utils.seed import common_random_noise

__all__ = ["ProjectedMethod"]

#: Control codes this class serves.
CONTROL_CODES: tuple[str, ...] = ("none", "C1", "C2", "C3")


class ProjectedMethod(MethodBase):
    """B3 and its siblings: project a site activation against the risk critic.

    Served ids (``08`` §2):

    * ``B3`` — projection during training and at inference (the proposed method).
    * ``B1+TTP`` — plain LoRA (``train_mode='0'``) with test-time projection only:
      separates "inference-time filter" from "training-time regulariser" (A7).
    * ``B3-trainonly`` — trained with projection, evaluated without it (A7, the
      other half).
    * ``C1``/``C2``/``C3`` — the controls above, driven by ``MethodCfg.control``.
    """

    # B4a/B4b are constructed here by :class:`ActionProjection` so that the
    # action-space arms share this code path (same operator, same ladder, same log)
    # and differ from B3 only in *where* the correction is applied.
    ids = ("B3", "B1+TTP", "B3-trainonly", "B4a", "B4b", "C1", "C2", "C3")

    def __init__(self, method_cfg: MethodCfg) -> None:
        super().__init__(method_cfg)
        if self.name not in self.ids:
            raise ValueError(f"ProjectedMethod serves {self.ids}; got {self.name!r}")
        self.proj_cfg: ProjCfg = method_cfg.projection
        self.train_projects = method_cfg.train_mode == "lat"
        self.infer_projects = method_cfg.inference_mode in ("lat", "act-post", "act-flow")
        if self.train_projects and self.name in ("B1+TTP", "B0"):
            raise ValueError(f"{self.name} is inference-only; train_mode must be '0' (08 §2)")
        # ``fallback=False`` is legal (ablation A12) but changes what the run means: an
        # unusable gradient then stays uncorrected, and the decision log carries that
        # (the operator still flags it, so it is visible rather than silent).

    # ------------------------------------------------------------------
    # the correction itself (shared by training and inference)
    # ------------------------------------------------------------------
    def _risk_fn(
        self,
        policy: Any,
        critic: RiskCritic,
        pol_ctx: Any,
        critic_ctx: dict[str, Any],
        site: str,
    ) -> Callable[[Tensor], Tensor]:
        """``h -> r`` through the adapter's downstream map (``07`` §1)."""
        fn = make_risk_fn(
            critic,
            critic_ctx,
            lambda h: policy.f_act(pol_ctx, h, site),
            space=self.proj_cfg.space,
            chunk_horizon=policy_horizon(policy),
        )
        # The C3 permutation is applied in ``_control_risk_fn``, not here. This method
        # only builds the *critic* risk; every entry point that corrects — training and
        # the direct ``_correct`` path used by inference — must get identical control
        # semantics, and `_correct` is the one place they all funnel through.
        return fn

    def _control_risk_fn(
        self, risk_fn: Callable[[Tensor], Tensor], *, n: int, extra: dict[str, float]
    ) -> Callable[[Tensor], Tensor]:
        """Apply the control's transform to the risk function (C3 only).

        C3 stands in for a critic trained on shuffled labels: the risk that drives
        example *i*'s correction is the risk computed for a *different* example. The
        multiset of risk values is preserved — it is a permutation, not a different
        critic — so only the example -> risk association is destroyed.

        A batch of one admits no derangement. That case is *logged* rather than
        silently collapsing into B3, because a C3 that is secretly B3 would poison the
        A1 comparison in exactly the direction that flatters the hypothesis.
        """
        if self.control != "C3":
            return risk_fn
        if n <= 1:
            extra["c3_degenerate"] = 1.0
            return risk_fn
        return lambda h: torch.roll(risk_fn(h), shifts=1, dims=0)

    @staticmethod
    def _windowed(risk_fn: Callable[[Tensor], Tensor], window: Tensor) -> Callable[[Tensor], Tensor]:
        """Zero the risk outside the active flow-step window.

        Rows outside the window are then trivially feasible and the operator returns
        them bitwise unchanged — which is the per-example version of "project only at
        the last ``m`` steps" (``05`` §4), without ever splitting the batch.
        """

        def windowed(h: Tensor) -> Tensor:
            risk = risk_fn(h)
            return torch.where(window, risk, torch.zeros_like(risk))

        return windowed

    def _correct(
        self,
        h_c: Tensor,
        h0: Tensor | None,
        risk_fn: Callable[[Tensor], Tensor],
        window: Tensor,
        *,
        seed_key: tuple[int, int],
    ) -> tuple[Tensor, ProjectionInfo | None, dict[str, float]]:
        """Apply the method's correction, honouring the active control code."""
        extra: dict[str, float] = {"control_code": {"none": 0.0, "C1": 1.0, "C2": 2.0, "C3": 3.0}[self.control]}
        if self.control == "C3":
            # Recorded unconditionally, including for decisions the flow window leaves
            # alone: a control that silently is not running must never be
            # indistinguishable from one that is (CLAUDE.md 4).
            extra["c3_permutation"] = 1.0
        if not bool(window.any()):
            return h_c, None, extra
        risk = self._windowed(
            self._control_risk_fn(risk_fn, n=int(h_c.shape[0]), extra=extra), window
        )
        h_safe, info = project(h_c, risk, self.proj_cfg)
        if self.control == "none":
            return h_safe, info, extra

        # Every control needs B3's correction norm to match, so the reference
        # projection above is *always* run first and then overridden.
        u_ref = (h_safe - h_c).detach()
        norm_ref = u_ref.flatten(1).norm(dim=1)
        active = info.usable if info.usable is not None else torch.zeros_like(window)
        if bool(active.any()) and float(norm_ref.max()) > 0.0:
            if self.control == "C1":
                direction = common_random_noise(
                    seed_key[0],
                    "C1",
                    seed_key[1],
                    tuple(h_c.shape),
                    flow_step=1,
                    device=h_c.device,
                )
                flat = direction.flatten(1)
                unit = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                u = (unit * norm_ref.unsqueeze(1)).reshape(h_c.shape)
                h_safe = h_c + torch.where(
                    active.view(-1, *([1] * (h_c.dim() - 1))), u, torch.zeros_like(u)
                )
                extra["c1_norm_matched"] = float(norm_ref.max())
                extra["c1_norm_ratio"] = float(
                    (u.flatten(1).norm(dim=1) / norm_ref.clamp_min(1e-12)).mean()
                )
            elif self.control == "C2":
                if h0 is None:
                    raise RuntimeError(
                        "control C2 (shrink Δh toward the base) needs the pre-LoRA activation: "
                        "the adapter must expose it as PolicyOut.extras['h0']. The toy and "
                        "SmolVLA adapters do; a new adapter must too, or C2 cannot be honest."
                    )
                delta = (h_c - h0).detach()
                delta_norm = delta.flatten(1).norm(dim=1)
                scale = (norm_ref / delta_norm.clamp_min(1e-12)).clamp(max=1.0)
                u = (delta * scale.view(-1, *([1] * (h_c.dim() - 1)))).to(h_c.dtype)
                h_safe = h_c - torch.where(
                    active.view(-1, *([1] * (h_c.dim() - 1))), u, torch.zeros_like(u)
                )
                extra["c2_mean_scale"] = float(scale.mean())
                extra["c2_norm_ratio"] = float(
                    (u.flatten(1).norm(dim=1) / norm_ref.clamp_min(1e-12)).mean()
                )
            else:
                # C3: the correction *is* the projection against the permuted risk
                # computed above, so ``h_safe`` already holds it — nothing to override.
                # The event was logged before the window check.
                pass
        return h_safe, info, extra

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def training_loss(
        self,
        policy: Any,
        batch: dict[str, Any],
        cfg: ExperimentCfg,
        critic: RiskCritic | None,
        step: int,
    ) -> tuple[Tensor, dict[str, float]]:
        site = self.site_or_default(policy)
        out = policy.forward(dict(batch), site=site)
        is_flow = str(getattr(policy, "action_head", "flow_matching")) == "flow_matching"
        target = flow_target(batch["a0"], batch["eps"]) if is_flow else batch["a0"].float()
        h = out.h if out.h is not None else out.extras.get("h")
        info: dict[str, float] = {}

        def objective(head_output: Tensor) -> Tensor:
            # Flow head: regress the velocity target. Deterministic head: regress the
            # chunk. Either way the *activation* the head sees is the corrected one.
            return torch.nn.functional.mse_loss(head_output.float(), target.float())

        def head_output(ctx_: Any, h_: Tensor) -> Tensor:
            if is_flow:
                velocity = policy.tail_velocity(ctx_, h_, site)
                if velocity is None:  # pragma: no cover - parity with flow_matching_loss
                    raise RuntimeError(
                        f"{type(policy).__name__} reports a flow head but produced no velocity"
                    )
                return velocity
            return policy.f_act(ctx_, h_, site)

        if not self.train_projects or critic is None or h is None:
            plain = out.velocity if is_flow else out.chunk
            if plain is None:  # pragma: no cover - guarded by the adapter contracts
                raise RuntimeError(f"{type(policy).__name__} returned neither velocity nor chunk")
            info["projected"] = 0.0
            return objective(plain), info

        if critic.training:
            raise RuntimeError(
                "the critic must be in eval() mode during projected training (06 §3)"
            )
        ctx = policy.make_context(dict(batch), site=site)
        critic_ctx = critic_context_from_batch(batch)
        risk_fn = self._risk_fn(policy, critic, ctx, critic_ctx, site)
        # Per-example flow window: rows above tau_proj are left alone (05 §4).
        window = (
            batch["tau"] <= self.proj_cfg.tau_proj
            if is_flow
            else torch.ones(h.shape[0], dtype=torch.bool)
        )
        h0 = out.extras.get("h0")
        h_safe, proj_info, extra = self._correct(
            h, h0 if isinstance(h0, Tensor) else None, risk_fn, window, seed_key=(cfg.seed, step)
        )
        loss = objective(head_output(ctx, h_safe))
        info.update(extra)
        info["loss_flow"] = float(loss.detach())
        info["projected"] = 1.0
        info["window_frac"] = float(window.float().mean())
        if proj_info is not None:
            summary = proj_info.summary()
            info["n_active"] = summary.get("n_active", 0.0)
            info["n_unusable"] = summary.get("n_unusable", 0.0)
            info["corr_rel"] = summary.get("mean_corr_rel", 0.0)
            if proj_info.r_cand is not None and proj_info.r_after is not None:
                # E2 in miniature, per training step: nonlinear vs linearised risk.
                info["r_cand"] = float(proj_info.r_cand.mean())
                info["r_after"] = float(proj_info.r_after.mean())
                info["r_pred"] = float(proj_info.r_pred.mean()) if proj_info.r_pred is not None else float("nan")
        return loss, info

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
        if self.train_projects and critic is None:
            raise ValueError(
                f"{self.name} projects during training, so it needs a critic (06 §3). "
                "Train the critic first, or use B1+TTP for test-time projection only."
            )
        return super().train(
            cfg,
            policy,
            train_seed=train_seed,
            data=data,
            critic=critic,
            out_dir=out_dir,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def wrap(
        self, policy: Any, *, critic: RiskCritic | None = None, base_seed: int = 0
    ) -> PolicyWrapper:
        mode = self.method_cfg.inference_mode
        if mode == "0":
            return super().wrap(policy, critic=critic, base_seed=base_seed)
        if critic is None:
            raise ValueError(
                f"{self.name} intervenes at inference in mode {mode!r}, which is defined by the "
                "critic; pass the trained critic (06 §3)"
            )
        if critic.training:
            raise RuntimeError("wrap() requires critic.eval(): a stochastic critic breaks the projection")
        ladder = FallbackLadder(window=10, enabled=bool(self.proj_cfg.fallback))
        wrapper = PolicyWrapper(
            self.name,
            policy,
            critic=critic,
            proj_cfg=self.proj_cfg,
            site=self.site_or_default(policy),
            control=self.control,
            mode=mode,
            ladder=ladder,
            base_seed=base_seed,
        )
        site = wrapper.site
        if mode == "lat":
            wrapper.intervention = _site_intervention(self, policy, critic, wrapper, site, base_seed)
        elif mode == "act-post":
            wrapper.intervention = _action_intervention(self, policy, critic, wrapper, site, base_seed)
        elif mode == "act-flow":
            wrapper.intervention = _flow_intervention(self, policy, critic, wrapper, site, base_seed)
        else:  # pragma: no cover - config validation restricts the vocabulary
            raise ValueError(f"ProjectedMethod cannot serve inference_mode {mode!r}")
        return wrapper

    # ------------------------------------------------------------------
    # fallback
    # ------------------------------------------------------------------
    def _consult_ladder(
        self,
        wrapper: PolicyWrapper,
        policy: Any,
        critic: RiskCritic,
        batch: dict[str, Any],
        candidate: Tensor,
        *,
        flow_step: int,
    ) -> Tensor:
        """F1 -> F2 -> F3 for a chunk the projection could not certify (``01`` §3)."""
        ladder = wrapper.ladder
        if ladder is None:
            return candidate
        base = _base_action(policy, batch, site=wrapper.site)
        critic_ctx = critic_context_from_batch(batch)
        base_risk = critic.risk(critic_ctx, base, space=self.proj_cfg.space)
        decision = ladder.update(
            usable=torch.zeros(candidate.shape[0], dtype=torch.bool),
            action_candidate=candidate,
            base_action=base,
            base_action_risk=base_risk,
            threshold=threshold_in_space(self.proj_cfg),
            evidence={"method": self.name, "flow_step": float(flow_step)},
        )
        wrapper.last_fallback = decision
        wrapper.log_fallback(decision, flow_step=flow_step)
        ladder.note_action(decision.action if decision.action is not None else candidate)
        return decision.action if decision.action is not None else candidate


# --------------------------------------------------------------------------
# interventions
# --------------------------------------------------------------------------


def _ensure_eps(
    batch: dict[str, Any],
    base_seed: int,
    episode_id: int | str,
    step: int,
    policy: Any = None,
) -> dict[str, Any]:
    """Fill ``batch['eps']`` from the shared CRN stream when the caller has none.

    Thin alias for :func:`riskproj.methods.base.ensure_eps`, kept so the inference
    closures read locally. The shared stream is what makes method comparisons paired
    (``08`` §4).
    """
    return ensure_eps(
        batch, base_seed=base_seed, episode_id=episode_id, step=step, policy=policy
    )


def _base_action(policy: Any, batch: dict[str, Any], *, site: str) -> Tensor:
    """Zero-LoRA action: the F1 reference (``01`` §3)."""
    with lora_disabled(policy):
        return policy.forward(dict(batch), site=site).chunk.detach()


def _site_intervention(
    method: ProjectedMethod,
    policy: Any,
    critic: RiskCritic,
    wrapper: PolicyWrapper,
    site: str,
    base_seed: int,
) -> Callable[[dict[str, Any]], PolicyOut]:
    """Mode ``lat``: project the site activation at the last ``m`` denoising steps."""
    last: dict[str, Any] = {"unusable": False, "risk": float("nan")}

    def _pre_lora_activation(ctx: Any) -> Tensor | None:
        """Pre-LoRA activation, needed by control C2 (shrink Δh toward zero).

        Adapters put it in ``PolicyOut.extras['h0']``; when the context does not
        carry it, one zero-LoRA forward recovers it rather than silently degrading
        C2 into a no-op.
        """
        h0 = ctx.extras.get("h0")
        if isinstance(h0, Tensor):
            return h0
        if method.control != "C2":
            return None
        with lora_disabled(policy):
            base_out = policy.forward(dict(ctx.batch), site=site)
        candidate = base_out.extras.get("h0")
        return candidate if isinstance(candidate, Tensor) else None

    def project_hook(h: Tensor, ctx: Any) -> tuple[Tensor, ProjectionInfo | None]:
        critic_ctx = critic_context_from_batch(ctx.batch)
        risk_fn = method._risk_fn(policy, critic, ctx, critic_ctx, site)
        tau = ctx.tau
        window = (
            torch.ones(h.shape[0], dtype=torch.bool)
            if tau is None
            else (tau.reshape(-1) <= method.proj_cfg.tau_proj)
        )
        h_safe, info, _extra = method._correct(
            h, _pre_lora_activation(ctx), risk_fn, window, seed_key=(base_seed, wrapper.step)
        )
        if info is not None:
            wrapper._log_projection(info, flow_step=int(ctx.extras.get("flow_step", -1)), tau=tau)
            usable = bool(info.usable.any()) if info.usable is not None else False
            if bool(info.active.any()) and not usable:
                last["unusable"] = True
            if info.r_cand is not None:
                last["risk"] = float(info.r_cand.mean())
        return h_safe, info

    def intervention(batch: dict[str, Any]) -> PolicyOut:
        last["unusable"] = False
        data = _ensure_eps(batch, base_seed, wrapper.episode_id, wrapper.step, policy)
        out = run_inference(
            policy, data, site=site, proj_cfg=method.proj_cfg, project_hook=project_hook
        )
        if last["unusable"]:
            out = PolicyOut(
                chunk=method._consult_ladder(
                    wrapper, policy, critic, data, out.chunk, flow_step=-1
                ),
                site=out.site,
                h=out.h,
                extras=dict(out.extras),
            )
        elif wrapper.ladder is not None:
            wrapper.ladder.note_action(out.chunk.detach())
        return out

    return intervention


def _action_intervention(
    method: ProjectedMethod,
    policy: Any,
    critic: RiskCritic,
    wrapper: PolicyWrapper,
    site: str,
    base_seed: int,
) -> Callable[[dict[str, Any]], PolicyOut]:
    """Mode ``act-post`` (``S0``): project the final chunk itself, Dalal-style."""

    def intervention(batch: dict[str, Any]) -> PolicyOut:
        data = _ensure_eps(batch, base_seed, wrapper.episode_id, wrapper.step, policy)
        out = policy.forward(data, site=site)
        critic_ctx = critic_context_from_batch(data)
        horizon = policy_horizon(policy)

        # S0's downstream map is the identity: the "site" is the action itself.
        def risk_fn(a: Tensor) -> Tensor:
            if a.dim() != 3 or a.shape[1] != horizon:
                raise ValueError(
                    f"action-space projection needs a [B, {horizon}, A] chunk; got {tuple(a.shape)}"
                )
            return critic.risk(critic_ctx, a, space=method.proj_cfg.space)

        window = torch.ones(out.chunk.shape[0], dtype=torch.bool)
        chunk_safe, info, _extra = method._correct(
            out.chunk, None, risk_fn, window, seed_key=(base_seed, wrapper.step)
        )
        if info is not None:
            wrapper._log_projection(info, flow_step=-1, tau=None)
        if info is not None and bool(info.active.any()) and not bool(info.usable.any()):
            chunk_safe = method._consult_ladder(
                wrapper, policy, critic, data, chunk_safe, flow_step=-1
            )
        elif wrapper.ladder is not None:
            wrapper.ladder.note_action(chunk_safe)
        return PolicyOut(
            chunk=chunk_safe, site=site, h=out.h, tau=out.tau, extras=dict(out.extras)
        )

    return intervention


def _flow_intervention(
    method: ProjectedMethod,
    policy: Any,
    critic: RiskCritic,
    wrapper: PolicyWrapper,
    site: str,
    base_seed: int,
) -> Callable[[dict[str, Any]], PolicyOut]:
    """Mode ``act-flow`` (``S0f``): critic-guided correction of ``x_tau`` in-flow."""
    last: dict[str, Any] = {"unusable": False}

    def flow_hook(x: Tensor, tau: Tensor, ctx: Any) -> tuple[Tensor, ProjectionInfo | None]:
        critic_ctx = critic_context_from_batch(ctx.batch)

        def risk_fn(xx: Tensor) -> Tensor:
            v = policy.step_velocity(ctx, xx, tau, site)
            return critic.risk(critic_ctx, terminal_estimate(xx, v, tau), space=method.proj_cfg.space)

        window = tau.reshape(-1) <= method.proj_cfg.tau_proj
        x_safe, info, _extra = method._correct(
            x, None, risk_fn, window, seed_key=(base_seed, wrapper.step)
        )
        if info is not None:
            wrapper._log_projection(info, flow_step=int(ctx.extras.get("flow_step", -1)), tau=tau)
            if bool(info.active.any()) and not bool(info.usable.any()):
                last["unusable"] = True
        return x_safe, info

    def intervention(batch: dict[str, Any]) -> PolicyOut:
        last["unusable"] = False
        data = _ensure_eps(batch, base_seed, wrapper.episode_id, wrapper.step, policy)
        out = run_inference(
            policy, data, site=site, proj_cfg=method.proj_cfg, flow_hook=flow_hook
        )
        if last["unusable"]:
            out = PolicyOut(
                chunk=method._consult_ladder(
                    wrapper, policy, critic, data, out.chunk, flow_step=-1
                ),
                site=out.site,
                h=out.h,
                extras=dict(out.extras),
            )
        elif wrapper.ladder is not None:
            wrapper.ladder.note_action(out.chunk)
        return out

    return intervention


@register("method", "B3")
def _make_b3(method_cfg: MethodCfg) -> ProjectedMethod:
    return ProjectedMethod(method_cfg)


@register("method", "B1+TTP")
def _make_b1_ttp(method_cfg: MethodCfg) -> ProjectedMethod:
    return ProjectedMethod(method_cfg)


@register("method", "B3-trainonly")
def _make_b3_trainonly(method_cfg: MethodCfg) -> ProjectedMethod:
    return ProjectedMethod(method_cfg)


@register("method", "C1")
def _make_c1(method_cfg: MethodCfg) -> ProjectedMethod:
    return ProjectedMethod(method_cfg)


@register("method", "C2")
def _make_c2(method_cfg: MethodCfg) -> ProjectedMethod:
    return ProjectedMethod(method_cfg)


@register("method", "C3")
def _make_c3(method_cfg: MethodCfg) -> ProjectedMethod:
    return ProjectedMethod(method_cfg)
