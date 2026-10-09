"""Episode driver: one method, one cell, one seed — and the raw logs it writes.

Everything downstream is derived from the two JSONL files this module appends to
(``skills/software_engineering.md`` §1.4): ``episodes.jsonl`` carries one
:class:`~riskproj.types.EpisodeRecord` per episode in the ``agents/09`` §3 schema,
``decisions.jsonl`` carries one :class:`~riskproj.types.DecisionRecord` per
projection decision, so a diagnostic can be recomputed offline without re-running
anything.

The driver is deliberately explicit about the things that have historically made
results uninterpretable:

* dynamics are applied and **read back on every episode** — a non-nominal cell
  that silently ran at nominal physics would invalidate every robustness number;
* flow noise comes from :func:`riskproj.utils.seed.common_random_noise`, keyed by
  ``(base_seed, env_seed, step, flow_step)`` and never by method, so two methods
  are compared on identical noise (``08`` §4);
* every projection and fallback event is logged (``CLAUDE.md`` #4), including the
  ones that did nothing.

The chunk planner is a small contract (:class:`ChunkPlanner`) so that
``methods/`` can implement B0-B6 and the negative controls without touching this
module; the built-in planners cover the inference-only arms so the pipeline is
runnable and testable on its own.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from riskproj.data.splits import (
    episode_seeds,
    role_of_cell,
    sample_dynamics_for_cell,
)
from riskproj.envs import (
    DynReadback,
    SimState,
    ViolationMonitor,
    any_violation,
    apply_dynamics,
    failure_class,
    save_state,
    set_cell_id,
)
from riskproj.obs import CtxBuilder, env_obs_to_step_obs
from riskproj.projection import FallbackLadder, active_at_flow_step, project
from riskproj.types import (
    VIOLATION_TYPES,
    CellId,
    DecisionRecord,
    EpisodeRecord,
    FallbackCode,
    DynamicsRanges,
    DynamicsSpec,
    ProjectionInfo,
    empty_violation_counts,
)
from riskproj.utils.config import ProjCfg
from riskproj.utils.io import append_jsonl, capture_provenance, run_dir, write_json
from riskproj.utils.profiling import CudaTimer, peak_vram_mb
from riskproj.utils.seed import common_random_noise

__all__ = [
    "ChunkPlanner",
    "LatencyTracker",
    "NoProjectionPlanner",
    "PlannedChunk",
    "RolloutOutput",
    "RolloutSpec",
    "build_planner",
    "cell_episode_plan",
    "evaluate_method",
    "latency_tracker",
    "run_episode",
]

EPS = 1e-12


# --------------------------------------------------------------------------
# latency accounting
# --------------------------------------------------------------------------


class LatencyTracker:
    """Per-phase wall/CUDA time for one decision or one episode (``09`` §1 systems)."""

    PHASES = ("policy", "critic", "proj")

    def __init__(self) -> None:
        self.totals: dict[str, float] = dict.fromkeys(self.PHASES, 0.0)
        self.counts: dict[str, int] = dict.fromkeys(self.PHASES, 0)

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name not in self.totals:
            raise KeyError(f"unknown latency phase {name!r}; known: {sorted(self.totals)}")
        timer = CudaTimer()
        with timer:
            yield
        self.totals[name] += float(timer.ms)
        self.counts[name] += 1

    def means(self) -> dict[str, float]:
        """Mean ms per call, per phase — the number that goes in the episode log."""
        return {p: (self.totals[p] / self.counts[p] if self.counts[p] else 0.0) for p in self.PHASES}

    def merged_with(self, other: LatencyTracker) -> LatencyTracker:
        out = LatencyTracker()
        for p in self.PHASES:
            out.totals[p] = self.totals[p] + other.totals[p]
            out.counts[p] = self.counts[p] + other.counts[p]
        return out


@contextlib.contextmanager
def latency_tracker() -> Iterator[LatencyTracker]:
    """``with latency_tracker() as t: with t.phase("policy"): ...``"""
    yield LatencyTracker()


# --------------------------------------------------------------------------
# planner contract (implemented by methods/ for B0-B6 and C1-C3)
# --------------------------------------------------------------------------


@dataclass
class PlannedChunk:
    """A chunk produced by a planner, plus the events it must be logged with."""

    chunk: Tensor  # [B, H, A], the actions that will actually be executed
    decisions: list[DecisionRecord] = field(default_factory=list)
    #: risk of the emitted chunk, per example (critic-predicted; a diagnostic only)
    risks: list[float] = field(default_factory=list)
    #: most severe fallback code emitted while planning this chunk
    fallback_code: int = FallbackCode.NONE
    site: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class ChunkPlanner(Protocol):
    """What ``methods/`` must provide to the rollout driver.

    Implementations receive everything the driver knows and return the actions to
    execute plus the decisions to log. They must not execute anything themselves:
    the driver owns the environment, so that CRN, dynamics read-back and logging
    stay in one place.
    """

    name: str

    def plan(
        self,
        *,
        policy: Any,
        batch: dict[str, Any],
        ctx_batch: Mapping[str, Any],
        step: int,
        env_seed: int,
        crn_base_seed: int,
        tracker: LatencyTracker | None = None,
    ) -> PlannedChunk: ...


# --------------------------------------------------------------------------
# built-in planners
# --------------------------------------------------------------------------


def _device_of(policy: Any) -> torch.device:
    device = getattr(policy, "device_id", None)
    if device is not None:
        return torch.device(device)
    params = list(getattr(policy, "parameters", lambda: [])())
    if params:
        return params[0].device
    return torch.device("cpu")


def _context_to_device(ctx: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a critic context onto the critic's device.

    A CUDA critic scoring a CPU context is a crash, not a slowdown, and the context is
    built from numpy observations. One conversion here keeps every planner honest.
    """
    return {
        key: (
            value.to(device)
            if isinstance(value, Tensor)
            else torch.as_tensor(np.asarray(value), dtype=torch.float32).to(device)
        )
        for key, value in ctx.items()
    }


def _is_flow(policy: Any) -> bool:
    return str(getattr(policy, "action_head", "flow_matching")) == "flow_matching"


def _horizon(policy: Any) -> int:
    return int(getattr(policy, "chunk_horizon"))


def _action_dim(policy: Any) -> int:
    return int(getattr(policy, "action_dim"))


def _site(policy: Any, site: str | None) -> str:
    return site or str(getattr(policy, "default_site", "S2"))


def _decision_from_info(
    info: ProjectionInfo,
    *,
    ep_id: str,
    step: int,
    flow_step: int,
    extras: Mapping[str, Any] | None = None,
) -> DecisionRecord:
    """Flatten a per-batch :class:`ProjectionInfo` for the single-example rollout log."""

    def scalar(name: str, default: float | None = None) -> float | None:
        value = getattr(info, name, None)
        if value is None or not isinstance(value, Tensor) or value.numel() == 0:
            return default
        return float(value.detach().float().reshape(-1)[0])

    r_after = getattr(info, "r_after", None)
    r_pred = getattr(info, "r_pred", None)
    active = scalar("active", 0.0)
    usable = scalar("usable", 0.0)
    return DecisionRecord(
        ep_id=ep_id,
        step=int(step),
        flow_step=int(flow_step),
        tau=float(extras.get("tau")) if extras and extras.get("tau") is not None else None,
        r_cand=scalar("r_cand"),
        gn2=scalar("gn2"),
        active=bool(active),
        usable=bool(usable),
        k_used=int(scalar("k_used", 0.0) or 0),
        corr_rel=float(scalar("corr_rel", 0.0) or 0.0),
        r_after=[float(x) for x in _as_1d(r_after)],
        r_pred=[float(x) for x in _as_1d(r_pred)],
        ens_std=scalar("ens_std"),
        fallback_code=int(scalar("fallback_code", float(FallbackCode.NONE)) or 0),
        extra=dict(extras or {}),
    )


def _as_1d(value: Any) -> list[float]:
    if value is None or not isinstance(value, Tensor) or value.numel() == 0:
        return []
    return [float(x) for x in value.detach().float().reshape(-1)]


class _FlowPlannerBase:
    """Shared Euler integration and the optional mid-flow projection hook."""

    name = "base"

    def __init__(
        self,
        *,
        site: str | None = None,
        proj_cfg: ProjCfg | None = None,
        critic: Any | None = None,
        horizon: int | None = None,
    ) -> None:
        self.site = site
        self.proj_cfg = proj_cfg
        self.critic = critic
        #: Optional cross-check against the policy's own chunk horizon (never a target).
        self.fixed_horizon = horizon
        self._action_dim_hint = int(getattr(critic, "action_dim", 0)) or 4

    # -- helpers ---------------------------------------------------------
    def _num_steps(self, policy: Any, batch: Mapping[str, Any]) -> int:
        return int(batch.get("num_flow_steps") or getattr(policy, "num_flow_steps", 10))

    def _initial_noise(
        self,
        *,
        batch_size: int,
        horizon: int,
        step: int,
        env_seed: int,
        crn_base_seed: int,
        sample: int = 0,
    ) -> Tensor:
        """Flow noise for one decision, keyed so every method sees the same draw."""
        if batch_size < 1 or horizon < 1:
            raise ValueError(f"bad noise shape ({batch_size}, {horizon}, A)")
        return common_random_noise(
            crn_base_seed,
            env_seed,
            step,
            (batch_size, horizon, self._action_dim_hint),
            flow_step=sample,
            dtype=torch.float32,
        )

    def _horizon(self, policy: Any | None, batch: Mapping[str, Any]) -> int:
        """Chunk length: the head's own horizon. A planner cannot change it.

        The action head emits a fixed ``chunk_horizon`` vector, so a planner that
        produced a different length would silently mis-broadcast against the velocity
        or the critic. How many of those actions are *executed* is
        ``RolloutSpec.exec_horizon``, which the driver applies when it trims the chunk.
        """
        if "x_tau" in batch and isinstance(batch["x_tau"], Tensor):
            return int(batch["x_tau"].shape[1])
        native = int(getattr(policy, "chunk_horizon"))
        if self.fixed_horizon is not None and int(self.fixed_horizon) != native:
            raise ValueError(
                f"planner horizon {self.fixed_horizon} != the policy's chunk horizon {native}: "
                "the action head fixes the chunk length, so use RolloutSpec.exec_horizon to "
                "choose how many of those actions are executed"
            )
        return native

    def _risk_fn(self, policy: Any, ctx_batch: Mapping[str, Any], ctx: Any, site: str) -> Callable[[Tensor], Tensor]:
        from riskproj.critic.model import make_risk_fn

        if self.critic is None:
            raise RuntimeError("projection planners need a critic")
        space = self.proj_cfg.space if self.proj_cfg is not None else "prob"
        return make_risk_fn(
            self.critic,
            dict(ctx_batch),
            lambda h: policy.f_act(ctx, h, site),
            space=space,
            chunk_horizon=int(getattr(policy, "chunk_horizon")),
        )

    def _flow_context(self, policy: Any, batch: Mapping[str, Any], x_tau: Tensor, tau: Tensor, site: str) -> Any:
        ctx = policy.make_context(dict(batch), site)
        ctx.x_tau = x_tau
        ctx.tau = tau
        return ctx

    def _euler(
        self,
        policy: Any,
        batch: dict[str, Any],
        *,
        step: int,
        env_seed: int,
        crn_base_seed: int,
        ctx_batch: Mapping[str, Any],
        ep_id: str,
        tracker: LatencyTracker | None,
        sample: int = 0,
        project_site: bool = False,
    ) -> tuple[Tensor, list[DecisionRecord]]:
        """Integrate ``tau: 1 -> 0`` and optionally project at the last m steps."""
        from riskproj.policies.flow import euler_taus

        horizon, action_dim = _horizon(policy), _action_dim(policy)
        site = _site(policy, self.site)
        num_steps = self._num_steps(policy, batch)
        taus = euler_taus(num_steps)
        self._action_dim_hint = action_dim
        x = self._initial_noise(
            batch_size=int(batch["state"].shape[0]),
            horizon=horizon,
            step=step,
            env_seed=env_seed,
            crn_base_seed=crn_base_seed,
            sample=sample,
        ).reshape(-1, horizon, action_dim)
        decisions: list[DecisionRecord] = []

        for i in range(num_steps):
            tau_i = taus[i].reshape(1).expand(1)
            self._fill_batch(batch, x, tau_i)
            ctx = self._flow_context(policy, batch, x, tau_i, site)
            if tracker is not None:
                with tracker.phase("policy"):
                    out = policy.forward(dict(batch), site=site)
                h_c = out.h
            else:
                out = policy.forward(dict(batch), site=site)
                h_c = out.h
            if h_c is None:
                raise RuntimeError(
                    f"policy {policy.name!r} forward did not expose a site activation; "
                    "projection needs h (07 §1)"
                )

            should_project = bool(
                project_site
                and self.proj_cfg is not None
                and self.critic is not None
                and bool(active_at_flow_step(tau_i, i, num_steps, self.proj_cfg).all())
            )
            if should_project:
                risk_fn = self._risk_fn(policy, ctx_batch, ctx, site)
                if tracker is not None:
                    with tracker.phase("proj"):
                        h_safe, info = project(h_c, risk_fn, self.proj_cfg)  # type: ignore[arg-type]
                else:
                    h_safe, info = project(h_c, risk_fn, self.proj_cfg)  # type: ignore[arg-type]
                if tracker is not None:
                    with tracker.phase("critic"):
                        # Nonlinear risk of the emitted activation, for the log: the
                        # critic is a diagnostic here, never a success metric.
                        r_after = float(risk_fn(h_safe.detach()).detach().float().reshape(-1)[0])
                else:
                    r_after = float(risk_fn(h_safe.detach()).detach().float().reshape(-1)[0])
                decisions.append(
                    _decision_from_info(
                        info,
                        ep_id=ep_id,
                        step=step,
                        flow_step=i,
                        extras={
                            "site": site,
                            "mode": "latent",
                            "r_after_nonlinear": r_after,
                            "tau": float(tau_i.reshape(-1)[0]),
                        },
                    )
                )
                if tracker is not None:
                    with tracker.phase("policy"):
                        velocity = policy.tail_velocity(ctx, h_safe, site)
                else:
                    velocity = policy.tail_velocity(ctx, h_safe, site)
            else:
                velocity = out.velocity
                if velocity is None:
                    raise RuntimeError(
                        "flow head did not return a velocity; the planner cannot integrate"
                    )
            dt = (taus[i + 1] - taus[i]).to(x.dtype)
            x = x + dt * velocity.detach()
        return x, decisions

    @staticmethod
    def _fill_batch(batch: dict[str, Any], x_tau: Tensor, tau: Tensor) -> None:
        batch["x_tau"] = x_tau
        batch["tau"] = tau
        batch["proj_tau_max"] = None
        batch["proj_steps"] = None

    # -- planning --------------------------------------------------------
    def plan(
        self,
        *,
        policy: Any,
        batch: dict[str, Any],
        ctx_batch: Mapping[str, Any],
        step: int,
        env_seed: int,
        crn_base_seed: int,
        tracker: LatencyTracker | None = None,
    ) -> PlannedChunk:  # pragma: no cover - overridden
        raise NotImplementedError


class NoProjectionPlanner(_FlowPlannerBase):
    """B0/B1: the policy's own chunk, no critic anywhere."""

    name = "none"

    def plan(
        self,
        *,
        policy: Any,
        batch: dict[str, Any],
        ctx_batch: Mapping[str, Any],
        step: int,
        env_seed: int,
        crn_base_seed: int,
        tracker: LatencyTracker | None = None,
    ) -> PlannedChunk:
        if not _is_flow(policy):
            if tracker is not None:
                with tracker.phase("policy"):
                    out = policy.forward(dict(batch), site=self.site)
            else:
                out = policy.forward(dict(batch), site=self.site)
            return PlannedChunk(chunk=out.chunk.detach(), site=self.site)
        chunk, decisions = self._euler(
            policy,
            batch,
            step=step,
            env_seed=env_seed,
            crn_base_seed=crn_base_seed,
            ctx_batch=ctx_batch,
            ep_id="",
            tracker=tracker,
            project_site=False,
        )
        return PlannedChunk(chunk=chunk.detach(), decisions=decisions, site=self.site)


class LatentProjectionPlanner(_FlowPlannerBase):
    """B3: project the site activation at the last ``steps_active`` flow steps."""

    name = "latent"

    def plan(
        self,
        *,
        policy: Any,
        batch: dict[str, Any],
        ctx_batch: Mapping[str, Any],
        step: int,
        env_seed: int,
        crn_base_seed: int,
        tracker: LatencyTracker | None = None,
    ) -> PlannedChunk:
        if self.proj_cfg is None or self.critic is None:
            raise RuntimeError("LatentProjectionPlanner needs proj_cfg and critic")
        if not _is_flow(policy):
            # Deterministic head (ACT/VLA-Adapter): there is no flow step to project
            # inside, so the only latent site is the head input — handled as a
            # single-step projection of the produced chunk.
            return self._regression_plan(
                policy, batch, ctx_batch, step=step, tracker=tracker
            )
        chunk, decisions = self._euler(
            policy,
            batch,
            step=step,
            env_seed=env_seed,
            crn_base_seed=crn_base_seed,
            ctx_batch=ctx_batch,
            ep_id="",
            tracker=tracker,
            project_site=True,
        )
        return PlannedChunk(chunk=chunk.detach(), decisions=decisions, site=self.site)

    def _regression_plan(
        self,
        policy: Any,
        batch: dict[str, Any],
        ctx_batch: Mapping[str, Any],
        *,
        step: int,
        tracker: LatencyTracker | None,
    ) -> PlannedChunk:
        site = _site(policy, self.site)
        with torch.enable_grad():
            out = policy.forward(dict(batch), site=site)
            h_c = out.h
            if h_c is None:
                raise RuntimeError("policy forward did not expose h")
            ctx = policy.make_context(dict(batch), site)
            risk_fn = self._risk_fn(policy, ctx_batch, ctx, site)
            h_safe, info = project(h_c, risk_fn, self.proj_cfg)  # type: ignore[arg-type]
            chunk = policy.f_act(ctx, h_safe, site)
        return PlannedChunk(
            chunk=chunk.detach(),
            decisions=[
                _decision_from_info(info, ep_id="", step=step, flow_step=-1, extras={"site": site})
            ],
            site=site,
        )


class ActionPostPlanner(_FlowPlannerBase):
    """B4a: plan normally, then project the *final chunk* (Dalal-style, ``08`` §2)."""

    name = "act-post"

    def plan(
        self,
        *,
        policy: Any,
        batch: dict[str, Any],
        ctx_batch: Mapping[str, Any],
        step: int,
        env_seed: int,
        crn_base_seed: int,
        tracker: LatencyTracker | None = None,
    ) -> PlannedChunk:
        if self.proj_cfg is None or self.critic is None:
            raise RuntimeError("ActionPostPlanner needs proj_cfg and critic")
        base = NoProjectionPlanner(site=self.site, horizon=self.fixed_horizon, critic=None)
        planned = base.plan(
            policy=policy,
            batch=batch,
            ctx_batch=ctx_batch,
            step=step,
            env_seed=env_seed,
            crn_base_seed=crn_base_seed,
            tracker=tracker,
        )
        from riskproj.critic.model import make_risk_fn

        space = self.proj_cfg.space
        risk_fn = make_risk_fn(
            self.critic,
            dict(ctx_batch),
            lambda chunk: chunk,
            space=space,
            chunk_horizon=int(getattr(policy, "chunk_horizon")),
        )
        with torch.enable_grad():
            chunk_safe, info = project(planned.chunk, risk_fn, self.proj_cfg)
        decision = _decision_from_info(
            info, ep_id="", step=step, flow_step=-1, extras={"site": "S0", "mode": "action_post"}
        )
        return PlannedChunk(
            chunk=chunk_safe.detach(), decisions=[decision], site="S0", extras={"r_cand": decision.r_cand}
        )


class ActionFlowPlanner(_FlowPlannerBase):
    """B4b: correct the noisy chunk ``x_tau`` at the last m flow steps."""

    name = "act-flow"

    def plan(
        self,
        *,
        policy: Any,
        batch: dict[str, Any],
        ctx_batch: Mapping[str, Any],
        step: int,
        env_seed: int,
        crn_base_seed: int,
        tracker: LatencyTracker | None = None,
    ) -> PlannedChunk:
        if self.proj_cfg is None or self.critic is None:
            raise RuntimeError("ActionFlowPlanner needs proj_cfg and critic")
        if not _is_flow(policy):
            raise ValueError(
                "B4b corrects x_tau, so it needs a stochastic (flow-matching) head; "
                "ACT/VLA-Adapter have none (08 §2)"
            )
        from riskproj.critic.model import make_risk_fn
        from riskproj.policies.flow import euler_taus, terminal_estimate

        horizon, action_dim = _horizon(policy), _action_dim(policy)
        site = _site(policy, self.site)
        num_steps = self._num_steps(policy, batch)
        taus = euler_taus(num_steps)
        self._action_dim_hint = action_dim
        x = self._initial_noise(
            batch_size=int(batch["state"].shape[0]),
            horizon=horizon,
            step=step,
            env_seed=env_seed,
            crn_base_seed=crn_base_seed,
        ).reshape(-1, horizon, action_dim)
        decisions: list[DecisionRecord] = []
        for i in range(num_steps):
            tau_i = taus[i].reshape(1).expand(1)
            self._fill_batch(batch, x, tau_i)
            ctx = self._flow_context(policy, batch, x, tau_i, site)
            if bool(active_at_flow_step(tau_i, i, num_steps, self.proj_cfg).all()):
                risk_fn = make_risk_fn(
                    self.critic,
                    dict(ctx_batch),
                    lambda xx: terminal_estimate(xx, policy.step_velocity(ctx, xx, tau_i, site), tau_i),
                    space=self.proj_cfg.space,
                    chunk_horizon=int(getattr(policy, "chunk_horizon")),
                )
                with torch.enable_grad():
                    x_safe, info = project(x, risk_fn, self.proj_cfg)
                decisions.append(
                    _decision_from_info(
                        info, ep_id="", step=step, flow_step=i, extras={"site": "S0f", "mode": "action_flow"}
                    )
                )
                x = x_safe
            velocity = policy.step_velocity(ctx, x, tau_i, site)
            x = x + (taus[i + 1] - taus[i]).to(x.dtype) * velocity.detach()
        return PlannedChunk(chunk=x.detach(), decisions=decisions, site="S0f")


class BestOfNPlanner(_FlowPlannerBase):
    """B5: sample N chunks and keep the one the critic likes (V-GPS style)."""

    name = "bon"

    def __init__(self, *, n: int = 8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if n < 1:
            raise ValueError("best_of_n must be >= 1")
        self.n = int(n)

    def plan(
        self,
        *,
        policy: Any,
        batch: dict[str, Any],
        ctx_batch: Mapping[str, Any],
        step: int,
        env_seed: int,
        crn_base_seed: int,
        tracker: LatencyTracker | None = None,
    ) -> PlannedChunk:
        if self.critic is None:
            raise RuntimeError("BestOfNPlanner needs a critic to rank samples")
        base = NoProjectionPlanner(site=self.site, horizon=self.fixed_horizon)
        risks: list[float] = []
        chunks: list[Tensor] = []
        for sample in range(self.n):
            chunk, _ = (
                base._euler(  # noqa: SLF001 -- same class hierarchy, deliberate reuse
                    policy,
                    batch,
                    step=step,
                    env_seed=env_seed,
                    crn_base_seed=crn_base_seed,
                    ctx_batch=ctx_batch,
                    ep_id="",
                    tracker=tracker,
                    sample=sample,
                    project_site=False,
                )
                if _is_flow(policy)
                else (policy.forward(dict(batch), site=self.site).chunk, [])
            )
            chunks.append(chunk.detach())
            with torch.no_grad():
                risks.append(
                    float(
                        self.critic.risk(dict(ctx_batch), chunk.detach()).float().reshape(-1)[0]
                    )
                )
        best = int(np.argmin(np.asarray(risks, dtype=float)))
        return PlannedChunk(
            chunk=chunks[best],
            risks=risks,
            site=self.site,
            extras={"chosen_sample": best, "n_samples": self.n, "risk_best": risks[best]},
        )


class GatePlanner(_FlowPlannerBase):
    """B6: no correction — the critic only decides whether to fall back."""

    name = "gate"

    def __init__(self, *, hold_action: Tensor | None = None, window: int = 10, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ladder = FallbackLadder(window=window)
        self.hold_action = hold_action

    def plan(
        self,
        *,
        policy: Any,
        batch: dict[str, Any],
        ctx_batch: Mapping[str, Any],
        step: int,
        env_seed: int,
        crn_base_seed: int,
        tracker: LatencyTracker | None = None,
    ) -> PlannedChunk:
        if self.critic is None or self.proj_cfg is None:
            raise RuntimeError("GatePlanner needs critic and proj_cfg (for tau)")
        base = NoProjectionPlanner(site=self.site, horizon=self.fixed_horizon)
        planned = base.plan(
            policy=policy,
            batch=batch,
            ctx_batch=ctx_batch,
            step=step,
            env_seed=env_seed,
            crn_base_seed=crn_base_seed,
            tracker=tracker,
        )
        with torch.no_grad():
            risk = self.critic.risk(dict(ctx_batch), planned.chunk).float()
        tau = float(self.proj_cfg.setpoint)
        usable = (risk <= tau).reshape(1)
        hold = self.hold_action
        if hold is None:
            hold = torch.zeros_like(planned.chunk)
        decision = self.ladder.update(
            usable=usable,
            action_candidate=planned.chunk,
            base_action=hold,
            base_action_risk=None,
            threshold=tau,
            evidence={"risk": float(risk.reshape(-1)[0]), "step": float(step)},
        )
        chose_fallback = bool((decision.per_example != FallbackCode.NONE).any())
        action = decision.action if decision.action is not None else planned.chunk
        record = DecisionRecord(
            ep_id="",
            step=int(step),
            flow_step=-1,
            r_cand=float(risk.reshape(-1)[0]),
            active=not bool(usable.all()),
            usable=False,
            fallback_code=int(decision.code),
            extra={"site": "S0", "mode": "gate", "reason": decision.reason},
        )
        return PlannedChunk(
            chunk=action.detach(),
            decisions=[*planned.decisions, record],
            risks=[float(risk.reshape(-1)[0])],
            fallback_code=int(decision.code),
            site="S0",
            extras={"gated": chose_fallback},
        )


_PLANNERS: dict[str, str] = {
    "0": "none",
    "lat": "latent",
    "act-post": "act-post",
    "act-flow": "act-flow",
    "bon": "bon",
    "gate": "gate",
}


def build_planner(
    inference_mode: str,
    *,
    proj_cfg: ProjCfg | None = None,
    critic: Any | None = None,
    site: str | None = None,
    best_of_n: int = 8,
    horizon: int | None = None,
    hold_action: Tensor | None = None,
) -> ChunkPlanner:
    """Map an ``agents/08`` inference mode onto a planner."""
    if inference_mode not in _PLANNERS:
        raise ValueError(f"unknown inference_mode {inference_mode!r}; expected one of {sorted(_PLANNERS)}")
    common = {"proj_cfg": proj_cfg, "critic": critic, "site": site, "horizon": horizon}
    if inference_mode == "0":
        return NoProjectionPlanner(**common)  # type: ignore[arg-type]
    if inference_mode == "lat":
        return LatentProjectionPlanner(**common)  # type: ignore[arg-type]
    if inference_mode == "act-post":
        return ActionPostPlanner(**common)  # type: ignore[arg-type]
    if inference_mode == "act-flow":
        return ActionFlowPlanner(**common)  # type: ignore[arg-type]
    if inference_mode == "bon":
        return BestOfNPlanner(n=best_of_n, **common)  # type: ignore[arg-type]
    return GatePlanner(hold_action=hold_action, **common)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# episode driver
# --------------------------------------------------------------------------


@dataclass
class RolloutSpec:
    """One episode's identity. Everything that could change a number lives here."""

    method: str
    train_seed: int
    cell: CellId
    split: str
    env_seed: int
    max_steps: int = 200
    #: steps executed open-loop before re-planning; the caller sets this from
    #: ``env.exec_horizon`` (and to 1 for the E0 re-planning comparison).
    exec_horizon: int = 10
    record_decisions: bool = True
    save_states: bool = False
    save_every: int = 1
    crn_base_seed: int = 0
    #: A crashed episode is logged and re-raised unless this is set (never set it
    #: for a pre-registered run: a silent failure is worse than a stopped run).
    allow_errors: bool = False
    run_id: str = "local"
    #: Extra fields merged into the episode record's ``extra``. The train x inference
    #: factorial (T3, ``01`` §6) is read from ``extra['train_mode']``/
    #: ``extra['inference_mode']``, so a run that wants that table must record the
    #: modes it actually used instead of letting an analyser infer them from a name.
    record_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RolloutOutput:
    episode: EpisodeRecord
    decisions: list[DecisionRecord] = field(default_factory=list)
    states: list[SimState] = field(default_factory=list)
    plan_actions: list[np.ndarray] = field(default_factory=list)


def cell_episode_plan(
    cells: Sequence[CellId], eps_per_cell: int, base_seed: int
) -> list[tuple[CellId, int]]:
    """Deterministic ``[(cell, env_seed)]`` plan, recorded with every run."""
    if eps_per_cell < 0:
        raise ValueError("eps_per_cell must be >= 0")
    plan: list[tuple[CellId, int]] = []
    for cell in cells:
        seed_key = int(abs(hash(cell)) % (2**31))  # noqa: S324 -- mixed with base_seed below
        seeds = episode_seeds(cell, base_seed=base_seed + seed_key, n=eps_per_cell)
        plan.extend((cell, s) for s in seeds)
    return plan


def _refresh_obs(env: Any) -> dict[str, Any]:
    """Observation after a state restore, without mutating the environment."""
    for name in ("get_obs", "_obs", "observe", "get_observation"):
        getter = getattr(env, name, None)
        if callable(getter):
            obs = getter()
            if isinstance(obs, Mapping):
                return dict(obs)
    raise RuntimeError(
        f"{type(env).__name__} exposes no observation getter; the counterfactual path "
        "needs the observation of the restored state (04 §6)"
    )


def _step_info(obs: Mapping[str, Any], info: Mapping[str, Any]) -> dict[str, Any]:
    """Merge obs and info into the ``ViolationMonitor`` contract (``04`` §4)."""
    step_info = {
        "eef_pos": obs.get("eef_pose", info.get("eef_pos")),
        "object_pos": obs.get("object_pose", info.get("object_pos")),
        "grasped": bool(obs.get("grasped", info.get("grasped", False))),
        "force": info.get("force", obs.get("force", 0.0)),
        "penetration": info.get("penetration", 0.0),
        "progress_m": info.get("progress_m", obs.get("progress_m", 0.0)),
        "grasp_established": bool(obs.get("grasped", info.get("grasped", False))),
        "object_height": float(np.asarray(obs.get("object_pose", [0.0, 0.0, 0.0]))[2]),
    }
    missing = [k for k, v in step_info.items() if v is None]
    if missing:
        raise KeyError(f"step info is incomplete; missing {missing} (04 §4)")
    return step_info


def _proj_summary(decisions: Sequence[DecisionRecord]) -> dict[str, float]:
    if not decisions:
        return {
            "n_dec": 0.0,
            "n_active": 0.0,
            "n_unusable": 0.0,
            "mean_corr_rel": 0.0,
            "mean_r_before": 0.0,
            "mean_r_after": 0.0,
            "mean_r_pred": 0.0,
        }

    def mean(values: Sequence[float]) -> float:
        vals = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "n_dec": float(len(decisions)),
        "n_active": float(sum(1 for d in decisions if d.active)),
        "n_unusable": float(sum(1 for d in decisions if d.active and not d.usable)),
        "mean_corr_rel": mean([d.corr_rel for d in decisions]),
        "mean_r_before": mean([d.r_cand for d in decisions if d.r_cand is not None]),
        "mean_r_after": mean([d.r_after[-1] for d in decisions if d.r_after]),
        "mean_r_pred": mean([d.r_pred[-1] for d in decisions if d.r_pred]),
    }


def run_episode(
    task: Any,
    policy: Any,
    spec: RolloutSpec,
    *,
    planner: ChunkPlanner | None = None,
    critic: Any | None = None,
    proj_cfg: ProjCfg | None = None,
    inference_mode: str = "0",
    site: str | None = None,
    best_of_n: int = 8,
    dyn: DynamicsSpec | None = None,
    ranges: DynamicsRanges | None = None,
    hist_len: int = 1,
) -> RolloutOutput:
    """Run one episode and return its record, its decisions and any saved states.

    Dynamics are applied and **read back** before the first step: a non-nominal cell
    that ran at nominal physics would make every robustness number meaningless
    (``04`` §2).

    ``inference_mode`` (``08`` §2) selects the planner. It is a required part of calling
    this function for anything other than the baseline: an earlier version hardcoded the
    no-projection planner here, so a ``B3`` run silently measured ``B0`` and logged no
    decisions at all.
    """
    if dyn is None and ranges is not None:
        dyn = sample_dynamics_for_cell(spec.cell, ranges, spec.env_seed)
    if dyn is None:
        dyn = DynamicsSpec()

    env = task.make_env(dyn, spec.env_seed)
    try:
        set_cell_id(env, spec.cell)
    except AttributeError:  # pragma: no cover - tasks without the metadata hook
        pass

    readback: DynReadback = apply_dynamics(env, dyn)
    readback.assert_applied()

    if planner is None:
        # A critic is the signal that the caller means to project: every method config
        # carries a ProjCfg by default, so the config alone proves nothing. Passing a
        # critic with the baseline planner is the mistake that would silently measure B0
        # under B3's name.
        if inference_mode == "0" and critic is not None:
            raise ValueError(
                "inference_mode='0' is the no-projection planner, but a critic was supplied: "
                "this run would measure the baseline under the method's name and log no "
                f"decisions. Pass the method's inference_mode (08 §2) or drop the critic "
                f"(method={spec.method!r})."
            )
        planner = build_planner(
            inference_mode,
            proj_cfg=proj_cfg,
            critic=critic,
            site=site,
            best_of_n=best_of_n,
            horizon=spec.exec_horizon,
        )

    ctx_builder = CtxBuilder(
        hist_len=hist_len,
        # Declared, not inferred: the first decision has no action history yet and the
        # context block still has to line up with the critic's feature layout.
        action_dim=int(getattr(policy, "action_dim", 0) or 0) or None,
    )
    tracker = LatencyTracker()
    decisions: list[DecisionRecord] = []
    states: list[SimState] = []
    actions: list[np.ndarray] = []

    obs, info = env.reset(seed=spec.env_seed)
    ctx_builder.reset()
    monitor = ViolationMonitor(task.violations)
    ep_id = f"{spec.run_id}:{spec.method}:{spec.cell}:{spec.env_seed}"

    steps = 0
    t_success: float | None = None
    stall_steps = 0
    last_progress: float | None = None
    fallback_counts = {"F1": 0, "F2": 0, "F3": 0}
    errors: dict[str, Any] = {}

    try:
        while steps < spec.max_steps:
            ctx_builder.observe(env_obs_to_step_obs(dict(obs)))
            # The critic's context is the *pre-action* observation only: no future
            # information, so a label cannot leak the outcome (04 §5).
            ctx_batch = _batch_context(
                ctx_builder,
                actions=actions,
                hist_len=hist_len,
                action_dim=int(getattr(env, "action_dim", 3)),
            )
            # The critic's context must live where the critic's weights do. The planner
            # may own the critic (when the caller passed a planner instead of a mode).
            risk_critic = critic if critic is not None else getattr(planner, "critic", None)
            if risk_critic is not None:
                ctx_batch = _context_to_device(ctx_batch, _device_of(risk_critic))
            state_t = torch.as_tensor(np.asarray(obs["state"], dtype=np.float32))
            batch: dict[str, Any] = {"state": state_t.reshape(1, -1).to(_device_of(policy))}
            if spec.save_states and len(states) % max(1, spec.save_every) == 0:
                states.append(save_state(env))

            planned = planner.plan(
                policy=policy,
                batch=batch,
                ctx_batch=ctx_batch,
                step=steps,
                env_seed=spec.env_seed,
                crn_base_seed=spec.crn_base_seed,
                tracker=tracker,
            )
            chunk = planned.chunk
            if chunk.dim() != 3 or chunk.shape[0] != 1:
                raise ValueError(f"planner must return a [1, H, A] chunk; got {tuple(chunk.shape)}")
            for decision in planned.decisions:
                decision.ep_id = ep_id
                decision.step = steps
            decisions.extend(planned.decisions)
            for name, code in (("F1", FallbackCode.F1_BASE_ACTION), ("F2", FallbackCode.F2_HOLD), ("F3", FallbackCode.F3_TERMINATE)):
                if int(planned.fallback_code) == code:
                    fallback_counts[name] += 1

            horizon = min(spec.exec_horizon, int(chunk.shape[1]))
            terminated = truncated = False
            for k in range(horizon):
                action = np.asarray(chunk[0, k].detach().cpu(), dtype=np.float64)
                obs, _reward, terminated, truncated, info = env.step(action)
                steps += 1
                actions.append(action.copy())
                monitor.observe(_step_info(dict(obs), dict(info)), action=action)
                ctx_builder.observe(env_obs_to_step_obs(dict(obs), action=action))
                progress = float(info.get("progress_m", obs.get("progress_m", 0.0)))
                if last_progress is not None and abs(progress - last_progress) <= 1e-9:
                    stall_steps += 1
                last_progress = progress
                if bool(info.get("success", False)) and t_success is None:
                    t_success = float(steps)
                if terminated or truncated or steps >= spec.max_steps:
                    break
            if terminated or truncated:
                break
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised by default
        errors = {"error": f"{type(exc).__name__}: {exc}"}
        if not spec.allow_errors:
            raise RuntimeError(
                f"episode {ep_id} failed after {steps} steps: {type(exc).__name__}: {exc}. "
                "A crashed episode is never silently dropped (CLAUDE.md #4)."
            ) from exc

    violations = monitor.finalize()
    success = bool(info.get("success", False))
    episode = EpisodeRecord(
        run_id=spec.run_id,
        method=spec.method,
        train_seed=int(spec.train_seed),
        env_seed=int(spec.env_seed),
        cell_id=spec.cell,
        split=spec.split,
        success=success,
        steps=int(steps),
        dyn_readback=dict(readback.effective),
        t_success=t_success,
        violations=violations,
        failure_class=failure_class(
            violations,
            success=success,
            progress=float(info.get("progress", 0.0)),
            max_steps_reached=steps >= spec.max_steps,
        ),
        fallback=fallback_counts,
        stall_steps=int(stall_steps),
        action_norm_mean=_mean_action_norm(actions),
        jerk=_jerk(actions),
        proj=_proj_summary(decisions),
        latency_ms=tracker.means(),
        peak_vram_mb=peak_vram_mb(),
        extra={
            "planner": getattr(planner, "name", "unknown"),
            "exec_horizon": int(spec.exec_horizon),
            "n_decisions": len(decisions),
            "progress": float(info.get("progress", 0.0)),
            "crn_base_seed": int(spec.crn_base_seed),
            **dict(spec.record_extra),
            # errors last: a crash annotation must never be overwritten by a caller's
            # extra fields.
            **errors,
        },
    )
    return RolloutOutput(episode=episode, decisions=decisions, states=states, plan_actions=actions)


def _batch_context(
    builder: CtxBuilder,
    *,
    actions: Sequence[np.ndarray] | None = None,
    hist_len: int = 1,
    action_dim: int | None = None,
) -> dict[str, np.ndarray]:
    """One-example critic context, batched (the critic always sees ``[B, ...]``).

    ``action_hist`` is rebuilt here with the **action** width rather than the
    proprio width: :class:`~riskproj.obs.CtxBuilder` sizes its history windows from
    the proprio vector, while the critic's ``FeatureSpec`` sizes the action block
    from the action dimension, and a 14-wide action block against ``action_dim=3``
    would make the feature layout disagree with the chunk it has to score. The
    proprio/velocity/vision/force blocks come straight from the builder.
    """
    ctx = builder.context()
    batched = {k: np.asarray(v)[None, ...] for k, v in ctx.items()}
    if action_dim is not None:
        n = max(int(hist_len), 1)
        block = np.zeros((1, n, int(action_dim)), dtype=np.float32)
        recent = list(actions or [])[-n:]
        for i, act in enumerate(recent):
            block[0, n - len(recent) + i, :] = np.asarray(act, dtype=np.float32).reshape(-1)[
                :action_dim
            ]
        batched["action_hist"] = block
    return batched


def _mean_action_norm(actions: Sequence[np.ndarray]) -> float:
    if not actions:
        return 0.0
    return float(np.mean([float(np.linalg.norm(a)) for a in actions]))


def _jerk(actions: Sequence[np.ndarray]) -> float:
    """Mean squared second difference of the action sequence (smoothness diagnostic)."""
    if len(actions) < 3:
        return 0.0
    arr = np.stack([np.asarray(a, dtype=np.float64).reshape(-1) for a in actions])
    return float(np.mean(np.sum(np.diff(arr, n=2, axis=0) ** 2, axis=1)))


# --------------------------------------------------------------------------
# method-level evaluation
# --------------------------------------------------------------------------


def evaluate_method(
    task: Any,
    policy: Any,
    *,
    cells: Sequence[CellId],
    eps_per_cell: int,
    base_seed: int,
    out_dir: str | Path,
    run_id: str,
    method: str,
    train_seed: int,
    spec_kwargs: Mapping[str, Any] | None = None,
    planner: ChunkPlanner | None = None,
    critic: Any | None = None,
    proj_cfg: ProjCfg | None = None,
    inference_mode: str = "0",
    site: str | None = None,
    best_of_n: int = 8,
    ranges: DynamicsRanges | None = None,
    dyn: DynamicsSpec | None = None,
    split: str | None = None,
    config: Mapping[str, Any] | None = None,
    hist_len: int = 1,
) -> dict[str, Any]:
    """Run ``cells x eps_per_cell`` episodes, appending to the raw JSONL logs.

    Raw logs are append-only and immutable; every derived number is recomputed from
    them (``skills/software_engineering.md`` §1.4).
    """
    folder = Path(run_dir(out_dir, run_id, method=method, seed=train_seed, create=True))
    episodes_path = folder / "episodes.jsonl"
    decisions_path = folder / "decisions.jsonl"
    extra = dict(spec_kwargs or {})

    plan = cell_episode_plan(cells, eps_per_cell, base_seed)
    summaries: list[dict[str, Any]] = []
    n_decisions = 0
    for cell, env_seed in plan:
        spec = RolloutSpec(
            method=method,
            train_seed=int(train_seed),
            cell=cell,
            split=split if split is not None else role_of_cell(cell),
            env_seed=int(env_seed),
            crn_base_seed=int(base_seed),
            run_id=run_id,
            **extra,
        )
        out = run_episode(
            task,
            policy,
            spec,
            planner=planner,
            critic=critic,
            proj_cfg=proj_cfg,
            inference_mode=inference_mode,
            site=site,
            best_of_n=best_of_n,
            ranges=ranges,
            dyn=dyn,
            hist_len=hist_len,
        )
        append_jsonl(episodes_path, out.episode.to_json())
        if spec.record_decisions and out.decisions:
            append_jsonl(decisions_path, [d.to_json() for d in out.decisions])
            n_decisions += len(out.decisions)
        summaries.append(
            {
                "cell": cell,
                "env_seed": int(env_seed),
                "success": bool(out.episode.success),
                "steps": int(out.episode.steps),
                "any_violation": any_violation(out.episode.violations),
                "fallback": dict(out.episode.fallback),
                "failure_class": out.episode.failure_class,
                "dyn_readback": dict(out.episode.dyn_readback),
            }
        )

    summary = {
        "run_id": run_id,
        "method": method,
        "train_seed": int(train_seed),
        "cells": list(cells),
        "eps_per_cell": int(eps_per_cell),
        "base_seed": int(base_seed),
        "n_episodes": len(summaries),
        "n_decisions": n_decisions,
        "success_rate": float(np.mean([s["success"] for s in summaries])) if summaries else 0.0,
        "violation_rate": float(np.mean([s["any_violation"] for s in summaries])) if summaries else 0.0,
        "fallback_rate": float(
            np.mean([any(v > 0 for v in s["fallback"].values()) for s in summaries])
        )
        if summaries
        else 0.0,
        "episodes_jsonl": str(episodes_path),
        "decisions_jsonl": str(decisions_path) if n_decisions else None,
        "per_episode": summaries,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(folder / "summary.json", summary)
    capture_provenance(folder, config=dict(config) if config else None, seeds={"base_seed": base_seed})
    return summary


def _violation_rates(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Per-type violation rates from archived episode records (offline recomputable)."""
    if not records:
        return dict.fromkeys(VIOLATION_TYPES, 0.0)
    out: dict[str, float] = {}
    for v in VIOLATION_TYPES:
        out[v] = float(
            np.mean([float(r["violations"][v]["count"]) > 0.0 for r in records])
        )
    return out


def _empty_violations() -> dict[str, dict[str, float]]:
    """Zeroed per-type record, in the exact shape the monitor emits (``09`` §3)."""
    return empty_violation_counts()
