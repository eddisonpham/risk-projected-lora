"""Method plumbing shared by every baseline, the proposed method and the controls.

``agents/08_methods_and_training.md`` §1 fixes the vocabulary: a method is a
training mode ``T`` in {0, pen, lat, act} crossed with an inference mode ``I`` in
{0, lat, act-post, act-flow, bon, gate}. Everything here exists so that the
*two* places a method can act — the site activation on the way to the action head,
and the action chunk itself — are handled once, with one loop, one logger and one
fallback ladder.

Three invariants this module is responsible for:

1. **Shared budget and shared noise.** All methods share ``ExperimentCfg.train``,
   the data split and the seed set, and flow noise comes from
   :func:`riskproj.utils.seed.common_random_noise`, keyed by
   ``(seed, episode, step, flow_step)`` and never by method. Otherwise a
   difference between methods can be a difference in noise.
2. **Every projection and fallback event is logged** (CLAUDE.md #4). Decisions go
   through :class:`DecisionRecord`; the ladder never resolves an unusable gradient
   silently.
3. **Autograd stays on around the projected tail.** ``05`` §6: no ``no_grad``
   around the tail, and all critic/projection math is fp32.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from riskproj.critic.model import RiskCritic
from riskproj.policies.flow import flow_interpolate, flow_target
from riskproj.projection import FallbackLadder, ProjectionInfo
from riskproj.types import DecisionRecord, FallbackCode, PolicyOut, as_float
from riskproj.utils.config import ExperimentCfg, MethodCfg, ProjCfg, TrainCfg
from riskproj.utils.registry import get
from riskproj.utils.seed import common_random_noise, seed_everything

__all__ = [
    "MethodBase",
    "MethodResult",
    "PolicyWrapper",
    "apply_method_controls",
    "base_action_from",
    "build_method",
    "critic_context_from_batch",
    "ensure_eps",
    "flow_matching_loss",
    "lora_disabled",
    "lr_scale",
    "next_batch",
    "policy_horizon",
    "prepare_flow_inputs",
    "run_inference",
]

#: Projection-cfg fields that select a *different method* rather than a variant of
#: the same one. Changing one of these without saying so would silently make the
#: run a different arm, so the wrapper records them in the decision log.
RECORDED_FIELDS: tuple[str, ...] = (
    "K",
    "space",
    "margin",
    "max_rel_step",
    "backward",
    "steps_active",
    "tau_proj",
    "fallback",
)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class MethodResult:
    """What training returns: where the checkpoint is, and what it cost."""

    method_id: str
    train_seed: int
    checkpoint_dir: Path | None
    steps: int
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "train_seed": int(self.train_seed),
            "checkpoint_dir": None if self.checkpoint_dir is None else str(self.checkpoint_dir),
            "steps": int(self.steps),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
        }


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def policy_horizon(policy: Any, cfg: ExperimentCfg | None = None) -> int:
    """Action-chunk horizon of an adapter, without guessing silently."""
    for name in ("chunk_horizon", "_chunk_horizon"):
        value = getattr(policy, name, None)
        if isinstance(value, int) and value > 0:
            return value
    if cfg is not None:
        return int(cfg.model.chunk_size)
    raise AttributeError(
        f"{type(policy).__name__} exposes neither chunk_horizon nor _chunk_horizon; "
        "cannot size the critic's candidate chunk"
    )


def critic_context_from_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Build the critic's deployable-input context for an inference batch.

    Real rollouts pass ``batch["critic_ctx"]`` explicitly (the eval layer builds it
    from the environment through :mod:`riskproj.obs`, one step at a time). The
    fallback here exists for the toy/mechanism path and for unit tests, and it only
    ever uses *deployable* inputs: proprioception from ``batch["state"]`` and zero
    blocks for the optional histories. It refuses to invent vision or privileged
    force (`04` §5).
    """
    explicit = batch.get("critic_ctx")
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            raise TypeError(f"batch['critic_ctx'] must be a mapping; got {type(explicit).__name__}")
        return dict(explicit)
    state = batch.get("state")
    if state is None:
        raise KeyError(
            "cannot build a critic context: the batch has no 'critic_ctx' and no 'state'. "
            "Rollouts must pass batch['critic_ctx'] (04 §5); this fallback is only for the "
            "toy/mechanism path."
        )
    if not isinstance(state, Tensor):
        state = torch.as_tensor(state, dtype=torch.float32)
    b = state.shape[0]
    action_dim = int(batch.get("action_dim", 4))
    ctx: dict[str, Any] = {
        "proprio": state.detach().float(),
        "ee_vel": torch.zeros(b, 3),
        "action_hist": torch.zeros(b, 1, action_dim),
        "proprio_hist": state.detach().float().unsqueeze(1),
        "vision_emb": torch.zeros(b, 0),
        "force": torch.zeros(b, 0),
    }
    return ctx


@contextlib.contextmanager
def lora_disabled(policy: Any) -> Iterator[None]:
    """Temporarily zero every LoRA delta: the base-action (F1) reference.

    ``01`` §3's F1 fallback is "the unadapted action", which is the only sensible
    reference when the adapted policy is the thing that went wrong. Rather than
    rebuilding the policy, the low-rank scale is set to zero for the duration of the
    block — equivalent to removing the adapter, and cheap.
    """
    modules = []
    lister = getattr(policy, "lora_modules", None)
    if callable(lister):
        modules = list(lister().values())
    saved = [getattr(m, "scale", None) for m in modules]
    try:
        for module in modules:
            if hasattr(module, "scale"):
                module.scale = 0.0
        yield
    finally:
        for module, value in zip(modules, saved, strict=True):
            if value is not None:
                module.scale = value


def ensure_eps(
    batch: Mapping[str, Any],
    *,
    base_seed: int,
    episode_id: int | str,
    step: int,
    policy: Any = None,
) -> dict[str, Any]:
    """Fill ``batch['eps']`` from the shared CRN stream when the caller has none.

    Keyed by ``(base_seed, episode_id, step)`` and never by method, so that two
    methods making the same decision integrate the *same* noise — the pairing that
    makes a paired test valid (``08`` §4). The noise shape comes from a target chunk
    when the batch carries one, otherwise from the policy's horizon and action
    dimension; a deterministic head needs no noise at all, and gets none.
    """
    head = str(getattr(policy, "action_head", "flow_matching")) if policy is not None else "flow_matching"
    if batch.get("eps") is not None or head != "flow_matching":
        return dict(batch)
    source = batch.get("a0", batch.get("chunk"))
    if source is not None:
        shape = tuple(torch.as_tensor(source, dtype=torch.float32).shape)
        device = torch.as_tensor(source).device
    elif policy is not None and batch.get("state") is not None:
        batch_size = int(torch.as_tensor(batch["state"]).shape[0])
        action_dim = int(getattr(policy, "action_dim", getattr(policy, "_action_dim", 4)))
        shape = (batch_size, policy_horizon(policy), action_dim)
        device = torch.as_tensor(batch["state"]).device
    else:
        raise KeyError(
            "cannot size the flow noise: the batch has neither 'eps', a target chunk, nor 'state'. "
            "Pass 'eps' explicitly so methods are compared on identical noise (08 §4)."
        )
    out = dict(batch)
    out["eps"] = common_random_noise(
        base_seed, episode_id, step, shape, flow_step=0, device=device
    )
    return out


def base_action_from(policy: Any, batch: Mapping[str, Any], *, site: str | None = None) -> Tensor:
    """The base (zero-LoRA) action for the F1 fallback."""
    with lora_disabled(policy):
        out = policy.forward(dict(batch), site=site)
    return out.chunk.detach()


def next_batch(data: Any, batch_size: int | None = None) -> dict[str, Any]:
    """Pull one training batch from whatever the caller supplied.

    Accepted forms, in order: a zero-argument callable returning a batch dict; an
    object exposing ``sample_batch``/``batch``/``next_batch`` (the toy policy and
    the synthetic critics data both do); or a finite/infinite iterable of batches
    (re-cycled). Anything else is a ``TypeError`` — a silent empty batch would
    produce a zero loss and a "successful" run.
    """
    if callable(data):
        out = data() if batch_size is None else data(batch_size)
    elif isinstance(data, Iterator):
        out = next(data)
    else:
        handler = None
        for name in ("sample_batch", "batch", "next_batch"):
            candidate = getattr(data, name, None)
            if callable(candidate):
                handler = candidate
                break
        if handler is not None:
            out = handler() if batch_size is None else handler(batch_size)
        elif isinstance(data, Sequence) and data:
            out = data[0]
        else:
            raise TypeError(
                "training data must be a callable returning a batch, an iterator of batches, "
                f"or an object with sample_batch()/batch(); got {type(data).__name__}"
            )
    if not isinstance(out, Mapping):
        raise TypeError(f"a training batch must be a mapping; got {type(out).__name__}")
    return dict(out)


def lr_scale(step: int, cfg: TrainCfg) -> float:
    """Warmup then cosine, floored at ``min_lr_ratio`` (``05`` §5)."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return float(step + 1) / float(cfg.warmup_steps)
    if cfg.schedule == "constant":
        return 1.0
    total = max(1, cfg.steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, (step - cfg.warmup_steps) / total))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


def build_optimizer(params: Sequence[nn.Parameter], cfg: TrainCfg) -> torch.optim.Optimizer:
    if not params:
        raise ValueError("no trainable parameters: attach LoRA before training (05 §5)")
    return torch.optim.AdamW(
        list(params), lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay
    )


def prepare_flow_inputs(
    batch: Mapping[str, Any],
    *,
    schedule: str,
    schedule_seed: int,
    step: int,
    ln_m: float = 1.5,
    ln_s: float = 1.0,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Complete a batch with a valid ``(a0, eps, tau, x_tau)`` flow quadruple.

    A demonstration batch carries actions but no flow noise, so the noise is drawn
    from the shared CRN stream. Because the stream is keyed by
    ``(seed, episode, step)`` and not by method, B1 and B3 train on the *same* noise
    for the same data — which is what makes a loss comparison between them mean
    anything.
    """
    from riskproj.policies.flow import tau_schedule

    out = dict(batch)
    a0 = out.get("a0", out.get("chunk", out.get("actions")))
    if a0 is None:
        raise KeyError(
            "a training batch needs a target chunk under 'a0', 'chunk' or 'actions'; "
            f"got keys {sorted(out)}"
        )
    a0 = torch.as_tensor(a0, dtype=torch.float32)
    dev = device if device is not None else a0.device
    eps = out.get("eps")
    if eps is None:
        eps = common_random_noise(
            schedule_seed, "train", step, tuple(a0.shape), flow_step=0, device=dev
        )
    eps = torch.as_tensor(eps, dtype=torch.float32).to(dev)
    tau = out.get("tau")
    if tau is None:
        generator = torch.Generator().manual_seed(int(schedule_seed) * 100_003 + int(step))
        tau = tau_schedule(
            schedule,
            (a0.shape[0],),
            generator=generator,
            device=dev,
            ln_m=ln_m,
            ln_s=ln_s,
        )
    tau = torch.as_tensor(tau, dtype=torch.float32).to(dev)
    out["a0"] = a0
    out["eps"] = eps
    out["tau"] = tau
    if out.get("x_tau") is None:
        out["x_tau"] = flow_interpolate(a0, eps, tau)
    return out


def _mean_float(value: Any) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().float().mean()) if value.numel() else float("nan")
    return float("nan")


# --------------------------------------------------------------------------
# the inference wrapper
# --------------------------------------------------------------------------


class PolicyWrapper:
    """Inference-time view of a method: the policy plus its intervention.

    The wrapper owns per-episode state (the fallback ladder's hold counter) and the
    decision log. ``__call__`` performs *one* decision: run the policy, apply the
    method's intervention, consult the ladder when the correction was not possible,
    log everything.
    """

    def __init__(
        self,
        method_id: str,
        policy: Any,
        *,
        intervention: Callable[[dict[str, Any]], PolicyOut] | None = None,
        critic: RiskCritic | None = None,
        proj_cfg: ProjCfg | None = None,
        site: str | None = None,
        control: str = "none",
        mode: str = "0",
        ladder: FallbackLadder | None = None,
        base_seed: int = 0,
    ) -> None:
        self.method_id = method_id
        self.policy = policy
        self.intervention = intervention
        self.critic = critic
        self.proj_cfg = proj_cfg
        self.site = site or getattr(policy, "default_site", "S2")
        self.control = control
        self.mode = mode
        self.ladder = ladder
        self.base_seed = int(base_seed)
        self.episode_id: int | str = 0
        self.step = 0
        #: Last ladder consultation (a FallbackDecision), so the rollout can act on
        #: an F3 termination instead of stepping with a placeholder action.
        self.last_fallback: Any | None = None
        self._decisions: list[dict[str, Any]] = []
        self._proj_infos: list[ProjectionInfo] = []
        #: Method-specific scalars (best-of-N spread, gate rate, ...) surfaced through
        #: ``projection_summary`` so every arm reports through one channel.
        self._extra: dict[str, float] = {}

    def record_extra(self, values: Mapping[str, float]) -> None:
        """Attach method-specific scalars to the summary (never silently dropped)."""
        for key, value in values.items():
            self._extra[str(key)] = float(value)

    # -- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        """Start a new episode: clear hold state, keep the accumulated log."""
        if self.ladder is not None:
            self.ladder.reset()
        self.step = 0

    def begin_episode(self, episode_id: int | str) -> None:
        """Name the episode decisions will be attributed to (log join key)."""
        self.episode_id = episode_id
        self.step = 0
        if self.ladder is not None:
            self.ladder.reset()

    def close_episode(self) -> None:
        """Advance the episode counter without discarding the log."""
        self.episode_id = int(self.episode_id) + 1 if isinstance(self.episode_id, int) else self.episode_id
        self.step = 0

    # -- inference --------------------------------------------------------
    def __call__(self, batch: Mapping[str, Any], **kwargs: Any) -> PolicyOut:
        data = dict(batch)
        # The step counter advances *before* the intervention so that decisions
        # logged inside it are attributed to the decision being taken, not the last one.
        self.step += 1
        if self.intervention is None:
            # A baseline still runs the *full* denoising loop, only without the
            # correction. Comparing a one-step terminal estimate against a corrected
            # multi-step integration would make "B1 vs B3" a comparison of inference
            # procedures and nothing else.
            data = ensure_eps(
                data,
                base_seed=self.base_seed,
                episode_id=self.episode_id,
                step=self.step,
                policy=self.policy,
            )
            return run_inference(self.policy, data, site=self.site)
        return self.intervention(data)

    def trainable_parameters(self) -> list[nn.Parameter]:
        getter = getattr(self.policy, "trainable_parameters", None)
        return list(getter()) if callable(getter) else []

    # -- reporting --------------------------------------------------------
    def _log_projection(self, info: ProjectionInfo, *, flow_step: int, tau: Any) -> None:
        self._proj_infos.append(info)
        decision = DecisionRecord(
            ep_id=str(self.episode_id),
            step=int(self.step),
            flow_step=int(flow_step),
            tau=None if tau is None else as_float(tau),
            # Aggregates are batch means (see `aggregation` in `extra`): a per-decision
            # log line has to be one row per decision, and the live tensors stay in
            # ProjectionInfo for anything that needs per-example detail.
            r_cand=_mean_float(info.r_cand) if info.r_cand is not None else None,
            gn2=_mean_float(info.gn2) if info.gn2 is not None else None,
            active=bool(info.active.any()) if info.active is not None else False,
            usable=bool(info.usable.any()) if info.usable is not None else False,
            k_used=int(info.k_used.max()) if info.k_used is not None else 0,
            corr_rel=_mean_float(info.corr_rel),
            r_after=[_mean_float(info.r_after)],
            r_pred=[_mean_float(info.r_pred)],
            ens_std=None if info.ens_std is None else _mean_float(info.ens_std),
            fallback_code=int(info.fallback_code.max()) if info.fallback_code is not None else 0,
            extra={
                "n_examples": float(info.r_cand.numel()) if info.r_cand is not None else 0.0,
                "method": self.method_id,
                "mode": self.mode,
                "control": self.control,
                # Aggregate decision numbers are means over the batch; the per-example
                # tensors stay in the live ProjectionInfo and are not serialised here.
                "aggregation": "mean_over_batch",
                "site": self.site,
            },
        )
        self._decisions.append(decision.to_json())

    def log_fallback(self, decided: Any, *, flow_step: int = -1) -> None:
        """Record a ladder consultation (the ladder itself keeps the event log)."""
        code = int(getattr(decided, "code", 0))
        self._decisions.append(
            DecisionRecord(
                ep_id=str(self.episode_id),
                step=int(self.step),
                flow_step=int(flow_step),
                active=True,
                usable=False,
                fallback_code=code,
                extra={
                    "method": self.method_id,
                    "mode": self.mode,
                    "reason": str(getattr(decided, "reason", "")),
                    "fallback_names": FallbackCode.NAMES.get(code, "unknown"),
                    "details": dict(getattr(decided, "details", {}) or {}),
                },
            ).to_json()
        )

    def projection_summary(self) -> dict[str, float]:
        """Aggregate the projection telemetry, plus the ladder's summary."""
        out: dict[str, float] = {}
        if self._proj_infos:
            keys = ("mean_r_before", "mean_r_after", "mean_r_pred", "n_dec", "n_active",
                    "n_unusable", "mean_corr_rel", "mean_ens_std")
            totals = {k: 0.0 for k in keys}
            for info in self._proj_infos:
                summary = info.summary()
                for key in keys:
                    totals[key] += float(summary.get(key, 0.0))
            n = float(len(self._proj_infos))
            out.update({key: value / n for key, value in totals.items()})
            out["n_projection_calls"] = n
            # Linearization error averaged over decisions: this is E2's headline.
            errs = [
                _mean_float(info.r_after) - _mean_float(info.r_pred)
                for info in self._proj_infos
                if info.r_after is not None and info.r_pred is not None
            ]
            errs = [e for e in errs if math.isfinite(e)]
            if errs:
                out["mean_linearization_error"] = sum(errs) / len(errs)
        if self.ladder is not None:
            out.update(self.ladder.summary())
        out.update(self._extra)
        return out

    def fallback_counts(self) -> dict[str, int]:
        return self.ladder.counts() if self.ladder is not None else {"none": 0, "F1": 0, "F2": 0, "F3": 0}

    def decision_records(self) -> list[dict[str, Any]]:
        """Every logged decision (projection and fallback) since the last drain."""
        return list(self._decisions)

    def drain_decisions(self) -> list[dict[str, Any]]:
        """Return the decisions and clear the buffer (the eval layer's per-episode path)."""
        out = self._decisions
        self._decisions = []
        return out

    def lora_scale_zeroed_action(self, batch: Mapping[str, Any]) -> Tensor:
        """Base action for F1, computed with the adapter switched off."""
        return base_action_from(self.policy, batch, site=self.site)


# --------------------------------------------------------------------------
# the method base
# --------------------------------------------------------------------------


class MethodBase:
    """Shared training loop, budget accounting and wrapper construction.

    Subclasses implement :meth:`training_loss` and, when they intervene at inference
    time, :meth:`build_wrapper`. The loop itself (optimiser, warmup+cosine schedule,
    gradient accumulation, checkpointing, metrics) is deliberately *shared*: ``08``
    §3 requires identical step and tuning budgets across methods, and a per-method
    loop is exactly how that divergence creeps in.
    """

    #: ids this class serves, used by ``methods/__init__`` for registration checks.
    ids: tuple[str, ...] = ()

    #: Method id. Declared at class level as well as assigned in ``__init__`` so
    #: that the *class* satisfies the ``Method`` Protocol: the registries store and
    #: isinstance-check classes as well as instances (skills/software_engineering.md
    #: §2), and an instance-only attribute makes those checks fail misleadingly.
    name: str = "method"

    def __init__(self, method_cfg: MethodCfg) -> None:
        self.method_cfg = method_cfg
        self.name = method_cfg.id
        self.control = apply_method_controls(method_cfg)
        #: VAL configurations consumed while tuning this method (``08`` §3.5).
        self.tuning_budget: list[dict[str, Any]] = []

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
        logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> MethodResult:
        """Run Stage B (or Stage A) training and return where the checkpoint is."""
        if self.name == "B0":
            raise RuntimeError(
                "B0 is the frozen Stage-A policy: it has nothing to train, and silently "
                "returning it unchanged would look like a successful run (08 §2). Use B1."
            )
        tcfg = cfg.train
        seed_everything(int(train_seed))
        adapted = policy.attach_lora(cfg.model.lora)
        if adapted == 0:
            raise RuntimeError(
                "attach_lora adapted no modules; the configured LoRA suffixes "
                f"{list(cfg.model.lora.target_suffixes)} matched nothing (05 §3)"
            )
        policy.assert_no_gradients_outside_lora()
        params = policy.trainable_parameters()
        optimizer = build_optimizer(params, tcfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: lr_scale(s, tcfg)
        )

        ckpt_dir = None if out_dir is None else Path(out_dir) / "checkpoints"
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        losses: list[float] = []
        gnorms: list[float] = []
        extras: dict[str, list[float]] = {}
        t0 = time.perf_counter()
        accumulator = 0
        for step in range(int(tcfg.steps)):
            batch = prepare_flow_inputs(
                next_batch(data, None),
                schedule=self.method_cfg.noise_schedule,
                schedule_seed=int(cfg.seed),
                step=step,
                ln_m=self.method_cfg.ln_m,
                ln_s=self.method_cfg.ln_s,
            )
            loss, info = self.training_loss(policy, batch, cfg, critic, step)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{self.name}: non-finite loss at step {step}; refusing to continue "
                    f"(loss={float(loss)})"
                )
            (loss / max(1, tcfg.grad_accum)).backward()
            accumulator += 1
            if accumulator >= max(1, tcfg.grad_accum):
                if tcfg.grad_clip is not None:
                    gnorms.append(
                        float(
                            torch.nn.utils.clip_grad_norm_(params, tcfg.grad_clip).detach()
                        )
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulator = 0
            losses.append(float(loss.detach()))
            for key, value in info.items():
                extras.setdefault(key, []).append(float(value))
            if logger is not None and tcfg.log_every and (step + 1) % tcfg.log_every == 0:
                logger({"method": self.name, "step": step, "loss": losses[-1], **info})
            if (
                ckpt_dir is not None
                and tcfg.save_every
                and (step + 1) % int(tcfg.save_every) == 0
            ):
                torch.save(
                    {
                        "method": self.name,
                        "step": step + 1,
                        "train_seed": int(train_seed),
                        "state_dict": _state_dict(policy),
                    },
                    ckpt_dir / f"step_{step + 1}.pt",
                )
        if ckpt_dir is not None:
            torch.save(
                {
                    "method": self.name,
                    "step": int(tcfg.steps),
                    "train_seed": int(train_seed),
                    "state_dict": _state_dict(policy),
                },
                ckpt_dir / "final.pt",
            )

        metrics: dict[str, float] = {
            "steps": float(tcfg.steps),
            "loss_first": losses[0] if losses else float("nan"),
            "loss_final": losses[-1] if losses else float("nan"),
            "loss_ema": _ema(losses),
            "gnorm_mean": sum(gnorms) / len(gnorms) if gnorms else float("nan"),
            "n_trainable": float(sum(p.numel() for p in params)),
            "wall_s": time.perf_counter() - t0,
        }
        for key, values in extras.items():
            metrics[f"{key}_mean"] = sum(values) / len(values)
        return MethodResult(
            method_id=self.name,
            train_seed=int(train_seed),
            checkpoint_dir=ckpt_dir,
            steps=int(tcfg.steps),
            metrics=metrics,
        )

    def training_loss(
        self,
        policy: Any,
        batch: dict[str, Any],
        cfg: ExperimentCfg,
        critic: RiskCritic | None,
        step: int,
    ) -> tuple[Tensor, dict[str, float]]:
        """The one thing every method must define. Default: plain flow-matching."""
        return flow_matching_loss(policy, batch, site=self.site_or_default(policy))

    # -- inference --------------------------------------------------------
    def site_or_default(self, policy: Any) -> str:
        return str(getattr(policy, "default_site", "S2"))

    def wrap(
        self, policy: Any, *, critic: RiskCritic | None = None, base_seed: int = 0
    ) -> PolicyWrapper:
        """Return the inference-time wrapper for this method."""
        return PolicyWrapper(
            self.name,
            policy,
            critic=critic,
            proj_cfg=self.method_cfg.projection,
            site=self.site_or_default(policy),
            control=self.control,
            mode=self.method_cfg.inference_mode,
            ladder=None,
            base_seed=base_seed,
        )

    def budget(self, cfg: ExperimentCfg) -> dict[str, float]:
        """Tuning budget consumed, so parity across methods is reportable (``08`` §3.5)."""
        return {
            "n_val_configs": float(len(self.tuning_budget)),
            "steps": float(cfg.train.steps),
            "grad_accum": float(cfg.train.grad_accum),
            "effective_batch": float(cfg.train.batch_size * cfg.train.grad_accum),
        }

    # -- registry ---------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(id={self.name!r}, control={self.control!r})"


def run_inference(
    policy: Any,
    batch: Mapping[str, Any],
    *,
    site: str | None = None,
    proj_cfg: ProjCfg | None = None,
    project_hook: Callable[[Tensor, Any], tuple[Tensor, ProjectionInfo | None]] | None = None,
    flow_hook: Callable[[Tensor, Tensor, Any], tuple[Tensor, ProjectionInfo | None]] | None = None,
    info_sink: Callable[[ProjectionInfo, int, Any], None] | None = None,
) -> PolicyOut:
    """Run one decision, with the intervention hooks applied at the configured steps.

    This is the single implementation of "project only at the last ``m`` denoising
    steps" on the inference path (``05`` §4, ``01`` §6 item 2). ``project_hook``
    intervenes on the *site activation* (B3/B1+TTP/C1-C3, and S0 for B4a);
    ``flow_hook`` intervenes on the noisy chunk ``x_tau`` itself (B4b).

    A flow-matching head runs the ``tau: 1 -> 0`` Euler loop here; a deterministic
    head takes one forward pass and comes back through ``f_act``. ``batch['eps']``
    must be supplied for a flow head: the wrapper fills it from the shared CRN
    stream, so two methods see the same noise at the same decision (``08`` §4).

    Cost note (``07`` §6): the site activation is obtained from a full forward, and
    after a correction the velocity is recomputed from the corrected activation via
    ``tail_velocity``. That is one extra head evaluation per corrected step, which is
    the price of not assuming anything about adapter internals.
    """
    site_id = site or str(getattr(policy, "default_site", "S2"))
    head = str(getattr(policy, "action_head", "flow_matching"))

    def _site_activation(out: PolicyOut) -> Tensor | None:
        if out.h is not None:
            return out.h
        h = out.extras.get("h")
        return h if isinstance(h, Tensor) else None

    if head != "flow_matching":
        # Deterministic head (ACT / VLA-Adapter / toy regression): one pass, then
        # the downstream map from the corrected activation.
        data = dict(batch)
        out = policy.forward(data, site=site_id)
        h = _site_activation(out)
        if project_hook is None or h is None:
            return out
        ctx = policy.make_context(data, site=site_id)
        h_safe, info = project_hook(h, ctx)
        if info is not None and info_sink is not None:
            info_sink(info, -1, batch.get("tau"))
        chunk = policy.f_act(ctx, h_safe, site_id)
        return PolicyOut(chunk=chunk, site=site_id, h=h_safe, extras=dict(out.extras))

    num_steps = int(getattr(policy, "num_flow_steps", 10))
    if num_steps < 1:
        raise ValueError(f"num_flow_steps must be >= 1; got {num_steps}")
    eps = batch.get("eps")
    if eps is None:
        raise KeyError(
            "a flow-head inference batch needs 'eps' (the initial noise): the wrapper draws "
            "it from the shared CRN stream so methods are compared on identical noise"
        )
    eps_t = torch.as_tensor(eps, dtype=torch.float32)
    x = eps_t
    dt = 1.0 / float(num_steps)
    velocity: Tensor | None = None
    h_used: Tensor | None = None
    last_out: PolicyOut | None = None
    for i in range(num_steps):
        tau_value = 1.0 - i * dt
        tau_vec = torch.full((eps_t.shape[0],), tau_value, dtype=torch.float32)
        step_batch = {**dict(batch), "x_tau": x, "tau": tau_vec}
        ctx = policy.make_context(step_batch, site=site_id)
        active = True
        if proj_cfg is not None and (project_hook is not None or flow_hook is not None):
            from riskproj.projection import active_at_flow_step

            active = bool(active_at_flow_step(tau_vec, i, num_steps, proj_cfg).any())

        if flow_hook is not None and active:
            x, info = flow_hook(x, tau_vec, ctx)
            if info is not None and info_sink is not None:
                info_sink(info, i, tau_vec)
            step_batch["x_tau"] = x

        out = policy.forward(step_batch, site=site_id)
        last_out = out
        velocity = out.velocity
        h = _site_activation(out)
        if velocity is None:
            raise RuntimeError(
                f"{type(policy).__name__} has a flow-matching head but returned no velocity"
            )
        if project_hook is not None and h is not None and active:
            h_safe, info = project_hook(h, ctx)
            if info is not None and info_sink is not None:
                info_sink(info, i, tau_vec)
            if info is not None and bool(info.active.any()):
                # Recompute the velocity from the corrected activation: this is the
                # projected tail of the flow step.
                velocity = policy.tail_velocity(ctx, h_safe, site_id)
                h_used = h_safe
        if h_used is None:
            h_used = h
        x = x - dt * velocity
    return PolicyOut(
        chunk=x,
        site=site_id,
        h=h_used,
        tau=torch.zeros(eps_t.shape[0], dtype=torch.float32),
        velocity=velocity,
        extras=dict(last_out.extras) if last_out is not None else {},
    )


def flow_matching_loss(
    policy: Any, batch: Mapping[str, Any], *, site: str | None = None
) -> tuple[Tensor, dict[str, float]]:
    """Stage-B objective for a flow-matching head: ``MSE(v_theta, eps - a0)``.

    ``05`` §4: the head predicts a *velocity*, and the target is
    ``v* = eps - a0`` under the ``tau=1``-is-noise convention. The regression-head
    fallback (ACT/VLA-Adapter) is not silent: it switches to MSE against the
    demonstration chunk and says so, because an unadapted flow target on a
    deterministic head would be a quietly wrong experiment.
    """
    out = policy.forward(dict(batch), site=site)
    head = getattr(policy, "action_head", "flow_matching")
    if head == "flow_matching":
        if out.velocity is None:
            raise RuntimeError(
                f"{type(policy).__name__} reports a flow-matching head but returned no velocity; "
                "the flow path is broken upstream"
            )
        target = flow_target(batch["a0"], batch["eps"])
        loss = torch.nn.functional.mse_loss(out.velocity.float(), target.float())
    else:
        # Deterministic head: no flow noise exists, so the flow quadruple is unused.
        loss = torch.nn.functional.mse_loss(out.chunk.float(), batch["a0"].float())
    return loss, {"loss": float(loss.detach())}


def _state_dict(policy: Any) -> dict[str, Any]:
    if isinstance(policy, nn.Module):
        return {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    raise TypeError(
        f"{type(policy).__name__} is not an nn.Module, so it has no checkpointable state"
    )


def _ema(values: Sequence[float], alpha: float = 0.1) -> float:
    if not values:
        return float("nan")
    out = values[0]
    for value in values[1:]:
        out = alpha * value + (1.0 - alpha) * out
    return float(out)


# --------------------------------------------------------------------------
# controls and construction
# --------------------------------------------------------------------------


def apply_method_controls(method_cfg: MethodCfg) -> str:
    """Resolve ``MethodCfg.control`` into the control actually applied.

    The control must live *inside* the B3 pipeline (``08`` §2): a control that used
    a different site, a different step size or a different number of iterations
    would be a different method, not a control for this one. Weaker (non-latent)
    modes are therefore rejected loudly rather than silently downgraded.
    """
    control = str(method_cfg.control)
    if control == "none":
        return "none"
    if control not in ("C1", "C2", "C3"):
        raise ValueError(f"unknown control {control!r}; expected 'none', 'C1', 'C2' or 'C3'")
    if method_cfg.inference_mode != "lat":
        raise ValueError(
            f"control {control} must run inside the B3 inference pipeline: "
            f"inference_mode must be 'lat', got {method_cfg.inference_mode!r} (08 §2)"
        )
    return control


def build_method(method_cfg: MethodCfg) -> MethodBase:
    """Construct the method named by ``method_cfg.id`` from the registry."""
    cls = get("method", method_cfg.id)
    method = cls(method_cfg)
    if not isinstance(method, MethodBase):
        raise TypeError(
            f"method {method_cfg.id!r} is registered to {type(method).__name__}, which is not a "
            "MethodBase"
        )
    return method
