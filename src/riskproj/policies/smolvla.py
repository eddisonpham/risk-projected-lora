"""SmolVLA adapter (tier A1) — the model every main experiment uses.

Deliberately thin: module discovery, LoRA swapping and the flow convention live in
:mod:`riskproj.policies.sites`, :mod:`riskproj.policies.base` and
:mod:`riskproj.policies.flow`. What is left here is the upstream wiring.

**Verified vs unverified.** From ``05`` §2 the following are [V]: SmolVLA is a
SmolVLM-2 backbone with a ~100M action expert, a 50-step action chunk, **10
inference steps**, and a frozen VLM prefix whose per-layer KV is computed once per
chunk and reused across denoising steps. The **module names and the exact
per-denoising-step entry point are [R]** — LeRobot's internal API is not verified
in this repository. So:

* site patterns come from ``configs/model/smolvla_sites.yaml`` when S3 has written
  it, otherwise from the documented heuristic substrings in
  ``sites.CANDIDATE_PATTERNS``, and every run records which one it used;
* the per-step entry point is discovered at load time from a list of plausible
  callables, and if none exists the adapter raises with that list rather than
  silently doing something else;
* the tail after a site is recomputed by **substituting the site module's output**
  and re-running one step. That is model-agnostic — a site is by definition
  consumed only downstream — and needs no knowledge of block internals.

Do not "fix" one of the guards below by guessing an API. Record the real name in
``agents/research_log/LOG.md`` and, if upstream must change, in ``patches/``.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from riskproj.policies import sites as sites_mod
from riskproj.policies.base import (
    AdapterContext,
    FlowPolicyMixin,
    LoRALinear,
    PolicyAdapterBase,
    discover_linear_targets,
    swap_module,
)
from riskproj.policies.flow import euler_taus, projection_active
from riskproj.types import PolicyOut
from riskproj.utils.config import LoraCfg, ModelCfg
from riskproj.utils.registry import register

logger = logging.getLogger(__name__)

__all__ = ["EXPERT_ENTRY_POINTS", "SmolVLAAdapter"]

#: Candidate names for "run one denoising step of the action expert". Unverified [R] —
#: extend this list (never replace it) once the real name is known, so the error
#: message keeps showing everything that was tried.
EXPERT_ENTRY_POINTS: tuple[str, ...] = (
    "denoise_step",
    "flow_step",
    "step_velocity",
    "forward_expert",
    "forward_action_expert",
    "expert_forward",
)

_SKIP_REMEDIATION = "install with: python tasks.py bootstrap-sim"
#: Spec defaults for the projection window, overridable per batch from ProjCfg.
DEFAULT_PROJ_TAU_MAX = 0.3
DEFAULT_PROJ_STEPS = 3


@register("policy", "smolvla")
class SmolVLAAdapter(PolicyAdapterBase, FlowPolicyMixin, nn.Module):
    """Adapter over LeRobot's ``SmolVLAPolicy`` exposing per-step site activations."""

    name = "smolvla"
    #: Set in __init__; class-level annotation keeps the type stable for mypy.
    _model: nn.Module

    def __init__(
        self,
        cfg: ModelCfg,
        *,
        action_dim: int = 4,
        state_dim: int | None = None,
        device: str = "auto",
        checkpoint: str | None = None,
        load: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        PolicyAdapterBase.__init__(
            self, chunk_horizon=cfg.chunk_size, action_dim=action_dim, state_dim=state_dim
        )
        self.cfg = cfg
        self.default_site = cfg.site.default
        self.num_flow_steps = int(cfg.num_flow_steps)
        self.dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[cfg.dtype]
        self.device_name = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        self.site_specs = sites_mod.resolve_sites(cfg.site, adapter=None)
        self.matched: dict[str, str] = {}
        self.site_errors: dict[str, str] = {}
        self._step_name: str | None = None
        self._packaging: Any = None  # how the site module wraps its output (tuple, ...)
        self._checkpoint = checkpoint or cfg.checkpoint
        if load:
            self._load()

    # ------------------------------------------------------------------
    # availability / loading
    # ------------------------------------------------------------------
    @staticmethod
    def is_available() -> bool:
        """True when the LeRobot/transformers stack is importable (tests skip otherwise)."""
        return (
            importlib.util.find_spec("lerobot") is not None
            and importlib.util.find_spec("transformers") is not None
        )

    def _load(self) -> None:
        if self._checkpoint is None:
            raise ValueError(
                "SmolVLAAdapter needs a checkpoint: set model.checkpoint in the config "
                "(the spec assumes `lerobot/smolvla_base` [R]) or pass checkpoint=..."
            )
        if not self.is_available():
            raise ImportError(f"SmolVLA needs LeRobot and transformers; {_SKIP_REMEDIATION}")
        self._model = self._build_policy(self._checkpoint)
        self._model.to(self.device_name)
        if self.dtype is not torch.float32:
            # Only the frozen backbone/expert run in reduced precision; the critic and
            # all projection math stay fp32 (05 §6).
            self._model.to(self.dtype)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)
        self._step_name = self._find_step_entry()
        self._discover_sites()

    def _build_policy(self, checkpoint: str) -> nn.Module:
        """Instantiate the upstream policy; tries the documented entry points in order."""
        errors: list[str] = []
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore

            return SmolVLAPolicy.from_pretrained(checkpoint)
        except Exception as exc:
            errors.append(f"lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy: {exc!r}")
        try:
            from lerobot.policies.factory import get_policy_class  # type: ignore

            return get_policy_class("smolvla").from_pretrained(checkpoint)
        except Exception as exc:
            errors.append(f"lerobot.policies.factory.get_policy_class('smolvla'): {exc!r}")
        raise ImportError(
            f"could not construct SmolVLA from checkpoint {checkpoint!r}. Tried:\n  - "
            + "\n  - ".join(errors)
            + f"\n{_SKIP_REMEDIATION}, then read third_party/lerobot to confirm the current "
            "entry point and record it in agents/research_log/LOG.md."
        )

    def _find_step_entry(self) -> str:
        for name in EXPERT_ENTRY_POINTS:
            if callable(getattr(self._model, name, None)):
                logger.info("SmolVLA per-step entry point: %s", name)
                return name
        raise NotImplementedError(
            "no per-denoising-step entry point found on the loaded SmolVLA policy; tried "
            f"{list(EXPERT_ENTRY_POINTS)}. Projection needs per-step access (05 §4). "
            "Inspect the module tree (scripts/inspect_model.py) and extend EXPERT_ENTRY_POINTS, "
            "or record this as an OPEN QUESTION in agents/research_log/LOG.md."
        )

    # ------------------------------------------------------------------
    # sites
    # ------------------------------------------------------------------
    def _discover_sites(self) -> None:
        """Fill ``self._sites`` from the resolved specs; failures are recorded, not swallowed."""
        self._sites.clear()
        self.matched, self.site_errors = sites_mod.register_discovered_sites(
            self, self._model, self.site_specs, action_dim=self.action_dim
        )

    def sites(self) -> dict[str, Any]:
        return dict(self._sites)

    def site_module(self, site: str) -> Any:
        if site in self._sites:
            return self._sites[site]
        if site in self.site_errors:
            raise LookupError(
                f"site {site!r} could not be resolved on the loaded model: {self.site_errors[site]}"
            )
        return super().site_module(site)

    def match_report(self) -> dict[str, Any]:
        """Provenance: which module each site resolved to, and what failed."""
        sites_file = Path(self.cfg.site.resolved_file)
        return {
            "matched": dict(self.matched),
            "errors": dict(self.site_errors),
            "step_entry": self._step_name,
            "sites_file": str(sites_file),
            "sites_file_exists": sites_file.exists(),
        }

    # ------------------------------------------------------------------
    # one denoising step, and the tail after a site
    # ------------------------------------------------------------------
    def _step_fn(self) -> Callable[..., Any]:
        if self._step_name is None:
            raise RuntimeError("adapter is not loaded; call _load() first")
        fn = getattr(self._model, self._step_name, None)
        if not callable(fn):
            raise NotImplementedError(
                f"the per-step entry point {self._step_name!r} disappeared from the model"
            )
        return fn

    def capture_step_site(self, ctx: AdapterContext, site: str) -> Tensor:
        """Run one unsubstituted step and return the site activation.

        Also learns the module's output packaging, which :meth:`tail_velocity` needs to
        substitute the activation without breaking downstream code.
        """
        tensor, raw = PolicyAdapterBase.capture_site_output(self, site, lambda: self._run_step(ctx))
        self._packaging = raw
        return tensor

    def _run_step(self, ctx: AdapterContext, _site: str | None = None) -> Tensor:
        """Run one denoising step; returns the raw velocity ``[B, H, A]``.

        ``_site`` is accepted (and ignored) only so callers can record which site a
        step was taken for; the site's effect is expressed through
        :meth:`site_output_override`, not by changing the step itself.
        """
        fn = self._step_fn()
        x_tau, tau = ctx.flow()
        kwargs = dict(ctx.extras.get("step_kwargs", {}))
        try:
            out = fn(x_tau, tau, **kwargs)
        except TypeError as exc:
            raise NotImplementedError(
                f"the per-step entry point {self._step_name!r} does not accept (x_tau, tau): {exc}. "
                "Record its real signature in LOG.md and adapt _run_step (see 05 §4)."
            ) from exc
        return _as_velocity(out, ctx)

    def step_velocity(self, ctx: AdapterContext, x_tau: Tensor, tau: Tensor, _site: str) -> Tensor:
        return self._run_step(ctx.with_flow(x_tau, tau))

    def tail_velocity(self, ctx: AdapterContext, h: Tensor, site: str) -> Tensor:
        """Velocity of the current step with the site activation replaced by ``h``."""
        if self._packaging is None:
            self.capture_step_site(ctx, site)  # learns the packaging and validates the site
        with self.site_output_override(site, h, self._packaging):
            return self._run_step(ctx)

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    def attach_lora(self, cfg: LoraCfg) -> int:
        """Wrap target ``nn.Linear`` leaves in place; returns the number adapted."""
        if cfg.r <= 0:
            return 0
        names = discover_linear_targets(
            self._model, cfg.target_suffixes, last_n_blocks=cfg.last_n_blocks
        )
        if not names:
            available = sorted(
                {
                    n.split(".")[-1]
                    for n, m in self._model.named_modules()
                    if isinstance(m, nn.Linear)
                }
            )
            raise RuntimeError(
                "no Linear matched the LoRA target suffixes "
                f"{list(cfg.target_suffixes)} (last_n_blocks={cfg.last_n_blocks}). "
                f"Linear leaf names present: {available}. Confirm the real names with "
                "scripts/inspect_model.py and correct configs/model/*.yaml — never edit upstream."
            )
        adapted = 0
        for name in names:
            old = self._model.get_submodule(name)
            if isinstance(old, LoRALinear):
                continue
            lora = LoRALinear(old, cfg.r, cfg.alpha, cfg.dropout)
            swap_module(self._model, name, lora)
            self.register_lora(name, lora, cfg)
            adapted += 1
        # Swapping can replace a site module; re-resolve so hooks act on the live tree.
        self._discover_sites()
        self.freeze_base()
        logger.info("attached LoRA to %d modules (r=%d, alpha=%d)", adapted, cfg.r, cfg.alpha)
        return adapted

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def make_context(self, batch: dict[str, Any], site: str | None = None) -> AdapterContext:
        ctx = super().make_context(batch, site)
        ctx.state = batch.get("state")
        return ctx

    def _initial_noise(self, batch: dict[str, Any]) -> Tensor:
        """Starting ``eps``. CRN requires the harness to pass ``batch['noise']``."""
        noise = batch.get("noise")
        if noise is None:
            raise KeyError(
                "flow sampling needs 'noise' in the batch: evaluation must pass seeded common "
                "random numbers so every method sees identical eps (05 §4 pairing rule)."
            )
        noise = noise.to(device=self.device_name, dtype=self.dtype)
        if noise.dim() != 3 or noise.shape[-1] != self.action_dim:
            raise ValueError(
                f"flow noise must be [B, H, A={self.action_dim}]; got {tuple(noise.shape)}. "
                "A wrong shape here would silently broadcast against the velocity."
            )
        return noise

    def projection_window(self, batch: dict[str, Any]) -> tuple[float, int]:
        """``(tau_proj, m)`` for this batch, from ProjCfg, with a loud default.

        The adapter does not own this decision — the method wrapper passes
        ``batch['proj_tau_max']``/``batch['proj_steps']`` from ``ProjCfg``
        (``tau_proj``/``steps_active``). Using the fallback means a caller forgot, so
        it is logged once per call rather than silently assumed.
        """
        tau_max = batch.get("proj_tau_max")
        steps = batch.get("proj_steps")
        if tau_max is None or steps is None:
            logger.warning(
                "batch carries no projection window (proj_tau_max/proj_steps); falling back to "
                "tau_proj=%s, m=%s. Pass ProjCfg values explicitly for a comparable run.",
                DEFAULT_PROJ_TAU_MAX,
                DEFAULT_PROJ_STEPS,
            )
            tau_max = DEFAULT_PROJ_TAU_MAX if tau_max is None else tau_max
            steps = DEFAULT_PROJ_STEPS if steps is None else steps
        return float(tau_max), int(steps)

    def forward(
        self,
        batch: dict[str, Any],
        site: str | None = None,
        hook: Callable[[Tensor], Tensor] | None = None,
    ) -> PolicyOut:
        """Sample a chunk, optionally projecting the site activation at low-noise steps.

        ``hook(h) -> Tensor`` is the projection: it returns a replacement activation (or
        ``None`` to observe only). The active window is ``m`` last denoising steps with
        ``tau <= tau_proj`` taken from the batch (``05`` §4).
        """
        site = site or self.default_site
        if site not in sites_mod.SITE_CATALOG:
            raise KeyError(f"unknown site {site!r}")
        tau_max, m = self.projection_window(batch)
        noise = self._initial_noise(batch)
        ctx = self.make_context(batch, site=site)
        taus = euler_taus(self.num_flow_steps)
        dt = 1.0 / self.num_flow_steps
        x = noise.clone()
        last_h: Tensor | None = None
        last_tau = taus[0]
        for i in range(self.num_flow_steps):
            tau_i = taus[i]
            tau_vec = tau_i.expand(x.shape[0]).to(device=x.device, dtype=x.dtype)
            step_ctx = ctx.with_flow(x, tau_vec)
            active = (
                hook is not None
                and i >= self.num_flow_steps - m
                and bool(projection_active(tau_vec, tau_max).all())
            )
            if site == "S0f" and active:
                # B4b: correct the noisy chunk itself before this step.
                x = self.apply_hook(x, hook)
            if site not in ("S0", "S0f") and active:
                h = self.capture_step_site(step_ctx, site)
                last_h = self.apply_hook(h, hook)
                velocity = self.tail_velocity(step_ctx, last_h, site)
            elif site not in ("S0", "S0f"):
                with self.forward_hook_at(site) as captured:
                    velocity = self._run_step(step_ctx)
                last_h = captured[-1] if captured else None
            else:
                velocity = self._run_step(step_ctx)
            last_tau = tau_i
            x = x - dt * velocity
        chunk = x
        if site == "S0" and hook is not None:
            # B4a: post-hoc correction of the finished chunk.
            chunk = self.apply_hook(chunk, hook)
        return PolicyOut(
            chunk=chunk,
            site=site,
            h=last_h,
            tau=last_tau.expand(chunk.shape[0]).to(device=chunk.device, dtype=chunk.dtype),
            velocity=None,
            extras={
                "num_flow_steps": self.num_flow_steps,
                "flow_convention": "tau1_is_noise",
                "matched_sites": dict(self.matched),
                "proj_tau_max": tau_max,
                "proj_steps": m,
            },
        )


def _as_velocity(out: Any, ctx: AdapterContext) -> Tensor:
    """Coerce a step entry point's return value into a velocity tensor ``[B, H, A]``."""
    if isinstance(out, Tensor):
        velocity = out
    elif isinstance(out, dict):
        for key in ("velocity", "v", "v_t", "pred"):
            if isinstance(out.get(key), Tensor):
                velocity = out[key]
                break
        else:
            raise NotImplementedError(
                f"the step entry point returned a dict without a velocity-like key; keys={sorted(out)}"
            )
    elif isinstance(out, (tuple, list)) and out and isinstance(out[0], Tensor):
        velocity = out[0]
    else:
        raise NotImplementedError(
            f"the step entry point returned {type(out).__name__}; cannot read a velocity"
        )
    if velocity.dim() == 2:
        velocity = velocity.reshape(velocity.shape[0], ctx.chunk_horizon, -1)
    if velocity.dim() != 3:
        raise ValueError(f"velocity must be [B, H, A]; got {tuple(velocity.shape)}")
    if velocity.shape[-1] != ctx.action_dim:
        raise ValueError(
            f"velocity has action dim {velocity.shape[-1]} but the adapter expects {ctx.action_dim}"
        )
    return velocity
