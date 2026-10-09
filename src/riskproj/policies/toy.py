"""A CPU-only toy policy: the mechanism testbed for the walking skeleton.

This exists so that every *mechanism* in the project (site projection, terminal
estimate, LoRA-at-a-site, per-example independence, flow convention) can be
executed and tested in CI in seconds, with no checkpoint, no simulator and no GPU
(``03`` §3 smoke test S6). It is **not** a baseline and its numbers are not
results: a random-init policy has nothing to adapt.

Known degeneracy, stated so nobody over-reads it: the toy has a single expert
block, so ``S1`` and ``S2`` are the same tensor and ``S5`` is not registered at
all. The depth sweep (ablation A9) therefore needs SmolVLA; on the toy it is
degenerate by construction.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor, nn

from riskproj.policies.base import AdapterContext, FlowPolicyMixin, LoRALinear, PolicyAdapterBase
from riskproj.policies.flow import flow_interpolate, flow_target, tau_schedule, terminal_estimate
from riskproj.types import PolicyOut
from riskproj.utils.config import LoraCfg, ModelCfg
from riskproj.utils.registry import register

__all__ = ["ToyPolicy"]


@register("policy", "toy")
class ToyPolicy(PolicyAdapterBase, FlowPolicyMixin, nn.Module):
    """A tiny site-bearing policy with optional LoRA at the site.

    ``action_head="flow_matching"`` gives a velocity head (so the terminal-estimate
    path is exercised); ``action_head="regression"`` gives a deterministic head
    (the ACT/VLA-Adapter analogue) where ``f_act`` is the head itself.
    """

    name = "toy"
    #: Exactly one of these is built, per ``action_head``. Class-level annotations
    #: keep the attribute type stable for mypy without registering a None submodule.
    action_out: nn.Linear | None
    velocity_out: nn.Linear | None

    # nn.Module.__init__ is called explicitly: the MRO is
    # ToyPolicy -> PolicyAdapterBase -> FlowPolicyMixin -> nn.Module and the mixins
    # do not chain super().__init__(), so cooperative init would never reach it.
    def __init__(
        self,
        cfg: ModelCfg | None = None,
        *,
        state_dim: int = 12,
        action_dim: int = 4,
        hidden: int = 64,
        chunk_horizon: int | None = None,
        num_flow_steps: int | None = None,
        action_head: Literal["flow_matching", "regression"] = "flow_matching",
        seed: int = 0,
        device: str = "cpu",
        default_site: str | None = None,
    ) -> None:
        nn.Module.__init__(self)
        horizon = int(
            chunk_horizon if chunk_horizon is not None else (cfg.chunk_size if cfg else 10)
        )
        steps = int(
            num_flow_steps if num_flow_steps is not None else (cfg.num_flow_steps if cfg else 10)
        )
        if action_head not in ("flow_matching", "regression"):
            raise ValueError(
                f"action_head must be 'flow_matching' or 'regression'; got {action_head!r}"
            )
        PolicyAdapterBase.__init__(
            self, chunk_horizon=horizon, action_dim=action_dim, state_dim=state_dim
        )

        self.action_head = action_head
        self.num_flow_steps = steps
        self.hidden = int(hidden)
        self.device_name = device
        if default_site is not None:
            self.default_site = default_site

        # Deterministic construction: same seed -> same parameters (contract suite).
        generator = torch.Generator().manual_seed(int(seed))
        self.seed = int(seed)

        self.encoder = nn.Linear(state_dim, self.hidden)
        self.site_proj = nn.Linear(self.hidden, self.hidden)
        if action_head == "regression":
            self.action_out = nn.Linear(self.hidden, horizon * action_dim)
            self.velocity_out = None
        else:
            self.velocity_out = nn.Linear(
                self.hidden + horizon * action_dim + 1, horizon * action_dim
            )
            self.action_out = None
        self.register_site("S1", self.site_proj)
        self.register_site("S2", self.site_proj)
        self._init_parameters(generator)

        self.to(device)

    # -- init ------------------------------------------------------------
    def _init_parameters(self, generator: torch.Generator) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    fan_in = module.in_features
                    bound = 1.0 / max(fan_in, 1) ** 0.5
                    module.weight.copy_(
                        torch.empty_like(module.weight).uniform_(-bound, bound, generator=generator)
                    )
                    if module.bias is not None:
                        module.bias.copy_(
                            torch.empty_like(module.bias).uniform_(
                                -bound, bound, generator=generator
                            )
                        )

    # -- geometry helpers -------------------------------------------------
    @property
    def device_id(self) -> torch.device:
        return next(self.parameters()).device

    def _velocity(self, h: Tensor, x_tau: Tensor, tau: Tensor) -> Tensor:
        if self.velocity_out is None:
            raise RuntimeError("this toy policy has a regression head; there is no velocity head")
        batch = h.shape[0]
        if x_tau.shape[0] != batch:
            raise ValueError(f"batch mismatch: h has {batch} rows, x_tau has {x_tau.shape[0]}")
        flat = x_tau.reshape(batch, -1)
        t = tau.reshape(batch, 1).to(dtype=h.dtype, device=h.device)
        return self.velocity_out(torch.cat([h, flat, t], dim=-1)).reshape(
            batch, self.chunk_horizon, self.action_dim
        )

    def _regression_chunk(self, h: Tensor) -> Tensor:
        if self.action_out is None:
            raise RuntimeError(
                "this toy policy has a flow-matching head; use terminal_estimate instead"
            )
        return self.action_out(h).reshape(h.shape[0], self.chunk_horizon, self.action_dim)

    def _site_activation(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(z, h0, h_delta)``: pre-site features, base activation, LoRA delta."""
        z = torch.tanh(self.encoder(state))
        if isinstance(self.site_proj, LoRALinear):
            base = self.site_proj.base(z)
            delta = self.site_proj.scale * self.site_proj.lora_B(self.site_proj.lora_A(z))
            return z, base, delta
        return z, self.site_proj(z), torch.zeros_like(z)

    # -- PolicyAdapter ----------------------------------------------------
    def attach_lora(self, cfg: LoraCfg) -> int:
        """Wrap the site projection in a LoRA layer; returns the number adapted (0 or 1)."""
        if cfg.r <= 0:
            return 0  # documented no-op
        if isinstance(self.site_proj, LoRALinear):
            raise ValueError(
                "LoRA is already attached at the site; re-attaching would nest low-rank layers"
            )
        lora = LoRALinear(self.site_proj, cfg.r, cfg.alpha, cfg.dropout)
        self.site_proj = lora
        self._sites["S1"] = lora
        self._sites["S2"] = lora
        self.register_lora("S2", lora, cfg)
        self.freeze_base()
        return 1

    def forward(
        self,
        batch: dict[str, Any],
        site: str | None = None,
        hook: Any = None,
    ) -> PolicyOut:
        """Full forward. ``chunk`` is always the *terminal* action estimate."""
        site = site or self.default_site
        if site not in ("S1", "S2"):
            raise KeyError(
                f"toy policy exposes only S1/S2 (single expert block); got {site!r}. "
                "Use SmolVLA for the depth sweep (ablation A9)."
            )
        if "state" not in batch:
            raise KeyError("toy batch must contain 'state'")
        state = batch["state"].to(self.device_id)
        z, h0, delta = self._site_activation(state)
        h = self.apply_hook(h0 + delta, hook)

        velocity: Tensor | None = None
        out_tau: Tensor | None = None
        if self.action_head == "flow_matching":
            x_tau = batch.get("x_tau")
            tau = batch.get("tau")
            if x_tau is None or tau is None:
                raise KeyError(
                    "flow-matching toy policy needs 'x_tau' and 'tau' in the batch; "
                    "use ToyPolicy.sample_batch() to build a valid one"
                )
            x_tau = x_tau.to(self.device_id)
            tau = tau.to(self.device_id)
            velocity = self._velocity(h, x_tau, tau)
            chunk = terminal_estimate(x_tau, velocity, tau)
            out_tau = tau.detach()
        else:
            chunk = self._regression_chunk(h)

        return PolicyOut(
            chunk=chunk,
            site=site,
            h=h,
            tau=out_tau,
            velocity=velocity,
            extras={"h0": h0, "h": h, "lora_delta": delta, "pre_site": z},
        )

    def f_act(self, ctx: AdapterContext, h: Tensor, site: str) -> Tensor:
        """Downstream map from a site activation to the terminal estimate (``05`` §4)."""
        if self.action_head == "regression":
            return self._regression_chunk(h)
        return FlowPolicyMixin.f_act(self, ctx, h, site)

    def tail_velocity(self, ctx: AdapterContext, h: Tensor, _site: str) -> Tensor:
        """Velocity of the current flow step given a site activation (the differentiable tail)."""
        x_tau, tau = ctx.flow()
        return self._velocity(h, x_tau.to(self.device_id), tau.to(self.device_id))

    def step_velocity(self, ctx: AdapterContext, x_tau: Tensor, tau: Tensor, _site: str) -> Tensor:
        """Velocity from a full forward at the given flow step (used by B4b / diagnostics)."""
        state = ctx.batch.get("state")
        if state is None:
            raise KeyError("context batch has no 'state'; cannot run the full forward")
        _, h0, delta = self._site_activation(state.to(self.device_id))
        return self._velocity(h0 + delta, x_tau.to(self.device_id), tau.to(self.device_id))

    def make_context(self, batch: dict[str, Any], site: str | None = None) -> AdapterContext:
        ctx = super().make_context(batch, site)
        ctx.state = batch.get("state")
        return ctx

    # -- test / smoke helpers ---------------------------------------------
    def sample_batch(
        self,
        batch_size: int = 4,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        reward_scale: float = 0.5,
    ) -> dict[str, Any]:
        """Build a valid synthetic batch (state, a0, eps, tau, x_tau, velocity_target).

        ``a0`` is a stand-in for a demonstration chunk, ``eps`` the flow noise, and
        ``x_tau`` the interpolant at ``tau`` — so the batch satisfies the convention
        by construction and a perfect velocity predictor recovers ``a0`` exactly.
        """
        if generator is None:
            generator = torch.Generator().manual_seed(int(seed if seed is not None else self.seed))
        device = self.device_id
        state = torch.randn(batch_size, int(self._state_dim or 12), generator=generator).to(device)
        a0 = reward_scale * torch.randn(
            batch_size, self.chunk_horizon, self.action_dim, generator=generator
        ).to(device)
        eps = torch.randn(batch_size, self.chunk_horizon, self.action_dim, generator=generator).to(
            device
        )
        tau = tau_schedule("beta", (batch_size,), generator=generator, device=device)
        x_tau = flow_interpolate(a0, eps, tau)
        return {
            "state": state,
            "a0": a0,
            "eps": eps,
            "tau": tau,
            "x_tau": x_tau,
            "velocity_target": flow_target(a0, eps),
        }
