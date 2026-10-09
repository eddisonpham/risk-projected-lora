"""Structural contracts every subsystem implements.

Depend on these Protocols, not on concrete classes (``skills/software_engineering.md``
§2). Concrete implementations register themselves in the registries from
:mod:`riskproj.utils.registry`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn

from riskproj.types import DynamicsSpec, PolicyOut, ProjectionInfo, ViolationSpec

#: A per-example risk functional over an activation: ``h -> r`` with ``r.shape == [B]``.
RiskFn = Callable[[Tensor], Tensor]


@runtime_checkable
class PolicyAdapter(Protocol):
    """Thin adapter over a VLA/BC policy exposing *sites* and *local forwards*.

    The projection operator only ever needs ``sites()`` and the downstream map;
    the training loop needs ``forward``/``parameters``.
    """

    name: str

    def sites(self) -> dict[str, Any]:
        """Named activation sites: site id -> module (or a documented hook handle)."""

    def site_tree(self) -> nn.Module:
        """The ``nn.Module`` whose ``named_modules()`` names the site patterns.

        One namespace, used by both the discovery script (which *writes*
        ``configs/model/<model>_sites.yaml``) and the adapter (which resolves those
        patterns at load time). An adapter that wraps the policy (the adapter is an
        ``nn.Module`` holding it as ``self._model``) must return the *wrapped* object,
        or the file comes out with an extra prefix and nothing resolves.
        """

    def attach_lora(self, cfg: Any) -> int:
        """Attach LoRA at the configured placement; returns number of adapted modules."""

    def forward(
        self, batch: Any, site: str | None = None, hook: Callable[..., Any] | None = None
    ) -> PolicyOut:
        """Full policy forward; ``site``/``hook`` expose the chosen activation."""

    def f_act(self, ctx: Any, h: Tensor, site: str) -> Tensor:
        """Downstream map from a site activation to the terminal action estimate.

        Must be differentiable in ``h`` and reuse cached context (frozen VLM KV,
        flow time, noisy chunk) so that ``f_act(h0) == forward(...).chunk``.
        """


@runtime_checkable
class Critic(Protocol):
    """Risk critic ``V_phi(c, a_hat) -> logits in R^{1+5}`` (any-violation + V1..V5)."""

    horizon: int

    def logits(self, ctx: Any, chunk: Tensor) -> Tensor:
        """Raw logits, shape ``[B, 1 + 5]``; column 0 is any-violation."""

    def risk(self, ctx: Any, chunk: Tensor, *, space: str = "prob") -> Tensor:
        """Scalar risk in ``[B]``: ``sigmoid(z0)`` in prob space, ``z0`` in logit space."""


@runtime_checkable
class Projector(Protocol):
    """Activation projection: risk-feasibility correction at a site."""

    def __call__(self, h_c: Tensor, risk_fn: RiskFn, cfg: Any) -> tuple[Tensor, ProjectionInfo]:
        """Return ``(h_safe, info)``; must be per-example independent and logged."""


@runtime_checkable
class Task(Protocol):
    """Simulator task: env construction, dynamics, violation spec, scripted expert."""

    name: str
    violations: ViolationSpec

    def make_env(self, dyn: DynamicsSpec, seed: int, **kwargs: Any) -> Any:
        """Build a (possibly vectorized) env with ``dyn`` applied and read back."""

    def expert(self) -> Any:
        """Scripted expert used for nominal data and threshold derivation."""


@runtime_checkable
class Metric(Protocol):
    """Streaming metric over archived records; never needs a live model."""

    name: str

    def update(self, rec: dict[str, Any]) -> None: ...

    def compute(self) -> dict[str, Any]: ...


@runtime_checkable
class Method(Protocol):
    """A training/inference method (B0-B6, C1-C3)."""

    name: str

    def train(self, cfg: Any) -> Path:
        """Run training; returns the checkpoint directory (or a run dir)."""

    def wrap(self, policy: PolicyAdapter) -> PolicyAdapter:
        """Return the inference-time policy wrapper for this method."""


__all__ = ["Critic", "Method", "Metric", "PolicyAdapter", "Projector", "RiskFn", "Task"]
