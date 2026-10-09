"""Deployable-input feature builder for the risk critic (``04`` §5, ``06`` §2).

The critic may consume **only** what a deployable policy would have:
proprioception, end-effector velocity, a short action/proprio history, an optional
*frozen* visual embedding, and the candidate action chunk. True friction / mass /
stiffness, the cell id, and object ground-truth pose are forbidden inputs;
simulated wrist force is a flagged privileged-sim ablation input.

Both forbidden blocks are therefore *guarded by configuration*, not by
convention: :func:`flatten_context_torch` raises if a caller hands over a
``vision_emb`` while ``CriticCfg.use_vision`` is False, or a ``force`` block while
``CriticCfg.privileged_force`` is False. A headline run cannot silently eat
simulator ground truth.

Column order is fixed and identical to :func:`riskproj.obs.context_vector`:
``proprio | ee_vel | flatten(action_hist) | flatten(proprio_hist) | vision_emb | force | flatten(chunk)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from riskproj.utils.config import CriticCfg

#: Bumped whenever the feature layout changes, so old checkpoints fail loudly.
FEATURE_LAYOUT_VERSION = 1


def _as_tensor(x: Any) -> Tensor:
    """to float32 tensor without detaching (the chunk may carry a graph to a site)."""
    if isinstance(x, Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


@dataclass(frozen=True)
class FeatureSpec:
    """The exact input layout a critic was trained with.

    Stored inside every checkpoint: loading a model against a different layout is
    an error, not a silent reshape (``06`` §3).
    """

    state_dim: int
    action_dim: int
    chunk_horizon: int
    hist_len: int = 1
    vision_dim: int = 0
    force_dim: int = 0
    use_vision: bool = False
    privileged_force: bool = False

    def __post_init__(self) -> None:
        for name in ("state_dim", "action_dim", "chunk_horizon"):
            if getattr(self, name) < 1:
                raise ValueError(f"FeatureSpec.{name} must be >= 1; got {getattr(self, name)}")
        for name in ("hist_len", "vision_dim", "force_dim"):
            if getattr(self, name) < 0:
                raise ValueError(f"FeatureSpec.{name} must be >= 0; got {getattr(self, name)}")
        if self.use_vision and self.vision_dim < 1:
            raise ValueError("use_vision=True requires vision_dim >= 1")
        if not self.use_vision and self.vision_dim != 0:
            raise ValueError(
                "vision_dim must be 0 when use_vision is False; the vision block is "
                "off by default because it is not part of the deployable set (04 §5)"
            )
        if self.privileged_force and self.force_dim < 1:
            raise ValueError("privileged_force=True requires force_dim >= 1")
        if not self.privileged_force and self.force_dim != 0:
            raise ValueError(
                "force_dim must be 0 when privileged_force is False; simulated force is a "
                "flagged ablation input only (04 §5)"
            )

    # -- widths ----------------------------------------------------------
    @property
    def history_steps(self) -> int:
        """CtxBuilder always emits at least one history row (zero-padded)."""
        return max(self.hist_len, 1)

    @property
    def context_dim(self) -> int:
        n = self.history_steps
        return (
            self.state_dim
            + 3
            + n * self.action_dim
            + n * self.state_dim
            + (self.vision_dim if self.use_vision else 0)
            + (self.force_dim if self.privileged_force else 0)
        )

    @property
    def chunk_dim(self) -> int:
        return self.chunk_horizon * self.action_dim

    @property
    def dim(self) -> int:
        return self.context_dim + self.chunk_dim

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureSpec:
        return cls(**d)


def _block_width(x: Any, batch: int) -> int:
    """Per-row width of an optional block; ``None`` and ``[B, 0]`` both mean 'absent'."""
    if x is None:
        return 0
    t = _as_tensor(x)
    if t.numel() == 0:
        return 0
    return int(t.numel() // max(batch, 1)) if t.dim() >= 1 else 1


def _promote_history(x: Any, *, key: str, steps: int, width: int) -> Tensor:
    """Coerce a history block to ``[B, steps, width]``.

    ``riskproj.obs.stack_contexts`` flattens each history block, so a batched
    context can arrive as ``[B, steps * width]``. That is unambiguous once
    ``steps`` and ``width`` are known from the spec, so it is unflattened rather
    than rejected (with ``steps == 1`` the two forms coincide).
    """
    t = _as_tensor(x)
    if t.dim() == 2 and steps > 1 and t.shape[-1] == steps * width:
        t = t.reshape(t.shape[0], steps, width)
    return _promote(t, key=key, per_step=False).float()


def _promote(x: Any, *, key: str, per_step: bool) -> Tensor:
    """Coerce a ctx block to a batch: ``[B, ...]`` (single examples are promoted)."""
    t = _as_tensor(x)
    if per_step:
        # per-step block: [D] -> [1, D]
        if t.dim() == 0:
            raise ValueError(f"ctx[{key!r}] must not be a scalar")
        return t.unsqueeze(0) if t.dim() == 1 else t
    # history block: [n, W] -> [1, n, W]; [W] -> [1, 1, W]
    if t.dim() == 1:
        return t.reshape(1, 1, -1)
    if t.dim() == 2:
        return t.unsqueeze(0)
    return t


def flatten_context_torch(ctx: dict[str, Any], spec: FeatureSpec) -> Tensor:
    """Flatten a batched context dict to ``[B, spec.context_dim]`` in the canonical order.

    Raises if a configured-off optional block is non-empty (the anti-leak guard).
    """
    missing = [k for k in ("proprio", "ee_vel", "action_hist", "proprio_hist") if k not in ctx]
    if missing:
        raise KeyError(
            f"context is missing required blocks {missing}; expected keys "
            f"{('proprio', 'ee_vel', 'action_hist', 'proprio_hist', 'vision_emb', 'force')}"
        )

    n = spec.history_steps
    proprio = _promote(ctx["proprio"], key="proprio", per_step=True).float()
    ee_vel = _promote(ctx["ee_vel"], key="ee_vel", per_step=True).float()
    ahist = _promote_history(ctx["action_hist"], key="action_hist", steps=n, width=spec.action_dim)
    phist = _promote_history(ctx["proprio_hist"], key="proprio_hist", steps=n, width=spec.state_dim)

    checks = [
        ("proprio", proprio, spec.state_dim, 1),
        ("ee_vel", ee_vel, 3, 1),
        ("action_hist", ahist, spec.action_dim, 2),
        ("proprio_hist", phist, spec.state_dim, 2),
    ]
    for name, tensor, width, rank in checks:
        if tensor.dim() != rank + 1:
            raise ValueError(
                f"ctx[{name!r}] must have {rank + 1} dims after promotion (B + {rank}); "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[-1] != width:
            raise ValueError(f"ctx[{name!r}] last dim is {tensor.shape[-1]}, expected {width}")
    for name, tensor in (("action_hist", ahist), ("proprio_hist", phist)):
        if tensor.shape[-2] != n:
            raise ValueError(
                f"ctx[{name!r}] has {tensor.shape[-2]} history rows, expected {n} "
                f"(hist_len={spec.hist_len}; CtxBuilder left-pads to at least one row)"
            )

    batch = {proprio.shape[0], ee_vel.shape[0], ahist.shape[0], phist.shape[0]}
    if len(batch) != 1:
        raise ValueError(f"inconsistent batch sizes across context blocks: {sorted(batch)}")
    B = batch.pop()

    blocks = [proprio, ee_vel, ahist.reshape(B, -1), phist.reshape(B, -1)]

    if not spec.use_vision:
        if _block_width(ctx.get("vision_emb"), B) != 0:
            raise ValueError(
                "ctx['vision_emb'] is non-empty but CriticCfg.use_vision is False; the frozen "
                "visual embedding is an ablation input (A18) and must be enabled explicitly"
            )
    else:
        width = _block_width(ctx.get("vision_emb"), B)
        if width != spec.vision_dim:
            raise ValueError(
                f"ctx['vision_emb'] has width {width}, expected {spec.vision_dim} "
                "(CriticCfg.use_vision is True, so the block is required)"
            )
        blocks.append(_promote(ctx["vision_emb"], key="vision_emb", per_step=True).float())

    if not spec.privileged_force:
        if _block_width(ctx.get("force"), B) != 0:
            raise ValueError(
                "ctx['force'] is non-empty but CriticCfg.privileged_force is False; simulated "
                "wrist force is a privileged-sim input used only by ablation E-critic-force"
            )
    else:
        width = _block_width(ctx.get("force"), B)
        if width != spec.force_dim:
            raise ValueError(
                f"ctx['force'] has width {width}, expected {spec.force_dim} "
                "(CriticCfg.privileged_force is True, so the block is required)"
            )
        blocks.append(_promote(ctx["force"], key="force", per_step=True).float())

    out = torch.cat([b.reshape(B, -1) for b in blocks], dim=1)
    if out.shape[1] != spec.context_dim:
        raise RuntimeError(
            f"context width {out.shape[1]} != FeatureSpec.context_dim {spec.context_dim}; "
            "the layout drifted, which would silently change the critic's meaning"
        )
    return out


class CriticFeatureBuilder:
    """Callable ``(ctx, chunk) -> [B, dim]`` with a stable, checkpointed layout."""

    def __init__(
        self,
        cfg: CriticCfg,
        *,
        state_dim: int,
        action_dim: int,
        chunk_horizon: int,
        vision_dim: int = 0,
        force_dim: int = 0,
    ) -> None:
        self.cfg = cfg
        self.spec = FeatureSpec(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_horizon=chunk_horizon,
            hist_len=cfg.use_history,
            vision_dim=vision_dim if cfg.use_vision else 0,
            force_dim=force_dim if cfg.privileged_force else 0,
            use_vision=cfg.use_vision,
            privileged_force=cfg.privileged_force,
        )

    @classmethod
    def from_spec(cls, cfg: CriticCfg, spec: FeatureSpec) -> CriticFeatureBuilder:
        """Rebuild a builder for an existing layout (used when loading a checkpoint).

        The stored :class:`FeatureSpec` wins over ``cfg`` on every layout field, so
        a checkpoint can never be reshaped by an edited config; the flags still
        gate the leak guards through the spec itself.
        """
        builder = cls.__new__(cls)
        builder.cfg = cfg
        builder.spec = spec
        return builder

    def dim(self) -> int:
        return self.spec.dim

    @property
    def context_dim(self) -> int:
        return self.spec.context_dim

    def __call__(self, ctx: dict[str, Any], chunk: Tensor | np.ndarray) -> Tensor:
        chunk_t = _as_tensor(chunk)
        if chunk_t.dim() != 3:
            raise ValueError(f"candidate chunk must be [B, H, A]; got {tuple(chunk_t.shape)}")
        if chunk_t.shape[1] != self.spec.chunk_horizon:
            raise ValueError(
                f"candidate chunk horizon is {chunk_t.shape[1]} but the critic was built for "
                f"{self.spec.chunk_horizon} (exec_horizon == critic_horizon, 04 §1)"
            )
        if chunk_t.shape[2] != self.spec.action_dim:
            raise ValueError(
                f"candidate chunk action dim is {chunk_t.shape[2]}, expected {self.spec.action_dim}"
            )
        ctx_flat = flatten_context_torch(ctx, self.spec)
        if ctx_flat.shape[0] != chunk_t.shape[0]:
            raise ValueError(f"context batch {ctx_flat.shape[0]} != chunk batch {chunk_t.shape[0]}")
        out = torch.cat([ctx_flat, chunk_t.reshape(chunk_t.shape[0], -1)], dim=1)
        if out.shape[1] != self.dim():
            raise RuntimeError(f"feature width {out.shape[1]} != FeatureSpec.dim {self.dim()}")
        return out


__all__ = [
    "FEATURE_LAYOUT_VERSION",
    "CriticFeatureBuilder",
    "FeatureSpec",
    "flatten_context_torch",
]
