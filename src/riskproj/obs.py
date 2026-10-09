"""Observation and context contract.

Two *different* input sets exist and must never be conflated (``04`` §5):

* **policy visible** — ``images``, ``state`` (proprioception), ``instruction``,
  short history. This is what the VLA sees.
* **critic visible (deployable)** — proprioception + end-effector velocity +
  short action/proprio history + an optional *frozen* visual embedding + the
  candidate future action chunk. It may **not** see true friction/mass/stiffness,
  the cell id, or object ground-truth pose unless the policy also receives it.

Simulated wrist force is a separate, flagged "privileged-sim" input used only by
ablation E-critic-force; it is off by default and must stay off for any headline
number.

Tasks emit env observations with the keys declared in :data:`TASK_OBS_KEYS`;
:func:`env_obs_to_step_obs` and :class:`CtxBuilder` turn those into the critic's
context, one place, so no subsystem can invent its own feature layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Keys a task environment must return from ``reset``/``step``.
TASK_OBS_KEYS: tuple[str, ...] = ("state", "eef_vel", "images", "instruction")
#: Optional keys: ``force`` (privileged-sim), ``vision_emb`` (frozen embedding),
#: ``progress`` (task progress in [0, 1]), ``grasped`` (bool), ``contacts`` (dict).
TASK_OBS_OPTIONAL_KEYS: tuple[str, ...] = (
    "force",
    "vision_emb",
    "progress",
    "grasped",
    "contacts",
    "object_pose",
    "eef_pose",
)

CTX_KEYS: tuple[str, ...] = (
    "proprio",
    "ee_vel",
    "action_hist",
    "proprio_hist",
    "vision_emb",
    "force",
)


@dataclass
class StepObs:
    """One environment step, in the form the context builder consumes."""

    proprio: np.ndarray  # [P]
    ee_vel: np.ndarray  # [3]
    action: np.ndarray | None = None  # [A], the action actually taken *this* step
    vision_emb: np.ndarray | None = None  # [D], frozen encoder output
    force: np.ndarray | None = None  # [F], privileged-sim only
    extra: dict[str, Any] = field(default_factory=dict)


def env_obs_to_step_obs(obs: dict[str, Any], action: np.ndarray | None = None) -> StepObs:
    """Convert a task observation dict into a :class:`StepObs` (fail loud)."""
    missing = [k for k in ("state", "eef_vel") if k not in obs]
    if missing:
        raise KeyError(f"task observation is missing required keys {missing}; got {sorted(obs)}")
    proprio = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    ee_vel = np.asarray(obs["eef_vel"], dtype=np.float32).reshape(-1)
    if ee_vel.shape[0] != 3:
        raise ValueError(f"eef_vel must have 3 components; got {ee_vel.shape}")
    extra = {
        k: obs[k]
        for k in ("progress", "grasped", "contacts", "object_pose", "eef_pose")
        if k in obs
    }
    return StepObs(
        proprio=proprio,
        ee_vel=ee_vel,
        action=None if action is None else np.asarray(action, dtype=np.float32).reshape(-1),
        vision_emb=None
        if obs.get("vision_emb") is None
        else np.asarray(obs["vision_emb"], dtype=np.float32),
        force=None
        if obs.get("force") is None
        else np.asarray(obs["force"], dtype=np.float32).reshape(-1),
        extra=extra,
    )


class CtxBuilder:
    """Maintains the short action/proprio history the critic is allowed to see.

    This is a *deployable-inputs-only* container: it will raise if asked for
    privileged force while the configuration has it disabled, so a headline run
    cannot silently consume simulator ground truth.
    """

    def __init__(
        self, hist_len: int = 1, *, use_vision: bool = False, privileged_force: bool = False
    ) -> None:
        if hist_len < 0:
            raise ValueError("hist_len must be >= 0")
        self.hist_len = int(hist_len)
        self.use_vision = bool(use_vision)
        self.privileged_force = bool(privileged_force)
        self._actions: list[np.ndarray] = []
        self._proprios: list[np.ndarray] = []
        self._last: StepObs | None = None

    def reset(self) -> None:
        self._actions.clear()
        self._proprios.clear()
        self._last = None

    def observe(self, step: StepObs) -> None:
        if step.action is not None:
            self._actions.append(step.action.astype(np.float32))
        self._proprios.append(step.proprio.astype(np.float32))
        self._last = step

    # -- accessors -------------------------------------------------------
    def _tail(self, buf: list[np.ndarray], width: int) -> np.ndarray:
        """Last ``hist_len`` entries, left-padded with zeros (never a ragged array)."""
        n = max(self.hist_len, 1)
        if not buf:
            return np.zeros((n, width), dtype=np.float32)
        rows = buf[-n:]
        pad = n - len(rows)
        out = np.zeros((n, width), dtype=np.float32)
        for i, r in enumerate(rows):
            out[pad + i, : r.shape[0]] = r[:width]
        return out

    def context(self) -> dict[str, np.ndarray]:
        """Batched-free context for a single decision point."""
        if self._last is None:
            raise RuntimeError("CtxBuilder.context() called before any observe()")
        last = self._last
        proprio = last.proprio.astype(np.float32)
        ctx: dict[str, np.ndarray] = {
            "proprio": proprio,
            "ee_vel": last.ee_vel.astype(np.float32),
            "action_hist": self._tail(self._actions, proprio.shape[0]),
            "proprio_hist": self._tail(self._proprios, proprio.shape[0]),
        }
        ctx["vision_emb"] = (
            np.zeros(0, dtype=np.float32)
            if (not self.use_vision or last.vision_emb is None)
            else last.vision_emb.astype(np.float32)
        )
        ctx["force"] = (
            np.zeros(0, dtype=np.float32)
            if (not self.privileged_force or last.force is None)
            else last.force.astype(np.float32)
        )
        return ctx

    def context_dim(self, *, action_dim: int, state_dim: int, vision_dim: int = 0) -> int:
        """Width of :func:`context_vector` for a given layout — used to size the critic."""
        n = max(self.hist_len, 1)
        dim = state_dim + 3 + n * action_dim + n * state_dim
        if self.use_vision:
            dim += vision_dim
        if self.privileged_force:
            raise ValueError("privileged force width is task-specific; pass it explicitly instead")
        return dim


def context_vector(ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten a context dict into a fixed-order feature vector.

    Order: ``proprio | ee_vel | flatten(action_hist) | flatten(proprio_hist) |
    vision_emb | force``. Empty optional blocks contribute nothing, so a config
    change of ``use_vision``/``privileged_force`` changes the width (the critic
    checks its own input width at load time).
    """
    parts = [
        np.asarray(ctx["proprio"], dtype=np.float32).reshape(-1),
        np.asarray(ctx["ee_vel"], dtype=np.float32).reshape(-1),
        np.asarray(ctx["action_hist"], dtype=np.float32).reshape(-1),
        np.asarray(ctx["proprio_hist"], dtype=np.float32).reshape(-1),
    ]
    for key in ("vision_emb", "force"):
        block = np.asarray(ctx.get(key, np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(
            -1
        )
        if block.size:
            parts.append(block)
    return np.concatenate(parts).astype(np.float32)


def stack_contexts(contexts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-decision contexts into a batch (ragged optionals become width 0)."""
    if not contexts:
        raise ValueError("no contexts to stack")
    out: dict[str, np.ndarray] = {}
    for key in CTX_KEYS:
        blocks = [np.asarray(c[key], dtype=np.float32).reshape(-1) for c in contexts]
        widths = {b.shape[0] for b in blocks}
        if len(widths) > 1 and key in ("vision_emb", "force"):
            # Optional block present for some rows only: pad to the widest.
            width = max(widths)
            blocks = [np.pad(b, (0, width - b.shape[0])) for b in blocks]
        out[key] = np.stack(blocks)
    return out


__all__ = [
    "CTX_KEYS",
    "TASK_OBS_KEYS",
    "TASK_OBS_OPTIONAL_KEYS",
    "CtxBuilder",
    "StepObs",
    "context_vector",
    "env_obs_to_step_obs",
    "stack_contexts",
]
