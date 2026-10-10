"""Actors that drive an env, including a deliberately failure-inducing one (``04`` §8).

G1's *coverage* criterion asks whether each violation event V1-V5 is reachable at all.
The scripted expert is supposed to be failure-free, so measuring reachability on it alone
answers a different question — how often a correct controller trips the events — and on
peg-insert-side it answered "never" for V1/V3/V5 (``research_log/LOG.md``, iteration 8).
:class:`PerturbedExpertActor` is the failure-inducing alternative: the expert's own
actions plus a coherent per-chunk deviation, using the **same sigma convention** as
``riskproj.data.gen_critic_data.perturb_chunk`` (noise scaled by the RMS of the actions),
so that "sigma" means the same thing in coverage as it does in the critic's unsafe class.

The one deliberate difference from ``perturb_chunk``: that function perturbs an expert
chunk that was *replayed from a saved state*, so the whole horizon exists before it is
perturbed. Here the chunk does not exist yet — the expert is a feedback controller driving
the live env — so one deviation vector is drawn at each chunk boundary and held for the
chunk's duration. That is a coherent offset rather than white noise; it is documented
rather than hidden, and both are logged in the sweep payload.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["PerturbedExpertActor", "clip_actions"]


def clip_actions(action: Any, low: Any, high: Any) -> np.ndarray:
    """Clip an action to the env's bounds, if bounds were given."""
    out = np.asarray(action, dtype=np.float64).reshape(-1)
    if low is None or high is None:
        return out
    return np.clip(out, np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64))


@dataclass
class PerturbedExpertActor:
    """Expert actions + a coherent per-chunk Gaussian deviation.

    ``sigma`` scales the deviation by the RMS of the last ``scale_window`` commanded
    actions (the expert's own, not the perturbed ones), which is the convention
    ``perturb_chunk`` uses; ``horizon`` is how many consecutive steps share one deviation
    vector. ``reseed`` is called per episode by the sweep so that a given (axis, scale,
    episode) is reproducible rather than depending on how many episodes ran before it.
    """

    expert: Any
    sigma: float = 0.2
    horizon: int = 10
    seed: int = 0
    action_low: Any = None
    action_high: Any = None
    scale_window: int = 10
    low_frequency: bool = True  # kept for payload/provenance parity with perturb_chunk
    _rng: np.random.Generator = field(init=False, repr=False)
    _recent: deque[np.ndarray] = field(init=False, repr=False)
    _offset: np.ndarray | None = field(init=False, default=None, repr=False)
    _left: int = field(init=False, default=0, repr=False)
    n_draws: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0; got {self.sigma}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {self.horizon}")
        self.reseed(self.seed)

    def reseed(self, seed: int) -> None:
        """Restart the deviation stream; called once per episode."""
        self._rng = np.random.default_rng(int(seed))
        self._recent = deque(maxlen=max(1, int(self.scale_window)))
        self._offset = None
        self._left = 0
        self.n_draws = 0

    def _next_offset(self, like: np.ndarray) -> np.ndarray:
        if self.sigma == 0.0:
            return np.zeros_like(like)
        if self._recent:
            sample = np.stack(self._recent)
            scale = float(np.sqrt(np.mean(sample**2)))
        else:
            scale = float(np.sqrt(np.mean(like**2)))
        scale = scale if scale > 0 else 1.0
        # A single vector per chunk: the deviation is coherent over `horizon` steps, which
        # is the "low_frequency" mode of perturb_chunk (a constant is the strongest such
        # smoothing) and is what a misplaced wrist rotation looks like in practice.
        self.n_draws += 1
        return self.sigma * scale * self._rng.standard_normal(like.shape)

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        base = np.asarray(self.expert.act(obs), dtype=np.float64).reshape(-1)
        if self._offset is None or self._offset.shape != base.shape or self._left <= 0:
            self._offset = self._next_offset(base)
            self._left = int(self.horizon)
        self._left -= 1
        self._recent.append(base.copy())
        return clip_actions(base + self._offset, self.action_low, self.action_high)
