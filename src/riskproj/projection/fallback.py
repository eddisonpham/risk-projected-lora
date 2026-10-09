"""The fallback ladder F1 -> F2 -> F3 (``01`` §3, ``07`` §4).

Used when the projection cannot help: the candidate risk exceeds ``tau`` but the
gradient is unusable (below the ``eta`` floor, unavailable, non-finite). The
ladder never calls such a decision "safe": F1 re-checks the base action against
the critic, F2 holds the previous command for at most ``window`` steps, F3
terminates the episode and is counted as a failure.

Every event is recorded with a reason — CLAUDE.md non-negotiable #4: nothing
fails silently. ``counts()`` is per example (so a mixed batch is not averaged
away) and ``events()`` is per call (so a report can reconstruct the sequence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from riskproj.types import FallbackCode

__all__ = ["FallbackDecision", "FallbackLadder"]


@dataclass
class FallbackDecision:
    """Outcome of one ladder consultation for a whole batch.

    ``code`` is the most severe per-example code (F1 < F2 < F3); ``per_example``
    carries the individual codes so a per-cell report can attribute fallbacks.
    ``action`` is the action to execute: the candidate for corrected examples, the
    held/previous action for F2 rows and the base action for passing F1 rows.
    Rows coded F3 must be terminated by the caller — the action is a placeholder
    for bookkeeping only, never something to keep stepping with.
    """

    code: int
    action: Tensor | None
    reason: str
    steps_held: int
    per_example: Tensor | None = None
    details: dict[str, Any] = field(default_factory=dict)


class FallbackLadder:
    """Per-episode fallback state.

    One instance per episode (or per evaluation rollout): it owns the hold
    counter, the last action and the event log. Counters are tensors so a
    per-example mix of unusable gradients is represented exactly.
    """

    def __init__(self, *, window: int = 10, enabled: bool = True) -> None:
        if window < 1:
            raise ValueError("window must be >= 1 (F2 may not hold forever)")
        self.window = int(window)
        self.enabled = bool(enabled)
        self._held: Tensor | None = None
        self._last_action: Tensor | None = None
        self._counts: dict[str, int] = {"none": 0, "F1": 0, "F2": 0, "F3": 0}
        self._events: list[dict[str, Any]] = []
        self.n_updates = 0

    # -- state -----------------------------------------------------------
    def reset(self) -> None:
        self._held = None
        self._last_action = None

    def _resize(self, n: int, device: torch.device) -> Tensor:
        if self._held is None or self._held.shape[0] != n or self._held.device != device:
            self._held = torch.zeros(n, dtype=torch.long, device=device)
        return self._held

    def note_action(self, action: Tensor) -> None:
        """Record the action actually executed (becomes the F2 hold target)."""
        self._last_action = action.detach().clone()

    # -- main entry point ------------------------------------------------
    def update(
        self,
        *,
        usable: Tensor,
        action_candidate: Tensor,
        base_action: Tensor,
        base_action_risk: Tensor | None,
        threshold: float,
        evidence: dict[str, float] | None = None,
    ) -> FallbackDecision:
        """Consult the ladder for the examples whose gradient was unusable.

        ``usable`` is a ``[B]`` bool: True means the projection produced a
        corrected activation and the candidate action should be used as-is.
        """
        if usable.dim() != 1:
            raise ValueError(f"usable must be [B]; got {tuple(usable.shape)}")
        if action_candidate.shape[0] != usable.shape[0]:
            raise ValueError(
                f"batch mismatch: usable has {usable.shape[0]} rows, candidate has "
                f"{action_candidate.shape[0]}"
            )
        n = usable.shape[0]
        device = usable.device
        self.n_updates += 1
        needs = ~usable.to(torch.bool)

        held = self._resize(n, device)
        action = action_candidate.clone()
        codes = torch.zeros(n, dtype=torch.long, device=device)
        reason_bits: list[str] = []

        if not bool(needs.any()):
            self._counts["none"] += n
            self._events.append(
                {
                    "call": self.n_updates,
                    "code": FallbackCode.NONE,
                    "n_examples": 0,
                    "reason": "projection usable for all examples; no fallback",
                    "evidence": dict(evidence or {}),
                }
            )
            return FallbackDecision(
                code=FallbackCode.NONE,
                action=action,
                reason="no fallback needed",
                steps_held=int(held.max().item()) if n else 0,
                per_example=codes,
                details={"n_needing": 0},
            )

        if not self.enabled:
            # Loud, not silent: an unusable gradient without a ladder is an
            # uncorrected risk that the report must be able to count.
            self._counts["none"] += int(needs.sum().item())
            self._events.append(
                {
                    "call": self.n_updates,
                    "code": FallbackCode.NONE,
                    "n_examples": int(needs.sum().item()),
                    "reason": "fallback disabled: unusable gradient left uncorrected",
                    "evidence": dict(evidence or {}),
                }
            )
            return FallbackDecision(
                code=FallbackCode.NONE,
                action=action,
                reason="fallback disabled; unusable gradients left uncorrected",
                steps_held=int(held.max().item()) if n else 0,
                per_example=codes,
                details={"n_needing": int(needs.sum().item()), "enabled": False},
            )

        # F1: the base (unadapted) action is acceptable to the critic.
        if base_action_risk is None:
            f1_ok = torch.zeros_like(needs)
            reason_bits.append("F1 unavailable (no critic risk for the base action)")
        else:
            risk = base_action_risk.reshape(-1).to(torch.float32)
            f1_ok = needs & torch.isfinite(risk) & (risk <= threshold)
            if bool(f1_ok.any()):
                reason_bits.append("F1 base action passes the critic")
        codes = torch.where(f1_ok, torch.full_like(codes, FallbackCode.F1_BASE_ACTION), codes)
        action = torch.where(
            f1_ok.view(-1, *([1] * (action.dim() - 1))), base_action.to(action.dtype), action
        )

        # F2: hold the previous command, but only for `window` steps.
        remaining = needs & ~f1_ok
        held = torch.where(remaining, held + 1, torch.zeros_like(held))
        f2 = remaining & (held <= self.window)
        codes = torch.where(f2, torch.full_like(codes, FallbackCode.F2_HOLD), codes)
        if bool(f2.any()):
            # Hold the last executed action. Before anything has been executed
            # there is nothing to hold, so fall back to the base action rather
            # than to the risky candidate.
            hold_target = self._last_action if self._last_action is not None else base_action
            action = torch.where(
                f2.view(-1, *([1] * (action.dim() - 1))),
                hold_target.to(action.dtype),
                action,
            )
            reason_bits.append(f"F2 hold <= {self.window} steps")

        # F3: hold budget exhausted -> terminate (a failure, never "safe").
        f3 = remaining & ~f2
        codes = torch.where(f3, torch.full_like(codes, FallbackCode.F3_TERMINATE), codes)
        if bool(f3.any()):
            reason_bits.append(f"F3 hold > {self.window} steps: terminate")

        self._counts["F1"] += int(f1_ok.sum().item())
        self._counts["F2"] += int(f2.sum().item())
        self._counts["F3"] += int(f3.sum().item())
        self._counts["none"] += n - int(needs.sum().item())
        self._held = held

        worst = int(codes.max().item())
        details = {
            "n_needing": int(needs.sum().item()),
            "n_F1": int(f1_ok.sum().item()),
            "n_F2": int(f2.sum().item()),
            "n_F3": int(f3.sum().item()),
        }
        self._events.append(
            {
                "call": self.n_updates,
                "code": worst,
                "n_examples": int(needs.sum().item()),
                "reason": "; ".join(reason_bits) or "unusable gradient",
                "counts": {k: v for k, v in details.items() if k != "n_needing"},
                "evidence": dict(evidence or {}),
            }
        )
        return FallbackDecision(
            code=worst,
            action=action,
            reason="; ".join(reason_bits) or "unusable gradient",
            steps_held=int(held.max().item()) if n else 0,
            per_example=codes,
            details=details,
        )

    # -- reporting -------------------------------------------------------
    def counts(self) -> dict[str, int]:
        """Per-example counts with the keys ``F1``, ``F2``, ``F3``, ``none``."""
        return dict(self._counts)

    def events(self) -> list[dict[str, Any]]:
        """Every consultation that mattered, in order, with its reason."""
        return list(self._events)

    def summary(self) -> dict[str, float]:
        """Log-line friendly view: rates plus the raw counts."""
        total = sum(self._counts.values())
        out = {f"fallback_{k}": float(v) for k, v in self._counts.items()}
        if total:
            out.update({f"fallback_rate_{k}": v / total for k, v in self._counts.items()})
        out["fallback_n_consultations"] = float(self.n_updates)
        return out
