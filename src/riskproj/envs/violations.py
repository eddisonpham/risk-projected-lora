"""Violation events V1-V5 (``04`` §4).

Thresholds come from :class:`riskproj.types.ViolationSpec` and are derived from
**nominal scripted-expert rollouts only**, then frozen, *before* any learned
policy is evaluated.

``step_info`` contract (a KeyError naming the missing key is raised otherwise):

===========================  =====================================================
key                          meaning
===========================  =====================================================
``eef_pos``                  (3,) end-effector position
``object_pos``               (3,) manipulated-object position
``grasped``                  bool, is the object currently held
``force``                    float *sum* over task-critical contact pairs, or an
                             array of per-pair normal forces (summed here)
``penetration``              float, or array of per-contact ``dist`` values; the
                             **absolute** penetration depth (metres). Kept
                             stiffness-independent on purpose, otherwise V4 would
                             be confounded with the stiffness axis
``progress_m``               metres of task progress (distance closed)
optional
``grip_offset``              (3,) object pose in the gripper frame; derived as
                             ``object_pos - eef_pos`` when omitted
``grasp_established``        bool; inferred from the first ``grasped`` step
``released``                 bool, the command intentionally opened the gripper
``object_height``            float, object height above the table
``progress``                 float in [0, 1] for logging only
``contacts``                 raw per-pair dict for provenance
===========================  =====================================================

Event definitions (``04`` §4):

* **V1 slip** — grasped-object drift in the gripper frame above
  ``spec.slip_drift`` within ``spec.window`` steps, while the object is grasped.
* **V2 drop** — a grasp had been established and the object left it: the
  object-gripper distance (windowed, and cleared at each new grasp so pre-grasp
  samples cannot count) exceeded ``spec.drop_dist``, or the grasp was lost
  without an intentional release.
  The absolute-height rule that ``04`` §4 also mentions is deliberately *not*
  implemented: a drop is a loss of the gripper-frame offset, and ``|Δz| <= ‖Δp‖``
  means the distance test already fires no later than any height test — while an
  absolute-height test additionally fires whenever a policy legitimately lowers a
  held object to place it (the toy expert does exactly that, ~25 mm, so the rule
  reported a drop in 12/12 nominal episodes). ``object_height`` remains an
  accepted, optional field of ``step_info`` for provenance.
* **V3 over-force** — summed normal contact force above ``spec.force_max``.
* **V4 penetration** — any task-critical contact depth above
  ``spec.penetration`` (absolute, so the stiffness axis cannot move the bar).
* **V5 jam** — force above ``spec.jam_force`` for ``spec.jam_steps`` consecutive
  steps while progress moved less than ``spec.jam_progress_eps`` (FACT's
  precision/jam signature).

``any_violation = V1 or ... or V5`` is the scalar critic label; per-type labels
are always stored, never collapsed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from riskproj.types import VIOLATION_TYPES, ViolationSpec, empty_violation_counts

#: Progress (in [0,1]) above which a timeout counts as "stop-short" (partially
#: inserted) rather than "idle-timeout" (``04`` §4 V6 / FACT taxonomy [V]).
PARTIAL_PROGRESS = 0.05

#: Priority order for :func:`failure_class`; the first firing type wins.
FAILURE_PRIORITY: tuple[tuple[str, str], ...] = (
    ("V5", "rim_jam"),
    ("V3", "over_force"),
    ("V4", "penetration"),
    ("V1", "slip"),
    ("V2", "drop"),
)

REQUIRED_STEP_INFO: tuple[str, ...] = (
    "eef_pos",
    "object_pos",
    "grasped",
    "force",
    "penetration",
    "progress_m",
)


def _vec3(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"{name} must have 3 components; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {arr}")
    return arr


def _sum_force(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"force contains non-finite values: {arr}")
    return float(arr.sum()) if arr.size > 1 else float(arr[0])


def _max_penetration(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"penetration contains non-finite values: {arr}")
    return float(np.max(arr))


class ViolationMonitor:
    """Streaming detector for V1-V5 over a single episode.

    Every decision is recorded; nothing fails silently (``CLAUDE.md`` #4). The
    returned record always has exactly the shape of
    :func:`riskproj.types.empty_violation_counts`: ``{V: {"first_step": float,
    "count": float}}`` with ``first_step = -1.0`` when the type never fired.
    """

    def __init__(self, spec: ViolationSpec) -> None:
        self.spec = spec
        self._step = 0
        self._rec: dict[str, dict[str, float]] = empty_violation_counts()
        self._fired: dict[str, bool] = dict.fromkeys(VIOLATION_TYPES, False)
        # rolling windows
        self._drift_win: deque[float] = deque(maxlen=max(1, int(spec.window)))
        self._dist_win: deque[float] = deque(maxlen=max(1, int(spec.window)))
        self._force_win: deque[float] = deque(maxlen=max(1, int(spec.jam_steps)))
        self._progress_win: deque[float] = deque(maxlen=max(1, int(spec.jam_steps)))
        # grasp bookkeeping
        self._grasp_ref: np.ndarray | None = None
        self._grasp_established = False
        self._grasp_established_prev = False

        self._released = False
        # per-step trace of fired types, for horizon-windowed critic labels
        self.step_events: list[frozenset[str]] = []
        self._finalized = False

    # -- introspection ----------------------------------------------------
    @property
    def step(self) -> int:
        return self._step

    @property
    def per_type(self) -> dict[str, dict[str, float]]:
        return {k: dict(v) for k, v in self._rec.items()}

    @property
    def any_violation(self) -> bool:
        return any(self._fired[t] for t in VIOLATION_TYPES)

    def counts(self) -> dict[str, int]:
        return {t: int(self._rec[t]["count"]) for t in VIOLATION_TYPES}

    # -- main loop --------------------------------------------------------
    def observe(
        self,
        step_info: Mapping[str, Any],
        *,
        action: Any | None = None,  # noqa: ARG002 -- API hook: callers may attach the action
    ) -> dict[str, dict[str, float]]:
        """Consume one environment step; returns the running per-type record.

        ``action`` is accepted (and currently unused) so that a rollout logger can
        hand the monitor the executed action at the same call site as the step
        info, keeping projection/fallback events joinable to decisions.
        """
        if self._finalized:
            raise RuntimeError(
                "observe() called after finalize(); create a new monitor per episode"
            )
        missing = [k for k in REQUIRED_STEP_INFO if k not in step_info]
        if missing:
            raise KeyError(f"step_info is missing required keys {missing}; got {sorted(step_info)}")

        step = self._step
        self._step += 1
        eef = _vec3(step_info["eef_pos"], name="eef_pos")
        obj = _vec3(step_info["object_pos"], name="object_pos")
        grasped = bool(step_info["grasped"])
        force = _sum_force(step_info["force"])
        pen = _max_penetration(step_info["penetration"])
        progress_m = float(step_info["progress_m"])
        if not np.isfinite(progress_m):
            raise ValueError("progress_m must be finite")
        released = bool(step_info.get("released", False))
        if released:
            self._released = True

        grip_offset = (
            _vec3(step_info["grip_offset"], name="grip_offset")
            if step_info.get("grip_offset") is not None
            else (obj - eef)
        )
        if step_info.get("grasp_established") is not None:
            self._grasp_established = bool(step_info["grasp_established"])
        elif grasped:
            self._grasp_established = True

        fired: set[str] = set()

        # ---- V1 slip: windowed drift in the gripper frame ---------------
        if grasped:
            if self._grasp_ref is None:
                # a fresh grasp defines the reference pose, and starts a clean window
                self._grasp_ref = grip_offset.copy()
                self._drift_win.clear()
            drift = float(np.linalg.norm(grip_offset - self._grasp_ref))
        else:
            drift = 0.0
            self._grasp_ref = None  # the next grasp gets a fresh reference
        self._drift_win.append(drift)
        if grasped and self._grasp_ref is not None and max(self._drift_win) > self.spec.slip_drift:
            fired.add("V1")

        # ---- V2 drop ----------------------------------------------------
        dist = float(np.linalg.norm(obj - eef))
        # The distance window must only ever contain *in-grasp* samples. Before the
        # grasp the end effector is far from the object by construction (~0.12 m in
        # the toy task), so those samples would trip `max(window) > drop_dist` the
        # instant a grasp is established -- reporting a drop at nominal dynamics.
        # V1 clears its drift window on a fresh grasp for the same reason.
        if self._grasp_established and not self._grasp_established_prev:
            self._dist_win.clear()
        self._grasp_established_prev = bool(self._grasp_established)
        self._dist_win.append(dist)
        dropped = False
        if self._grasp_established:
            if max(self._dist_win) > self.spec.drop_dist:
                dropped = True
            if not grasped and not released:
                dropped = True
        if dropped:
            fired.add("V2")
            # the dropped object re-enters as a fresh grasp on the next grasp
            self._grasp_ref = None
            self._grasp_established = False

        # ---- V3 over-force ---------------------------------------------
        if force > self.spec.force_max:
            fired.add("V3")

        # ---- V4 penetration (absolute depth) ---------------------------
        if pen > self.spec.penetration:
            fired.add("V4")

        # ---- V5 jam: sustained force without progress -------------------
        self._force_win.append(force)
        self._progress_win.append(progress_m)
        if len(self._force_win) == self._force_win.maxlen:
            sustained = all(f > self.spec.jam_force for f in self._force_win)
            stalled = (self._progress_win[-1] - self._progress_win[0]) < self.spec.jam_progress_eps
            if sustained and stalled:
                fired.add("V5")

        for v in fired:
            if not self._fired[v]:
                self._rec[v]["first_step"] = float(step)
                self._fired[v] = True
            self._rec[v]["count"] += 1.0
        self.step_events.append(frozenset(fired))
        return self.per_type

    def finalize(self) -> dict[str, dict[str, float]]:
        """Freeze the record. Per-type labels are preserved exactly as detected."""
        self._finalized = True
        return self.per_type

    # -- horizon windows --------------------------------------------------
    def window(self, start_step: int, end_step: int) -> dict[str, dict[str, float]]:
        """Per-type record restricted to ``[start_step, end_step)``.

        This is how the critic's H_c labels are produced: the label for a chunk
        executed from a saved state covers exactly the executed steps (``04`` §4).
        """
        return violations_in_window(self.step_events, start_step, end_step)


def violations_in_window(
    step_events: Sequence[Iterable[str]], start_step: int, end_step: int
) -> dict[str, dict[str, float]]:
    """Aggregate a slice of per-step fired-type sets into the record shape."""
    if start_step < 0 or end_step < start_step:
        raise ValueError(f"bad window [{start_step}, {end_step})")
    rec = empty_violation_counts()
    for offset, fired in enumerate(step_events):
        step = offset
        if step < start_step or step >= end_step:
            continue
        for v in fired:
            if v not in rec:
                raise KeyError(f"unknown violation type {v!r} in step_events")
            if rec[v]["first_step"] < 0:
                rec[v]["first_step"] = float(step)
            rec[v]["count"] += 1.0
    return rec


def any_violation(violations: Mapping[str, Mapping[str, float]]) -> bool:
    """``any_violation = V1 or ... or V5``: the scalar critic label."""
    for v in VIOLATION_TYPES:
        entry = violations.get(v)
        if entry and float(entry.get("count", 0.0)) > 0.0:
            return True
    return False


def violation_rate(records: Iterable[Mapping[str, Mapping[str, float]]]) -> dict[str, float]:
    """Fraction of episodes in which each type fired, plus ``any``."""
    recs = list(records)
    if not recs:
        return {**{v: float("nan") for v in VIOLATION_TYPES}, "any": float("nan")}
    out = dict.fromkeys(VIOLATION_TYPES, 0.0)
    for rec in recs:
        for v in VIOLATION_TYPES:
            entry = rec.get(v)
            if entry and float(entry.get("count", 0.0)) > 0.0:
                out[v] += 1.0
    out = {k: v / len(recs) for k, v in out.items()}
    out["any"] = sum(
        1.0
        for rec in recs
        if any(float((rec.get(v) or {}).get("count", 0.0)) > 0.0 for v in VIOLATION_TYPES)
    ) / len(recs)
    return out


def failure_class(
    violations: Mapping[str, Mapping[str, float]],
    *,
    success: bool,
    progress: float,
    max_steps_reached: bool,
    partial_progress: float = PARTIAL_PROGRESS,
) -> str:
    """FACT-style failure taxonomy label for the episode (``04`` §4).

    Only meaningful for failed episodes: a successful episode is labelled
    ``"none"`` even if a transient violation fired.
    """
    if success:
        return "none"
    for v, label in FAILURE_PRIORITY:
        entry = violations.get(v)
        if entry and float(entry.get("count", 0.0)) > 0.0:
            return label
    if max_steps_reached:
        return "stop_short" if float(progress) >= partial_progress else "idle_timeout"
    return "none"


@dataclass
class ViolationSummary:
    """Compact per-episode summary used by the rollout log (``09`` §3)."""

    violations: dict[str, dict[str, float]]
    any_violation: bool
    failure_class: str

    def to_json(self) -> dict[str, Any]:
        return {
            "violations": self.violations,
            "any_violation": self.any_violation,
            "failure_class": self.failure_class,
        }


__all__ = [
    "FAILURE_PRIORITY",
    "PARTIAL_PROGRESS",
    "REQUIRED_STEP_INFO",
    "ViolationMonitor",
    "ViolationSummary",
    "any_violation",
    "failure_class",
    "violation_rate",
    "violations_in_window",
]
