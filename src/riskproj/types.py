"""Shared vocabulary for riskproj.

Everything in this module is imported by more than one subsystem, so it lives at
the top of the dependency order (``utils <- envs/data <- policies/critic <-
projection <- methods <- eval <- scripts``) and must not import from any of them.

Conventions fixed here (single source of truth):
  * dynamics axes are ``mu`` (friction), ``mass`` (payload), ``stiff`` (contact
    stiffness); a *level* is one of ``xL, L, M, H, xH`` where ``x*`` are the
    extrapolation levels used only by TEST-range cells;
  * a *cell id* is ``"<mu>-<mass>-<stiff>"``, e.g. ``"H-M-xL"``;
  * risk thresholds live on the probability scale unless ``ProjCfg.space`` says
    otherwise;
  * flow-matching convention: ``tau=1`` is noise, ``a_tau = (1-tau) a0 + tau eps``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import torch
from torch import Tensor

# --------------------------------------------------------------------------
# dynamics axes, levels, cells
# --------------------------------------------------------------------------

AXES: tuple[str, str, str] = ("mu", "mass", "stiff")
AXIS_LONG_NAME: dict[str, str] = {
    "mu": "sliding friction",
    "mass": "payload mass",
    "stiff": "contact stiffness",
}
#: Ordered low -> high. ``xL``/``xH`` are the extrapolation levels (TEST-range).
LEVELS: tuple[str, ...] = ("xL", "L", "M", "H", "xH")
LEVEL_RANK: dict[str, int] = {lv: i for i, lv in enumerate(LEVELS)}
#: Multiplicative sign used by the split rules (``01`` §3 / ``04`` §3).
LEVEL_SIGN: dict[str, float] = {"xL": -1.0, "L": -1.0, "M": 0.0, "H": 1.0, "xH": 1.0}
EXT_LEVELS: tuple[str, str] = ("xL", "xH")
IN_RANGE_LEVELS: tuple[str, str, str] = ("L", "M", "H")

CellId = str


def make_cell_id(mu_level: str, mass_level: str, stiff_level: str) -> CellId:
    """Canonical cell id, e.g. ``make_cell_id('H', 'M', 'xL') == 'H-M-xL'``."""
    for lv in (mu_level, mass_level, stiff_level):
        if lv not in LEVELS:
            raise ValueError(f"unknown dynamics level {lv!r}; expected one of {LEVELS}")
    return f"{mu_level}-{mass_level}-{stiff_level}"


def parse_cell_id(cell: CellId) -> dict[str, str]:
    """Inverse of :func:`make_cell_id`."""
    parts = cell.split("-")
    if len(parts) != 3 or any(p not in LEVELS for p in parts):
        raise ValueError(f"malformed cell id {cell!r}; expected '<mu>-<mass>-<stiff>'")
    return dict(zip(AXES, parts, strict=True))


def cell_levels(cell: CellId) -> tuple[str, str, str]:
    parsed = parse_cell_id(cell)
    return parsed["mu"], parsed["mass"], parsed["stiff"]


def cell_signs(cell: CellId) -> dict[str, float]:
    return {ax: LEVEL_SIGN[lv] for ax, lv in parse_cell_id(cell).items()}


def is_extrapolation_cell(cell: CellId) -> bool:
    """True if any axis sits at an extrapolation level (TEST-range by construction)."""
    return any(lv in EXT_LEVELS for lv in parse_cell_id(cell).values())


def n_non_mid(cell: CellId) -> int:
    return sum(1 for lv in parse_cell_id(cell).values() if lv != "M")


ALL_CELLS: tuple[CellId, ...] = tuple(
    make_cell_id(m, d, s) for m in LEVELS for d in LEVELS for s in LEVELS
)


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicsSpec:
    """Multiplicative perturbations applied to the simulator at reset time.

    ``1.0`` means nominal on every axis. Values are *scales*, never absolute
    physics values, so the same spec applies to any task.
    """

    mu_scale: float = 1.0
    mass_scale: float = 1.0
    stiff_scale: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "mu_scale": self.mu_scale,
            "mass_scale": self.mass_scale,
            "stiff_scale": self.stiff_scale,
        }

    def scaled(self, **kwargs: float) -> DynamicsSpec:
        base = self.as_dict()
        base.update(kwargs)
        return DynamicsSpec(**base)

    def is_nominal(self) -> bool:
        return all(v == 1.0 for v in self.as_dict().values())


AxisRanges = dict[str, tuple[float, float]]


@dataclass(frozen=True)
class DynamicsRanges:
    """Per-axis, per-level multiplicative ranges; one sample is drawn per episode.

    Values are **placeholders set at G1** from the dynamics-sensitivity sweep
    (``04`` §3). Levels are sampled uniformly inside their interval, so shifts
    are continuous *within* a cell and discrete *between* cells.
    """

    mu: AxisRanges
    mass: AxisRanges
    stiff: AxisRanges

    def for_axis(self, axis: str) -> AxisRanges:
        if axis not in AXES:
            raise ValueError(f"unknown axis {axis!r}")
        return getattr(self, axis)

    def sample(self, cell: CellId, rng: Any) -> DynamicsSpec:
        """Draw one :class:`DynamicsSpec` from this cell's level intervals."""
        levels = parse_cell_id(cell)
        draws: dict[str, float] = {}
        for axis, level in levels.items():
            lo, hi = self.for_axis(axis)[level]
            draws[f"{axis}_scale"] = float(rng.uniform(lo, hi))
        return DynamicsSpec(**draws)

    def midpoint_spec(self, cell: CellId) -> DynamicsSpec:
        """Deterministic representative spec (used for smoke tests / baselines)."""
        draws: dict[str, float] = {}
        for axis, level in parse_cell_id(cell).items():
            lo, hi = self.for_axis(axis)[level]
            draws[f"{axis}_scale"] = 0.5 * (lo + hi)
        return DynamicsSpec(**draws)


# --------------------------------------------------------------------------
# policies / models
# --------------------------------------------------------------------------


@dataclass
class PolicyOut:
    """Output of one policy forward pass, at a chosen site.

    ``chunk`` is always the *terminal action estimate* (``a0``): for regression
    heads that is the forward pass itself, for flow-matching heads it is
    ``x_tau - tau * v`` (``05`` §4). That keeps ``risk_fn(h) = V(c, chunk)``
    valid for both head families.
    """

    chunk: Tensor  # [B, H, A]
    site: str | None = None
    h: Tensor | None = None  # flattened-or-structured site activation used to build `chunk`
    tau: Tensor | None = None  # [B] flow time (1 = noise)
    velocity: Tensor | None = None  # [B, H, A] raw head output when applicable
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chunk.dim() != 3:
            raise ValueError(f"chunk must be [B, H, A]; got {tuple(self.chunk.shape)}")


# --------------------------------------------------------------------------
# violations
# --------------------------------------------------------------------------

ViolationType = Literal["V1", "V2", "V3", "V4", "V5"]
VIOLATION_TYPES: tuple[ViolationType, ...] = ("V1", "V2", "V3", "V4", "V5")
VIOLATION_NAME: dict[str, str] = {
    "V1": "slip",
    "V2": "drop",
    "V3": "over_force",
    "V4": "penetration",
    "V5": "jam",
    "V6": "stop_short",  # episode level only
}
VIOLATION_DESCRIPTION: dict[str, str] = {
    "V1": "grasped-object drift in the gripper frame beyond delta_slip while grasped",
    "V2": "grasp lost: object-gripper distance or height drop beyond delta_drop",
    "V3": "sum of normal contact force on task-critical pairs above F_max",
    "V4": "task-critical contact penetration depth beyond delta_pen",
    "V5": "force above F_jam for W steps with task progress below eps (jam)",
    "V6": "episode ended on timeout with the object only partially placed (stop-short)",
}


@dataclass(frozen=True)
class ViolationSpec:
    """Thresholds for the violation events (``04`` §4).

    Derived from **nominal scripted-expert rollouts only**, then frozen, before
    any learned policy is evaluated. The defaults below are placeholders pending
    G1 and must be replaced by percentiles measured by
    ``scripts/derive_violation_thresholds.py``.
    """

    slip_drift: float = 0.02  # V1, metres, gripper frame
    drop_dist: float = 0.03  # V2, metres
    force_max: float = 60.0  # V3, newtons (sum over pairs)
    penetration: float = 0.005  # V4, metres, absolute (kept stiffness-independent)
    jam_force: float = 40.0  # V5, newtons
    jam_progress_eps: float = 0.002  # V5, metres of progress
    jam_steps: int = 5  # V5, W
    window: int = 10  # steps over which V1/V2 are evaluated
    horizon: int = 10  # H_c: label horizon == exec horizon
    #: Derived-once values recorded for provenance (percentiles, margins, git sha).
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "slip_drift": self.slip_drift,
            "drop_dist": self.drop_dist,
            "force_max": self.force_max,
            "penetration": self.penetration,
            "jam_force": self.jam_force,
            "jam_progress_eps": self.jam_progress_eps,
            "jam_steps": self.jam_steps,
            "window": self.window,
            "horizon": self.horizon,
        }


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

BackwardMode = Literal["ste", "projected", "second_order"]


@dataclass
class ProjectionInfo:
    """Per-decision projection telemetry (``07`` §2, ``09`` §3).

    All tensors are detached and shaped ``[B]``; ``trace[k]`` holds the
    iteration-level record. ``r_after`` is the **nonlinear** risk re-evaluated
    after the step and ``r_pred`` is the first-order prediction, so that their
    difference is the linearization error (E2).
    """

    r_cand: Tensor | None = None
    r_after: Tensor | None = None
    r_pred: Tensor | None = None
    gn2: Tensor | None = None
    active: Tensor | None = None
    usable: Tensor | None = None
    unusable: Tensor | None = None
    k_used: Tensor | None = None
    corr_rel: Tensor | None = None
    ens_std: Tensor | None = None
    fallback_code: Tensor | None = None
    trace: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """Aggregate to scalars for the per-episode log line."""
        out: dict[str, float] = {}
        if self.r_cand is not None:
            out["mean_r_before"] = float(self.r_cand.float().mean())
        if self.r_after is not None:
            out["mean_r_after"] = float(self.r_after.float().mean())
        if self.r_pred is not None:
            out["mean_r_pred"] = float(self.r_pred.float().mean())
        if self.active is not None:
            out["n_dec"] = float(self.active.numel())
        if self.usable is not None:
            out["n_active"] = float(self.usable.sum())
        if self.unusable is not None:
            out["n_unusable"] = float(self.unusable.sum())
        if self.corr_rel is not None and self.corr_rel.numel():
            out["mean_corr_rel"] = float(self.corr_rel.float().mean())
        if self.ens_std is not None and self.ens_std.numel():
            out["mean_ens_std"] = float(self.ens_std.float().mean())
        return out


# --------------------------------------------------------------------------
# fallback ladder (``01`` §3)
# --------------------------------------------------------------------------


class FallbackCode:
    """Codes for the fallback ladder; 0 means 'no fallback taken'."""

    NONE = 0
    F1_BASE_ACTION = 1  # base policy action, re-checked against the critic
    F2_HOLD = 2  # hold the previous commanded action for <= W steps
    F3_TERMINATE = 3  # give up: counted as a failure, never as "safe"

    NAMES: ClassVar[dict[int, str]] = {
        0: "none",
        1: "F1_base_action",
        2: "F2_hold",
        3: "F3_terminate",
    }


# --------------------------------------------------------------------------
# logging schema (``09`` §3)
# --------------------------------------------------------------------------


def empty_violation_counts() -> dict[str, dict[str, float]]:
    return {v: {"first_step": -1.0, "count": 0.0} for v in VIOLATION_TYPES}


@dataclass
class EpisodeRecord:
    """One row of the episode-level JSONL log."""

    run_id: str
    method: str
    train_seed: int
    env_seed: int
    cell_id: CellId
    split: str
    success: bool
    steps: int
    dyn_readback: dict[str, float] = field(default_factory=dict)
    t_success: float | None = None
    violations: dict[str, dict[str, float]] = field(default_factory=empty_violation_counts)
    failure_class: str = "none"
    fallback: dict[str, int] = field(default_factory=lambda: {"F1": 0, "F2": 0, "F3": 0})
    stall_steps: int = 0
    action_norm_mean: float = 0.0
    jerk: float = 0.0
    proj: dict[str, float] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    peak_vram_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class DecisionRecord:
    """One row of the per-decision JSONL log (projection events, ``CLAUDE.md`` #4)."""

    ep_id: str
    step: int
    flow_step: int = -1
    tau: float | None = None
    r_cand: float | None = None
    gn2: float | None = None
    active: bool = False
    usable: bool = False
    k_used: int = 0
    corr_rel: float = 0.0
    r_after: list[float] = field(default_factory=list)
    r_pred: list[float] = field(default_factory=list)
    ens_std: float | None = None
    fallback_code: int = FallbackCode.NONE
    true_violation_next_H: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def as_float(x: Any) -> float:
    """Defensive tensor -> float for logging (keeps the log schema stable."""
    if isinstance(x, Tensor):
        return float(x.detach().float().reshape(-1)[0].item()) if x.numel() else float("nan")
    return float(x)


__all__ = [
    "ALL_CELLS",
    "AXES",
    "AXIS_LONG_NAME",
    "EXT_LEVELS",
    "IN_RANGE_LEVELS",
    "LEVELS",
    "LEVEL_RANK",
    "LEVEL_SIGN",
    "VIOLATION_DESCRIPTION",
    "VIOLATION_NAME",
    "VIOLATION_TYPES",
    "AxisRanges",
    "BackwardMode",
    "CellId",
    "DecisionRecord",
    "DynamicsRanges",
    "DynamicsSpec",
    "EpisodeRecord",
    "FallbackCode",
    "PolicyOut",
    "ProjectionInfo",
    "ViolationSpec",
    "ViolationType",
    "as_float",
    "cell_levels",
    "cell_signs",
    "empty_violation_counts",
    "is_extrapolation_cell",
    "make_cell_id",
    "n_non_mid",
    "parse_cell_id",
    "torch",
]
