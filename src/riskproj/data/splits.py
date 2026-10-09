"""Cell grid and split roles (``agents/04_env_dynamics_data.md`` §3).

The roles are **derived from the sign rules**, not read from a table: a hand-typed
list of 35 cells is exactly the kind of artifact that silently drifts from the
protocol. Only the two compound extrapolation cells are a configuration choice,
because the rules do not determine them.

Designated grid
---------------
Each axis has five levels, but the *designed* grid is:

* 27 in-range cells (``L``/``M``/``H`` on each axis),
* 6 single-axis extrapolation cells (one axis at ``xL``/``xH``, the others ``M``),
* 2 compound extrapolation cells from ``SplitsCfg.range_compound``.

That is 35 cells, and the derivation partitions them exactly 7 / 6 / 14 / 8 as
``04`` §3 requires. Any other combination (for example ``xL-xL-M``) is *not*
part of the design; :func:`role_of_cell` raises rather than inventing a role, so
a typo cannot quietly create a 36th cell.

Sign convention: ``LO = -1``, ``HI = +1`` per axis (``04`` §3). VAL takes the
two-axis cells whose two non-MID sign products are ``+1`` (``L-L``, ``H-H``);
TEST-combo takes sign product ``-1`` plus every three-axis cell.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from riskproj.types import (
    AXES,
    EXT_LEVELS,
    CellId,
    DynamicsRanges,
    DynamicsSpec,
    cell_signs,
    make_cell_id,
    n_non_mid,
    parse_cell_id,
)
from riskproj.utils.config import SplitsCfg

ROLE_TRAIN = "train"
ROLE_VAL = "val"
ROLE_TEST_COMBO = "test_combo"
ROLE_TEST_RANGE = "test_range"
ROLES: tuple[str, ...] = (ROLE_TRAIN, ROLE_VAL, ROLE_TEST_COMBO, ROLE_TEST_RANGE)

#: Expected cell count per role — the protocol numbers from ``04`` §3.
EXPECTED_COUNTS: dict[str, int] = {
    ROLE_TRAIN: 7,
    ROLE_VAL: 6,
    ROLE_TEST_COMBO: 14,
    ROLE_TEST_RANGE: 8,
}

#: Roles whose cells may be used for critic/policy training (``06`` §2).
TRAINABLE_ROLES: tuple[str, ...] = (ROLE_TRAIN,)
#: Roles allowed for tuning (tau/lambda/eta/K/site selection) — VAL only.
TUNABLE_ROLES: tuple[str, ...] = (ROLE_VAL,)
#: Roles that must never be touched before the pre-registration tag is cut.
SEALED_ROLES: tuple[str, ...] = (ROLE_TEST_COMBO, ROLE_TEST_RANGE)


def _cfg(cfg: SplitsCfg | None) -> SplitsCfg:
    return cfg if cfg is not None else SplitsCfg()


# --------------------------------------------------------------------------
# the designated grid
# --------------------------------------------------------------------------


def in_range_cells() -> list[CellId]:
    """The 27 ``L``/``M``/``H`` cells."""
    return [
        make_cell_id(m, d, s)
        for m in ("L", "M", "H")
        for d in ("L", "M", "H")
        for s in ("L", "M", "H")
    ]


def single_extrapolation_cells() -> list[CellId]:
    """The 6 cells with exactly one axis at an extrapolation level, others ``M``."""
    out: list[CellId] = []
    for axis_idx, _axis in enumerate(AXES):
        for level in EXT_LEVELS:
            levels = ["M", "M", "M"]
            levels[axis_idx] = level
            out.append(make_cell_id(*levels))  # type: ignore[arg-type]
    return out


def designated_cells(cfg: SplitsCfg | None = None) -> list[CellId]:
    """All 35 cells the protocol defines, in role order."""
    return in_range_cells() + single_extrapolation_cells() + list(_cfg(cfg).range_compound)


def is_designated(cell: CellId, cfg: SplitsCfg | None = None) -> bool:
    return cell in set(designated_cells(cfg))


# --------------------------------------------------------------------------
# role derivation
# --------------------------------------------------------------------------


def role_of_cell(cell: CellId, cfg: SplitsCfg | None = None) -> str:
    """Derive the split role of one cell, or raise if the cell is not designated.

    Raises:
        ValueError: malformed cell id, or a cell outside the 35-cell design.
    """
    cfg = _cfg(cfg)
    levels = parse_cell_id(cell)  # raises on malformed ids
    ext_axes = [ax for ax, lv in levels.items() if lv in EXT_LEVELS]

    if ext_axes:
        if len(ext_axes) == 1:
            others = [ax for ax in AXES if ax not in ext_axes]
            if all(levels[ax] == "M" for ax in others):
                return ROLE_TEST_RANGE
            raise ValueError(
                f"{cell}: an extrapolation level must be the only non-MID axis of its cell "
                f"(others are {[levels[ax] for ax in others]}); designated extrapolation cells "
                f"are {single_extrapolation_cells()} plus {list(cfg.range_compound)}"
            )
        if len(ext_axes) == 2 and cell in cfg.range_compound:
            return ROLE_TEST_RANGE
        raise ValueError(
            f"{cell}: multi-axis extrapolation cells are limited to the two compound cells "
            f"declared in SplitsCfg.range_compound ({list(cfg.range_compound)}); got {cell}"
        )

    n = n_non_mid(cell)
    if n <= 1:
        return ROLE_TRAIN
    signs = [cell_signs(cell)[ax] for ax in AXES if levels[ax] != "M"]
    if n == 2:
        return ROLE_VAL if signs[0] * signs[1] > 0 else ROLE_TEST_COMBO
    return ROLE_TEST_COMBO


def cells_for_role(role: str, cfg: SplitsCfg | None = None) -> list[CellId]:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    return [c for c in designated_cells(cfg) if role_of_cell(c, cfg) == role]


def split_table(cfg: SplitsCfg | None = None) -> dict[str, list[CellId]]:
    """``{role: [cells]}`` — the frozen protocol view used by every phase."""
    return {role: cells_for_role(role, cfg) for role in ROLES}


def cell_role_index(cfg: SplitsCfg | None = None) -> dict[CellId, str]:
    cfg = _cfg(cfg)
    return {cell: role_of_cell(cell, cfg) for cell in designated_cells(cfg)}


def assert_splits_valid(cfg: SplitsCfg | None = None) -> None:
    """Fail loud if the derived roles do not match the protocol (``04`` §3).

    Raises:
        ValueError: wrong per-role counts, overlapping roles, or a designated cell
            that the rules leave unassigned.
    """
    cfg = _cfg(cfg)
    table = split_table(cfg)
    problems: list[str] = []
    for role, expected in EXPECTED_COUNTS.items():
        got = len(table[role])
        if got != expected:
            problems.append(f"{role}: expected {expected} cells, derived {got} ({table[role]})")

    seen: dict[CellId, str] = {}
    for role in ROLES:
        for cell in table[role]:
            if cell in seen:
                problems.append(f"{cell} appears in both {seen[cell]} and {role}")
            seen[cell] = role

    designated = set(designated_cells(cfg))
    unassigned = sorted(designated - set(seen))
    if unassigned:
        problems.append(f"designated but unassigned: {unassigned}")
    extra = sorted(set(seen) - designated)
    if extra:
        problems.append(f"assigned but not designated: {extra}")

    if problems:
        raise ValueError("split rules are inconsistent with 04 §3: " + "; ".join(problems))


def assert_no_leakage(cells: Sequence[CellId], role: str) -> None:
    """Raise if any cell's derived role differs from the role being used.

    Guards the one rule that invalidates the whole study: TEST cells must not be
    used for training or tuning (``06`` §2, ``CLAUDE.md`` #2).
    """
    bad = [(c, role_of_cell(c)) for c in cells if role_of_cell(c) != role]
    if bad:
        raise ValueError(f"leakage risk: cells {bad} are not part of role {role!r}")


def assert_trainable(cells: Sequence[CellId]) -> None:
    """Raise unless every cell is a TRAIN cell (what training data may come from)."""
    assert_no_leakage(cells, ROLE_TRAIN)


def assert_tunable(cells: Sequence[CellId]) -> None:
    """Raise unless every cell is a VAL cell (tuning is VAL-only)."""
    assert_no_leakage(cells, ROLE_VAL)


def summarise_splits(cfg: SplitsCfg | None = None) -> dict[str, Any]:
    """Counts + one human-readable line per cell, used by the smoke report and data cards."""
    cfg = _cfg(cfg)
    table = split_table(cfg)
    return {
        "n_cells_total": len(designated_cells(cfg)),
        "counts": {role: len(cells) for role, cells in table.items()},
        "expected_counts": dict(EXPECTED_COUNTS),
        "roles": {role: list(cells) for role, cells in table.items()},
        "lines": [f"{role}: " + ", ".join(cells) for role, cells in table.items()],
        "sign_balance": sign_balance_report(cfg),
        "range_compound": list(cfg.range_compound),
    }


def sign_balance_report(cfg: SplitsCfg | None = None) -> dict[str, Any]:
    """Per-role +/- balance on each axis (G1 check: no role is systematically easier).

    Returns, per role: the count of cells with each axis at LO vs HI and the mean
    sign, plus ``balanced`` = whether every axis has both signs represented
    (or no sign at all, as for the all-MID TRAIN cell).
    """
    cfg = _cfg(cfg)
    out: dict[str, Any] = {}
    for role in ROLES:
        cells = cells_for_role(role, cfg)
        per_axis: dict[str, Any] = {}
        balanced = True
        for axis in AXES:
            vals = [cell_signs(c)[axis] for c in cells]
            n_lo = sum(1 for v in vals if v < 0)
            n_hi = sum(1 for v in vals if v > 0)
            n_mid = sum(1 for v in vals if v == 0)
            # both signs present, or only MID: balanced. One sign alone is not.
            axis_balanced = bool(n_lo and n_hi) or (n_lo == 0 and n_hi == 0)
            balanced = balanced and axis_balanced
            per_axis[axis] = {
                "n_lo": n_lo,
                "n_hi": n_hi,
                "n_mid": n_mid,
                "mean_sign": float(np.mean(vals)) if vals else 0.0,
                "balanced": axis_balanced,
            }
        out[role] = {"n_cells": len(cells), "per_axis": per_axis, "balanced": balanced}
    return out


# --------------------------------------------------------------------------
# seeding / dynamics sampling
# --------------------------------------------------------------------------


def sample_dynamics_for_cell(cell: CellId, ranges: DynamicsRanges, seed: int) -> DynamicsSpec:
    """Draw one episode's dynamics inside the cell's level intervals.

    Seeded, so a cell's dynamics are reproducible from ``(cell, seed)`` alone.
    """
    role_of_cell(cell)  # validates the cell against the design
    return ranges.sample(cell, np.random.default_rng(seed))


def episode_seeds(cell: CellId, *, base_seed: int, n: int) -> list[int]:
    """Deterministic episode seeds for a cell, so replays and CRN pairing agree.

    Seeded through sha256 rather than ``hash()``: Python randomises string hashing
    per process, which would silently give a *different* dataset on every run.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    key = f"episode_seeds|{cell}|{int(base_seed)}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]


__all__ = [
    "EXPECTED_COUNTS",
    "ROLES",
    "ROLE_TEST_COMBO",
    "ROLE_TEST_RANGE",
    "ROLE_TRAIN",
    "ROLE_VAL",
    "SEALED_ROLES",
    "TRAINABLE_ROLES",
    "TUNABLE_ROLES",
    "assert_no_leakage",
    "assert_splits_valid",
    "assert_trainable",
    "assert_tunable",
    "cell_role_index",
    "cells_for_role",
    "designated_cells",
    "episode_seeds",
    "in_range_cells",
    "is_designated",
    "role_of_cell",
    "sample_dynamics_for_cell",
    "sign_balance_report",
    "single_extrapolation_cells",
    "split_table",
    "summarise_splits",
]
