"""Dynamics perturbations: friction (mu), payload mass, contact stiffness (``04`` §2).

Three rules this module exists to enforce:

1. **Always read back.** Perturbing a model is not the same as changing the
   physics (MuJoCo may clamp, mix or recompute). Every application returns a
   :class:`DynReadback` with the *effective* values, and a non-nominal spec that
   changes nothing raises instead of silently running a no-op experiment.
2. **Always scale from a nominal baseline.** Applying ``mu_scale=2`` twice must
   not give ``4x``. The first application snapshots the pristine MuJoCo values on
   the target (``_riskproj_dyn_baseline``) and every later application computes
   ``value = baseline * scale``, which makes repeated resets idempotent.
3. **Task-critical geoms are discovered, not assumed** (``04`` §2): they come from
   ``configs/env/task_geoms.yaml``, written by ``scripts/inspect_contacts.py``.

Two targets are supported:
  * MuJoCo-backed objects exposing ``model``/``data`` (searched through common
    gym/LeRobot wrappers);
  * objects implementing ``apply_dynamics(self, spec) -> dict`` (the toy env),
    whose returned mapping must contain an ``effective`` block with the three
    achieved scales and a ``changed`` flag.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from riskproj.types import DynamicsSpec
from riskproj.utils.io import ROOT

#: Where the discovered task-critical geoms/bodies are recorded (``04`` §2).
TASK_GEOMS_PATH = ROOT / "configs" / "env" / "task_geoms.yaml"

#: Relative tolerance for read-back equality; the physics edit must be exact.
DYN_READBACK_TOL = 1e-9

#: Attribute we stash the pristine baseline under, on the target object.
BASELINE_ATTR = "_riskproj_dyn_baseline"


@dataclass
class DynReadback:
    """Effective physics values after applying a :class:`DynamicsSpec`.

    ``mu``/``mass`` are always in the requesting units (multipliers of the
    nominal baseline), so they are directly comparable to the requested scales.
    ``solref`` holds the raw ``(timeconst, dampratio)`` pairs actually in the
    model, which is what a contact read-back can be checked against.
    """

    spec: DynamicsSpec
    mu: dict[str, float] = field(default_factory=dict)
    mass: dict[str, float] = field(default_factory=dict)
    solref: dict[str, tuple[float, float]] = field(default_factory=dict)
    changed: bool = False
    verified: bool = False
    notes: list[str] = field(default_factory=list)
    #: Achieved scales per axis (mean over the touched elements) for G1 logging.
    effective: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "mu": dict(self.mu),
            "mass": dict(self.mass),
            "solref": {k: list(v) for k, v in self.solref.items()},
            "changed": bool(self.changed),
            "verified": bool(self.verified),
            "effective": dict(self.effective),
            "notes": list(self.notes),
        }

    def assert_applied(self) -> DynReadback:
        """Fail loud when a requested perturbation did not take effect."""
        if not self.spec.is_nominal() and not self.changed:
            raise RuntimeError(
                "apply_dynamics requested a non-nominal DynamicsSpec but nothing changed; "
                f"spec={self.spec.as_dict()} notes={self.notes}"
            )
        if not self.verified:
            raise RuntimeError(
                "apply_dynamics read-back does not match the requested scales: "
                f"requested={self.spec.as_dict()} effective={self.effective} notes={self.notes}"
            )
        return self


# --------------------------------------------------------------------------
# task-critical geometry discovery
# --------------------------------------------------------------------------


def load_task_geoms(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``configs/env/task_geoms.yaml``.

    Accepted schema (either/or, both are merged)::

        task: peg-insert-side-v3
        geoms: [grip_pad_l, grip_pad_r, peg, hole_wall, table]   # flat list
        groups:                                                  # or grouped
          gripper: [grip_pad_l, grip_pad_r]
          object: [peg]
          fixture: [hole_wall]
          table: [table]
        bodies: [peg]        # manipulated bodies receiving the mass scale

    Missing file is an error: guessing geom names is exactly what ``04`` §2
    forbids.
    """
    p = Path(path) if path is not None else TASK_GEOMS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Task-critical geoms must be *discovered*, not assumed (04 §2): "
            "run `python scripts/inspect_contacts.py --task <task>` on the nominal scripted "
            "expert to write this file, or pass task_geoms explicitly."
        )
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{p}: top level must be a mapping")
    geoms: list[str] = [str(g) for g in (data.get("geoms") or [])]
    groups = data.get("groups") or {}
    if groups:
        if not isinstance(groups, Mapping):
            raise TypeError(f"{p}: 'groups' must be a mapping of name -> [geom, ...]")
        for name, members in groups.items():
            if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
                raise TypeError(f"{p}: groups[{name!r}] must be a list of geom names")
            geoms.extend(str(m) for m in members)
    bodies = [str(b) for b in (data.get("bodies") or [])]
    if not geoms:
        raise ValueError(f"{p}: no geoms listed (need 'geoms' or 'groups')")
    return {
        "task": data.get("task"),
        "geoms": sorted(set(geoms)),
        "bodies": sorted(set(bodies)),
        "groups": groups,
    }


def task_critical_geoms(target: Any = None, *, path: str | Path | None = None) -> dict[str, Any]:
    """Task-critical geoms/bodies for ``target`` (loaded from the config file)."""
    spec = load_task_geoms(path)
    if target is not None and spec.get("task"):
        name = getattr(target, "task_name", None) or getattr(
            getattr(target, "unwrapped", None), "task_name", None
        )
        if name and not _task_names_match(str(name), str(spec["task"])):
            raise ValueError(
                f"task_geoms.yaml was written for task {spec['task']!r} but the env is {name!r}"
            )
    return spec


#: Tokens that carry no task identity in a Meta-World *class* name.
_TASK_NAME_NOISE = frozenset({"sawyer", "env", "v2", "v3", "mt1"})


def _task_names_match(env_name: str, config_task: str) -> bool:
    """Do an env's self-reported name and a config task id denote the same task?

    A Meta-World env reports its **class** name (``SawyerPegInsertionSideEnvV3``) while
    a config uses the **registered id** (``peg-insert-side-v3``), so exact comparison
    rejects a perfectly good geoms file. Nor does normalizing help: the class says
    "Insertion" and the id says "insert". So the *id's* tokens must all appear in the
    class name — ``assembly-v3`` against the peg env fails on ``assembly``, which is the
    mismatch worth failing on.
    """
    haystack = "".join(ch for ch in env_name.lower() if ch.isalnum())
    tokens = [t for t in re.split(r"[^a-z0-9]+", config_task.lower()) if t]
    tokens = [t for t in tokens if t not in _TASK_NAME_NOISE]
    return bool(tokens) and all(token in haystack for token in tokens)


# --------------------------------------------------------------------------
# MuJoCo handles
# --------------------------------------------------------------------------

_WRAPPER_ATTRS = ("unwrapped", "env", "_env", "wrapped_env")


def _mujoco_handles(target: Any) -> tuple[Any | None, Any | None]:
    """Find the (model, data) pair on ``target`` or one wrapper level deep."""
    seen: list[Any] = []
    queue: list[Any] = [target]
    while queue:
        obj = queue.pop(0)
        if obj is None or any(obj is s for s in seen):
            continue
        seen.append(obj)
        model = getattr(obj, "model", None)
        data = getattr(obj, "data", None)
        if model is not None and data is not None:
            return model, data
        for attr in _WRAPPER_ATTRS:
            child = getattr(obj, attr, None)
            if child is not None:
                queue.append(child)
    return None, None


#: Token prefix for elements the model does not name (``geom39``).
UNNAMED_PREFIX = "geom"


def _unnamed_index(token: str, *, model: Any, mujoco: Any, obj: Any) -> int | None:
    """Resolve a ``geom<id>`` token, or ``None`` if it is not one.

    Meta-World adds the peg aperture's collision boxes **unnamed**, so they cannot be
    addressed by name at all; ``scripts/inspect_contacts.py`` emits ``geom<id>`` for
    them and this is the only place that convention is interpreted. It is validated
    twice (in range, and *actually* unnamed) so a typo is an error rather than a
    silent no-op on some other geom.
    """
    if not token.startswith(UNNAMED_PREFIX):
        return None
    suffix = token[len(UNNAMED_PREFIX) :]
    if not suffix.isdigit():
        return None
    index = int(suffix)
    count = int(getattr(model, "ngeom", 0)) if obj == mujoco.mjtObj.mjOBJ_GEOM else int(
        getattr(model, "nbody", 0)
    )
    if not 0 <= index < count:
        raise KeyError(f"{token!r} is out of range (the model has {count} of them)")
    if mujoco.mj_id2name(model, obj, index) is not None:
        raise KeyError(
            f"{token!r} is a positional reference but that element *is* named; address it by "
            "name so the config survives a model rebuild"
        )
    return index


def _geom_ids(model: Any, mujoco: Any, names: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for name in names:
        token = str(name)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, token)
        if gid < 0:
            gid = _unnamed_index(
                token, model=model, mujoco=mujoco, obj=mujoco.mjtObj.mjOBJ_GEOM
            )
        if gid is None or gid < 0:
            raise KeyError(
                f"geom {name!r} not found in the model; check configs/env/task_geoms.yaml "
                "(unnamed geoms must be written as geom<id>, the token "
                "scripts/inspect_contacts.py prints)"
            )
        ids.append(int(gid))
    return ids


def _body_ids(model: Any, mujoco: Any, names: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for name in names:
        token = str(name)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, token)
        if bid < 0:
            bid = _unnamed_index(token, model=model, mujoco=mujoco, obj=mujoco.mjtObj.mjOBJ_BODY)
        if bid is None or bid < 0:
            raise KeyError(
                f"body {name!r} not found in the model; check configs/env/task_geoms.yaml"
            )
        ids.append(int(bid))
    return ids


def resolve_geom_ids(model: Any, names: Sequence[str]) -> list[int]:
    """Public name -> geom-id resolver (``geom<id>`` included); see :func:`_geom_ids`."""
    import mujoco

    return _geom_ids(model, mujoco, names)


def _baseline(
    target: Any, model: Any, geom_ids: Sequence[int], body_ids: Sequence[int]
) -> dict[str, Any]:
    """Pristine values, snapshotted once per target (idempotent re-application)."""
    existing = getattr(target, BASELINE_ATTR, None)
    if existing is not None:
        if list(existing["geom_ids"]) != list(geom_ids) or list(existing["body_ids"]) != list(
            body_ids
        ):
            raise RuntimeError(
                "dynamics baseline was captured for a different geom/body set; create a fresh env "
                "instead of reusing this one across tasks"
            )
        return existing
    base = {
        "geom_ids": list(geom_ids),
        "body_ids": list(body_ids),
        "mu": np.array([float(model.geom_friction[g, 0]) for g in geom_ids], dtype=np.float64),
        "solref": np.array(
            [list(map(float, model.geom_solref[g])) for g in geom_ids], dtype=np.float64
        ),
        "mass": np.array([float(model.body_mass[b]) for b in body_ids], dtype=np.float64),
        "inertia": np.array(
            [list(map(float, model.body_inertia[b])) for b in body_ids], dtype=np.float64
        ),
    }
    setattr(target, BASELINE_ATTR, base)
    return base


def _apply_mujoco(
    target: Any,
    spec: DynamicsSpec,
    task_geoms: Mapping[str, Any] | None,
    tol: float,
) -> DynReadback:
    import mujoco  # lazy: the simulator extra

    model, data = _mujoco_handles(target)
    if model is None or data is None:  # pragma: no cover - guarded by the caller
        raise TypeError("no MuJoCo model/data found on the target")

    geom_spec = dict(task_geoms) if task_geoms is not None else task_critical_geoms(target)
    geom_names = list(geom_spec.get("geoms") or [])
    body_names = list(geom_spec.get("bodies") or [])
    if not geom_names:
        raise ValueError("task_geoms lists no geoms; run scripts/inspect_contacts.py first")
    if not body_names:
        raise ValueError(
            "task_geoms lists no bodies, but the mass axis scales body_mass/body_inertia of the "
            "manipulated body (04 §2). Add e.g. `bodies: [peg]` to configs/env/task_geoms.yaml."
        )
    geom_ids = _geom_ids(model, mujoco, geom_names)
    body_ids = _body_ids(model, mujoco, body_names)
    base = _baseline(target, model, geom_ids, body_ids)

    timestep = float(model.opt.timestep)
    min_timeconst = 2.0 * timestep  # solref guard from 04 §2 [V2]
    notes: list[str] = []

    # --- friction: geom_friction[:, 0] is the sliding coefficient mu ------
    for i, g in enumerate(geom_ids):
        model.geom_friction[g, 0] = base["mu"][i] * float(spec.mu_scale)

    # --- mass: body_mass and body_inertia together -----------------------
    for i, b in enumerate(body_ids):
        model.body_mass[b] = base["mass"][i] * float(spec.mass_scale)
        model.body_inertia[b] = base["inertia"][i] * float(spec.mass_scale)

    # --- stiffness: timeconst' = timeconst / sqrt(stiff_scale) ------------
    stiff = float(spec.stiff_scale)
    if stiff <= 0:
        raise ValueError(f"stiff_scale must be > 0; got {stiff}")
    clamped = 0
    for i, g in enumerate(geom_ids):
        raw = float(base["solref"][i, 0]) / math.sqrt(stiff)
        if raw < min_timeconst:
            clamped += 1
            raw = min_timeconst
        model.geom_solref[g, 0] = raw
    if clamped:
        notes.append(
            f"timeconst clamped to >= 2*timestep ({min_timeconst:.3e}) on {clamped}/{len(geom_ids)} "
            "geoms: the requested stiff_scale is not reachable at this timestep (04 §2)"
        )
    if float(model.opt.timestep) != timestep:
        model.opt.timestep = timestep

    mujoco.mj_setConst(model, data)

    # --- read back --------------------------------------------------------
    eff_mu = {
        name: float(model.geom_friction[g, 0]) for name, g in zip(geom_names, geom_ids, strict=True)
    }
    eff_mass = {
        name: float(model.body_mass[b]) for name, b in zip(body_names, body_ids, strict=True)
    }
    eff_solref = {
        name: (float(model.geom_solref[g, 0]), float(model.geom_solref[g, 1]))
        for name, g in zip(geom_names, geom_ids, strict=True)
    }

    changed = bool(
        np.any(np.array(list(eff_mu.values())) != base["mu"])
        or np.any(np.array(list(eff_mass.values())) != base["mass"])
        or np.any(np.array([v[0] for v in eff_solref.values()]) != base["solref"][:, 0])
    )

    # Mean over geoms of the *per-geom* achieved scale. Stiffness goes as 1/timeconst**2
    # (04 §2), so the achieved stiff_scale is the mean of (baseline_tc / effective_tc)**2;
    # averaging the timeconsts first would report a scale no geom actually has.
    eff_tc = np.array([v[0] for v in eff_solref.values()], dtype=np.float64)
    effective = {
        "mu_scale": float(np.mean(np.array(list(eff_mu.values())) / base["mu"])),
        "mass_scale": float(np.mean(np.array(list(eff_mass.values())) / base["mass"])),
        "stiff_scale": float(np.mean((base["solref"][:, 0] / eff_tc) ** 2)),
    }
    verified = (
        abs(effective["mu_scale"] - spec.mu_scale) <= max(tol, 1e-6 * abs(spec.mu_scale))
        and abs(effective["mass_scale"] - spec.mass_scale) <= max(tol, 1e-6 * abs(spec.mass_scale))
        and (
            clamped > 0
            or abs(effective["stiff_scale"] - spec.stiff_scale) <= 1e-6 * abs(spec.stiff_scale)
        )
    )
    if clamped:
        notes.append("stiff_scale verification relaxed because the timeconst guard bound")

    return DynReadback(
        spec=spec,
        mu=eff_mu,
        mass=eff_mass,
        solref=eff_solref,
        changed=changed,
        verified=verified,
        notes=notes,
        effective=effective,
    )


def _apply_delegated(target: Any, spec: DynamicsSpec, fn: Any, tol: float) -> DynReadback:
    """Delegate to a target's own ``apply_dynamics`` (toy env).

    The delegated mapping must provide ``effective`` (achieved scales) and
    ``changed``; ``mu``/``mass``/``solref`` are passed through when present.
    """
    raw = fn(spec)
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"{type(target).__name__}.apply_dynamics must return a mapping; got {type(raw)!r}"
        )
    effective = dict(raw.get("effective") or {})
    missing = [k for k in ("mu_scale", "mass_scale", "stiff_scale") if k not in effective]
    if missing:
        raise ValueError(
            f"{type(target).__name__}.apply_dynamics must report achieved scales in 'effective'; "
            f"missing {missing}"
        )
    # ``changed`` means "the physics is no longer nominal", *not* "the values moved
    # since the previous call". A target that stores scales directly (the toy env)
    # is idempotent: applying the same spec twice legitimately reports no delta,
    # while the physics is still perturbed. Guarding on a delta would either raise
    # spuriously on the second reset or, worse, hide a silent no-op. Comparing the
    # achieved scales against nominal is the property the caller actually needs.
    nominal_effective = all(abs(float(effective[k]) - 1.0) <= max(tol, 1e-9) for k in effective)
    changed = (not nominal_effective and not spec.is_nominal()) or bool(raw.get("changed", False))
    verified = all(
        abs(float(effective[k]) - float(getattr(spec, k))) <= max(tol, 1e-9) for k in effective
    )
    solref = {str(k): (float(v[0]), float(v[1])) for k, v in dict(raw.get("solref") or {}).items()}
    return DynReadback(
        spec=spec,
        mu={str(k): float(v) for k, v in dict(raw.get("mu") or {}).items()},
        mass={str(k): float(v) for k, v in dict(raw.get("mass") or {}).items()},
        solref=solref,
        changed=changed,
        verified=verified,
        notes=[str(n) for n in (raw.get("notes") or [])],
        effective={k: float(v) for k, v in effective.items()},
    )


def apply_dynamics(
    target: Any,
    spec: DynamicsSpec,
    *,
    task_geoms: Mapping[str, Any] | None = None,
    tol: float = DYN_READBACK_TOL,
) -> DynReadback:
    """Apply ``spec`` to ``target`` and return the verified read-back (``04`` §2).

    Raises ``TypeError`` if the target is neither MuJoCo-backed nor delegating,
    and ``RuntimeError`` if a non-nominal spec left the physics unchanged.
    """
    model, data = _mujoco_handles(target)
    if model is not None and data is not None:
        readback = _apply_mujoco(target, spec, task_geoms, tol)
    else:
        delegate = getattr(target, "apply_dynamics", None)
        if not callable(delegate):
            raise TypeError(
                f"cannot apply dynamics to {type(target).__name__}: no MuJoCo model/data and no "
                "apply_dynamics(spec) method. Supported: MuJoCo-backed envs and the toy env."
            )
        readback = _apply_delegated(target, spec, delegate, tol)
    return readback.assert_applied()


def read_back_dynamics(target: Any, *, task_geoms: Mapping[str, Any] | None = None) -> DynReadback:
    """Read the current effective dynamics without modifying anything."""
    model, data = _mujoco_handles(target)
    spec = getattr(target, "_riskproj_requested_dynamics", None) or DynamicsSpec()
    if model is None or data is None:
        getter = getattr(target, "effective_dynamics", None)
        if not callable(getter):
            raise TypeError(
                f"{type(target).__name__} exposes neither model/data nor effective_dynamics()"
            )
        raw = dict(getter())
        return DynReadback(
            spec=spec,
            mu={str(k): float(v) for k, v in dict(raw.get("mu") or {}).items()},
            mass={str(k): float(v) for k, v in dict(raw.get("mass") or {}).items()},
            solref={
                str(k): (float(v[0]), float(v[1])) for k, v in dict(raw.get("solref") or {}).items()
            },
            changed=bool(raw.get("changed", False)),
            verified=bool(raw.get("verified", True)),
            notes=[str(n) for n in (raw.get("notes") or [])],
            effective={str(k): float(v) for k, v in dict(raw.get("effective") or {}).items()},
        )

    import mujoco

    geom_spec = dict(task_geoms) if task_geoms is not None else task_critical_geoms(target)
    geom_names = list(geom_spec.get("geoms") or [])
    body_names = list(geom_spec.get("bodies") or [])
    geom_ids = _geom_ids(model, mujoco, geom_names)
    body_ids = _body_ids(model, mujoco, body_names)
    base = _baseline(target, model, geom_ids, body_ids)
    eff_mu = np.array([float(model.geom_friction[g, 0]) for g in geom_ids])
    eff_mass = np.array([float(model.body_mass[b]) for b in body_ids])
    eff_tc = np.array([float(model.geom_solref[g, 0]) for g in geom_ids])
    return DynReadback(
        spec=spec,
        mu={n: float(v) for n, v in zip(geom_names, eff_mu, strict=True)},
        mass={n: float(v) for n, v in zip(body_names, eff_mass, strict=True)},
        solref={
            n: (float(model.geom_solref[g, 0]), float(model.geom_solref[g, 1]))
            for n, g in zip(geom_names, geom_ids, strict=True)
        },
        changed=True,
        verified=True,
        notes=[],
        effective={
            "mu_scale": float(np.mean(eff_mu / base["mu"])),
            "mass_scale": float(np.mean(eff_mass / base["mass"])),
            "stiff_scale": float((base["solref"][:, 0].mean() / eff_tc.mean()) ** 2),
        },
    )


def contact_friction_readback(data: Any, *, max_pairs: int = 64) -> list[dict[str, float]]:
    """Per-contact effective friction, for verifying MuJoCo's geom-pair mixing.

    Contact friction is a mixing rule over two geoms (max by default [R]), so the
    geom-level value is not the friction a contact actually experiences.
    """
    out: list[dict[str, float]] = []
    for i in range(int(data.ncon)):
        if len(out) >= max_pairs:
            break
        c = data.contact[i]
        out.append(
            {
                "geom1": float(c.geom1),
                "geom2": float(c.geom2),
                "dist": float(c.dist),
                "friction_slide": float(c.friction[0]),
                "solref_timeconst": float(c.solref[0]),
                "solref_dampratio": float(c.solref[1]),
            }
        )
    return out


__all__ = [
    "BASELINE_ATTR",
    "DYN_READBACK_TOL",
    "TASK_GEOMS_PATH",
    "DynReadback",
    "apply_dynamics",
    "contact_friction_readback",
    "load_task_geoms",
    "read_back_dynamics",
    "task_critical_geoms",
]
