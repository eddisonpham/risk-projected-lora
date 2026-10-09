"""Simulator state save/restore (``04`` §6, smoke test S2).

Counterfactual labels and the E1 paired comparison both require executing an
arbitrary candidate chunk from an *identical* saved state and getting identical
results every time. Two consequences:

* the state must include everything the integrator reads — ``qpos``, ``qvel``,
  ``act`` and **mocap position/quaternion** (``04`` §6) — plus, in ``extra``,
  ``ctrl`` and ``qacc_warmstart``, because MuJoCo's contact warm-start makes the
  first step after a restore differ otherwise;
* the toy env's state is stored in the same container (``kind="toy"``), with a
  flat ``extra`` mapping, so downstream code has one code path.

Nested values are rejected: a flat, array-only container is what makes hashing
and npz round-tripping exact.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from riskproj.envs.dynamics import _mujoco_handles

#: Fields written/read for a MuJoCo state, in this order.
MUJOCO_FIELDS: tuple[str, ...] = ("qpos", "qvel", "act", "mocap_pos", "mocap_quat")

#: Flat auxiliary arrays restored for exactness (warm-start, controls).
MUJOCO_EXTRA_FIELDS: tuple[str, ...] = ("ctrl", "qacc_warmstart")

#: A JSON blob describing the state is stored under this key inside the npz.
META_KEY = "__meta__"


def _flat(value: Any, *, name: str) -> np.ndarray:
    """Coerce to a contiguous ndarray, rejecting nested mappings."""
    if isinstance(value, Mapping):
        raise TypeError(
            f"{name}: SimState.extra must be arrays/scalars (nested mappings are not "
            f"round-trippable); got {type(value).__name__}"
        )
    arr = np.asarray(value)
    if arr.dtype == object:
        raise TypeError(f"{name}: object dtype is not supported in SimState")
    return np.ascontiguousarray(arr)


@dataclass
class SimState:
    """A complete, restorable simulator state."""

    qpos: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    qvel: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    act: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    mocap_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    mocap_quat: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float64))
    time: float = 0.0
    #: ``"mujoco"`` or ``"toy"``; restore dispatches on it.
    kind: str = "mujoco"
    #: Flat auxiliary arrays (MuJoCo: ctrl/qacc_warmstart; toy: its whole state).
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    # -- conversion -------------------------------------------------------
    def to_dict(self) -> dict[str, np.ndarray | float | str]:
        out: dict[str, Any] = {f: getattr(self, f) for f in MUJOCO_FIELDS}
        out["time"] = float(self.time)
        out["kind"] = self.kind
        for k, v in self.extra.items():
            out[f"extra/{k}"] = v
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimState:
        kwargs: dict[str, Any] = {}
        for f in MUJOCO_FIELDS:
            kwargs[f] = _flat(data[f], name=f) if f in data else np.zeros(0, dtype=np.float64)
        extras: dict[str, np.ndarray] = {}
        for key, value in data.items():
            if key.startswith("extra/"):
                extras[key.split("/", 1)[1]] = _flat(value, name=key)
        kind_arr = np.asarray(data.get("kind", "mujoco"))
        kind = str(kind_arr.reshape(-1)[0]) if kind_arr.size else "mujoco"
        time_arr = np.asarray(data.get("time", 0.0))
        return cls(
            time=float(time_arr.reshape(-1)[0]) if time_arr.size else 0.0,
            kind=kind,
            extra=extras,
            **kwargs,
        )

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {f: np.asarray(getattr(self, f)) for f in MUJOCO_FIELDS}
        payload["time"] = np.asarray(float(self.time))
        payload["kind"] = np.asarray(self.kind)
        for k, v in self.extra.items():
            payload[f"extra/{k}"] = np.asarray(v)
        meta = {
            "kind": self.kind,
            "extra_keys": sorted(self.extra),
            "dtypes": {k: str(np.asarray(v).dtype) for k, v in payload.items()},
        }
        payload[META_KEY] = np.asarray(json.dumps(meta, sort_keys=True))
        np.savez_compressed(p, **payload)
        return p

    @classmethod
    def load(cls, path: str | Path, *, allow_pickle: bool = False) -> SimState:
        with np.load(Path(path), allow_pickle=allow_pickle) as npz:
            if META_KEY not in npz:
                raise ValueError(f"{path}: missing {META_KEY}; not a SimState file")
            meta = json.loads(str(npz[META_KEY].item()))
            data: dict[str, Any] = {k: npz[k] for k in npz.files if k != META_KEY}
        state = cls.from_dict(data)
        if state.kind != meta.get("kind"):
            raise ValueError(f"{path}: kind mismatch ({state.kind!r} vs meta {meta.get('kind')!r})")
        return state

    # -- identity ---------------------------------------------------------
    def hash(self) -> str:
        """sha256 over every array and the clock: the reproducibility check for S2."""
        h = hashlib.sha256()
        payload = self.to_dict()
        for key in sorted(payload):
            arr = np.asarray(payload[key])
            h.update(key.encode())
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def nbytes(self) -> int:
        return int(sum(np.asarray(v).nbytes for v in self.to_dict().values()))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SimState(kind={self.kind}, qpos={self.qpos.shape}, qvel={self.qvel.shape}, "
            f"act={self.act.shape}, mocap_pos={self.mocap_pos.shape}, time={self.time:.4f}, "
            f"extra={sorted(self.extra)})"
        )


# --------------------------------------------------------------------------
# save / restore
# --------------------------------------------------------------------------


def save_state(env: Any) -> SimState:
    """Capture the full simulator state of ``env`` (MuJoCo or toy)."""
    model, data = _mujoco_handles(env)
    if model is not None and data is not None:
        extra = {}
        for name in MUJOCO_EXTRA_FIELDS:
            value = getattr(data, name, None)
            if value is not None and np.asarray(value).size:
                extra[name] = np.array(value)  # copy: the live buffer keeps mutating
        return SimState(
            qpos=np.array(data.qpos),
            qvel=np.array(data.qvel),
            act=np.array(data.act),
            mocap_pos=np.array(data.mocap_pos),
            mocap_quat=np.array(data.mocap_quat),
            time=float(data.time),
            kind="mujoco",
            extra=extra,
        )
    getter = getattr(env, "get_state", None)
    if not callable(getter):
        raise TypeError(
            f"cannot save state of {type(env).__name__}: no MuJoCo model/data and no get_state()"
        )
    raw = dict(getter())
    if not raw:
        raise ValueError(f"{type(env).__name__}.get_state() returned an empty state")
    extra = {str(k): _flat(v, name=f"get_state()[{k!r}]") for k, v in raw.items()}
    return SimState(kind="toy", extra=extra, time=float(extra.get("time", np.asarray(0.0))))


def restore_state(env: Any, state: SimState) -> dict[str, Any] | None:
    """Restore ``state`` into ``env``; raises on shape/kind mismatch (fail loud).

    Returns the refreshed observation when the target can produce one (a
    delegating env whose ``set_state`` returns its observation, or one that
    exposes ``get_obs``/``_get_obs``/``observe``/``get_observation``), otherwise
    ``None``. Counterfactual labelling must build its context from the *restored*
    state (``04`` §6), so this return value is the contract that keeps callers
    from labelling a stale observation.
    """
    model, data = _mujoco_handles(env)
    if state.kind == "mujoco":
        if model is None or data is None:
            raise TypeError(f"cannot restore a MuJoCo state into {type(env).__name__}")
        import mujoco

        for name in MUJOCO_FIELDS:
            value = np.asarray(getattr(state, name))
            target = getattr(data, name)
            if value.shape != target.shape:
                raise ValueError(
                    f"{name}: shape mismatch between saved state {value.shape} and env {target.shape}"
                )
            target[:] = value
        data.time = float(state.time)
        for name, value in state.extra.items():
            target = getattr(data, name, None)
            if target is None:
                continue
            value = np.asarray(value)
            if value.shape == np.asarray(target).shape:
                target[:] = value
        mujoco.mj_forward(model, data)
        return _refreshed_obs(env)

    setter = getattr(env, "set_state", None)
    if not callable(setter):
        raise TypeError(f"cannot restore a toy state into {type(env).__name__}: no set_state()")
    result = setter({k: np.array(v) for k, v in state.extra.items()})
    if isinstance(result, dict):
        return result
    return _refreshed_obs(env)


def _refreshed_obs(env: Any) -> dict[str, Any] | None:
    """Observation getters, tried on the env and its unwrapped form."""
    for target in (env, getattr(env, "unwrapped", None)):
        if target is None:
            continue
        for name in ("get_obs", "_get_obs", "observe", "get_observation"):
            getter = getattr(target, name, None)
            if callable(getter):
                obs = getter()
                if isinstance(obs, dict):
                    return obs
    return None


def state_hash(env: Any) -> str:
    return save_state(env).hash()


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


def rollout_from_state(
    env: Any,
    state: SimState,
    actions: Sequence[Any],
    *,
    observe_hook: Any | None = None,
) -> list[dict[str, Any]]:
    """Execute ``actions`` open-loop from ``state``; return per-step records.

    This is the primitive behind counterfactual labels and the E1 paired test:
    the same saved state with the original chunk and with the corrected chunk.
    ``observe_hook(step, obs, reward, terminated, truncated, info)`` is called
    after every step when provided (e.g. to feed a :class:`ViolationMonitor`).
    """
    restore_state(env, state)
    records: list[dict[str, Any]] = []
    for t, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        record = {
            "step": t,
            "obs": obs,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        }
        records.append(record)
        if observe_hook is not None:
            observe_hook(t, obs, float(reward), bool(terminated), bool(truncated), info)
        if terminated or truncated:
            break
    return records


def states_equal(a: SimState, b: SimState, *, tol: float = 0.0) -> bool:
    """Compare two states; ``tol=0`` demands bitwise equality (S2)."""
    if a.kind != b.kind:
        return False
    if set(a.extra) != set(b.extra):
        return False
    if tol == 0.0:
        return a.hash() == b.hash()
    for name in MUJOCO_FIELDS:
        if not np.allclose(getattr(a, name), getattr(b, name), atol=tol, rtol=0.0):
            return False
    if abs(a.time - b.time) > tol:
        return False
    return all(
        np.allclose(np.asarray(a.extra[k]), np.asarray(b.extra[k]), atol=tol, rtol=0.0)
        for k in a.extra
    )


def as_flat_state(mapping: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Public helper for toy-env state dicts (flat, array-valued)."""
    return {str(k): _flat(v, name=str(k)) for k, v in mapping.items()}


__all__ = [
    "META_KEY",
    "MUJOCO_EXTRA_FIELDS",
    "MUJOCO_FIELDS",
    "SimState",
    "as_flat_state",
    "restore_state",
    "rollout_from_state",
    "save_state",
    "state_hash",
    "states_equal",
]
