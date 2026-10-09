"""Demonstration generation (D-nom for Stage A, D-B for Stage B).

``04`` §7: D-B is the **same dataset for every method**, drawn from TRAIN cells
only, metadata = cell id (never a model input). D-nom is nominal-dynamics expert
data used to fit ``pi_base``.

Env plumbing that both generators need lives here (``reset_env``, ``step_env``,
``expert_action``, ``readback_dict``) so the two files cannot drift apart.

Task contract used here (enforced loudly):
  * ``task.make_env(dyn, seed)`` -> gymnasium-style env with
    ``reset(seed=...) -> (obs, info)`` and ``step(a) -> (obs, reward, terminated,
    truncated, info)``; observations contain the keys in ``obs.TASK_OBS_KEYS``;
  * the env exposes the applied-dynamics read-back as ``env.dyn_readback``;
  * ``task.expert()`` -> callable ``expert(obs: dict) -> np.ndarray``;
  * the env reports episode success as ``info["success"]`` (or ``env.success``).
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from riskproj.data.schema import (
    DATASET_VERSION,
    DemoRecord,
    data_card,
    register_dataset,
    write_data_card,
    write_demos,
)
from riskproj.data.splits import episode_seeds, sample_dynamics_for_cell
from riskproj.protocols import Task
from riskproj.types import CellId, DynamicsRanges, DynamicsSpec
from riskproj.utils.io import ROOT, manifest_hash
from riskproj.utils.registry import register

DEFAULT_MANIFEST = ROOT / "data" / "MANIFEST.json"


# --------------------------------------------------------------------------
# env plumbing (shared with gen_critic_data)
# --------------------------------------------------------------------------


def reset_env(env: Any, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """``reset`` with or without an ``info`` return, normalised to ``(obs, info)``."""
    try:
        out = env.reset(seed=seed)
    except TypeError:  # older gymnasium signature
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    if not isinstance(obs, dict) or "state" not in obs:
        raise TypeError(
            "task env must return a dict observation containing 'state'; got "
            f"{type(obs).__name__} with keys {sorted(obs) if isinstance(obs, dict) else 'n/a'}"
        )
    return obs, dict(info)


def step_env(
    env: Any, action: np.ndarray
) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    """``step`` normalised to the gymnasium 5-tuple."""
    out = env.step(np.asarray(action, dtype=np.float32))
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
    elif isinstance(out, tuple) and len(out) == 4:  # pre-0.26 gym API
        obs, reward, done, info = out
        terminated, truncated = bool(done), False
    else:
        raise TypeError(f"env.step must return a 4- or 5-tuple; got {type(out).__name__}")
    return obs, float(reward), bool(terminated), bool(truncated), dict(info)


def expert_action(expert: Any, obs: dict[str, Any], action_dim: int | None = None) -> np.ndarray:
    """Call a task expert. The contract is ``expert(obs: dict) -> [A]``."""
    if not callable(expert):
        raise TypeError(
            "Task.expert() must return a callable taking the observation dict; "
            f"got {type(expert).__name__}"
        )
    try:
        action = expert(obs)
    except (TypeError, KeyError) as exc:  # a clear contract error beats a wrong number
        raise TypeError(
            "task expert must accept the observation dict (keys: "
            f"{sorted(obs)}); calling it raised {type(exc).__name__}: {exc}"
        ) from exc
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_dim is not None and action.shape[0] != action_dim:
        raise ValueError(
            f"expert returned action of width {action.shape[0]}, expected {action_dim}"
        )
    return action


def readback_dict(env: Any) -> dict[str, Any]:
    """Effective physics after ``apply_dynamics`` (``04`` §2: always read back)."""
    rb = getattr(env, "dyn_readback", None)
    unwrapped = getattr(env, "unwrapped", None)
    if rb is None and unwrapped is not None:
        rb = getattr(unwrapped, "dyn_readback", None)
    if rb is None:
        return {}
    if dataclasses.is_dataclass(rb) and not isinstance(rb, type):
        return dataclasses.asdict(rb)
    if isinstance(rb, dict):
        return dict(rb)
    return {"value": str(rb)}


def assert_dynamics_applied(spec: DynamicsSpec, readback: dict[str, Any], cell: CellId) -> None:
    """Fail loud if a non-nominal cell was rolled out at nominal physics.

    This is the check that catches the classic silent no-op: ``apply_dynamics``
    raising nothing while the model is unchanged, which would make every
    "robustness" number meaningless.
    """
    if spec.is_nominal():
        return
    if not readback:
        raise ValueError(
            f"{cell}: no dynamics read-back available. The env must expose the effective "
            "physics as env.dyn_readback after apply_dynamics (04 §2); without it a "
            "non-nominal rollout cannot be distinguished from a nominal one."
        )
    scales = _effective_scales(readback)
    if scales is None:
        raise ValueError(
            f"{cell}: dyn read-back {sorted(readback)} is present but has no recognisable "
            "effective scales; expose 'mu', 'mass' and 'solref' in env.dyn_readback (04 §2)"
        )
    # Checked **per axis**: a read-back in which only the friction edit landed is a
    # partial silent failure, and the aggregate "everything is still nominal" test
    # would let it through because the other two axes are perturbed.
    requested = {axis: float(getattr(spec, f"{axis}_scale")) for axis in ("mu", "mass", "stiff")}
    not_applied = [
        axis
        for axis, want in requested.items()
        if abs(want - 1.0) > 1e-9
        and scales.get(axis) is not None
        and abs(float(scales[axis]) - 1.0) < 1e-6
    ]
    if not_applied:
        raise RuntimeError(
            f"{cell}: dynamics were not applied on {not_applied} — read-back effective scales "
            f"are nominal there ({scales}) while the requested spec was {spec.as_dict()}"
        )


def _effective_scales(readback: dict[str, Any]) -> dict[str, float | None] | None:
    """Pull per-axis effective scales out of an arbitrary read-back structure."""
    out: dict[str, float | None] = {}
    found = False
    for axis in ("mu", "mass", "stiff"):
        node = readback.get(axis)
        if isinstance(node, dict):
            for key in ("scale", "ratio", "mean", "value"):
                if key in node and isinstance(node[key], (int, float)):
                    out[axis] = float(node[key])
                    found = True
                    break
            else:
                nums = [float(v) for v in node.values() if isinstance(v, (int, float))]
                if nums:
                    out[axis] = float(np.mean(nums))
                    found = True
        elif isinstance(node, (int, float)):
            out[axis] = float(node)
            found = True
    for key in ("scales", "effective", "applied"):
        node = readback.get(key)
        if isinstance(node, dict):
            inner = _effective_scales(node)
            if inner is not None:
                return inner
    return out if found else None


def episode_success(info: dict[str, Any], env: Any, *, saw_success_key: list[bool]) -> bool:
    """Read the task success signal, recording that a signal was present at all."""
    for key in ("success", "is_success"):
        if key in info:
            saw_success_key[0] = True
            return bool(info[key])
    attr = getattr(env, "success", None)
    if attr is not None:
        saw_success_key[0] = True
        return bool(attr)
    unwrapped = getattr(env, "unwrapped", None)
    attr = getattr(unwrapped, "success", None) if unwrapped is not None else None
    if attr is not None:
        saw_success_key[0] = True
        return bool(attr)
    return False


def task_instruction(task: Task, env: Any) -> str:
    for source in (env, getattr(env, "unwrapped", None), task):
        if source is None:
            continue
        value = getattr(source, "instruction", None)
        if isinstance(value, str):
            return value
    return ""


def close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


@register("dataset", "demos")
def generate_demos(
    task: Task,
    *,
    cells: Sequence[CellId],
    ranges: DynamicsRanges,
    episodes_per_cell: int,
    base_seed: int,
    out_dir: str | Path,
    max_steps: int,
    config_hash: str,
    generator: str = "gen_demos",
    split: str | None = None,
    keep_failures: bool = False,
    record_images: bool = False,
    record_progress: bool = True,
    verify_dynamics: bool = True,
    action_dim: int | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    data_root: str | Path = ROOT,
    max_episode_seconds: float | None = None,
) -> dict[str, Any]:
    """Roll out the scripted expert on the given cells and write D-B/D-nom shards.

    Returns a summary dict with ``n_records``, ``per_cell``, ``success_rate``,
    ``paths`` and ``manifest_hash`` — the record the run directory keeps.
    """
    t_start = time.perf_counter()
    if episodes_per_cell < 1:
        raise ValueError("episodes_per_cell must be >= 1")
    out_dir = Path(out_dir)
    records: list[DemoRecord] = []
    per_cell: dict[str, dict[str, float]] = {}
    readbacks: dict[str, list[dict[str, Any]]] = {}
    expert = task.expert()
    saw_success_key: list[bool] = [False]

    for cell in cells:
        seeds = episode_seeds(cell, base_seed=base_seed, n=episodes_per_cell)
        n_ok = 0
        n_kept = 0
        for ep_idx, seed in enumerate(seeds):
            dyn = sample_dynamics_for_cell(cell, ranges, seed)
            env = task.make_env(dyn, seed)
            ep_start = time.perf_counter()
            try:
                obs, info = reset_env(env, seed)
                rb = readback_dict(env)
                if verify_dynamics:
                    assert_dynamics_applied(dyn, rb, cell)
                readbacks.setdefault(cell, []).append(rb)
                instruction = task_instruction(task, env)
                actions: list[np.ndarray] = []
                proprios: list[np.ndarray] = []
                progresses: list[float] = []
                images: list[np.ndarray] = []
                success = False
                terminated = truncated = False
                step = 0
                for _step in range(max_steps):
                    action = expert_action(expert, obs, action_dim)
                    obs, _reward, terminated, truncated, info = step_env(env, action)
                    actions.append(action)
                    proprios.append(np.asarray(obs["state"], dtype=np.float32).reshape(-1))
                    if record_progress:
                        progresses.append(float(info.get("progress", 0.0)))
                    if record_images and obs.get("images") is not None:
                        images.append(
                            np.asarray(
                                next(iter(obs["images"].values()))
                                if isinstance(obs["images"], dict)
                                else obs["images"]
                            )
                        )
                    success = episode_success(info, env, saw_success_key=saw_success_key)
                    if (
                        max_episode_seconds is not None
                        and time.perf_counter() - ep_start > max_episode_seconds
                    ):
                        break
                    if terminated or truncated:
                        break
                n_ok += int(success)
                if success or keep_failures:
                    n_kept += 1
                    records.append(
                        DemoRecord(
                            ep_id=f"{cell}-s{seed}-e{ep_idx}",
                            cell_id=cell,
                            split=split or "train",
                            seed=int(seed),
                            task=task.name,
                            success=bool(success),
                            steps=step + 1,
                            actions=np.stack(actions),
                            proprio=np.stack(proprios),
                            instruction=instruction,
                            images=np.stack(images) if images else None,
                            progress=np.array(progresses, dtype=np.float32) if progresses else None,
                            dyn=dyn.as_dict(),
                            extra={"dyn_readback": rb},
                        )
                    )
            finally:
                close_env(env)
        per_cell[cell] = {
            "episodes": float(len(seeds)),
            "success_rate": n_ok / len(seeds),
            "kept": float(n_kept),
        }

    if not saw_success_key[0]:
        raise RuntimeError(
            "no episode exposed a success signal: the env must report info['success'] "
            "(or expose env.success) — without it, D-B and every success metric are undefined"
        )
    if not records:
        raise RuntimeError(
            f"no demonstrations were recorded for cells {list(cells)}: the scripted expert "
            f"never succeeded (success rates: {per_cell}). Reduce the dynamics range or keep_failures=True."
        )

    shard = "demos"
    npz_path = write_demos(out_dir, records, shard=shard)
    card = data_card(
        records,
        generator=generator,
        config_hash=config_hash,
        extra={
            "ranges_source": "DynamicsRanges",
            "base_seed": int(base_seed),
            "cells": list(cells),
            "episodes_per_cell": int(episodes_per_cell),
            "max_steps": int(max_steps),
            "keep_failures": bool(keep_failures),
            "per_cell": per_cell,
            "split_role": split,
            "dyn_readback": {cell: rbs[:1] for cell, rbs in readbacks.items()},
        },
    )
    write_data_card(out_dir / "DATA_CARD.json", card)
    register_dataset(
        npz_path,
        generator=generator,
        generator_version=DATASET_VERSION,
        config_hash=config_hash,
        manifest_path=manifest_path,
        base=data_root,
        n_records=len(records),
        extra={"kind": "demos", "cells": list(cells)},
    )
    return {
        "n_records": len(records),
        "per_cell": per_cell,
        "success_rate": float(np.mean([r.success for r in records])),
        "paths": [str(npz_path), str(out_dir / "DATA_CARD.json")],
        "data_card": card,
        "manifest_hash": manifest_hash(manifest_path),
        "elapsed_s": time.perf_counter() - t_start,
    }


def generate_nominal_demos(
    task: Task,
    *,
    episodes: int,
    base_seed: int,
    out_dir: str | Path,
    max_steps: int,
    config_hash: str,
    action_dim: int | None = None,
    keep_failures: bool = False,
    record_images: bool = False,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    data_root: str | Path = ROOT,
) -> dict[str, Any]:
    """D-nom: scripted-expert episodes at nominal dynamics (Stage A SFT data).

    ``04`` §7 also allows the LeRobot ``metaworld_mt50`` episodes; this path exists
    so a Stage-A corpus can be regenerated deterministically without network access.
    """
    out_dir = Path(out_dir)
    records: list[DemoRecord] = []
    readbacks: list[dict[str, Any]] = []
    expert = task.expert()
    saw_success_key: list[bool] = [False]
    nominal = DynamicsSpec()
    seeds = episode_seeds("M-M-M", base_seed=base_seed, n=episodes)

    for ep_idx, seed in enumerate(seeds):
        env = task.make_env(nominal, seed)
        try:
            obs, _info = reset_env(env, seed)
            rb = readback_dict(env)
            readbacks.append(rb)
            instruction = task_instruction(task, env)
            actions: list[np.ndarray] = []
            proprios: list[np.ndarray] = []
            images: list[np.ndarray] = []
            success = False
            step = 0
            for _step in range(max_steps):
                action = expert_action(expert, obs, action_dim)
                obs, _r, terminated, truncated, info = step_env(env, action)
                actions.append(action)
                proprios.append(np.asarray(obs["state"], dtype=np.float32).reshape(-1))
                if record_images and obs.get("images") is not None:
                    images.append(np.asarray(obs["images"]))
                success = episode_success(info, env, saw_success_key=saw_success_key)
                if terminated or truncated:
                    break
            if success or keep_failures:
                records.append(
                    DemoRecord(
                        ep_id=f"nominal-s{seed}-e{ep_idx}",
                        cell_id="M-M-M",
                        split="train",
                        seed=int(seed),
                        task=task.name,
                        success=bool(success),
                        steps=step + 1,
                        actions=np.stack(actions),
                        proprio=np.stack(proprios),
                        instruction=instruction,
                        images=np.stack(images) if images else None,
                        dyn=nominal.as_dict(),
                        extra={"dyn_readback": rb},
                    )
                )
        finally:
            close_env(env)

    if not saw_success_key[0]:
        raise RuntimeError("no episode exposed a success signal (info['success'] / env.success)")
    if not records:
        raise RuntimeError("no nominal episodes recorded: the scripted expert never succeeded")

    npz_path = write_demos(out_dir, records, shard="demos")
    card = data_card(
        records,
        generator="gen_demos.nominal",
        config_hash=config_hash,
        extra={"episodes": episodes, "base_seed": base_seed, "nominal": True},
    )
    write_data_card(out_dir / "DATA_CARD.json", card)
    register_dataset(
        npz_path,
        generator="gen_demos.nominal",
        generator_version=DATASET_VERSION,
        config_hash=config_hash,
        manifest_path=manifest_path,
        base=data_root,
        n_records=len(records),
        extra={"kind": "demos_nominal"},
    )
    return {
        "n_records": len(records),
        "success_rate": float(np.mean([r.success for r in records])),
        "paths": [str(npz_path)],
        "data_card": card,
        "manifest_hash": manifest_hash(manifest_path),
    }


__all__ = [
    "DEFAULT_MANIFEST",
    "assert_dynamics_applied",
    "close_env",
    "episode_success",
    "expert_action",
    "generate_demos",
    "generate_nominal_demos",
    "readback_dict",
    "reset_env",
    "step_env",
    "task_instruction",
]
