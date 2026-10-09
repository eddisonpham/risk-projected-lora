"""Critic-data generation (D-crit) with counterfactual open-loop labelling.

``04`` §6-§7: every candidate chunk is labelled by **restoring a saved simulator
state and executing the chunk open-loop for ``H_c`` steps**, recording V1-V5. That
is the only label source that makes the critic's target a physical outcome rather
than a prediction, and it is also what makes E1 (counterfactual validity) possible
later: original vs. corrected chunk from the *same* state.

Two contracts this module needs from other packages, both fail-loud:

* ``riskproj.envs.state_io`` provides ``save_state(env)`` / ``restore_state(env,
  state)``. :class:`SimStateReplay` also accepts envs that implement
  ``get_state()``/``set_state()`` directly, and raises if the env supports neither
  (a missing snapshot must never silently degrade E1 into a confounded test).
* ``riskproj.envs.violations.ViolationMonitor`` provides ``update(step_info)`` and
  ``finalize() -> {V1..V5: ...}``. It is injected through ``monitor_factory`` so
  the labelling logic is unit-testable with a stub monitor.

Critic-context rule: features come from the **pre-action** observation and the
prior action/proprio history only. Nothing from inside the labelled horizon may
enter the features, or the critic would be reading its own label.

The expert "chunk" at a snapshot is the sequence of expert actions the scripted
expert actually took over the following ``H_c`` real steps (padded by repeating the
last action if the episode ends early). Building it by re-querying the expert on
predicted states would need a forward model we do not have.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from riskproj.data.gen_demos import (
    DEFAULT_MANIFEST,
    assert_dynamics_applied,
    close_env,
    expert_action,
    readback_dict,
    reset_env,
    step_env,
)
from riskproj.data.schema import (
    DATASET_VERSION,
    CriticDataset,
    CriticTuple,
    assert_deployable_ctx,
    data_card,
    register_dataset,
    write_data_card,
)
from riskproj.data.splits import episode_seeds, sample_dynamics_for_cell
from riskproj.obs import CtxBuilder, StepObs, env_obs_to_step_obs
from riskproj.protocols import Task
from riskproj.types import VIOLATION_TYPES, CellId, DynamicsRanges, ViolationSpec
from riskproj.utils.io import ROOT, manifest_hash
from riskproj.utils.registry import register

#: A candidate-chunk provider: ``policy(obs_dict) -> [H_c, A]``.
PolicyChunkFn = Callable[[dict[str, Any]], np.ndarray]


# --------------------------------------------------------------------------
# adapters owned by other slices
# --------------------------------------------------------------------------


def default_monitor_factory(spec: ViolationSpec) -> Any:
    """Build the real ``ViolationMonitor`` (``envs/violations.py``)."""
    from riskproj.envs.violations import ViolationMonitor

    return ViolationMonitor(spec)


class SimStateReplay:
    """Snapshot/restore for counterfactual replay, with a documented env contract.

    Prefers ``riskproj.envs.state_io``; falls back to an env that implements
    ``get_state``/``set_state``; otherwise raises. ``restore`` returns the
    observation dict so the caller never has to guess how to re-observe.
    """

    def __init__(self, env: Any, *, recompute_obs: bool = True) -> None:
        self.env = env
        self.recompute_obs = recompute_obs
        try:
            from riskproj.envs import state_io as _state_io

            self._state_io: Any | None = _state_io
        except ImportError:  # pragma: no cover - resolved once the envs slice lands
            self._state_io = None

    @property
    def available(self) -> bool:
        has_io = self._state_io is not None and all(
            hasattr(self._state_io, name) for name in ("save_state", "restore_state")
        )
        unwrapped = getattr(self.env, "unwrapped", self.env)
        has_env_methods = hasattr(self.env, "get_state") or hasattr(unwrapped, "get_state")
        return bool(has_io or has_env_methods)

    def snapshot(self) -> Any:
        if self._state_io is not None and hasattr(self._state_io, "save_state"):
            return self._state_io.save_state(self.env)
        for target in (self.env, getattr(self.env, "unwrapped", None)):
            if target is not None and hasattr(target, "get_state"):
                return target.get_state()
        raise RuntimeError(
            "no state snapshot mechanism: install riskproj.envs.state_io (save_state/restore_state) "
            "or give the env get_state()/set_state(); E1 depends on exact replay (04 §6)"
        )

    def restore(self, state: Any) -> dict[str, Any]:
        out: Any = None
        if self._state_io is not None and hasattr(self._state_io, "restore_state"):
            out = self._state_io.restore_state(self.env, state)
        else:
            restored = False
            for target in (self.env, getattr(self.env, "unwrapped", None)):
                if target is not None and hasattr(target, "set_state"):
                    out = target.set_state(state)
                    restored = True
                    break
            if not restored:
                raise RuntimeError(
                    "no state restore mechanism available (see SimStateReplay.snapshot)"
                )
        if isinstance(out, dict):
            return out
        if not self.recompute_obs:
            raise RuntimeError("restore_state returned no observation and recompute_obs=False")
        for target in (self.env, getattr(self.env, "unwrapped", None)):
            if target is None:
                continue
            for name in ("get_obs", "_get_obs", "observe", "get_observation"):
                getter = getattr(target, name, None)
                if callable(getter):
                    obs = getter()
                    if isinstance(obs, dict):
                        return obs
        raise RuntimeError(
            "restore_state returned None and no observation getter (get_obs/_get_obs/observe) was "
            "found; restore must refresh the observation so labels come from the restored state (04 §6)"
        )


# --------------------------------------------------------------------------
# candidate chunk families
# --------------------------------------------------------------------------


def perturb_chunk(
    chunk: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    *,
    low_frequency: bool = False,
) -> np.ndarray:
    """Add Gaussian noise scaled by the chunk's own per-action RMS.

    ``low_frequency=True`` smooths the noise along the horizon (moving average),
    producing coherent deviations rather than white noise. The realised magnitude
    is renormalised so ``sigma`` means the same thing in both modes.
    """
    c = np.asarray(chunk, dtype=np.float32)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    scale = float(np.sqrt(np.mean(c**2))) if c.size else 0.0
    scale = scale if scale > 0 else 1.0
    noise = rng.standard_normal(c.shape).astype(np.float32)
    if low_frequency and c.shape[0] >= 3:
        kernel = np.ones(3, dtype=np.float32) / 3.0
        smoothed = np.empty_like(noise)
        for j in range(noise.shape[1]):
            smoothed[:, j] = np.convolve(noise[:, j], kernel, mode="same")
        cur = float(np.sqrt(np.mean(smoothed**2)))
        if cur > 0:
            smoothed *= 1.0 / cur
        noise = smoothed
    return (c + sigma * scale * noise).astype(np.float32)


def unsafe_chunks(
    expert_chunk: np.ndarray,
    action_spec: dict[str, Any],
    rng: np.random.Generator,
    *,
    n_variants: int = 2,
) -> list[np.ndarray]:
    """Deliberately unsafe chunks: over-push and early/late release (``04`` §7).

    ``action_spec`` keys: ``gripper_index`` (int | None), ``gripper_open`` (float),
    ``overpush_range`` (tuple[float, float], default ``(1.5, 2.5)``).
    """
    c = np.asarray(expert_chunk, dtype=np.float32)
    if c.ndim != 2 or c.size == 0:
        return []
    lo, hi = action_spec.get("overpush_range", (1.5, 2.5))
    lo, hi = min(lo, hi), max(lo, hi)
    out: list[np.ndarray] = []
    for _ in range(max(0, n_variants)):
        out.append((c * float(rng.uniform(lo, hi))).astype(np.float32))
    gi = action_spec.get("gripper_index")
    if gi is not None and 0 <= int(gi) < c.shape[1]:
        gi = int(gi)
        open_val = float(action_spec.get("gripper_open", 1.0))
        early = c.copy()
        half = max(1, early.shape[0] // 2)
        early[half:, gi] = open_val  # release early: the peg is dropped mid-insertion
        out.append(early.astype(np.float32))
    return out


def _rng_for(*parts: Any) -> np.random.Generator:
    """Stable seed from arbitrary parts (sha256, not python's randomised hash())."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(seed)


def _normalize_labels(raw: Any) -> dict[str, bool]:
    """Accept ``{V: bool}`` or ``{V: {count, first_step}}`` and return bools."""
    labels = dict.fromkeys(VIOLATION_TYPES, False)
    if raw is None:
        return labels
    if isinstance(raw, dict):
        for v in VIOLATION_TYPES:
            node = raw.get(v)
            if node is None:
                continue
            if isinstance(node, dict):
                count = float(node.get("count", 0.0))
                first = float(node.get("first_step", -1.0))
                labels[v] = bool(count > 0) or bool(first >= 0)
            else:
                labels[v] = bool(node)
        return labels
    if isinstance(raw, (list, tuple, set)):  # tolerated shape
        for v in raw:
            if v in labels:
                labels[v] = True
    return labels


def _pad_chunk(chunk: np.ndarray, horizon: int) -> np.ndarray:
    """Pad/trim a chunk to exactly ``horizon`` steps, repeating the last action."""
    c = np.asarray(chunk, dtype=np.float32)
    if c.ndim != 2:
        raise ValueError(f"chunk must be [H, A]; got {c.shape}")
    if c.shape[0] >= horizon:
        return c[:horizon]
    if c.shape[0] == 0:
        raise ValueError("cannot pad an empty chunk")
    pad = np.repeat(c[-1:], horizon - c.shape[0], axis=0)
    return np.concatenate([c, pad], axis=0)


def _copy_obs_value(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, dict):
        return {k: _copy_obs_value(x) for k, x in v.items()}
    return v


# --------------------------------------------------------------------------
# state collection
# --------------------------------------------------------------------------


def collect_states(
    task: Task,
    cell: CellId,
    *,
    ranges: DynamicsRanges,
    seed: int,
    n_states: int,
    state_every: int,
    max_steps: int,
    horizon: int,
    env: Any | None = None,
    action_dim: int | None = None,
    verify_dynamics: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    """Roll out the scripted expert once and snapshot states along the trajectory.

    Each snapshot carries (i) the exact simulator state, (ii) the pre-action
    observation, (iii) the history of steps strictly *before* that observation,
    and (iv) the expert's next ``horizon`` actions. The env is returned **open** so
    the caller labels chunks by restoring into the very same env instance.
    """
    owns_env = env is None
    dyn = sample_dynamics_for_cell(cell, ranges, seed)
    if env is None:
        env = task.make_env(dyn, seed)
    snapshots: list[dict[str, Any]] = []
    try:
        obs, _info = reset_env(env, seed)
        rb = readback_dict(env)
        if verify_dynamics:
            assert_dynamics_applied(dyn, rb, cell)
        replay = SimStateReplay(env)
        if not replay.available:
            raise RuntimeError(
                f"{cell}: env exposes no state snapshot (see SimStateReplay); counterfactual "
                "labelling is impossible without it (04 §6)"
            )
        expert = task.expert()
        hist: list[StepObs] = [env_obs_to_step_obs(obs)]
        actions_taken: list[np.ndarray] = []
        want_at = sorted(set(range(0, max_steps, max(1, state_every))))[:n_states]
        for step in range(max_steps):
            if step in want_at:
                snapshots.append(
                    {
                        "state": replay.snapshot(),
                        "obs": {k: _copy_obs_value(v) for k, v in obs.items()},
                        "hist": list(hist[:-1]),  # strictly before the pre-action obs
                        "action_index": step,
                        "dyn_readback": rb,
                        "dyn": dyn.as_dict(),
                    }
                )
            action = expert_action(expert, obs, action_dim)
            actions_taken.append(action)
            obs, _r, terminated, truncated, _info = step_env(env, action)
            hist.append(env_obs_to_step_obs(obs, action=action))
            if terminated or truncated or len(snapshots) >= n_states:
                break
        for snap in snapshots:
            idx = int(snap["action_index"])
            tail = actions_taken[idx : idx + horizon]
            snap["chunk"] = (
                _pad_chunk(np.stack(tail), horizon) if tail else np.zeros((0, 0), dtype=np.float32)
            )
    except Exception:
        if owns_env:
            close_env(env)
        raise
    meta = {
        "cell": cell,
        "seed": int(seed),
        "dyn": dyn.as_dict(),
        "n_states": len(snapshots),
        "expert_steps": len(actions_taken),
    }
    return snapshots, meta, env


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def _chunks_per_rollout(max_steps: int, state_every: int, n_states: int) -> int:
    steps = len(range(0, max_steps, max(1, state_every)))
    return max(1, min(n_states, steps))


@register("dataset", "critic_tuples")
def generate_critic_data(
    task: Task,
    *,
    cells: Sequence[CellId],
    ranges: DynamicsRanges,
    states_per_cell: int,
    chunks_per_state: int,
    base_seed: int,
    out_dir: str | Path,
    horizon: int,
    violation_spec: ViolationSpec,
    config_hash: str,
    policy: PolicyChunkFn | None = None,
    sigma_grid: Sequence[float] = (0.02, 0.05, 0.1, 0.2),
    include_unsafe: bool = True,
    include_policy: bool = True,
    include_expert: bool = True,
    max_steps: int = 200,
    state_every: int = 10,
    hist_len: int = 1,
    use_vision: bool = False,
    privileged_force: bool = False,
    action_spec: dict[str, Any] | None = None,
    action_dim: int | None = None,
    verify_dynamics: bool = True,
    min_pos_rate: float = 0.15,
    shard_size: int = 5000,
    monitor_factory: Callable[[ViolationSpec], Any] = default_monitor_factory,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    data_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Build D-crit: candidate chunks labelled by open-loop execution (``04`` §6-§7).

    Returns a summary dict with class balance, per-cell positives, shard paths and
    the manifest hash. Raises if positivity falls below ``min_pos_rate`` — an
    all-safe critic dataset is a silent failure mode that would otherwise surface
    much later as a mysterious G3 failure.
    """
    t_start = time.perf_counter()
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if chunks_per_state < 1:
        raise ValueError("chunks_per_state must be >= 1")
    if not cells:
        raise ValueError("no cells supplied")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    action_spec = dict(action_spec or {"gripper_index": None, "gripper_open": 1.0})
    per_rollout = _chunks_per_rollout(max_steps, state_every, states_per_cell)
    n_rollouts = max(1, -(-states_per_cell // per_rollout))

    tuples: list[CriticTuple] = []
    per_cell: dict[str, dict[str, float]] = {}
    n_restores = 0

    for cell in cells:
        seeds = episode_seeds(cell, base_seed=base_seed, n=n_rollouts)
        cell_pos = 0
        cell_n = 0
        for ep_seed in seeds:
            env = task.make_env(sample_dynamics_for_cell(cell, ranges, int(ep_seed)), int(ep_seed))
            try:
                snapshots, _meta, env = collect_states(
                    task,
                    cell,
                    ranges=ranges,
                    seed=int(ep_seed),
                    n_states=per_rollout,
                    state_every=state_every,
                    max_steps=max_steps,
                    horizon=horizon,
                    env=env,
                    action_dim=action_dim,
                    verify_dynamics=verify_dynamics,
                )
                replay = SimStateReplay(env)
                for s_idx, snap in enumerate(snapshots):
                    obs_dict = snap["obs"]
                    hist: list[StepObs] = snap["hist"]
                    candidates = _candidate_chunks(
                        snap,
                        chunks_per_state=chunks_per_state,
                        sigma_grid=sigma_grid,
                        include_expert=include_expert,
                        include_policy=include_policy,
                        include_unsafe=include_unsafe,
                        policy=policy,
                        obs_dict=obs_dict,
                        action_spec=action_spec,
                        key=(base_seed, cell, int(ep_seed), s_idx),
                        horizon=horizon,
                    )
                    for c_idx, (chunk, source, sigma) in enumerate(candidates):
                        ctx = _build_ctx(
                            hist,
                            obs_dict,
                            hist_len=hist_len,
                            use_vision=use_vision,
                            privileged_force=privileged_force,
                        )
                        assert_deployable_ctx(
                            ctx, use_vision=use_vision, privileged_force=privileged_force
                        )
                        labels = _label_chunk(
                            env,
                            replay,
                            snap["state"],
                            chunk,
                            horizon=horizon,
                            monitor_factory=monitor_factory,
                            violation_spec=violation_spec,
                        )
                        n_restores += 1
                        cell_n += 1
                        cell_pos += int(any(labels.values()))
                        tuples.append(
                            CriticTuple(
                                tuple_id=f"{cell}-s{ep_seed}-st{s_idx}-c{c_idx}",
                                cell_id=cell,
                                split="train",
                                seed=int(ep_seed),
                                ctx=ctx,
                                chunk=chunk,
                                labels=labels,
                                chunk_source=source,
                                perturbation_sigma=sigma,
                                extra={
                                    "chunk_len": int(chunk.shape[0]),
                                    "dyn_readback": snap.get("dyn_readback", {}),
                                    "progress": float(obs_dict.get("progress", 0.0)),
                                },
                            )
                        )
            finally:
                close_env(env)
        per_cell[cell] = {
            "n": float(cell_n),
            "pos_rate": (cell_pos / cell_n) if cell_n else float("nan"),
        }

    if not tuples:
        raise RuntimeError("no critic tuples were produced; check cells/state collection")

    dataset = CriticDataset(tuples)
    paths: list[str] = []
    for shard_idx in range(0, len(tuples), shard_size):
        shard = CriticDataset(tuples[shard_idx : shard_idx + shard_size])
        stem = out_dir / f"tuples_{shard_idx // shard_size:04d}"
        npz_path = shard.save(stem)
        paths.append(str(npz_path))
        register_dataset(
            npz_path,
            generator="gen_critic_data",
            generator_version=DATASET_VERSION,
            config_hash=config_hash,
            manifest_path=manifest_path,
            base=data_root,
            n_records=len(shard),
            extra={"kind": "critic_tuples", "shard": shard_idx // shard_size},
        )

    balance = dataset.class_balance()
    card = data_card(
        dataset,
        generator="gen_critic_data",
        config_hash=config_hash,
        extra={
            "cells": list(cells),
            "horizon": int(horizon),
            "sigma_grid": list(sigma_grid),
            "include_unsafe": bool(include_unsafe),
            "include_policy": bool(include_policy),
            "include_expert": bool(include_expert),
            "base_seed": int(base_seed),
            "per_cell_from_generator": per_cell,
            "n_state_restores": n_restores,
            "hist_len": int(hist_len),
            "use_vision": bool(use_vision),
            "privileged_force": bool(privileged_force),
        },
    )
    write_data_card(out_dir / "DATA_CARD.json", card)

    if balance["pos_rate"] < min_pos_rate:
        raise RuntimeError(
            f"D-crit positivity {balance['pos_rate']:.3f} < required {min_pos_rate:.2f}. "
            f"n={int(balance['n'])}, per-type={dataset.per_type_balance()}, "
            f"per-source={dataset.source_balance()}, per-cell={dataset.per_cell_positives()}. "
            f"Fixes: widen sigma_grid (currently {list(sigma_grid)}), increase include_unsafe, or "
            "check the violation thresholds are not above anything the expert produces (04 §7)."
        )

    return {
        "n_tuples": int(balance["n"]),
        "pos_rate": balance["pos_rate"],
        "per_type_balance": dataset.per_type_balance(),
        "per_source_balance": dataset.source_balance(),
        "per_cell_positives": dataset.per_cell_positives(),
        "cell_histogram": dataset.cell_histogram(),
        "feature_spec": dataset.feature_spec_kwargs(),
        "paths": [*paths, str(out_dir / "DATA_CARD.json")],
        "data_card": card,
        "manifest_hash": manifest_hash(manifest_path),
        "elapsed_s": time.perf_counter() - t_start,
        "n_state_restores": n_restores,
    }


def _candidate_chunks(
    snap: dict[str, Any],
    *,
    chunks_per_state: int,
    sigma_grid: Sequence[float],
    include_expert: bool,
    include_policy: bool,
    include_unsafe: bool,
    policy: PolicyChunkFn | None,
    obs_dict: dict[str, Any],
    action_spec: dict[str, Any],
    key: tuple[Any, ...],
    horizon: int,
) -> list[tuple[np.ndarray, str, float | None]]:
    """The candidate families for one state, in a deterministic order."""
    out: list[tuple[np.ndarray, str, float | None]] = []
    expert_chunk = np.asarray(snap["chunk"], dtype=np.float32)
    if expert_chunk.ndim != 2 or expert_chunk.size == 0:
        expert_chunk = np.zeros((0, 0), dtype=np.float32)

    if include_expert and expert_chunk.size:
        out.append((_pad_chunk(expert_chunk, horizon), "expert", None))

    if include_policy and policy is not None:
        pol_chunk = np.asarray(policy(obs_dict), dtype=np.float32)
        if pol_chunk.ndim != 2 or pol_chunk.size == 0:
            raise ValueError(f"policy chunk must be [H, A]; got {pol_chunk.shape}")
        out.append((_pad_chunk(pol_chunk, horizon), "policy", None))

    if expert_chunk.size:
        n_extra = max(0, chunks_per_state - len(out))
        sigmas = list(sigma_grid)[:n_extra]
        for k, sigma in enumerate(sigmas):
            rng = _rng_for(*key, "perturb", k)
            out.append(
                (
                    perturb_chunk(expert_chunk, float(sigma), rng, low_frequency=bool(k % 2)),
                    "perturbed",
                    float(sigma),
                )
            )
        if include_unsafe:
            rng = _rng_for(*key, "unsafe")
            for chunk in unsafe_chunks(expert_chunk, action_spec, rng, n_variants=2):
                out.append((_pad_chunk(chunk, horizon), "unsafe", None))
    return out


def _build_ctx(
    hist: list[StepObs],
    obs_dict: dict[str, Any],
    *,
    hist_len: int,
    use_vision: bool,
    privileged_force: bool,
) -> dict[str, np.ndarray]:
    """Critic context from the pre-action observation and the preceding history."""
    builder = CtxBuilder(
        hist_len=hist_len, use_vision=use_vision, privileged_force=privileged_force
    )
    for step in hist:
        builder.observe(step)
    builder.observe(env_obs_to_step_obs(obs_dict))
    return builder.context()


def _label_chunk(
    env: Any,
    replay: SimStateReplay,
    state: Any,
    chunk: np.ndarray,
    *,
    horizon: int,
    monitor_factory: Callable[[ViolationSpec], Any],
    violation_spec: ViolationSpec,
) -> dict[str, bool]:
    """Restore the state and execute the chunk open-loop for ``horizon`` steps."""
    replay.restore(state)
    monitor = monitor_factory(violation_spec)
    labels: dict[str, bool] = dict.fromkeys(VIOLATION_TYPES, False)
    for t in range(min(horizon, chunk.shape[0])):
        action = np.asarray(chunk[t], dtype=np.float32)
        obs, _r, terminated, truncated, info = step_env(env, action)
        payload = dict(obs)
        payload.update(info)
        payload["action"] = action
        payload["step"] = t
        result = monitor.update(payload)
        if isinstance(result, dict) and result:
            labels = _normalize_labels(result)
        if terminated or truncated:
            break
    finalize = getattr(monitor, "finalize", None)
    if callable(finalize):
        out = finalize()
        if isinstance(out, dict) and out:
            labels = _normalize_labels(out)
    return labels


__all__ = [
    "PolicyChunkFn",
    "SimStateReplay",
    "collect_states",
    "default_monitor_factory",
    "generate_critic_data",
    "perturb_chunk",
    "unsafe_chunks",
]
