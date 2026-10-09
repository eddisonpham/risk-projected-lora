"""E1 — paired counterfactual validity (``04`` §6, ``09`` §1).

The question this answers is the one that decides whether the project is worth
continuing: *does the correction reduce real, simulator-measured violations?* It is
asked without a closed-loop rollout, so a success difference cannot come from
compounding, from episode length, or from lucky states.

Protocol, per saved simulator state:

1. restore the state, plan the baseline chunk, execute it open-loop for ``horizon``
   steps, record the violation events;
2. restore the *same* state again (asserting the saved hash is unchanged) and repeat
   with the corrected chunk.

Both arms see identical states and identical flow noise, so the comparison is
paired at the level of individual states. The norm-matched random-direction control
(C1) runs through the same function with a different planner; if it matches, the
correction was perturbation rather than risk information.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from riskproj.envs import SimState, ViolationMonitor, restore_state, save_state
# `_refresh_obs`/`_step_info` are the driver's shared internals: the counterfactual
# path must observe restored states exactly the way a rollout does, so it reuses them
# rather than growing a second, subtly different copy.
from riskproj.eval.rollout import (
    ChunkPlanner,
    NoProjectionPlanner,
    _refresh_obs,
    _step_info,
)
from riskproj.types import VIOLATION_TYPES, empty_violation_counts
from riskproj.utils.config import ProjCfg
from riskproj.utils.io import append_jsonl, write_json
from riskproj.utils.metrics import bootstrap_ci, mcnemar_exact

__all__ = ["CounterfactualRecord", "pairwise_summary", "paired_counterfactual"]

#: A step budget floor: a state may be saved late in an episode, so the open-loop
#: horizon is what matters here, not the remaining episode length.
DEFAULT_HORIZON = 10


@dataclass
class CounterfactualRecord:
    """One paired (state, original chunk, corrected chunk) comparison."""

    ep_id: str
    state_id: str
    cell: str
    method: str
    chunk_original: list[list[float]] = field(default_factory=list)
    chunk_corrected: list[list[float]] = field(default_factory=list)
    violations_original: dict[str, dict[str, float]] = field(default_factory=empty_violation_counts)
    violations_corrected: dict[str, dict[str, float]] = field(default_factory=empty_violation_counts)
    any_original: bool = False
    any_corrected: bool = False
    pair_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _chunk_from_planner(
    *,
    policy: Any,
    planner: ChunkPlanner,
    obs: Mapping[str, Any],
    step: int,
    env_seed: int,
    crn_base_seed: int,
    ctx_batch: Mapping[str, Any],
) -> Tensor:
    state_t = torch.as_tensor(np.asarray(obs["state"], dtype=np.float32))
    device = getattr(policy, "device_id", torch.device("cpu"))
    batch: dict[str, Any] = {"state": state_t.reshape(1, -1).to(device)}
    planned = planner.plan(
        policy=policy,
        batch=batch,
        ctx_batch=ctx_batch,
        step=step,  # same step index for both arms -> identical CRN
        env_seed=env_seed,
        crn_base_seed=crn_base_seed,
        tracker=None,
    )
    if planned.chunk.dim() != 3 or planned.chunk.shape[0] != 1:
        raise ValueError(f"planner returned {tuple(planned.chunk.shape)}; expected [1, H, A]")
    return planned.chunk.detach()


def _execute_open_loop(
    env: Any,
    task: Any,
    chunk: Tensor,
    horizon: int,
) -> tuple[dict[str, dict[str, float]], bool]:
    """Execute up to ``horizon`` actions open-loop; return per-type violations and the OR.

    ``horizon`` is capped by the chunk the planner emitted: the two arms may plan
    different numbers of actions, and executing past the end of one chunk would be an
    index error rather than a comparison.
    """
    n = min(int(horizon), int(chunk.shape[1]))
    if n < 1:
        raise ValueError(f"nothing to execute: horizon={horizon}, chunk={tuple(chunk.shape)}")
    monitor = ViolationMonitor(task.violations)
    for k in range(n):
        act = np.asarray(chunk[0, k].detach().cpu(), dtype=np.float64)
        obs, _reward, terminated, truncated, info = env.step(act)
        monitor.observe(_step_info(dict(obs), dict(info)), action=act)
        if terminated or truncated:
            break
    violations = monitor.finalize()
    return violations, any(float(v["count"]) > 0 for v in violations.values())


def paired_counterfactual(
    task: Any,
    policy: Any,
    states: Sequence[SimState | tuple[str, SimState]],
    *,
    planner: ChunkPlanner | None = None,
    baseline_planner: ChunkPlanner | None = None,
    critic: Any | None = None,
    proj_cfg: ProjCfg | None = None,
    horizon: int = DEFAULT_HORIZON,
    out_dir: str | Path | None = None,
    crn_base_seed: int = 0,
    cell: str = "M-M-M",
    method: str = "unknown",
    ctx_batch: Mapping[str, Any] | None = None,
    dyn: Any | None = None,
) -> dict[str, Any]:
    """Compare original and corrected chunks from identical saved states (E1)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not states:
        raise ValueError("no saved states supplied; E1 needs states from a rollout (04 §6)")

    method_planner = planner or NoProjectionPlanner()
    base_planner = baseline_planner or NoProjectionPlanner()
    if method_planner is base_planner:
        raise ValueError(
            "the corrected arm and the baseline arm must be different planners; "
            "otherwise the paired difference is identically zero by construction"
        )

    ctx = dict(ctx_batch) if ctx_batch is not None else None
    records: list[CounterfactualRecord] = []

    for idx, item in enumerate(states):
        state_id, state = item if isinstance(item, tuple) else (f"s{idx:04d}", item)
        # The caller knows the cell's dynamics; inventing a perturbed spec here would
        # silently change the physics between the two arms, so the default is nominal.
        env = task.make_env(dyn if dyn is not None else state_spec(state), idx)
        try:
            # ---- arm 1: the original (uncorrected) chunk --------------------
            obs = restore_state(env, state)
            if obs is None:
                obs = _refresh_obs(env)
            if ctx is None:
                ctx = _context_for(obs, policy, action_dim=int(getattr(env, "action_dim", 3)))
            chunk_original = _chunk_from_planner(
                policy=policy,
                planner=base_planner,
                obs=obs,
                step=0,
                env_seed=idx,
                crn_base_seed=crn_base_seed,
                ctx_batch=ctx,
            )
            v_original, any_original = _execute_open_loop(env, task, chunk_original, horizon)

            # ---- arm 2: the same state, the corrected chunk ----------------
            obs2 = restore_state(env, state)
            if obs2 is None:
                obs2 = _refresh_obs(env)
            # Restoring must be exact, otherwise the two arms are not paired and the
            # whole test is void (04 §6, smoke test S2). Fail loud rather than
            # report a comparison between different states.
            if save_state(env).hash() != state.hash():
                raise RuntimeError(
                    f"state {state_id}: restore is not exact (hash changed), so the paired "
                    "comparison would compare different states"
                )
            chunk_corrected = _chunk_from_planner(
                policy=policy,
                planner=method_planner,
                obs=obs2,
                step=0,
                env_seed=idx,
                crn_base_seed=crn_base_seed,
                ctx_batch=ctx,
            )
            v_corrected, any_corrected = _execute_open_loop(env, task, chunk_corrected, horizon)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

        pair_id = hashlib.sha256(
            f"{state_id}|{crn_base_seed}|{cell}".encode()
        ).hexdigest()[:12]
        records.append(
            CounterfactualRecord(
                ep_id=str(state_id),
                state_id=str(state_id),
                cell=str(cell),
                method=str(method),
                chunk_original=chunk_original[0].cpu().numpy().astype(float).round(6).tolist(),
                chunk_corrected=chunk_corrected[0].cpu().numpy().astype(float).round(6).tolist(),
                violations_original=v_original,
                violations_corrected=v_corrected,
                any_original=bool(any_original),
                any_corrected=bool(any_corrected),
                pair_id=pair_id,
                extras={
                    "state_hash": state.hash(),
                    "horizon": int(horizon),
                    "correction_norm": float((chunk_corrected - chunk_original).norm()),
                    "original_norm": float(chunk_original.norm()),
                },
            )
        )

    if out_dir is not None:
        folder = Path(out_dir)
        folder.mkdir(parents=True, exist_ok=True)
        jsonl = folder / f"counterfactual_{method}.jsonl"
        append_jsonl(jsonl, [r.to_json() for r in records])
        summary = pairwise_summary(records, seed=crn_base_seed)
        write_json(folder / f"counterfactual_{method}_summary.json", summary)
        summary["jsonl"] = str(jsonl)
    else:
        summary = pairwise_summary(records, seed=crn_base_seed)
    summary["n_pairs"] = len(records)
    summary["method"] = str(method)
    summary["cell"] = str(cell)
    return summary


def state_spec(state: SimState) -> Any:
    """Nominal dynamics for a restored state — the only honest default.

    A :class:`SimState` does not record which cell it came from, and inventing a
    perturbed spec here would silently change the physics between the two arms. A
    caller that saved the states knows the cell and should pass ``dyn``.
    """
    del state  # the argument exists for symmetry with future per-state metadata
    from riskproj.types import DynamicsSpec

    return DynamicsSpec()


def _context_for(
    obs: Mapping[str, Any], policy: Any, *, action_dim: int | None = None
) -> dict[str, np.ndarray]:
    """Critic context from a restored observation (pre-action, deployable only).

    No action history is available before the first action at a restored state, so
    the block is zeros — at the correct *action* width, which is what the critic's
    ``FeatureSpec`` expects (see ``eval.rollout._batch_context``).
    """
    from riskproj.eval.rollout import _batch_context
    from riskproj.obs import CtxBuilder, env_obs_to_step_obs

    hist = int(getattr(policy, "hist_len", 1) or 1)
    width = action_dim if action_dim is not None else int(getattr(policy, "action_dim", 0) or 0)
    builder = CtxBuilder(hist_len=hist, action_dim=width or None)
    builder.reset()
    builder.observe(env_obs_to_step_obs(dict(obs)))
    return _batch_context(
        builder, actions=None, hist_len=hist, action_dim=width if width else None
    )


def pairwise_summary(
    records: Sequence[CounterfactualRecord | Mapping[str, Any]], *, n_boot: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """Paired violation deltas with CIs, per type, plus the McNemar discordance.

    The interval is a percentile bootstrap over *pairs*, which is the unit of
    replication here (``09`` §2: never treat the episode as the unit when the state
    is).
    """
    rows = [_as_record(r) for r in records]
    if not rows:
        return {"n_pairs": 0, "note": "no counterfactual pairs"}

    original = np.asarray([1.0 if r.any_original else 0.0 for r in rows])
    corrected = np.asarray([1.0 if r.any_corrected else 0.0 for r in rows])
    delta = original - corrected  # positive = fewer violations after correction
    ci = bootstrap_ci(delta.tolist(), n_boot=n_boot, seed=seed)
    b = int(np.sum((original == 1) & (corrected == 0)))  # improved
    c = int(np.sum((original == 0) & (corrected == 1)))  # worsened

    per_type: dict[str, dict[str, float]] = {}
    for v in VIOLATION_TYPES:
        orig_v = np.asarray([_count(r.violations_original, v) for r in rows], dtype=float)
        corr_v = np.asarray([_count(r.violations_corrected, v) for r in rows], dtype=float)
        per_type[v] = {
            "rate_original": float(np.mean(orig_v > 0)),
            "rate_corrected": float(np.mean(corr_v > 0)),
            "mean_count_delta": float(np.mean(orig_v - corr_v)),
            "n_flipped_down": float(np.sum((orig_v > 0) & (corr_v == 0))),
            "n_flipped_up": float(np.sum((orig_v == 0) & (corr_v > 0))),
        }

    return {
        "n_pairs": len(rows),
        "any_violation_rate_original": float(original.mean()),
        "any_violation_rate_corrected": float(corrected.mean()),
        "paired_delta": {
            "point": float(delta.mean()),
            "lo": float(ci["lo"]),
            "hi": float(ci["hi"]),
            "excludes_zero": bool(ci["lo"] > 0.0 or ci["hi"] < 0.0),
            "ci_level": 0.95,
        },
        "discordant": {"improved": b, "worsened": c, **mcnemar_exact(b, c)},
        "per_type": per_type,
        "verdict": _verdict(ci, b, c),
        "guard": (
            "E1 decides on measured violations from identical states; a critic-score drop "
            "without a violation drop is a negative result about the critic (09 §5)"
        ),
    }


def _verdict(ci: Mapping[str, float], improved: int, worsened: int) -> str:
    if ci["lo"] > 0.0 and improved > worsened:
        return "measured violations reduced (paired CI excludes 0)"
    if ci["hi"] < 0.0:
        return "measured violations INCREASED (paired CI excludes 0)"
    return "no detectable change in measured violations (CI includes 0)"


def _count(violations: Mapping[str, Any], key: str) -> float:
    entry = violations.get(key) or {}
    return float(entry.get("count", 0.0))


def _as_record(record: Any) -> CounterfactualRecord:
    if isinstance(record, CounterfactualRecord):
        return record
    if isinstance(record, Mapping):
        return CounterfactualRecord(
            ep_id=str(record.get("ep_id", "")),
            state_id=str(record.get("state_id", "")),
            cell=str(record.get("cell", "")),
            method=str(record.get("method", "")),
            violations_original=dict(record.get("violations_original") or empty_violation_counts()),
            violations_corrected=dict(record.get("violations_corrected") or empty_violation_counts()),
            any_original=bool(record.get("any_original", False)),
            any_corrected=bool(record.get("any_corrected", False)),
            pair_id=str(record.get("pair_id", "")),
        )
    raise TypeError(f"unsupported record type {type(record).__name__}")
