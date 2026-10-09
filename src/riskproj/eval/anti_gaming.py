"""Anti-gaming checks (``09`` §1, §5).

A "fewer violations" or "higher success" number is only interesting if it is not
produced by doing nothing, by shrinking the actions, or by burning the episode on
fallbacks. These functions make those explanations *measurable* and label them in
the returned structure, so the label travels with the number instead of living in
a paragraph someone forgets to write.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from riskproj.envs import ViolationMonitor, apply_dynamics, set_cell_id
from riskproj.envs.violations import any_violation
from riskproj.types import DynamicsSpec, empty_violation_counts

__all__ = [
    "STALL_EPS",
    "action_statistics",
    "do_nothing_baseline",
    "fallback_conditioned_success",
    "suppression_report",
    "success_violation_pareto",
]

#: Below this per-step change (in metres/radians of the raw action vector) an action
#: is treated as a hold. Deliberately tiny: a policy that *legitimately* holds still
#: is not gaming anything, so only a near-exact repeat counts.
STALL_EPS = 1e-6


def _as_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    to_json = getattr(record, "to_json", None)
    if callable(to_json):
        return dict(to_json())
    raise TypeError(f"unsupported record type {type(record).__name__}")


def action_statistics(actions: Sequence[Any]) -> dict[str, float]:
    """Norm, jerk, path length and stall fraction of an executed action sequence."""
    if len(actions) == 0:
        return {
            "n": 0.0,
            "mean_norm": float("nan"),
            "max_norm": float("nan"),
            "path_length": 0.0,
            "jerk": float("nan"),
            "stall_fraction": float("nan"),
        }
    arr = np.stack([np.asarray(a, dtype=np.float64).reshape(-1) for a in actions])
    norms = np.linalg.norm(arr, axis=1)
    steps = np.linalg.norm(np.diff(arr, axis=0), axis=1) if len(arr) > 1 else np.zeros(1)
    if len(arr) > 2:
        jerk = float(np.mean(np.sum(np.diff(arr, n=2, axis=0) ** 2, axis=1)))
    else:
        jerk = float("nan")
    return {
        "n": float(len(arr)),
        "mean_norm": float(norms.mean()),
        "max_norm": float(norms.max()),
        "path_length": float(steps.sum()),
        "jerk": jerk,
        "stall_fraction": float(np.mean(steps <= STALL_EPS)) if len(steps) else float("nan"),
        "stall_eps": float(STALL_EPS),
    }


def do_nothing_baseline(
    task: Any,
    cells: Sequence[str],
    *,
    episodes: int,
    base_seed: int,
    max_steps: int = 200,
    ranges: Any | None = None,
    dyn: DynamicsSpec | None = None,
    action_dim: int | None = None,
) -> dict[str, Any]:
    """Zero-action baseline: what happens if the policy never moves.

    This is the floor a "fewer violations" claim has to beat. It is deliberately
    *not* a success-rate comparison: doing nothing is trivially safe on this task,
    which is exactly why ``01`` §2 rejects the idle action as a baseline and only
    uses it as a gaming check.
    """
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    per_cell: dict[str, Any] = {}
    all_violations = empty_violation_counts()
    n_success = 0
    n_episodes = 0
    for cell in cells:
        cell_dyn = dyn
        if cell_dyn is None and ranges is not None:
            rng = np.random.default_rng(abs(hash((cell, base_seed))) % (2**31))
            cell_dyn = ranges.sample(cell, rng)
        cell_dyn = cell_dyn or DynamicsSpec()
        cell_success = 0
        cell_violations = empty_violation_counts()
        for ep in range(episodes):
            env = task.make_env(cell_dyn, base_seed + ep)
            try:
                set_cell_id(env, cell)
            except AttributeError:  # pragma: no cover
                pass
            apply_dynamics(env, cell_dyn).assert_applied()
            monitor = ViolationMonitor(task.violations)
            obs, info = env.reset(seed=base_seed + ep)
            dim = int(action_dim or getattr(env, "action_dim", 3))
            steps = 0
            terminated = truncated = False
            while steps < max_steps and not (terminated or truncated):
                obs, _r, terminated, truncated, info = env.step(np.zeros(dim, dtype=np.float64))
                steps += 1
                monitor.observe(
                    {
                        "eef_pos": obs.get("eef_pose", info.get("eef_pos")),
                        "object_pos": obs.get("object_pose", info.get("object_pos")),
                        "grasped": bool(obs.get("grasped", False)),
                        "force": info.get("force", 0.0),
                        "penetration": info.get("penetration", 0.0),
                        "progress_m": info.get("progress_m", 0.0),
                        "grasp_established": bool(obs.get("grasped", False)),
                    }
                )
            rec = monitor.finalize()
            cell_success += int(bool(info.get("success", False)))
            for v, entry in rec.items():
                cell_violations[v]["count"] += float(entry["count"])
                all_violations[v]["count"] += float(entry["count"])
                if entry["first_step"] >= 0 and all_violations[v]["first_step"] < 0:
                    all_violations[v]["first_step"] = float(entry["first_step"])
        n_success += cell_success
        n_episodes += episodes
        per_cell[cell] = {
            "success_rate": cell_success / episodes,
            "violation_types_fired": sorted(
                v for v, e in cell_violations.items() if float(e["count"]) > 0
            ),
        }
    return {
        "episodes": n_episodes,
        "success_rate": (n_success / n_episodes) if n_episodes else float("nan"),
        "any_violation_rate": float(any(float(e["count"]) > 0 for e in all_violations.values())),
        "violations": all_violations,
        "per_cell": per_cell,
        "note": (
            "zero-action floor; a method whose violation gain disappears against this "
            "baseline has gained nothing (09 §5)"
        ),
    }


def success_violation_pareto(records: Sequence[Any]) -> dict[str, Any]:
    """Figure F6: success against violation rate per method, per split.

    Splits are kept separate: a Pareto point computed on pooled cells would hide a
    method that trades TEST-range violations for TEST-combo success.
    """
    by_method_split: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for raw in records:
        row = _as_dict(raw)
        violations = row.get("violations") or {}
        violated = 1.0 if any(float(v["count"]) > 0 for v in violations.values()) else 0.0
        split = str(row.get("split", "unknown"))
        by_method_split.setdefault(str(row["method"]), {}).setdefault(split, []).append(
            (1.0 if row.get("success") else 0.0, violated)
        )
    table: dict[str, dict[str, dict[str, float]]] = {}
    for method, splits in by_method_split.items():
        table[method] = {}
        for split, pairs in splits.items():
            arr = np.asarray(pairs, dtype=float)
            table[method][split] = {
                "n": float(arr.shape[0]),
                "success_rate": float(arr[:, 0].mean()),
                "violation_rate": float(arr[:, 1].mean()),
                # A method that wins on both axes is the only unambiguous win.
                "dominates_do_nothing": float(arr[:, 0].mean() > 0.0 and arr[:, 1].mean() < 1.0),
            }
    return {"table": table, "guard": "never pool splits; never report only the best seed/cell"}


def fallback_conditioned_success(records: Sequence[Any], *, max_fallback_rate: float = 0.30) -> dict[str, Any]:
    """Separate adaptation gains from fallback-driven ones (``09`` §5).

    A success rate that arrives with a high F2/F3 rate is labelled
    ``not_an_adaptation_gain`` in the returned structure.
    """
    rows = [_as_dict(r) for r in records]
    if not rows:
        return {"n": 0, "labelled": {}, "guard": "no records"}
    per_method: dict[str, dict[str, float]] = {}
    for method in sorted({str(r["method"]) for r in rows}):
        subset = [r for r in rows if str(r["method"]) == method]
        fallback = np.asarray([float(sum(int(v) for v in r["fallback"].values())) for r in subset])
        f23 = np.asarray(
            [float(int(r["fallback"]["F2"]) + int(r["fallback"]["F3"])) for r in subset]
        )
        success = np.asarray([1.0 if r.get("success") else 0.0 for r in subset])
        rate = float(f23.mean())
        per_method[method] = {
            "n": float(len(subset)),
            "success_rate": float(success.mean()),
            "mean_fallbacks": float(fallback.mean()),
            "f2_f3_rate": rate,
            "success_given_no_fallback": float(success[f23 == 0].mean()) if (f23 == 0).any() else float("nan"),
        }
    labelled = {
        method: "not_an_adaptation_gain" if vals["f2_f3_rate"] > max_fallback_rate else "ok"
        for method, vals in per_method.items()
    }
    return {
        "n": len(rows),
        "per_method": per_method,
        "labelled": labelled,
        "max_fallback_rate": float(max_fallback_rate),
        "guard": "gains with a high F2/F3 fallback rate are not adaptation gains (09 §5)",
    }


def suppression_report(records: Sequence[Any], *, baseline: str = "B1") -> dict[str, Any]:
    """Relative change in mean action norm against a baseline — the G6 criterion.

    Action suppression is the cheapest way to fake safety, so it is checked
    against the baseline method rather than against zero.
    """
    rows = [_as_dict(r) for r in records]
    norms: dict[str, list[float]] = {}
    for r in rows:
        norms.setdefault(str(r["method"]), []).append(float(r.get("action_norm_mean", 0.0)))
    if baseline not in norms:
        return {
            "baseline": baseline,
            "available": sorted(norms),
            "error": f"baseline method {baseline!r} not present in the records",
        }
    base = float(np.mean(norms[baseline]))
    out: dict[str, dict[str, float]] = {}
    for method, vals in sorted(norms.items()):
        mean = float(np.mean(vals))
        out[method] = {
            "n": float(len(vals)),
            "mean_action_norm": mean,
            "relative_change": ((mean - base) / base) if base else float("nan"),
        }
    return {
        "baseline": baseline,
        "baseline_mean_action_norm": base,
        "per_method": out,
        "suppression_flag": [
            m
            for m, v in out.items()
            if v["relative_change"] < -0.25  # G6: max_action_suppression_rel = 0.25
        ],
        "guard": "action suppression is not a safety gain (09 §5, G6)",
    }
