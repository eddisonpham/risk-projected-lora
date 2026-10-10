#!/usr/bin/env python
"""G1: the dynamics-sensitivity sweep that decides whether this setup can test the
hypothesis at all (``agents/04_env_dynamics_data.md`` §8).

For each axis it runs the scripted expert (and pi_base, when a checkpoint exists)
across log-spaced scales in ``[scale_lo, scale_hi]``, logging success, per-type
violation rates and the physics read-backs. It then evaluates the four criteria from
the config, and **fails an unimplemented criterion rather than skipping it**: a
sweep that cannot evaluate the band criterion for ``pi_base`` has not passed G1.

``gates.G1.coverage_actors`` decides which actors the *coverage* criterion is measured
over (``04`` §8). The scripted expert alone cannot evoke three of the five events on
peg-insert-side (``research_log/LOG.md``, iteration 8), and an expert is *supposed* to be
failure-free, so the criterion is measured over the failure-inducing
:class:`~riskproj.envs.actors.PerturbedExpertActor` as well. Every actor that runs is
reported per scale; the criterion names the actor that produced each event's rate.

    python scripts/sensitivity_sweep.py --config configs/exp/e0_baseline_repro.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

AXES = ("mu", "mass", "stiff")


def _scale_spec(axis: str, scale: float) -> Any:
    from riskproj.types import DynamicsSpec

    return DynamicsSpec(**{f"{axis}_scale": float(scale)})


def _episode(task: Any, spec: Any, actor: Any, *, seed: int, max_steps: int) -> dict[str, Any]:
    env = task.make_env(spec, seed=seed)
    # A stochastic actor must restart its stream per episode, otherwise its result for a
    # given (axis, scale, episode) depends on how many episodes ran before it.
    reseed = getattr(actor, "reseed", None)
    if callable(reseed):
        reseed(seed)
    obs, _info = env.reset(seed=seed)
    steps = 0
    for _ in range(max_steps):
        action = actor.act(obs)
        obs, _r, term, trunc, info = env.step(action)
        steps += 1
        if term or trunc:
            break
    summary = env.episode_summary() if hasattr(env, "episode_summary") else {}
    violations = summary.get("violations", info.get("violations", {}))
    # ``dyn_readback`` is a DynReadback dataclass, not a mapping; dropping it here made the
    # G1 read-back criterion report `nan` for every off-nominal scale — a *fail* caused by
    # the reporter rather than by the physics.
    rb = getattr(env, "dyn_readback", None)
    if rb is not None and hasattr(rb, "as_dict"):
        rb = rb.as_dict()
    return {
        "success": bool(summary.get("success", info.get("success", False))),
        "steps": steps,
        "violations": violations,
        "readback": rb if isinstance(rb, dict) else {},
    }


def _rate(results: list[dict[str, Any]], key: str) -> float:
    if not results:
        return float("nan")
    if key == "success":
        return sum(1 for r in results if r["success"]) / len(results)
    return sum(1 for r in results if float(r["violations"].get(key, {}).get("count", 0)) > 0) / len(
        results
    )


def sweep_axis(
    task: Any,
    axis: str,
    *,
    scales: list[float],
    episodes: int,
    actors: dict[str, Any],
    max_steps: int,
    base_seed: int,
) -> dict[str, Any]:
    import numpy as np

    out: dict[str, Any] = {"axis": axis, "scales": [float(s) for s in scales], "by_actor": {}}
    for actor_name, actor in actors.items():
        per_scale: list[dict[str, Any]] = []
        for si, scale in enumerate(scales):
            spec = _scale_spec(axis, scale)
            results = [
                _episode(task, spec, actor, seed=base_seed + 1000 * si + ep, max_steps=max_steps)
                for ep in range(episodes)
            ]
            per_scale.append(
                {
                    "scale": float(scale),
                    "success_rate": _rate(results, "success"),
                    "violation_rate": {
                        v: _rate(results, v) for v in ("V1", "V2", "V3", "V4", "V5")
                    },
                    "readback": results[0]["readback"],
                }
            )
            print(
                f"  {axis}={scale:<6.3g} {actor_name:<10} success="
                f"{per_scale[-1]['success_rate']:.2f}"
            )
        # Sensitivity: the largest shift in success or in any violation rate.
        succ = [p["success_rate"] for p in per_scale]
        shifts = {"success": float(max(succ) - min(succ)) if succ else float("nan")}
        for v in ("V1", "V2", "V3", "V4", "V5"):
            rates = [p["violation_rate"][v] for p in per_scale]
            shifts[v] = float(max(rates) - min(rates)) if rates else float("nan")
        out["by_actor"][actor_name] = {
            "per_scale": per_scale,
            "shifts": shifts,
            "max_shift": float(np.nanmax(list(shifts.values()))) if shifts else float("nan"),
        }
    return out


def evaluate_g1(cfg: Any, payload: dict[str, Any], *, axes: list[str]) -> dict[str, Any]:
    """Apply the G1 pass criteria from the config; unimplemented == failed."""
    import numpy as np

    gate = cfg.gates.G1
    lo, hi = float(gate.pi_base_success_band[0]), float(gate.pi_base_success_band[1])
    criteria: list[dict[str, Any]] = []

    band_ok = False
    band_note = "no pi_base actor was available in this sweep"
    for axis in axes:
        entry = payload["axes"].get(axis, {}).get("by_actor", {}).get("pi_base")
        if entry:
            nom = _closest_scale(entry["per_scale"], 1.0)
            if nom is not None:
                rate = float(nom["success_rate"])
                band_ok = lo <= rate <= hi
                band_note = f"pi_base nominal success {rate:.2f} (band [{lo:.2f}, {hi:.2f}])"
                break
    criteria.append(
        {
            "name": "pi_base_success_band",
            "passed": bool(band_ok),
            "detail": band_note,
            "unimplemented": "pi_base" not in str(payload.get("actors", [])),
        }
    )

    shifts = [
        (axis, float(payload["axes"][axis]["by_actor"][name]["max_shift"]))
        for axis in axes
        for name in payload["axes"][axis]["by_actor"]
    ]
    shift_ok = any(s >= float(gate.axis_shift_pp) / 100.0 for _ax, s in shifts)
    criteria.append(
        {
            "name": "axis_shift",
            "passed": bool(shift_ok),
            "detail": f"max shift {max((s for _a, s in shifts), default=float('nan')):.3f} "
            f"(threshold {float(gate.axis_shift_pp) / 100.0:.2f}); per-axis "
            f"{ {a: round(s, 3) for a, s in shifts} }",
        }
    )

    coverage_actors = list(getattr(gate, "coverage_actors", None) or ["expert"])

    def _rates_for(names: list[str]) -> dict[str, float]:
        return {
            v: max(
                (
                    float(per["violation_rate"][v])
                    for axis in axes
                    for name in names
                    if name in payload["axes"][axis]["by_actor"]
                    for per in payload["axes"][axis]["by_actor"][name]["per_scale"]
                ),
                default=float("nan"),
            )
            for v in ("V1", "V2", "V3", "V4", "V5")
        }

    # A configured actor may stand for a family: ``perturbed`` runs as ``perturbed@0.2``,
    # ``perturbed@0.4``, ... Matching only exact names made the criterion silently fall back
    # to the expert alone while the perturbed runs sat unread in the same payload.
    def _ran(name: str) -> bool:
        return any(a == name or a.startswith(f"{name}@") for a in payload.get("actors", []))

    present = [a for a in payload.get("actors", []) if any(a.split("@")[0] == n for n in coverage_actors)]
    unknown = [n for n in coverage_actors if not _ran(n)]
    seen = _rates_for(present or list(payload.get("actors", [])))
    occ_ok = all(np.isfinite(r) and r >= float(gate.min_violation_rate) for r in seen.values())
    detail = (
        f"per-type max rates { {k: round(v, 3) for k, v in seen.items()} } "
        f"(each must reach {gate.min_violation_rate}) over actors {present}"
    )
    if unknown:
        # A configured actor that never ran must not look like a clean result: the
        # criterion is then evaluated over the actors that did run, and says so.
        detail += f"; configured actors that did not run: {unknown}"
    criteria.append({"name": "violation_coverage", "passed": bool(occ_ok), "detail": detail})

    rb_ok = True
    rb_notes: list[str] = []
    for axis in axes:
        entry = payload["axes"][axis]["by_actor"]
        # Off-nominal scales must show off-nominal effective physics (read-back).
        for name, block in entry.items():
            for per in block["per_scale"]:
                if abs(per["scale"] - 1.0) < 1e-9:
                    continue
                eff = (per.get("readback") or {}).get("effective", per.get("readback") or {})
                got = float((eff or {}).get(f"{axis}_scale", float("nan")))
                if not np.isfinite(got) or abs(got - per["scale"]) > 1e-6:
                    rb_ok = False
                    rb_notes.append(f"{axis}={per['scale']} read back {got}")
    criteria.append(
        {
            "name": "readback_confirms_physics",
            "passed": bool(rb_ok),
            "detail": "; ".join(rb_notes[:6]) or "every off-nominal scale read back correctly",
        }
    )

    passed = all(c["passed"] for c in criteria)
    return {
        "gate": "G1",
        "passed": passed,
        "criteria": criteria,
        "failures": [c["name"] for c in criteria if not c["passed"]],
        "notes": [
            "On fail: change task (assembly-v3 / hand-insert-v3 / box-close-v3, then ManiSkill3) "
            "before any further work (agents/11)."
        ],
    }


def _coverage_actors(task: Any, cfg: Any, gate: Any) -> dict[str, Any]:
    """Actors named by ``gates.G1.coverage_actors`` beyond the expert (``04`` §8)."""
    from riskproj.envs.actors import PerturbedExpertActor

    wanted = [str(a) for a in (getattr(gate, "coverage_actors", None) or ["expert"])]
    out: dict[str, Any] = {}
    if not any(a.startswith("perturbed") for a in wanted):
        return out
    sigmas = [float(s) for s in (getattr(gate, "perturbed_sigmas", None) or [0.2])]
    low, high = _action_bounds(task, cfg)
    horizon = int(getattr(cfg.env, "exec_horizon", 10) or 10)
    for sigma in sigmas:
        name = f"perturbed@{sigma:g}"
        # ``perturbed`` without a sigma means "every sigma in the gate's list".
        if name not in wanted and "perturbed" not in wanted:
            continue
        out[name] = PerturbedExpertActor(
            expert=task.expert(),
            sigma=sigma,
            horizon=horizon,
            seed=int(cfg.seed),
            action_low=low,
            action_high=high,
        )
    return out


def _action_bounds(task: Any, cfg: Any) -> tuple[Any, Any]:
    """The env's action box, read from the live model rather than assumed."""
    env = task.make_env(cfg.dynamics.nominal, seed=0)
    try:
        space = getattr(env.unwrapped, "action_space", None)
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        return low, high
    finally:
        env.close()


def _closest_scale(per_scale: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    if not per_scale:
        return None
    return min(per_scale, key=lambda p: abs(float(p["scale"]) - target))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sensitivity_sweep.py", description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--axis", default="all", choices=(*AXES, "all"))
    parser.add_argument("--scales", type=int, default=None, help="default: gates.G1.n_scales")
    parser.add_argument("--episodes", type=int, default=None, help="default: gates.G1.episodes_per_scale")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--include-policy", action="store_true", help="also sweep pi_base")
    parser.add_argument("--out", default=str(ROOT / "runs" / "_g1" / "sensitivity.json"))
    ns = parser.parse_args(argv)

    import numpy as np

    from riskproj.envs import build_task
    from riskproj.utils.config import load_config

    cfg = load_config(ns.config)
    task = build_task(cfg)
    gate = cfg.gates.G1
    n_scales = int(ns.scales or gate.n_scales)
    episodes = int(ns.episodes or gate.episodes_per_scale)
    max_steps = int(ns.max_steps or cfg.env.max_steps)
    axes = list(AXES) if ns.axis == "all" else [ns.axis]

    actors: dict[str, Any] = {"expert": task.expert()}
    for name, actor in _coverage_actors(task, cfg, gate).items():
        actors[name] = actor
    if ns.include_policy:
        try:
            from riskproj.policies import build_policy, is_available

            if is_available(cfg.model.name):
                policy = build_policy(cfg.model)
                actors["pi_base"] = _PolicyActor(policy, cfg)
            else:
                print(
                    f"[sensitivity] pi_base unavailable ({cfg.model.name} deps missing); the band "
                    "criterion will FAIL as unimplemented, which is the honest outcome",
                    file=sys.stderr,
                )
        except Exception as exc:  # a missing checkpoint must not look like a pass
            print(f"[sensitivity] pi_base could not be built: {exc}", file=sys.stderr)

    started = time.perf_counter()
    axes_payload: dict[str, Any] = {}
    for axis in axes:
        scales = np.geomspace(float(gate.scale_lo), float(gate.scale_hi), n_scales)
        print(f"[sensitivity] axis {axis}: {n_scales} scales, {episodes} episodes each")
        axes_payload[axis] = sweep_axis(
            task,
            axis,
            scales=[float(s) for s in scales],
            episodes=episodes,
            actors=actors,
            max_steps=max_steps,
            base_seed=int(cfg.seed),
        )

    coverage_actors = [str(a) for a in (getattr(gate, "coverage_actors", None) or ["expert"])]
    payload = {
        "config": str(ns.config),
        "config_hash": cfg.config_hash(),
        "task": str(cfg.env.task),
        "actors": sorted(actors),
        "coverage_actors": coverage_actors,
        "perturbed_sigmas": [float(s) for s in (getattr(gate, "perturbed_sigmas", None) or [])],
        "episodes_per_scale": episodes,
        "n_scales": n_scales,
        "axes": axes_payload,
        "seconds": round(time.perf_counter() - started, 1),
    }
    report = evaluate_g1(cfg, payload, axes=axes)
    payload["gate_report"] = report

    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"[sensitivity] wrote {out}")
    return 0 if report["passed"] else 1


class _PolicyActor:
    """Adapts a policy to the ``act(obs)`` interface the sweep drives."""

    def __init__(self, policy: Any, cfg: Any) -> None:
        self.policy = policy
        self.cfg = cfg

    def act(self, obs: dict[str, Any]) -> Any:
        import torch

        state = torch.as_tensor(obs["state"], dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            out = self.policy.forward({"state": state}, site=self.cfg.model.site.default)
        return out.chunk[0, 0].numpy()


if __name__ == "__main__":
    raise SystemExit(main())
