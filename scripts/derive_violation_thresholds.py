#!/usr/bin/env python
"""Derive violation thresholds from **nominal scripted-expert rollouts only**.

``04`` §4 is explicit: the thresholds come from nominal expert behaviour, are
computed before any learned policy is evaluated, and are then frozen. Deriving them
from a learned policy, or from a perturbed cell, would make the labels a function of
the thing being measured — so this script refuses to run with a learned checkpoint in
the loop and records what it used in ``provenance``.

    python scripts/derive_violation_thresholds.py --config configs/exp/e0_baseline_repro.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_OUT = ROOT / "configs" / "violations_derived.yaml"

#: Percentile rule per event, straight from ``04`` §4.
PCT = 99.5
MARGIN = {"V1": 1.5, "V2": 1.5, "V3": 1.5, "V4": 1.0, "V5": 1.5}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="derive_violation_thresholds.py", description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    parser.add_argument(
        "--allow-learned",
        action="store_true",
        help="permit a learned policy (INVALID for the frozen thresholds; diagnostic only)",
    )
    ns = parser.parse_args(argv)

    import numpy as np

    from riskproj.envs import build_task
    from riskproj.utils.config import load_config
    from riskproj.utils.io import git_sha

    cfg = load_config(ns.config)
    if not ns.allow_learned and cfg.model.checkpoint:
        print(
            f"[derive-thresholds] ignoring the configured checkpoint {cfg.model.checkpoint!r}: "
            "thresholds must come from the scripted expert at nominal dynamics (04 §4). "
            "Pass --allow-learned only for a diagnostic comparison.",
            file=sys.stderr,
        )
    task = build_task(cfg)
    expert = task.expert()
    max_steps = int(ns.max_steps or cfg.env.max_steps)

    window = int(cfg.violations.window)
    slip, drop, force, pen = [], [], [], []
    jam_force_windows: list[float] = []
    for ep in range(ns.episodes):
        # Nominal dynamics, nominal cell: the reference behaviour.
        env = task.make_env(cfg.dynamics.nominal, seed=ep)
        obs, _info = env.reset(seed=ep)
        # Mirrors ViolationMonitor's own bookkeeping: the slip reference is the grip
        # offset recorded at each *new* grasp, and both windows restart then, so the
        # percentiles describe the same quantities the monitor later thresholds.
        ref: Any = None
        drift_win: deque[float] = deque(maxlen=max(1, window))
        dist_win: deque[float] = deque(maxlen=max(1, window))
        grasp_established = False
        prev_established = False
        for _ in range(max_steps):
            obs, _r, term, trunc, info = env.step(expert.act(obs))
            force.append(float(info.get("force", 0.0)))
            pen.append(float(info.get("penetration", 0.0)))
            grasped = bool(info.get("grasped"))
            grip_offset = np.asarray(info.get("grip_offset"), dtype=np.float64).reshape(-1)
            if grip_offset.shape[0] != 3:
                raise SystemExit(
                    "the env returned no 3-D grip_offset; run inspect_contacts/S1 first "
                    "(04 §4 thresholds need the gripper-frame offset)"
                )
            dist = float(np.linalg.norm(grip_offset))
            if grasped:
                if ref is None:
                    ref = grip_offset.copy()
                    drift_win.clear()
                drift_win.append(float(np.linalg.norm(grip_offset - ref)))
            else:
                ref = None
                drift_win.append(0.0)
            if grasped:
                slip.append(max(drift_win))
            if grasped:
                grasp_established = True
            if grasp_established and not prev_established:
                dist_win.clear()
            prev_established = grasp_established
            dist_win.append(dist)
            if grasp_established:
                drop.append(max(dist_win))
            jam_force_windows.append(float(info.get("force", 0.0)))
            if term or trunc:
                break

    if not force:
        raise SystemExit("no steps were collected; the expert never produced a step")

    def pct(values: list[float], q: float = PCT) -> float:
        return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0

    derived = {
        "slip_drift": pct(slip) * MARGIN["V1"],
        "drop_dist": pct(drop) * MARGIN["V2"],
        "force_max": pct(force) * MARGIN["V3"],
        "penetration": max(pct(pen) * MARGIN["V4"], 1e-6),
        "jam_force": pct(jam_force_windows) * MARGIN["V5"],
        "jam_progress_eps": float(cfg.violations.jam_progress_eps),
        "jam_steps": int(cfg.violations.jam_steps),
        "window": int(cfg.violations.window),
        "horizon": int(cfg.env.critic_horizon),
        "provenance": {
            "status": "derived",
            "generator": "scripts/derive_violation_thresholds.py",
            "generator_version": 1,
            "source": "scripted expert, nominal dynamics, nominal cell ONLY",
            "n_episodes": ns.episodes,
            "n_steps": len(force),
            "percentile": PCT,
            "margin": MARGIN,
            # Absolute depth for V4 keeps the bar stiffness-independent (04 §4).
            "v4_is_absolute_depth": True,
            "learned_checkpoint_in_loop": bool(ns.allow_learned and cfg.model.checkpoint),
            "config_hash": cfg.config_hash(),
            "git_sha": str(git_sha().get("sha", "")),
        },
    }
    print(json.dumps(derived, indent=2, default=str))
    if derived["provenance"]["learned_checkpoint_in_loop"]:
        print(
            "[derive-thresholds] WARNING: a learned checkpoint was involved; these thresholds are "
            "NOT valid for the frozen protocol (04 §4).",
            file=sys.stderr,
        )
    if ns.dry_run:
        return 0

    import yaml

    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump({"violations": derived}, sort_keys=True), encoding="utf-8"
    )
    print(f"[derive-thresholds] wrote {out} — review, then copy the values into "
          "configs/violations.yaml and freeze them before any policy evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
