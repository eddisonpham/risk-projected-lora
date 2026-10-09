#!/usr/bin/env python
"""Run one experiment config: the single entry point for every phase.

    python scripts/run_experiment.py --config configs/exp/e4_main.yaml --dry-run
    python scripts/run_experiment.py --config configs/exp/toy_smoke.yaml --method B3 --seed 0
    python scripts/run_experiment.py --config configs/exp/e4_main.yaml --shard 0/4

Properties that matter for a research repo (``skills/software_engineering.md`` §1, §3):

* the run directory is ``runs/<name>-<config-hash>/`` and a run is **idempotent by
  hash**: a method/seed whose ``summary.json`` exists is skipped unless ``--force``,
  so an interrupted sweep resumes instead of silently re-rolling;
* provenance (git sha, pip freeze, gpu, seeds, data-manifest hash) is captured
  before anything trains;
* TEST splits are refused unless ``--allow-test`` is passed, with a reminder that
  the prereg tag must exist first (``CLAUDE.md`` #2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _split_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _cells_for(cfg: Any, split: str) -> list[str]:
    from riskproj.data.splits import cells_for_role

    return cells_for_role(split, cfg.splits)


def build_plan(cfg: Any, *, methods: list[str] | None, seeds: list[int] | None) -> dict[str, Any]:
    """Resolve the config into the concrete work list, without touching disk."""
    chosen_methods = [m for m in cfg.methods if methods is None or m.id in methods]
    if not chosen_methods:
        raise SystemExit(f"no methods selected; config has {[m.id for m in cfg.methods]}")
    chosen_seeds = seeds or cfg.seed_list()
    per_split: dict[str, list[str]] = {
        split: _cells_for(cfg, split) for split in cfg.eval.splits
    }
    cells_total = sum(len(v) for v in per_split.values())
    episodes = cells_total * int(cfg.eval.eps_per_cell)
    jobs: list[dict[str, Any]] = []
    for method in chosen_methods:
        for seed in chosen_seeds:
            jobs.append(
                {
                    "method": method.id,
                    "train_seed": int(seed),
                    "inference_mode": method.inference_mode,
                    "train_mode": method.train_mode,
                    "episodes": episodes,
                }
            )
    return {
        "name": cfg.name,
        "config_hash": cfg.config_hash(),
        "run_dir": str(ROOT / cfg.out_dir / cfg.run_name()),
        "splits": {k: len(v) for k, v in per_split.items()},
        "cells": per_split,
        "seeds": [int(s) for s in chosen_seeds],
        "methods": [m.id for m in chosen_methods],
        "episodes_per_job": episodes,
        "total_episodes": episodes * len(jobs),
        "jobs": jobs,
        "prereg_tag": cfg.prereg_tag,
    }


def _shard(items: list[Any], shard: str | None) -> list[Any]:
    if not shard:
        return items
    try:
        index, total = (int(x) for x in shard.split("/"))
    except ValueError as exc:
        raise SystemExit(f"--shard expects i/n; got {shard!r}") from exc
    if total < 1 or not 0 <= index < total:
        raise SystemExit(f"--shard {shard!r} is out of range")
    return [item for i, item in enumerate(items) if i % total == index]


def run_job(
    cfg: Any,
    *,
    method_cfg: Any,
    train_seed: int,
    run_dir: Path,
    task: Any,
    allow_test: bool,
    force: bool,
    critic: Any | None = None,
    allow_untrained: bool = False,
) -> dict[str, Any]:
    """Train (if the method trains) and evaluate one method x seed.

    Evaluation goes through ``evaluate_method``'s **planner** path, driven by the
    method's own ``inference_mode`` (``08`` §2). An arm whose required artifacts do not
    exist yet is refused with the reason: an earlier version passed the method's name
    with the no-projection planner, so every arm would have measured the baseline while
    the run directory claimed ``B3`` — no correction, no decision logs, no error.

    ``allow_untrained`` (set only by ``--smoke``) evaluates whatever is in memory and
    stamps the summary as a wiring check, because **every** arm needs a Stage-A policy
    and the projecting arms also need a trained critic: without those, a number here is
    a number about a randomly initialised network.
    """
    summary_path = run_dir / method_cfg.id / str(train_seed) / "summary.json"
    if summary_path.exists() and not force:
        return {"status": "skipped_existing", "summary": str(summary_path)}

    from riskproj.eval.rollout import evaluate_method
    from riskproj.policies import build_policy_for_env

    if not allow_untrained and method_cfg.id != "B0":
        raise RuntimeError(
            f"{method_cfg.id} needs training (Stage A/B), but policy training is not wired: "
            "``MethodBase.train`` consumes a batch-provider callable and no demo batch provider "
            "exists yet (OPEN QUESTIONS, agents/research_log/LOG.md). Use --smoke for a wiring "
            "check; do **not** evaluate an untrained policy and label it this arm."
        )
    if not allow_untrained and critic is None and method_cfg.inference_mode == "0":
        raise RuntimeError(
            f"{method_cfg.id} must be evaluated on a Stage-A finetuned policy, and none is "
            "available yet (P2 has not run). Use --smoke for a wiring check."
        )
    if method_cfg.inference_mode != "0" and critic is None:
        raise RuntimeError(
            f"{method_cfg.id} intervenes at inference in mode "
            f"{method_cfg.inference_mode!r}, which is defined by the critic, but no critic was "
            "supplied: pass --critic <run>/critic.pt. Without it this job would measure the "
            "baseline under the method's name."
        )

    import torch

    ranges = cfg.dynamics.to_ranges()
    # Geometry comes from the env: the adapter must match what the task produces. The
    # device is decided here so the policy and the critic always share one — a CUDA
    # critic scored a CPU chunk once already, and that failure is not worth rediscovering.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = build_policy_for_env(cfg, device=device)
    if hasattr(policy, "to"):
        policy.to(device)

    splits = [s for s in cfg.eval.splits if allow_test or not s.startswith("test")]
    refused = sorted(set(cfg.eval.splits) - set(splits))
    if refused and not allow_test:
        print(f"[exp] REFUSING test splits {refused} without --allow-test (CLAUDE.md #2)")

    spec_kwargs = {
        "max_steps": cfg.eval.max_steps or int(cfg.env.max_steps),
        "exec_horizon": int(cfg.env.exec_horizon),
        "record_decisions": bool(cfg.eval.record_decisions),
        "save_states": bool(cfg.eval.save_states),
        "save_every": int(cfg.eval.state_save_every),
        # T3 (train x inference factorial) reads these from the episode's extra block.
        "record_extra": {
            "train_mode": method_cfg.train_mode,
            "inference_mode": method_cfg.inference_mode,
            "method_id": method_cfg.id,
        },
    }

    results: dict[str, Any] = {}
    if allow_untrained:
        results["wiring_check"] = {
            "untrained_policy": True,
            "note": (
                "--smoke: this arm was evaluated without its Stage-A policy (and without a "
                "critic unless --critic was passed). The numbers are a plumbing check and "
                "must never be reported as evidence (12 §claim ladder)."
            ),
        }
    for split in splits:
        results[split] = evaluate_method(
            task,
            policy,
            cells=_cells_for(cfg, split),
            eps_per_cell=int(cfg.eval.eps_per_cell),
            base_seed=int(train_seed),
            out_dir=run_dir / method_cfg.id / str(train_seed),
            run_id=split,
            method=method_cfg.id,
            train_seed=int(train_seed),
            spec_kwargs=spec_kwargs,
            critic=critic,
            # The baseline arms carry a default ProjCfg in every config; passing it to a
            # no-projection planner is meaningless, so only the projecting modes get it.
            proj_cfg=method_cfg.projection if method_cfg.inference_mode != "0" else None,
            inference_mode=method_cfg.inference_mode,
            best_of_n=int(method_cfg.best_of_n),
            ranges=ranges,
            split=split,
            config=cfg.canonical(),
            hist_len=max(1, int(cfg.critic.use_history)),
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return {"status": "ok", "summary": str(summary_path), "result": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_experiment.py", description=__doc__)
    parser.add_argument("--config", required=True, help="path to configs/exp/*.yaml")
    parser.add_argument("--method", default="", help="comma-separated method ids (default: all)")
    parser.add_argument("--seed", default="", help="comma-separated training seeds (default: config)")
    parser.add_argument("--shard", default=None, help="i/n — run every n-th job")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--smoke", action="store_true", help="tiny run: 1 episode per cell")
    parser.add_argument("--allow-test", action="store_true", help="permit TEST splits (needs prereg)")
    parser.add_argument("--force", action="store_true", help="re-run jobs whose summary exists")
    parser.add_argument(
        "--critic",
        default="",
        help="path to the frozen critic checkpoint (critic.pt); required by any arm that "
        "projects, and refused loudly when missing rather than silently measuring B0",
    )
    ns = parser.parse_args(argv)

    from riskproj.utils.config import dump_config, load_config

    cfg = load_config(ns.config)
    methods = _split_list(ns.method) or None
    seeds = [int(s) for s in _split_list(ns.seed)] or None
    if ns.smoke:
        # One episode per cell: a wiring check, not a measurement. Rebuild the plan
        # so the printed episode counts match what will actually run.
        cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"eps_per_cell": 1})})
    plan = build_plan(cfg, methods=methods, seeds=seeds)
    plan["jobs"] = _shard(plan["jobs"], ns.shard)
    if ns.smoke:
        plan["note"] = "--smoke: 1 episode per cell; do not read these numbers as evidence"

    print(json.dumps(plan, indent=2, default=str))
    if ns.dry_run:
        return 0

    from riskproj.envs import build_task
    from riskproj.utils.io import capture_provenance, manifest_hash

    run_dir = ROOT / cfg.out_dir / cfg.run_name()
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    data_manifest = ROOT / "data" / "MANIFEST.json"
    capture_provenance(
        run_dir,
        config=cfg.canonical(),
        seeds={"seeds": plan["seeds"], "env_seed_base": cfg.seed},
        data_manifest_hash=manifest_hash(data_manifest) if data_manifest.exists() else "no-manifest",
        extra={"plan": plan, "argv": sys.argv[1:]},
    )

    critic = None
    if ns.critic:
        import torch

        from riskproj.critic.model import RiskCritic

        critic = RiskCritic.from_checkpoint(
            ns.critic, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        critic.eval()
        print(f"[exp] critic loaded from {ns.critic} (frozen, eval mode)")

    task = build_task(cfg)
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for job in plan["jobs"]:
        method_cfg = next(m for m in cfg.methods if m.id == job["method"])
        label = f"{job['method']}/seed{job['train_seed']}"
        try:
            outcome = run_job(
                cfg,
                method_cfg=method_cfg,
                train_seed=job["train_seed"],
                run_dir=run_dir,
                task=task,
                allow_test=ns.allow_test,
                force=ns.force,
                critic=critic,
                allow_untrained=bool(ns.smoke),
            )
            print(f"[exp] {label}: {outcome['status']} -> {outcome.get('summary')}")
        except Exception as exc:  # keep going: one broken arm must not hide the others
            failures.append({"job": label, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[exp] {label}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            if ns.force:
                traceback.print_exc()

    elapsed = time.perf_counter() - started
    manifest = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "config_hash": cfg.config_hash(),
        "seconds": round(elapsed, 1),
        "n_jobs": len(plan["jobs"]),
        "failures": failures,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[exp] {len(plan['jobs'])} job(s) in {elapsed:.1f}s -> {run_dir}")
    if failures:
        print(f"[exp] {len(failures)} job(s) failed; see run_summary.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
