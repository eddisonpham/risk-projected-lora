"""End-to-end pipeline test: rollout -> logs -> tables -> report, on the toy task.

This is the walking skeleton of `agents/03` §3 S6 at the evaluation layer: the real
driver, the real planner, the real projection operator and the real critic, over the
real toy environment. It asserts the two things that matter for later phases — that
the ``agents/09`` §3 record schema is actually filled in, and that the report is
derived from the logs rather than from live objects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from riskproj.critic import FeatureSpec, RiskCritic
from riskproj.data.splits import cells_for_role
from riskproj.envs import ToyReachTask
from riskproj.eval import (
    NoProjectionPlanner,
    RolloutSpec,
    build_planner,
    build_tables,
    cell_episode_plan,
    evaluate_method,
    paired_counterfactual,
    run_episode,
    write_report,
)
from riskproj.policies.toy import ToyPolicy
from riskproj.types import DynamicsSpec
from riskproj.utils.config import CriticCfg, ModelCfg, ProjCfg

pytestmark = pytest.mark.integration

STATE_DIM = 14
ACTION_DIM = 3
HORIZON = 10
CHUNK = 10
#: Every field of the episode schema in `agents/09` §3.
EPISODE_SCHEMA_KEYS = (
    "run_id",
    "method",
    "train_seed",
    "env_seed",
    "cell_id",
    "split",
    "dyn_readback",
    "success",
    "steps",
    "t_success",
    "violations",
    "failure_class",
    "fallback",
    "stall_steps",
    "action_norm_mean",
    "jerk",
    "proj",
    "latency_ms",
    "peak_vram_mb",
)


def toy_policy(seed: int = 0) -> ToyPolicy:
    """The toy adapter built directly, so this test does not depend on the registry path."""
    return ToyPolicy(
        ModelCfg(name="toy", chunk_size=CHUNK, num_flow_steps=10),
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden=16,
        chunk_horizon=CHUNK,
        seed=seed,
        device="cpu",
    )


def toy_critic(seed: int = 0) -> RiskCritic:
    spec = FeatureSpec(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        chunk_horizon=CHUNK,
        hist_len=1,
        vision_dim=0,
        force_dim=0,
        use_vision=False,
        privileged_force=False,
    )
    critic = RiskCritic(CriticCfg(hidden=16, depth=2, ensemble=2, seed=seed), spec)
    return critic.eval()


def proj_cfg() -> ProjCfg:
    return ProjCfg(tau=0.5, eta=1e-8, K=1, space="prob", backward="projected")


def make_spec(**over: object) -> RolloutSpec:
    base: dict[str, object] = {
        "method": "B3",
        "train_seed": 0,
        "cell": "L-L-M",
        "split": "val",
        "env_seed": 0,
        "max_steps": 20,
        "exec_horizon": HORIZON,
        "record_decisions": True,
        "save_states": True,
        "save_every": 1,
        "crn_base_seed": 0,
        "run_id": "itest",
    }
    base.update(over)
    return RolloutSpec(**base)  # type: ignore[arg-type]


def test_run_episode_fills_the_schema_and_applies_dynamics() -> None:
    task = ToyReachTask()
    policy = toy_policy()
    critic = toy_critic()
    dyn = DynamicsSpec(mu_scale=0.6, mass_scale=1.8, stiff_scale=0.4)
    planner = build_planner(
        "lat",
        proj_cfg=proj_cfg(),
        critic=critic,
        site="S2",
        horizon=HORIZON,
    )
    out = run_episode(
        task, policy, make_spec(), planner=planner, critic=critic, proj_cfg=proj_cfg(), dyn=dyn
    )
    record = out.episode.to_json()
    missing = [k for k in EPISODE_SCHEMA_KEYS if k not in record]
    assert not missing, f"episode record is missing {missing}"

    # dynamics actually changed: a silent no-op would invalidate every number
    assert record["dyn_readback"]["mu_scale"] == pytest.approx(0.6)
    assert record["dyn_readback"]["mass_scale"] == pytest.approx(1.8)
    assert record["dyn_readback"]["stiff_scale"] == pytest.approx(0.4)

    assert record["steps"] > 0
    assert set(record["violations"]) == {"V1", "V2", "V3", "V4", "V5"}
    assert set(record["latency_ms"]) == {"policy", "critic", "proj"}
    assert record["extra"]["planner"] == "latent"
    assert out.states, "save_states=True must produce states for the E1 counterfactual"
    # CRN is keyed by the base seed, so it must be recorded for reproducibility
    assert record["extra"]["crn_base_seed"] == 0


def test_projection_events_are_logged_with_risk_before_and_after() -> None:
    task = ToyReachTask()
    policy = toy_policy()
    critic = toy_critic()
    planner = build_planner("lat", proj_cfg=proj_cfg(), critic=critic, site="S2", horizon=HORIZON)
    out = run_episode(
        task,
        policy,
        make_spec(),
        planner=planner,
        critic=critic,
        proj_cfg=proj_cfg(),
        dyn=DynamicsSpec(),
    )
    assert out.decisions, "the projection path must log a decision per flow step"
    for decision in out.decisions:
        assert decision.ep_id.endswith(":0") or ":itest:" in decision.ep_id
        assert decision.r_cand is not None, "r_cand must be logged (CLAUDE.md #4)"
        assert decision.r_after, "the nonlinear post-projection risk must be logged"
        assert decision.r_pred, "the linearized prediction must be logged next to it"
        assert decision.extra["site"] == "S2"
    summary = out.episode.proj
    assert summary["n_dec"] == float(len(out.decisions))
    assert summary["n_active"] >= 0.0


def test_no_projection_planner_matches_the_policy_chunk() -> None:
    task = ToyReachTask()
    policy = toy_policy()
    baseline = NoProjectionPlanner(horizon=HORIZON)
    out = run_episode(
        task,
        policy,
        make_spec(),
        planner=baseline,
        dyn=DynamicsSpec(),
        critic=None,
        proj_cfg=None,
    )
    assert out.decisions == [], "B0/B1 must not emit projection decisions"
    assert out.episode.proj["n_dec"] == 0.0


def test_cell_episode_plan_is_deterministic_and_seeded() -> None:
    cells = cells_for_role("val")[:2]
    first = cell_episode_plan(cells, 2, base_seed=7)
    second = cell_episode_plan(cells, 2, base_seed=7)
    assert first == second
    assert len(first) == 4
    assert {c for c, _ in first} == set(cells)
    assert len({s for _, s in first}) == 4, "episode seeds must differ within a cell"


def test_evaluate_method_writes_raw_logs_and_the_report_is_derived(tmp_path: Path) -> None:
    task = ToyReachTask()
    policy = toy_policy()
    critic = toy_critic()
    proj = proj_cfg()
    planner = build_planner("lat", proj_cfg=proj, critic=critic, site="S2", horizon=HORIZON)
    cells = cells_for_role("val")[:1]

    summary = evaluate_method(
        task,
        policy,
        cells=cells,
        eps_per_cell=2,
        base_seed=3,
        out_dir=tmp_path,
        run_id="itest-run",
        method="B3",
        train_seed=0,
        planner=planner,
        critic=critic,
        proj_cfg=proj,
        spec_kwargs={
            "max_steps": 20,
            "exec_horizon": HORIZON,
            "record_decisions": True,
            "save_states": True,
        },
        config={"name": "itest-run", "methods": [{"id": "B3", "projection": {"tau": 0.5}}]},
    )
    assert summary["n_episodes"] == 2
    assert summary["n_decisions"] > 0

    folder = Path(summary["episodes_jsonl"]).parent
    assert (folder / "episodes.jsonl").exists()
    assert (folder / "decisions.jsonl").exists()
    assert (folder / "summary.json").exists()
    # provenance is mandatory in every run dir (`03` §6)
    for name in ("git_sha.txt", "pip_freeze.txt", "gpu.txt", "seeds.json"):
        assert (folder / name).exists(), f"missing provenance file {name}"

    run_root = Path(summary["episodes_jsonl"]).parents[2]
    tables = build_tables(run_root)
    assert tables["n_episodes"] == 2
    assert tables["n_decisions"] == summary["n_decisions"]
    assert tables["T2_main"]["table"]["B3"]["val"]["success"]["n"] == 2.0
    assert tables["proj_diagnostics"]["n_decisions"] == summary["n_decisions"]
    assert tables["claim_rung"] in ("L0", "L1", "L2", "L3")

    report = write_report(run_root)
    text = report.read_text(encoding="utf-8")
    assert "**derived artifact**" in text
    assert "guarantees" in text  # the forbidden-language list is carried in the report


def test_counterfactual_pairs_original_and_corrected_chunks() -> None:
    """E1 wiring: same states, two arms, paired summary."""
    task = ToyReachTask()
    policy = toy_policy()
    critic = toy_critic()
    dyn = DynamicsSpec(mu_scale=0.7)
    out = run_episode(
        task,
        policy,
        make_spec(max_steps=15),
        planner=NoProjectionPlanner(horizon=HORIZON),
        dyn=dyn,
    )
    assert out.states, "the counterfactual test needs saved states"

    corrected = build_planner("lat", proj_cfg=proj_cfg(), critic=critic, site="S2", horizon=HORIZON)
    summary = paired_counterfactual(
        task,
        policy,
        [(f"s{i}", state) for i, state in enumerate(out.states[:2])],
        planner=corrected,
        baseline_planner=NoProjectionPlanner(),
        critic=critic,
        proj_cfg=proj_cfg(),
        horizon=5,
        crn_base_seed=0,
        cell="L-L-M",
        method="B3",
        dyn=dyn,
    )
    assert summary["n_pairs"] == 2
    assert "paired_delta" in summary
    assert summary["paired_delta"]["ci_level"] == 0.95
    assert set(summary["per_type"]) == {"V1", "V2", "V3", "V4", "V5"}
    assert "measured violations" in summary["verdict"]


def test_counterfactual_refuses_a_single_arm() -> None:
    task = ToyReachTask()
    policy = toy_policy()
    planner = NoProjectionPlanner(horizon=HORIZON)
    out = run_episode(task, policy, make_spec(max_steps=12), planner=planner, dyn=DynamicsSpec())
    assert out.states
    with pytest.raises(ValueError, match="different planners"):
        paired_counterfactual(
            task,
            policy,
            out.states[:1],
            planner=planner,
            baseline_planner=planner,
            horizon=3,
        )


def test_do_nothing_baseline_scores_zero_action_episodes() -> None:
    from riskproj.eval.anti_gaming import do_nothing_baseline

    out = do_nothing_baseline(
        ToyReachTask(), ["M-M-M"], episodes=2, base_seed=0, max_steps=20, action_dim=ACTION_DIM
    )
    assert out["episodes"] == 2
    assert 0.0 <= out["success_rate"] <= 1.0
    assert isinstance(out["any_violation_rate"], float)
    assert not np.isnan(out["any_violation_rate"])
