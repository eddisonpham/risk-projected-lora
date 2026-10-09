"""Toy env: determinism, obs contract, expert competence, dynamics sensitivity, V-reachability.

These tests are the CI stand-in for the G1 dynamics-sensitivity gate: if the toy
task stops responding to the three axes, the walking skeleton (S6) becomes
vacuous and every downstream toy number is meaningless.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from riskproj.envs import ToyEnv, ToyPhysics, ToyReachTask
from riskproj.envs.state_io import save_state
from riskproj.envs.toy import TOY_VIOLATION_SPEC
from riskproj.types import VIOLATION_TYPES, DynamicsSpec, empty_violation_counts

pytestmark = pytest.mark.unit

N_SEEDS = 12


def run_expert_episode(task: ToyReachTask, dyn: DynamicsSpec, seed: int) -> dict[str, Any]:
    env = task.make_env(dyn, seed)
    expert = task.expert()
    expert.reset()
    obs, _ = env.reset(seed=seed)
    for _ in range(env.max_episode_steps):
        obs, _reward, terminated, truncated, _info = env.step(expert.act(obs))
        if terminated or truncated:
            break
    return env.episode_summary()


def success_rate(task: ToyReachTask, dyn: DynamicsSpec, n: int = N_SEEDS) -> float:
    return sum(bool(run_expert_episode(task, dyn, s)["success"]) for s in range(n)) / n


def run_press_sequence(action: np.ndarray, steps: int = 12) -> dict[str, Any]:
    """Grasp the puck away from the slot, then press it into the table."""
    env = ToyEnv(seed=0)
    env.reset(seed=0)
    st = env.get_state()
    st["p"] = np.array(env.physics.object_start, dtype=float) + np.array([0.0, 0.0, 0.01])
    st["q"] = np.array(env.physics.object_start, dtype=float)
    st["v"] = np.zeros(3)
    st["qv"] = np.zeros(3)
    st["grasped"] = np.asarray(1.0)
    env.set_state(st)
    for _ in range(steps):
        _obs, _r, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    return env.episode_summary()


def test_toy_env_is_deterministic_given_seed_and_actions() -> None:
    actions = [np.array([0.01, 0.0, -0.005])] * 20

    def rollout() -> tuple[np.ndarray, str]:
        env = ToyEnv(seed=7)
        env.reset(seed=7)
        for a in actions:
            obs, *_ = env.step(a)
        return obs["state"], save_state(env).hash()

    s1, h1 = rollout()
    s2, h2 = rollout()
    assert np.array_equal(s1, s2)
    assert h1 == h2


def test_reset_jitter_depends_on_seed() -> None:
    a = ToyEnv(seed=1)
    b = ToyEnv(seed=2)
    qa = a.reset(seed=1)[0]["object_pose"]
    qb = b.reset(seed=2)[0]["object_pose"]
    assert not np.allclose(qa, qb)


def test_observation_contract() -> None:
    env = ToyEnv(seed=3)
    obs, info = env.reset(seed=3)
    for key in ("state", "eef_vel", "images", "instruction", "object_pose", "eef_pose"):
        assert key in obs, f"missing obs key {key}"
    assert obs["state"].shape == (env.state_dim,)
    assert obs["eef_vel"].shape == (3,)
    assert isinstance(obs["images"], dict) and "top" in obs["images"]
    assert obs["images"]["top"].dtype == np.uint8
    assert isinstance(obs["instruction"], str) and obs["instruction"]
    assert np.isfinite(obs["state"]).all()
    assert "dyn_readback" in info


def test_action_validation_fails_loud() -> None:
    env = ToyEnv(seed=0)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros(2))
    with pytest.raises(ValueError):
        env.step(np.array([np.nan, 0.0, 0.0]))


def test_step_after_done_raises() -> None:
    env = ToyReachTask(max_episode_steps=2).make_env(DynamicsSpec(), seed=0)
    env.reset(seed=0)
    for _ in range(2):
        env.step(np.zeros(3))
    with pytest.raises(RuntimeError):
        env.step(np.zeros(3))


def test_action_is_clipped_to_the_actuation_limit() -> None:
    physics = ToyPhysics()
    env = ToyEnv(physics, seed=0)
    env.reset(seed=0)
    before = env.get_state()["p"].copy()
    env.step(np.array([10.0, 0.0, 0.0]))
    after = env.get_state()["p"]
    assert float(np.linalg.norm(after - before)) <= physics.max_delta + 1e-9


def test_expert_solves_the_task_at_nominal_dynamics() -> None:
    task = ToyReachTask()
    rate = success_rate(task, DynamicsSpec())
    assert rate >= 0.8, f"scripted expert only solved {rate:.0%} of nominal episodes"


def test_nominal_expert_is_clean() -> None:
    """The reference controller must not manufacture contact violations."""
    task = ToyReachTask()
    summaries = [run_expert_episode(task, DynamicsSpec(), s) for s in range(N_SEEDS)]
    for v in ("V3", "V4", "V5"):
        fired = sum(float(r["violations"][v]["count"]) > 0 for r in summaries)
        assert fired == 0, f"{v} fired in {fired}/{N_SEEDS} nominal expert episodes"
    dirty = sum(bool(r["any_violation"]) for r in summaries) / N_SEEDS
    assert dirty <= 0.25, f"nominal expert trips a violation in {dirty:.0%} of episodes"


def test_friction_axis_moves_success() -> None:
    """G1 analogue: the friction axis must shift success by >= 15pp on the toy task."""
    task = ToyReachTask()
    low = success_rate(task, DynamicsSpec(mu_scale=0.5))
    high = success_rate(task, DynamicsSpec(mu_scale=2.0))
    assert high - low >= 0.15, f"mu_scale 0.5 -> {low:.0%} vs 2.0 -> {high:.0%}: no sensitivity"


def test_mass_axis_moves_success() -> None:
    task = ToyReachTask()
    nominal = success_rate(task, DynamicsSpec())
    heavy = success_rate(task, DynamicsSpec(mass_scale=2.5))
    assert nominal - heavy >= 0.15, f"mass_scale 2.5 -> {heavy:.0%} vs nominal {nominal:.0%}"


def test_grip_load_ratio_explains_the_friction_effect() -> None:
    """The mechanism, not just the outcome: low mu must overload the grip."""
    results = {}
    for mu in (0.5, 2.0):
        env = ToyReachTask().make_env(DynamicsSpec(mu_scale=mu), seed=0)
        expert = ToyReachTask().expert()
        obs, _ = env.reset(seed=0)
        dropped = False
        for _ in range(env.max_episode_steps):
            obs, _r, terminated, truncated, info = env.step(expert.act(obs))
            dropped = dropped or bool(info["dropped"])
            assert np.isfinite(info["grip_load_ratio"])
            if terminated or truncated:
                break
        results[mu] = dropped
    assert results[0.5] is True, "mu_scale=0.5 should overload and break the grip"
    assert results[2.0] is False, "mu_scale=2.0 should hold the puck for the whole episode"


def test_stiffness_axis_changes_contact_force_and_penetration() -> None:
    """Stiffer contact => same press produces more force and less penetration."""
    press = np.array([0.0, 0.0, -0.02])
    peaks: list[tuple[float, float]] = []
    for stiff in (0.25, 4.0):
        env = ToyEnv(seed=0)
        env.apply_dynamics(DynamicsSpec(stiff_scale=stiff))
        env.reset(seed=0)
        st = env.get_state()
        st["p"] = np.array(env.physics.object_start, dtype=float) + np.array([0.0, 0.0, 0.01])
        st["q"] = np.array(env.physics.object_start, dtype=float)
        st["grasped"] = np.asarray(1.0)
        env.set_state(st)
        peak_force = 0.0
        peak_pen = 0.0
        for _ in range(8):
            _obs, _r, terminated, truncated, info = env.step(press)
            peak_force = max(peak_force, float(info["force"]))
            peak_pen = max(peak_pen, float(info["penetration"]))
            if terminated or truncated:
                break
        peaks.append((peak_force, peak_pen))
        assert env.effective_dynamics()["effective"]["stiff_scale"] == pytest.approx(stiff)
    (soft_force, soft_pen), (stiff_force, stiff_pen) = peaks
    assert stiff_force > soft_force, f"stiffer contact gave less force: {peaks}"
    assert soft_pen >= stiff_pen, f"softer contact should not penetrate less: {peaks}"


def test_all_violation_types_are_reachable_in_the_toy_env() -> None:
    """A grasp-and-press policy must be able to reach every event (D-crit needs this)."""
    summary = run_press_sequence(np.array([0.0, 0.0, -0.02]))
    fired = {v for v, entry in summary["violations"].items() if float(entry["count"]) > 0}
    assert fired == set(VIOLATION_TYPES), f"only {sorted(fired)} of {sorted(VIOLATION_TYPES)} fired"


def test_do_nothing_episode_truncates_and_is_logged() -> None:
    task = ToyReachTask(max_episode_steps=25)
    env = task.make_env(DynamicsSpec(), seed=0)
    _obs, _ = env.reset(seed=0)
    for _ in range(env.max_episode_steps):
        _obs, _r, terminated, truncated, _info = env.step(np.zeros(3))
        if terminated or truncated:
            break
    summary = env.episode_summary()
    assert summary["success"] is False
    assert summary["steps"] == 25
    assert summary["failure_class"] in {"idle_timeout", "stop_short"}


def test_episode_summary_record_shape() -> None:
    summary = run_expert_episode(ToyReachTask(), DynamicsSpec(), 0)
    expected = empty_violation_counts()
    assert set(summary["violations"]) == set(VIOLATION_TYPES)
    for key, value in summary["violations"].items():
        assert set(value) == set(expected[key])
        assert isinstance(value["first_step"], float)
        assert isinstance(value["count"], float)
    assert isinstance(summary["failure_class"], str)
    assert set(summary["dyn_readback"]) == {"mu_scale", "mass_scale", "stiff_scale"}


def test_cell_id_is_metadata_only() -> None:
    from riskproj.envs.toy import set_cell_id

    env = ToyEnv(seed=0)
    set_cell_id(env, "H-L-M")
    obs, _ = env.reset(seed=0)
    assert env._cell_id == "H-L-M"
    assert env.episode_summary()["cell_id"] == "H-L-M"
    assert "H-L-M" not in str(obs)  # never leaks into the policy-visible observation


def test_toy_spec_uses_toy_units() -> None:
    assert str(TOY_VIOLATION_SPEC.provenance.get("scale", "")).startswith("toy")
    task = ToyReachTask()
    assert task.name == "toy"
    assert task.violations is TOY_VIOLATION_SPEC


def test_support_height_switches_on_alignment() -> None:
    env = ToyEnv(seed=0)
    env.reset(seed=0)
    assert env.is_aligned is False
    assert env.support_height(env.physics.object_start[:2]) == pytest.approx(env.physics.table_z)
    assert env.is_aligned is False
    assert env.support_height(np.array(env.physics.target[:2])) == pytest.approx(
        env.physics.hole_floor_z
    )
