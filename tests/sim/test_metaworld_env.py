"""Meta-World v3 smoke checks: this file *is* smoke tests S1 and S2 (``03`` §3).

Marked ``sim``: it needs gymnasium + metaworld + a discovered
``configs/env/task_geoms.yaml``. It skips cleanly (rather than failing) when the
simulator extra or the discovered geometry file is absent, because the CI path is
CPU-only; ``tasks.py smoke`` reports the skip as an unmet gate rather than a pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from riskproj.envs import MetaWorldTask, states_equal
from riskproj.types import DynamicsSpec

pytestmark = pytest.mark.sim

TASK = "peg-insert-side-v3"
EXPERT_STEPS = 120
ACTION = np.array([0.05, 0.0, 0.0, 1.0], dtype=np.float64)


@pytest.fixture(scope="module")
def task_geoms_path(repo_root: Path) -> Path:
    path = repo_root / "configs" / "env" / "task_geoms.yaml"
    if not path.exists():
        pytest.skip(
            f"{path} not found: task-critical geoms must be discovered by "
            "scripts/inspect_contacts.py before S1 can run"
        )
    return path


@pytest.fixture(scope="module")
def task(task_geoms_path: Path) -> MetaWorldTask:
    pytest.importorskip("metaworld")
    pytest.importorskip("gymnasium")
    return MetaWorldTask(TASK)


def test_s1_task_builds_and_dynamics_read_back(task: MetaWorldTask) -> None:
    """S1: the task builds, the physics really changed, and the read-back is verified."""
    env = task.make_env(DynamicsSpec(mu_scale=1.5, mass_scale=2.0, stiff_scale=4.0), seed=0)
    try:
        eff = env.effective_readback()
        assert eff["mu_scale"] == pytest.approx(1.5, rel=1e-3)
        assert eff["mass_scale"] == pytest.approx(2.0, rel=1e-3)
        assert eff["stiff_scale"] == pytest.approx(4.0, rel=1e-3)
        assert env.dyn_readback.verified is True
        assert env.dyn_readback.changed is True
    finally:
        env.close()


def test_s1_dynamics_are_idempotent_across_resets(task: MetaWorldTask) -> None:
    env = task.make_env(DynamicsSpec(mu_scale=1.5), seed=0)
    try:
        first = env.effective_readback()["mu_scale"]
        for _ in range(3):
            env.reset()
        again = env.effective_readback()["mu_scale"]
        assert first == pytest.approx(again, rel=1e-9), "dynamics must not compound across resets"
    finally:
        env.close()


def test_s1_scripted_expert_solves_the_task_at_nominal_dynamics(task: MetaWorldTask) -> None:
    """S1: the scripted expert must actually solve the task at nominal dynamics."""
    env = task.make_env(DynamicsSpec(), seed=0)
    expert = task.expert()
    successes = 0
    episodes = 3
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=episode)
            expert.reset()
            info: dict[str, object] = {"success": False}
            for _ in range(EXPERT_STEPS):
                obs, _reward, terminated, truncated, info = env.step(expert.act(obs))
                if terminated or truncated:
                    break
            successes += int(bool(info["success"]))
    finally:
        env.close()
    assert successes >= 1, (
        f"the scripted expert solved {successes}/{episodes} nominal episodes; the task, the expert "
        "resolution or the observation contract is wrong (S1 must pass before any modelling)"
    )


def test_s2_state_restore_reproduces_twenty_steps(task: MetaWorldTask) -> None:
    """S2: qpos/qvel/act/mocap pos+quat restore must reproduce a trajectory exactly."""
    env = task.make_env(DynamicsSpec(), seed=3)
    expert = task.expert()
    try:
        obs, _ = env.reset(seed=3)
        for _ in range(5):
            obs, *_ = env.step(expert.act(obs))
        state = env.save_state()

        for _ in range(20):
            obs, *_ = env.step(ACTION)
        after_first = env.save_state()

        env.restore_state(state)
        for _ in range(20):
            obs, *_ = env.step(ACTION)
        after_second = env.save_state()

        assert states_equal(after_first, after_second, tol=1e-12), (
            "state restore did not reproduce the trajectory: counterfactual labels (E1) would be "
            "built on non-reproducible states. Check queued actions, ctrl and qacc_warmstart."
        )
    finally:
        env.close()


def test_grasp_is_contact_and_force_is_the_largest_pair(task: MetaWorldTask) -> None:
    """The two read-backs V1/V3/V5 consume, measured on the real model.

    Both are easy to get subtly wrong in ways only the simulator reveals: a distance-only
    grasp test reports the arm's approach motion as slip (measured: 0.077 m inside a
    0.06 m ball), and a summed contact force is dominated by whichever stiff pair is
    meshing, so it *falls* as ``stiff_scale`` rises.
    """
    env = task.make_env(DynamicsSpec(), seed=0)
    expert = task.expert()
    grasped_steps = 0
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(EXPERT_STEPS):
            obs, _reward, _terminated, _truncated, info = env.step(expert.act(obs))
            assert info["grasp_test"] == "contact", (
                "task_geoms.yaml carries no gripper/object groups, so the grasp test fell back "
                "to the distance heuristic; V1/V2 would then measure free-space geometry"
            )
            pairs = np.asarray(info["force_pairs"], dtype=np.float64)
            expected = float(pairs.max()) if pairs.size else 0.0
            assert float(info["force"]) == pytest.approx(expected)
            assert float(info["force"]) <= float(info["force_sum"]) + 1e-9
            grasped_steps += int(bool(info["grasped"]))
    finally:
        env.close()
    assert grasped_steps > 0, (
        "the contact-based grasp test never fired during a nominal expert episode: pads do touch "
        "the peg (see the pair histogram in configs/env/task_geoms.yaml), so the geom groups or "
        "the model's contact list are being read wrongly"
    )


def test_contact_read_back_lists_pairs(task: MetaWorldTask) -> None:
    env = task.make_env(DynamicsSpec(), seed=0)
    expert = task.expert()
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(40):
            obs, *_ = env.step(expert.act(obs))
        pairs = env.contact_summary()
        assert isinstance(pairs, list)
        for entry in pairs:
            assert "friction_slide" in entry and "solref_timeconst" in entry
    finally:
        env.close()
