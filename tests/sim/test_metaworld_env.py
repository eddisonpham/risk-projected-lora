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


def test_episode_truncates_at_the_wrapper_budget(task: MetaWorldTask) -> None:
    """The wrapper must enforce ``max_episode_steps`` itself.

    Meta-World's own env reported ``truncated=False`` after 200 steps, so a consumer that
    trusted that flag saw a stuck episode as ``failure_class="none"`` — a silent outcome.
    The toy env truncates at its budget; this one has to as well.
    """
    budget = 25
    env = task.make_env(DynamicsSpec(), seed=0, max_steps=budget)
    try:
        assert env.max_episode_steps == budget
        obs, _ = env.reset(seed=0)
        truncated_flags: list[bool] = []
        for _ in range(budget + 5):
            # A fixed action: it must not accidentally solve the task and terminate early.
            obs, _r, terminated, truncated, _info = env.step(ACTION)
            truncated_flags.append(bool(truncated))
            if terminated or truncated:
                break
        assert truncated_flags[-1] is True, (
            f"the episode ran {len(truncated_flags)} steps of a {budget}-step budget without "
            "reporting truncation"
        )
        assert len(truncated_flags) == budget, (
            f"truncation must land exactly on the budget, got {len(truncated_flags)} steps"
        )
        assert not any(truncated_flags[:-1]), "an episode must not truncate before its budget"

        # And the counter must restart, or episode 2 would truncate immediately.
        obs, _ = env.reset(seed=1)
        obs, _r, terminated, truncated, _info = env.step(ACTION)
        assert not (terminated or truncated), "a reset must clear the step counter"
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


def test_thresholded_contacts_exclude_the_grasp_pair(task: MetaWorldTask) -> None:
    """V3/V4 threshold *manipulation* contacts, so the families must be separable.

    Regression guard for the measurement error behind G1's first freeze: the gripper's own
    squeeze was inside the penetration reduction, so V4's bar sat above the depth the task
    itself produces and its rate tracked the gripper, not the task. Measured here as a
    relation rather than a constant: the all-contact values can only be >= the manipulation
    ones, and while the peg is held they must be strictly larger at some step, which is the
    pads indenting into it.
    """
    env = task.make_env(DynamicsSpec(), seed=0)
    expert = task.expert()
    split_seen = False
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(EXPERT_STEPS):
            obs, _r, _t, _tr, info = env.step(expert.act(obs))
            assert info["contact_family"].startswith("manipulation")
            manip = np.asarray(info["force_pairs"], dtype=np.float64)
            every = np.asarray(info["force_pairs_all"], dtype=np.float64)
            assert float(info["force"]) == pytest.approx(manip.max() if manip.size else 0.0)
            assert float(info["penetration"]) <= float(info["penetration_all"]) + 1e-12
            if manip.size < every.size and float(info["penetration_all"]) > float(
                info["penetration"]
            ):
                split_seen = True
    finally:
        env.close()
    assert split_seen, (
        "no step showed the grasp pair carrying more penetration than the manipulation "
        "contacts; either the gripper/object groups stopped resolving or the family split "
        "was lost, and V3/V4 would silently go back to thresholding the gripper's squeeze"
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
