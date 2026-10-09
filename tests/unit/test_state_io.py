"""State save/restore: bitwise replay exactness (the S2 property) and fail-loud paths."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from riskproj.envs import ToyEnv, restore_state, rollout_from_state, save_state, states_equal
from riskproj.envs.state_io import SimState, as_flat_state

pytestmark = pytest.mark.unit

ACTIONS = [np.array([0.01, 0.005, -0.004])] * 6


def stepped_env(steps: int = 3) -> ToyEnv:
    env = ToyEnv(seed=11)
    env.reset(seed=11)
    for _ in range(steps):
        env.step(np.array([0.01, 0.0, -0.005]))
    return env


def test_simstate_npz_roundtrip_preserves_hash(tmp_path: Any) -> None:
    state = save_state(stepped_env())
    path = state.save(tmp_path / "state.npz")
    loaded = SimState.load(path)
    assert loaded.kind == "toy"
    assert loaded.hash() == state.hash()
    assert states_equal(loaded, state)
    assert loaded.nbytes() == state.nbytes()


def test_toy_state_roundtrip_in_memory() -> None:
    env = stepped_env()
    state = save_state(env)
    rebuilt = SimState.from_dict(state.to_dict())
    assert rebuilt.hash() == state.hash()


def test_replay_from_a_saved_state_is_bitwise_exact() -> None:
    env = stepped_env()
    state = save_state(env)

    for a in ACTIONS:
        env.step(a)
    first = save_state(env).hash()

    restore_state(env, state)
    for a in ACTIONS:
        env.step(a)
    second = save_state(env).hash()
    assert first == second, (
        "replay from a saved state diverged: counterfactual labels would be invalid"
    )


def test_restore_into_a_fresh_env_matches() -> None:
    env = stepped_env()
    state = save_state(env)
    for a in ACTIONS:
        env.step(a)
    expected = save_state(env).hash()

    fresh = ToyEnv(seed=99)
    fresh.reset(seed=99)
    restore_state(fresh, state)
    for a in ACTIONS:
        fresh.step(a)
    assert save_state(fresh).hash() == expected


def test_rollout_from_state_returns_per_step_records() -> None:
    env = stepped_env()
    state = save_state(env)
    records = rollout_from_state(env, state, ACTIONS)
    assert len(records) == len(ACTIONS)
    for i, rec in enumerate(records):
        assert rec["step"] == i
        assert isinstance(rec["reward"], float)
        assert isinstance(rec["terminated"], bool)
        assert isinstance(rec["info"], dict)


def test_rollout_from_state_stops_at_termination() -> None:
    env = ToyEnv(seed=0)
    env.reset(seed=0)
    state = save_state(env)
    records = rollout_from_state(env, state, [np.zeros(3)] * 200)
    assert len(records) < 200


def test_rollout_hook_is_called_per_step() -> None:
    env = stepped_env()
    state = save_state(env)
    seen: list[int] = []
    rollout_from_state(env, state, ACTIONS, observe_hook=lambda t, *_: seen.append(t))
    assert seen == list(range(len(ACTIONS)))


def test_states_equal_detects_a_difference() -> None:
    a = save_state(stepped_env(2))
    b = save_state(stepped_env(4))
    assert not states_equal(a, b)
    assert not states_equal(a, b, tol=1e-12)


def test_restore_rejects_a_shape_mismatch() -> None:
    env = stepped_env()
    state = save_state(env)
    state.extra["p"] = np.zeros(5)
    with pytest.raises(ValueError):
        restore_state(env, state)


def test_restore_rejects_a_kind_mismatch() -> None:
    env = stepped_env()
    state = save_state(env)
    state.kind = "mujoco"
    with pytest.raises(TypeError):
        restore_state(env, state)


def test_extra_rejects_nested_mappings() -> None:
    with pytest.raises(TypeError, match="nested mappings"):
        SimState.from_dict({"extra/bad": {"a": 1}})


def test_load_requires_the_meta_blob(tmp_path: Any) -> None:
    path = tmp_path / "bogus.npz"
    np.savez(path, qpos=np.zeros(3))
    with pytest.raises(ValueError, match="meta"):
        SimState.load(path)


def test_as_flat_state_makes_arrays() -> None:
    flat = as_flat_state({"a": 1.0, "b": [1, 2, 3]})
    assert flat["a"].dtype.kind == "f"
    assert flat["b"].shape == (3,)


def test_restore_into_an_unsupported_object_raises() -> None:
    with pytest.raises(TypeError):
        restore_state(object(), save_state(stepped_env()))
