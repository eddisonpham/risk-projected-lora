"""The failure-inducing actor used to evaluate G1's coverage criterion (``04`` §8)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from riskproj.envs.actors import PerturbedExpertActor, clip_actions

pytestmark = pytest.mark.unit


class _ConstantExpert:
    def __init__(self, action: Any) -> None:
        self.action = np.asarray(action, dtype=np.float64)
        self.calls = 0

    def act(self, obs: dict[str, Any]) -> np.ndarray:  # noqa: ARG002
        self.calls += 1
        return self.action.copy()


def _roll(actor: PerturbedExpertActor, n: int, obs: dict[str, Any] | None = None) -> np.ndarray:
    obs = obs or {}
    return np.stack([actor.act(obs) for _ in range(n)])


def test_sigma_zero_reproduces_the_expert_exactly() -> None:
    expert = _ConstantExpert([0.1, -0.2, 0.3, 1.0])
    actor = PerturbedExpertActor(expert=expert, sigma=0.0, horizon=4, seed=0)
    out = _roll(actor, 12)
    assert np.allclose(out, expert.action, atol=0.0, rtol=0.0)


def test_the_deviation_is_coherent_over_the_chunk() -> None:
    """One offset per chunk: a chunk's delta is constant, not white noise per step."""
    expert = _ConstantExpert([0.3, 0.0, 0.0, 1.0])
    actor = PerturbedExpertActor(expert=expert, sigma=0.5, horizon=4, seed=3)
    out = _roll(actor, 12)
    delta = out - expert.action
    assert np.allclose(delta[:4], delta[0]), "the first chunk must share one offset"
    assert np.allclose(delta[4:8], delta[4]), "the second chunk must share one offset"
    assert not np.allclose(delta[:4], delta[4:8]), "a new chunk must draw a new offset"


def test_actions_are_clipped_to_the_env_bounds() -> None:
    expert = _ConstantExpert([0.95, 0.0, 0.0, 1.0])
    actor = PerturbedExpertActor(
        expert=expert,
        sigma=0.9,
        horizon=2,
        seed=1,
        action_low=[-1.0] * 4,
        action_high=[1.0] * 4,
    )
    out = _roll(actor, 50)
    assert np.all(out <= 1.0 + 1e-12) and np.all(out >= -1.0 - 1e-12)


def test_same_seed_reproduces_and_a_different_seed_does_not() -> None:
    expert = _ConstantExpert([0.2, 0.2, 0.2, 0.5])
    first = _roll(PerturbedExpertActor(expert=expert, sigma=0.3, horizon=3, seed=7), 15)
    again = _roll(PerturbedExpertActor(expert=expert, sigma=0.3, horizon=3, seed=7), 15)
    other = _roll(PerturbedExpertActor(expert=expert, sigma=0.3, horizon=3, seed=8), 15)
    assert np.allclose(first, again)
    assert not np.allclose(first, other)


def test_reseed_restarts_the_stream() -> None:
    expert = _ConstantExpert([0.2, 0.2, 0.2, 0.5])
    actor = PerturbedExpertActor(expert=expert, sigma=0.3, horizon=3, seed=7)
    first = _roll(actor, 9)
    actor.reseed(7)
    again = _roll(actor, 9)
    assert np.allclose(first, again), "reseed must make an episode reproducible"


def test_sigma_is_relative_to_the_action_rms() -> None:
    """``sigma`` must mean the same thing here as in ``perturb_chunk``.

    The implementation of that function (``data.gen_critic_data``) adds
    ``sigma * rms(chunk) * noise``, so an action of RMS 1 with sigma 0.2 deviates by an
    RMS of about 0.2. Checked statistically over many chunks rather than per draw.
    """
    expert = _ConstantExpert([1.0, -1.0, 1.0, -1.0])  # RMS = 1
    actor = PerturbedExpertActor(expert=expert, sigma=0.2, horizon=1, seed=11)
    delta = _roll(actor, 2000) - expert.action
    assert float(np.sqrt(np.mean(delta**2))) == pytest.approx(0.2, rel=0.15)


def test_invalid_parameters_fail_loud() -> None:
    expert = _ConstantExpert([0.0] * 4)
    with pytest.raises(ValueError, match="sigma"):
        PerturbedExpertActor(expert=expert, sigma=-0.1)
    with pytest.raises(ValueError, match="horizon"):
        PerturbedExpertActor(expert=expert, sigma=0.1, horizon=0)


def test_clip_actions_is_a_no_op_without_bounds() -> None:
    out = clip_actions([2.0, -2.0], None, None)
    assert np.allclose(out, [2.0, -2.0])
    bounded = clip_actions([2.0, -2.0], [-1.0, -1.0], [1.0, 1.0])
    assert np.allclose(bounded, [1.0, -1.0])
