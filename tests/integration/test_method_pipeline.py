"""Integration: toy environment -> policy -> critic -> training -> projected inference.

This is the walking-skeleton path from ``03`` §3 (S6) restricted to the methods
layer: it proves the pieces are wired together (state in, decisions out, logs
written), not that the method works. Nothing here is evidence about the hypothesis,
and the numbers it produces are meaningless — a random-init toy policy has nothing
to adapt.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from riskproj.critic.features import FeatureSpec
from riskproj.critic.model import RiskCritic
from riskproj.critic.train import freeze
from riskproj.methods import build_method
from riskproj.methods.base import critic_context_from_batch
from riskproj.policies.toy import ToyPolicy
from riskproj.types import DynamicsSpec
from riskproj.utils.config import CriticCfg, ExperimentCfg, MethodCfg, ProjCfg, TrainCfg
from riskproj.utils.seed import common_random_noise

pytestmark = pytest.mark.integration

HORIZON = 10
FLOW_STEPS = 4


def _toy_batch(seed: int = 0) -> tuple[dict[str, Any], int, int]:
    """One batch built from a real toy-environment observation."""
    from riskproj.envs.toy import ToyReachTask

    task = ToyReachTask()
    env = task.make_env(DynamicsSpec(), seed=seed)
    obs, _info = env.reset(seed=seed)
    state = np.asarray(obs["state"], dtype=np.float32)
    ee_vel = np.asarray(obs["eef_vel"], dtype=np.float32)
    state_dim = int(state.shape[0])
    action_dim = int(env.action_dim)

    policy = ToyPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=16,
        chunk_horizon=HORIZON,
        num_flow_steps=FLOW_STEPS,
        seed=seed,
    )
    batch = policy.sample_batch(2, seed=seed)
    batch["state"] = torch.tensor(np.stack([state, state]), dtype=torch.float32)
    batch["action_dim"] = action_dim
    batch["critic_ctx"] = {
        "proprio": torch.tensor(np.stack([state, state])),
        "ee_vel": torch.tensor(np.stack([ee_vel, ee_vel])),
        "action_hist": torch.zeros(2, 1, action_dim),
        "proprio_hist": torch.tensor(np.stack([state, state])).unsqueeze(1),
        "vision_emb": torch.zeros(2, 0),
        "force": torch.zeros(2, 0),
    }
    env.close()
    return batch, state_dim, action_dim


def _critic(state_dim: int, action_dim: int) -> RiskCritic:
    spec = FeatureSpec(
        state_dim=state_dim, action_dim=action_dim, chunk_horizon=HORIZON, hist_len=1
    )
    cfg = CriticCfg(hidden=16, depth=1, ensemble=2, horizon=HORIZON)
    return freeze(RiskCritic(cfg, spec))


def _exp_cfg() -> ExperimentCfg:
    return ExperimentCfg(
        name="integration_methods",
        train=TrainCfg(steps=6, batch_size=2, grad_accum=1, lr=1e-2, save_every=0),
    )


def test_train_then_project_end_to_end(tmp_path: Any) -> None:
    batch, state_dim, action_dim = _toy_batch()
    policy = ToyPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=16,
        chunk_horizon=HORIZON,
        num_flow_steps=FLOW_STEPS,
        seed=0,
    )
    critic = _critic(state_dim, action_dim)
    cfg = _exp_cfg()

    b1 = build_method(MethodCfg(id="B1", train_mode="0", inference_mode="0"))
    result = b1.train(
        cfg,
        policy,
        train_seed=0,
        data=lambda: dict(batch),
        out_dir=tmp_path / "B1" / "0",
    )
    assert result.metrics["loss_final"] <= result.metrics["loss_first"]
    assert (tmp_path / "B1" / "0" / "checkpoints" / "final.pt").exists()

    # B3 on the same checkpoint: projection must engage (tau below tau_proj) and log.
    method = build_method(
        MethodCfg(
            id="B3",
            train_mode="lat",
            inference_mode="lat",
            projection=ProjCfg(tau=0.1, K=1, tau_proj=0.3),
        )
    )
    wrapper = method.wrap(policy, critic=critic, base_seed=0)
    wrapper.begin_episode(0)
    risky = dict(batch)
    risky["tau"] = torch.full((2,), 0.1)
    from riskproj.policies.flow import flow_interpolate

    risky["x_tau"] = flow_interpolate(risky["a0"], risky["eps"], risky["tau"])

    out = wrapper(risky)
    assert out.chunk.shape == (2, HORIZON, action_dim)
    assert torch.isfinite(out.chunk).all()

    summary = wrapper.projection_summary()
    assert summary["n_projection_calls"] >= 1.0, "the projection never ran; the arm is B1"
    assert summary["mean_corr_rel"] > 0.0, "no correction norm was recorded"
    assert "fallback_none" in summary, "the ladder's counters must be reported"

    decisions = wrapper.drain_decisions()
    assert decisions, "every projection event must be logged (CLAUDE.md #4)"
    assert {d["ep_id"] for d in decisions} == {"0"}
    assert wrapper.decision_records() == [], "drain must clear the buffer"


def test_b1_and_b3_see_the_same_noise_and_act_differently() -> None:
    """Pairing (same CRN stream) plus a real behavioural difference."""
    batch, state_dim, action_dim = _toy_batch(seed=1)
    policy = ToyPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=16,
        chunk_horizon=HORIZON,
        num_flow_steps=FLOW_STEPS,
        seed=1,
    )
    critic = _critic(state_dim, action_dim)
    b1 = build_method(MethodCfg(id="B1", train_mode="0", inference_mode="0")).wrap(
        policy, critic=critic, base_seed=0
    )
    b3 = build_method(
        MethodCfg(
            id="B3",
            train_mode="lat",
            inference_mode="lat",
            projection=ProjCfg(tau=0.1, K=1, tau_proj=0.3),
        )
    ).wrap(policy, critic=critic, base_seed=0)

    b1.begin_episode(0)
    b3.begin_episode(0)
    risky = dict(batch)
    risky["tau"] = torch.full((2,), 0.05)
    from riskproj.policies.flow import flow_interpolate

    risky["x_tau"] = flow_interpolate(risky["a0"], risky["eps"], risky["tau"])

    out_b1 = b1(risky)
    out_b3 = b3(risky)
    assert not torch.allclose(out_b1.chunk, out_b3.chunk), (
        "B3 produced the same action as B1: the projection did not change the decision"
    )
    # Both wrappers must have drawn the same flow noise for the same decision.
    assert torch.equal(risky["eps"], batch["eps"])


def test_critic_context_helper_matches_the_feature_layout() -> None:
    """The fallback context builder must produce exactly what the critic was built for."""
    batch, state_dim, action_dim = _toy_batch(seed=2)
    critic = _critic(state_dim, action_dim)
    ctx = critic_context_from_batch(batch)
    assert ctx["proprio"].shape == (2, state_dim)
    assert ctx["proprio_hist"].shape == (2, 1, state_dim)
    assert ctx["action_hist"].shape == (2, 1, action_dim)
    risk = critic.risk(ctx, batch["a0"])
    assert risk.shape == (2,)
    assert torch.isfinite(risk).all()


def test_epsilon_helper_pairs_methods_on_identical_noise() -> None:
    from riskproj.methods.controls import _ensure_eps

    batch, state_dim, action_dim = _toy_batch(seed=3)
    stripped = {k: v for k, v in batch.items() if k != "eps"}
    a = _ensure_eps(dict(stripped), base_seed=9, episode_id=0, step=0)
    b = _ensure_eps(dict(stripped), base_seed=9, episode_id=0, step=0)
    assert torch.equal(a["eps"], b["eps"])
    expected = common_random_noise(9, 0, 0, (2, HORIZON, action_dim))
    torch.testing.assert_close(a["eps"], expected)
