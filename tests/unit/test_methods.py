"""Unit tests for the method layer: registry, training, controls, ladder, CRN.

Everything here runs on CPU with the toy policy and a tiny critic — no simulator, no
downloads, no GPU. The tests are written to fail for *mechanism* reasons: if the
projection stops engaging, if a control stops matching B3's correction norm, or if an
unusable gradient stops reaching the fallback ladder, these break.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from riskproj.critic.features import FeatureSpec
from riskproj.critic.model import RiskCritic
from riskproj.critic.train import freeze
from riskproj.methods import METHOD_IDS, build_method
from riskproj.methods.base import critic_context_from_batch, flow_matching_loss, run_inference
from riskproj.methods.lora_sft import low_noise_mass
from riskproj.policies.flow import flow_interpolate
from riskproj.policies.toy import ToyPolicy
from riskproj.projection import FallbackCode
from riskproj.utils.config import CriticCfg, ExperimentCfg, MethodCfg, ProjCfg, TrainCfg
from riskproj.utils.seed import common_random_noise

pytestmark = pytest.mark.unit

STATE_DIM = 6
ACTION_DIM = 4
HORIZON = 5
FLOW_STEPS = 4
BATCH = 4

#: Method id -> (train_mode, inference_mode), the vocabulary of ``08`` §2.
MODES: dict[str, tuple[str, str]] = {
    "B0": ("0", "0"),
    "B1": ("0", "0"),
    "B1-LN": ("0", "0"),
    "B1+TTP": ("0", "lat"),
    "B2": ("pen", "0"),
    "B2-L": ("pen", "0"),
    "B3": ("lat", "lat"),
    "B3-trainonly": ("lat", "0"),
    "B4a": ("0", "act-post"),
    "B4b": ("0", "act-flow"),
    "B5": ("0", "bon"),
    "B6": ("0", "gate"),
    "C1": ("lat", "lat"),
    "C2": ("lat", "lat"),
    "C3": ("lat", "lat"),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_policy(*, action_head: str = "flow_matching", seed: int = 0) -> ToyPolicy:
    return ToyPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden=16,
        chunk_horizon=HORIZON,
        num_flow_steps=FLOW_STEPS,
        action_head=action_head,  # type: ignore[arg-type]
        seed=seed,
    )


def make_critic(*, seed: int = 0, ensemble: int = 2) -> RiskCritic:
    spec = FeatureSpec(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, chunk_horizon=HORIZON, hist_len=1
    )
    cfg = CriticCfg(hidden=16, depth=1, ensemble=ensemble, horizon=HORIZON, seed=seed)
    return freeze(RiskCritic(cfg, spec))


def fixed_batch(policy: ToyPolicy, batch_size: int = BATCH, seed: int = 123) -> dict[str, Any]:
    """A batch that satisfies the flow convention by construction, identical every call."""
    return policy.sample_batch(batch_size, seed=seed)


def in_window_batch(policy: ToyPolicy, tau: float = 0.1) -> dict[str, Any]:
    """Force the low-noise regime explicitly instead of hoping the sampler lands there.

    With the Beta sampler ``P(tau <= 0.3) = 0.16`` per sample, so a 4-row batch is
    outside the projection window about half the time; a test that depends on luck is
    not a test.
    """
    batch = fixed_batch(policy)
    batch["tau"] = torch.full((batch["a0"].shape[0],), float(tau))
    batch["x_tau"] = flow_interpolate(batch["a0"], batch["eps"], batch["tau"])
    return batch


def method_cfg(method_id: str, **overrides: Any) -> MethodCfg:
    train_mode, inference_mode = overrides.pop("mode", MODES[method_id])
    base: dict[str, Any] = {
        "id": method_id,
        "train_mode": train_mode,
        "inference_mode": inference_mode,
        "projection": ProjCfg(tau=0.5, K=1),
    }
    base.update(overrides)
    return MethodCfg(**base)


def exp_cfg(**train_overrides: Any) -> ExperimentCfg:
    base: dict[str, Any] = {"steps": 4, "batch_size": BATCH, "grad_accum": 1, "lr": 1e-2, "save_every": 0}
    base.update(train_overrides)
    return ExperimentCfg(name="unit_methods", train=TrainCfg(**base))


def convex_risk(centres: torch.Tensor, k: float = 1.0) -> Any:
    """Per-example quadratic risk: above any small tau and *guaranteed* to decrease."""

    def risk(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * k * (h - centres).square().sum(-1) + 0.5

    return risk


class StubCritic:
    """Constant-risk critic: forces the *unusable gradient* path deterministically.

    Deliberately a stub, not a RiskCritic: the test is about the ladder, and a
    trained critic would not reliably produce an unusable gradient on demand.
    """

    training = False
    horizon = HORIZON

    def __init__(self, value: float = 0.9) -> None:
        self.value = float(value)

    def risk(self, ctx: Any, chunk: torch.Tensor, *, space: str = "prob") -> torch.Tensor:
        return torch.full((chunk.shape[0],), self.value, dtype=torch.float32)

    def logits(self, ctx: Any, chunk: torch.Tensor) -> torch.Tensor:
        z = torch.full((chunk.shape[0], 6), float(torch.logit(torch.tensor(self.value))))
        return z


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_every_method_id_builds_and_the_registry_matches() -> None:
    from riskproj.utils.registry import available

    assert sorted(available("method")) == sorted(METHOD_IDS)
    for method_id in METHOD_IDS:
        overrides: dict[str, Any] = {}
        if method_id == "B1-LN":
            overrides["noise_schedule"] = "logit_normal"
        if method_id in ("C1", "C2", "C3"):
            overrides["control"] = method_id
        method = build_method(method_cfg(method_id, **overrides))
        assert method.name == method_id
        expected_control = method_id if method_id in ("C1", "C2", "C3") else "none"
        assert method.control == expected_control


def test_controls_must_run_inside_the_b3_pipeline() -> None:
    with pytest.raises(ValueError, match="inference_mode"):
        build_method(method_cfg("C1", mode=("lat", "act-post"), control="C1"))


def test_b1_ln_requires_the_logit_normal_schedule() -> None:
    with pytest.raises(ValueError, match="logit_normal"):
        build_method(method_cfg("B1-LN"))


def test_b0_has_nothing_to_train_and_says_so() -> None:
    policy = make_policy()
    method = build_method(method_cfg("B0"))
    with pytest.raises(RuntimeError, match="B0"):
        method.train(exp_cfg(), policy, train_seed=0, data=lambda: fixed_batch(policy))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def test_b1_overfits_one_batch(tmp_path: Any) -> None:
    """Sanity before new numbers (skills/software_engineering.md §Implement)."""
    policy = make_policy()
    batch = fixed_batch(policy, batch_size=8)
    method = build_method(method_cfg("B1"))
    result = method.train(
        exp_cfg(steps=60, lr=1e-2),
        policy,
        train_seed=0,
        data=lambda: dict(batch),
        out_dir=tmp_path / "b1",
    )
    assert result.metrics["loss_final"] < result.metrics["loss_first"], "the loss did not go down"
    assert (tmp_path / "b1" / "checkpoints" / "final.pt").exists()
    assert result.metrics["n_trainable"] > 0.0
    assert result.metrics["steps"] == 60.0


def test_b1_checkpoints_every_save_every(tmp_path: Any) -> None:
    policy = make_policy()
    batch = fixed_batch(policy)
    method = build_method(method_cfg("B1"))
    method.train(
        exp_cfg(steps=4, save_every=2),
        policy,
        train_seed=0,
        data=lambda: dict(batch),
        out_dir=tmp_path / "ckpt",
    )
    ckpt = tmp_path / "ckpt" / "checkpoints"
    assert (ckpt / "step_2.pt").exists() and (ckpt / "step_4.pt").exists()


def test_b1_ln_changes_only_the_schedule() -> None:
    beta = low_noise_mass("beta")
    ln = low_noise_mass("logit_normal", ln_m=1.5, ln_s=1.0)
    # FACT's measured numbers: ~8.9% of the gradient mass below tau=0.2 for Beta,
    # ~6x more under Logit-Normal (02 §A).
    assert 0.05 < beta < 0.15, beta
    assert ln > 0.4, ln
    assert ln / max(beta, 1e-9) > 3.0


def test_b2_lam_zero_matches_b1_and_the_critic_never_receives_gradients() -> None:
    policy_a, policy_b = make_policy(), make_policy()
    batch = fixed_batch(policy_a)
    critic = make_critic()
    b1 = build_method(method_cfg("B1"))
    b2 = build_method(method_cfg("B2", lam=0.0))

    loss_b1, _ = b1.training_loss(policy_a, dict(batch), exp_cfg(), None, 0)
    loss_b2, info = b2.training_loss(policy_b, dict(batch), exp_cfg(), critic, 0)
    torch.testing.assert_close(loss_b2, loss_b1, rtol=1e-5, atol=1e-6)
    assert info["penalty"] == pytest.approx(0.0, abs=1e-9)

    # With lam > 0 the penalty must be active, and the critic must stay untouched.
    b2_lam = build_method(method_cfg("B2", lam=1.0, projection=ProjCfg(tau=0.01, K=1)))
    before = [p.detach().clone() for p in critic.parameters()]
    loss, info = b2_lam.training_loss(policy_b, dict(batch), exp_cfg(), critic, 1)
    loss.backward()
    assert float(info["penalty"]) > 0.0, "the penalty must be active for this to test anything"
    for param, snapshot in zip(critic.parameters(), before, strict=True):
        assert param.grad is None, "the critic must never receive task gradients (06 §3)"
        torch.testing.assert_close(param, snapshot)
        assert not param.requires_grad


def test_b2_requires_a_critic() -> None:
    method = build_method(method_cfg("B2"))
    with pytest.raises(ValueError, match="critic"):
        method.train(exp_cfg(), make_policy(), train_seed=0, data=lambda: {})


def test_b2l_dual_ascent_moves_lambda_toward_the_budget() -> None:
    policy = make_policy()
    batch = fixed_batch(policy)
    critic = make_critic()
    method = build_method(
        method_cfg("B2-L", lam=0.0, lambda_dual_alpha=0.5, violation_budget=0.0,
                   projection=ProjCfg(tau=0.5, K=1))
    )
    _, first = method.training_loss(policy, dict(batch), exp_cfg(), critic, 0)
    lam_after_first = method.lam
    _, second = method.training_loss(policy, dict(batch), exp_cfg(), critic, 1)
    assert lam_after_first > 0.0, "the dual must rise when the risk exceeds the budget"
    assert second["lam"] > first["lam"]
    assert len(method.lam_trajectory) == 2


def test_b3_training_uses_the_projected_activation() -> None:
    policy_proj, policy_plain = make_policy(), make_policy()
    batch_proj = in_window_batch(policy_proj)
    batch_plain = in_window_batch(policy_plain)
    critic = make_critic()
    projected = build_method(method_cfg("B3", projection=ProjCfg(tau=0.01, K=1)))
    plain = build_method(method_cfg("B1"))

    loss_proj, info = projected.training_loss(policy_proj, batch_proj, exp_cfg(), critic, 0)
    loss_plain, _ = plain.training_loss(policy_plain, batch_plain, exp_cfg(), None, 0)

    assert info["projected"] == 1.0
    assert info["window_frac"] == 1.0, "the batch must be inside the projection window"
    assert info["n_active"] > 0.0, "the projection never engaged; the arm would be B1 in disguise"
    assert info["n_unusable"] == 0.0
    assert info["corr_rel"] > 0.0, "no correction was applied"
    assert float(loss_proj) != pytest.approx(float(loss_plain), abs=1e-9), (
        "the projected loss equals the unprojected loss: the corrected activation is not reaching "
        "the flow-matching objective"
    )


def test_b3_requires_a_critic_for_training() -> None:
    method = build_method(method_cfg("B3"))
    with pytest.raises(ValueError, match="critic"):
        method.train(exp_cfg(), make_policy(), train_seed=0, data=lambda: {})


# --------------------------------------------------------------------------
# the correction, called directly: a convex risk *guarantees* a decrease
# --------------------------------------------------------------------------


def test_correct_reduces_a_convex_risk_and_reports_the_trace() -> None:
    method = build_method(method_cfg("B3", projection=ProjCfg(tau=0.5, K=2)))
    h = torch.randn(3, 8)
    h_safe, info, extra = method._correct(
        h, None, convex_risk(torch.zeros(3, 8)), torch.ones(3, dtype=torch.bool), seed_key=(0, 0)
    )
    assert extra["control_code"] == 0.0
    assert info is not None
    assert bool(info.usable.all())
    assert float(info.r_after.max()) <= float(info.r_cand.max()) + 1e-6
    assert not torch.allclose(h_safe, h)
    assert float(info.corr_rel.mean()) > 0.0
    assert info.r_pred is not None
    assert len(info.trace) >= 1


def test_rows_outside_the_flow_window_are_untouched() -> None:
    method = build_method(method_cfg("B3", projection=ProjCfg(tau=0.5, K=1)))
    h = torch.randn(2, 5)
    h_safe, info, _ = method._correct(
        h,
        None,
        convex_risk(torch.zeros(2, 5)),
        torch.tensor([True, False]),
        seed_key=(0, 0),
    )
    assert info is not None
    assert torch.equal(h_safe[1], h[1]), "a row outside the window must be returned unchanged"
    assert not torch.equal(h_safe[0], h[0])
    assert bool(info.active[0]) and not bool(info.active[1])


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------


def test_c1_matches_b3_norm_with_a_different_direction() -> None:
    h = torch.randn(3, 6)
    window = torch.ones(3, dtype=torch.bool)
    risk = convex_risk(torch.zeros(3, 6))

    reference = build_method(method_cfg("B3", projection=ProjCfg(tau=0.5, K=1)))
    h_ref, info_ref, _ = reference._correct(h, None, risk, window, seed_key=(11, 5))
    c1 = build_method(method_cfg("C1", projection=ProjCfg(tau=0.5, K=1), control="C1"))
    h_c1, info_c1, extra = c1._correct(h, None, risk, window, seed_key=(11, 5))

    assert info_ref is not None and info_c1 is not None
    assert extra["c1_norm_ratio"] == pytest.approx(1.0, abs=1e-5), "the control must match B3's norm"
    assert float((h_c1 - h).flatten(1).norm(dim=1).mean()) == pytest.approx(
        float((h_ref - h).flatten(1).norm(dim=1).mean()), rel=1e-4
    )
    assert not torch.allclose(h_c1, h_ref), "C1 must not reproduce the gradient direction"


def test_c2_shrinks_the_delta_with_a_matched_norm() -> None:
    h = torch.randn(3, 6)
    # A delta far larger than B3's correction, so the clamp is not what matches the norm.
    h0 = h - 5.0 * torch.randn(3, 6)
    method = build_method(method_cfg("C2", projection=ProjCfg(tau=0.5, K=1), control="C2"))
    h_safe, info, extra = method._correct(
        h, h0, convex_risk(torch.zeros(3, 6)), torch.ones(3, dtype=torch.bool), seed_key=(1, 1)
    )
    assert info is not None
    assert extra["c2_mean_scale"] <= 1.0
    assert extra["c2_norm_ratio"] == pytest.approx(1.0, rel=1e-3)
    assert float((h_safe - h0).abs().mean()) < float((h - h0).abs().mean())


def test_c2_without_the_base_activation_fails_loud() -> None:
    method = build_method(method_cfg("C2", projection=ProjCfg(tau=0.5, K=1), control="C2"))
    with pytest.raises(RuntimeError, match="h0"):
        method._correct(
            torch.randn(2, 4),
            None,
            convex_risk(torch.zeros(2, 4)),
            torch.ones(2, dtype=torch.bool),
            seed_key=(0, 0),
        )


def test_c3_permutes_the_critic_so_the_correction_differs() -> None:
    h = torch.randn(3, 6)
    window = torch.ones(3, dtype=torch.bool)
    risk = convex_risk(torch.stack([torch.zeros(6), torch.full((6,), 2.0), torch.full((6,), -2.0)]))

    reference = build_method(method_cfg("B3", projection=ProjCfg(tau=0.5, K=1)))
    h_ref, info_ref, _ = reference._correct(h, None, risk, window, seed_key=(2, 2))
    c3 = build_method(method_cfg("C3", projection=ProjCfg(tau=0.5, K=1), control="C3"))
    h_c3, info_c3, extra = c3._correct(h, None, risk, window, seed_key=(2, 2))

    assert info_ref is not None and info_c3 is not None
    assert extra["c3_permutation"] == 1.0
    assert not torch.allclose(h_ref, h_c3), "the permuted critic produced an identical correction"
    # A rotation preserves the multiset of per-example risks — that is what makes it a
    # permutation rather than a different critic.
    assert sorted(info_ref.r_cand.tolist()) == pytest.approx(sorted(info_c3.r_cand.tolist()), rel=1e-6)


def test_c3_logs_the_degenerate_batch_instead_of_silently_equalling_b3() -> None:
    """A batch of one admits no derangement.

    C3 then *is* B3, which is exactly the failure mode that would bias A1 ("is it just
    the gradient?" / "is it just perturbation?") in the direction that flatters the
    hypothesis. The run has to say so rather than look like a control that ran.
    """
    h = torch.randn(1, 4)
    window = torch.ones(1, dtype=torch.bool)
    risk = convex_risk(torch.zeros(1, 4))

    reference = build_method(method_cfg("B3", projection=ProjCfg(tau=0.5, K=1)))
    h_ref, _info_ref, _ = reference._correct(h, None, risk, window, seed_key=(1, 1))
    c3 = build_method(method_cfg("C3", projection=ProjCfg(tau=0.5, K=1), control="C3"))
    h_c3, _info_c3, extra = c3._correct(h, None, risk, window, seed_key=(1, 1))

    assert torch.allclose(h_ref, h_c3), (
        "a batch of one cannot be deranged, so the correction must equal B3's"
    )
    assert extra["c3_degenerate"] == 1.0, (
        "the degenerate C3 case must be logged: an unlogged C3 that equals B3 would make "
        "A1's comparison meaningless"
    )
    assert extra["c3_permutation"] == 1.0, "the control was still selected for this run"


# --------------------------------------------------------------------------
# unusable gradients must reach the ladder, loudly
# --------------------------------------------------------------------------


def test_constant_risk_is_unusable_and_flagged() -> None:
    method = build_method(method_cfg("B3", projection=ProjCfg(tau=0.5, K=1)))
    h = torch.randn(2, 5)
    h_safe, info, _ = method._correct(
        h,
        None,
        lambda hh: torch.full((hh.shape[0],), 0.9),
        torch.ones(2, dtype=torch.bool),
        seed_key=(0, 0),
    )
    assert info is not None
    assert torch.equal(h_safe, h), "an unusable gradient must leave the activation unchanged"
    assert bool(info.unusable.all())
    assert (info.fallback_code == FallbackCode.F1_BASE_ACTION).all()


def test_ladder_decision_is_recorded_never_silent() -> None:
    policy = make_policy()
    critic = StubCritic(0.9)
    method = build_method(method_cfg("B3", projection=ProjCfg(tau=0.1, K=1)))
    wrapper = method.wrap(policy, critic=critic, base_seed=0)  # type: ignore[arg-type]
    wrapper.begin_episode(0)
    batch = fixed_batch(policy)
    batch["critic_ctx"] = critic_context_from_batch(batch)
    candidate = torch.zeros(BATCH, HORIZON, ACTION_DIM)

    out = method._consult_ladder(
        wrapper, policy, critic, batch, candidate, flow_step=-1  # type: ignore[arg-type]
    )
    assert wrapper.last_fallback is not None
    assert int(wrapper.last_fallback.code) != FallbackCode.NONE
    records = wrapper.decision_records()
    assert records, "a fallback that is not logged is the failure mode CLAUDE.md #4 bans"
    assert any(int(r["fallback_code"]) != 0 for r in records)
    assert torch.isfinite(out).all()
    counts = wrapper.fallback_counts()
    assert counts["F2"] + counts["F3"] > 0, counts


def test_disabled_ladder_still_records_that_it_was_disabled() -> None:
    policy = make_policy()
    critic = StubCritic(0.9)
    method = build_method(method_cfg("B3", projection=ProjCfg(tau=0.1, K=1, fallback=False)))
    wrapper = method.wrap(policy, critic=critic, base_seed=0)  # type: ignore[arg-type]
    assert wrapper.ladder is not None and not wrapper.ladder.enabled
    batch = fixed_batch(policy)
    batch["critic_ctx"] = critic_context_from_batch(batch)
    method._consult_ladder(
        wrapper, policy, critic, batch, torch.zeros(BATCH, HORIZON, ACTION_DIM), flow_step=-1  # type: ignore[arg-type]
    )
    events = wrapper.ladder.events()
    assert events and "disabled" in events[-1]["reason"]


# --------------------------------------------------------------------------
# B5 / B6
# --------------------------------------------------------------------------


def test_b5_picks_the_lowest_risk_candidate() -> None:
    policy = make_policy()
    critic = make_critic()
    method = build_method(method_cfg("B5", best_of_n=4))
    wrapper = method.wrap(policy, critic=critic, base_seed=3)
    batch = fixed_batch(policy, batch_size=3)
    batch["critic_ctx"] = critic_context_from_batch(batch)

    out = wrapper(batch)
    assert out.chunk.shape == (3, HORIZON, ACTION_DIM)
    summary = wrapper.projection_summary()
    assert summary["bon_n"] == 4.0
    assert summary["bon_best_risk"] <= summary["bon_mean_risk"] + 1e-6
    assert summary["bon_best_risk"] <= summary["bon_worst_risk"] + 1e-6
    # The returned chunk must be the argmin candidate, not a random one.
    risk = critic.risk(batch["critic_ctx"], out.chunk)
    torch.testing.assert_close(
        risk.mean(), torch.tensor(summary["bon_best_risk"]), rtol=1e-4, atol=1e-5
    )


def test_b5_refuses_a_deterministic_head() -> None:
    policy = make_policy(action_head="regression")
    method = build_method(method_cfg("B5", best_of_n=2))
    with pytest.raises(ValueError, match="stochastic head"):
        method.wrap(policy, critic=make_critic())


def test_b6_gates_without_correcting_and_logs_it_as_gating() -> None:
    policy = make_policy()
    critic = make_critic()
    method = build_method(method_cfg("B6", projection=ProjCfg(tau=0.01, K=1)))
    wrapper = method.wrap(policy, critic=critic, base_seed=0)
    batch = fixed_batch(policy)
    batch["critic_ctx"] = critic_context_from_batch(batch)

    out = wrapper(batch)
    assert out.chunk.shape == (BATCH, HORIZON, ACTION_DIM)
    summary = wrapper.projection_summary()
    assert summary["gate_blocked_frac"] > 0.0, "the gate never fired; B6 would be B1 in disguise"
    records = wrapper.decision_records()
    assert records and all(r["active"] and not r["usable"] for r in records), (
        "gating must be recorded as active-but-uncorrected, never as a projection"
    )
    assert wrapper.last_fallback is not None


# --------------------------------------------------------------------------
# common random numbers
# --------------------------------------------------------------------------


def test_common_random_noise_is_shared_across_methods() -> None:
    shape = (2, HORIZON, ACTION_DIM)
    assert torch.equal(
        common_random_noise(7, 0, 3, shape, flow_step=1),
        common_random_noise(7, 0, 3, shape, flow_step=1),
    )
    assert not torch.equal(
        common_random_noise(7, 0, 3, shape, flow_step=1),
        common_random_noise(7, 0, 4, shape, flow_step=1),
    )
    assert not torch.equal(
        common_random_noise(7, 0, 3, shape, flow_step=1),
        common_random_noise(7, 1, 3, shape, flow_step=1),
    )


def test_epsilon_stream_is_keyed_by_episode_and_step_not_by_method() -> None:
    from riskproj.methods.controls import _ensure_eps

    shape = (2, HORIZON, ACTION_DIM)
    batch = {"a0": torch.zeros(shape)}
    for episode, step in ((0, 0), (1, 0), (0, 2)):
        filled = _ensure_eps(dict(batch), base_seed=5, episode_id=episode, step=step)
        torch.testing.assert_close(filled["eps"], common_random_noise(5, episode, step, shape))


# --------------------------------------------------------------------------
# run_inference
# --------------------------------------------------------------------------


def test_run_inference_on_a_flow_head_returns_the_terminal_chunk() -> None:
    policy = make_policy()
    out = run_inference(policy, fixed_batch(policy), site="S2")
    assert out.chunk.shape == (BATCH, HORIZON, ACTION_DIM)
    assert out.velocity is not None
    assert torch.isfinite(out.chunk).all()


def test_run_inference_requires_eps_for_a_flow_head() -> None:
    policy = make_policy()
    batch = {k: v for k, v in fixed_batch(policy).items() if k != "eps"}
    with pytest.raises(KeyError, match="eps"):
        run_inference(policy, batch, site="S2")


def test_run_inference_with_a_deterministic_head_goes_through_f_act() -> None:
    policy = make_policy(action_head="regression")
    batch = fixed_batch(policy)
    seen: list[torch.Tensor] = []

    def hook(h: torch.Tensor, ctx: Any) -> tuple[torch.Tensor, None]:
        seen.append(h)
        return h, None

    out = run_inference(policy, batch, site="S2", project_hook=hook)
    assert seen, "the deterministic-head project hook was never called"
    torch.testing.assert_close(out.chunk, policy.forward(batch, site="S2").chunk)


def test_flow_matching_loss_is_the_flow_objective() -> None:
    """The loss must be ``MSE(v_theta, eps - a0)`` and not something else in disguise."""
    policy = make_policy()
    batch = fixed_batch(policy)
    loss, info = flow_matching_loss(policy, batch, site="S2")
    expected = torch.nn.functional.mse_loss(
        policy.forward(batch, site="S2").velocity, batch["velocity_target"]
    )
    torch.testing.assert_close(loss, expected)
    assert loss.dim() == 0 and float(loss) > 0.0
    assert set(info) == {"loss"}
    assert float(batch["tau"].max()) <= 1.0
