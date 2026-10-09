"""Unit tests for the projection operator (``agents/07_projection_operator.md`` §5).

These pin the *contract*, not the implementation: safe candidates are untouched,
the step matches the closed form on a linear risk, unusable gradients are flagged
rather than silently skipped, examples are independent, and the backward-pass
Jacobian is exactly what ``07`` §3 derives.
"""

from __future__ import annotations

import math

import pytest
import torch

from riskproj.projection import (
    FallbackCode,
    ProjCfg,
    active_at_flow_step,
    project,
    project_fn,
    setpoint,
    threshold_in_space,
)

try:  # tolerances live in tests/, which pytest puts on sys.path via conftest
    from tolerances import EXACT, MATCH
except ImportError:  # pragma: no cover
    from tests.tolerances import EXACT, MATCH

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def linear_risk(w: torch.Tensor, b: torch.Tensor):
    """Per-example linear risk ``r_i = w_i · h_i + b_i`` (dtype agnostic)."""

    def risk(h: torch.Tensor) -> torch.Tensor:
        return (h * w).sum(-1) + b

    return risk


def quadratic_risk(centre: torch.Tensor, k: float = 1.0):
    """Per-example convex risk ``r_i = 0.5 k ||h_i - c_i||²``."""

    def risk(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * k * (h - centre).square().sum(-1)

    return risk


def sigmoid_risk(a: torch.Tensor, c: torch.Tensor):
    """Dtype-agnostic smooth risk in (0,1), used for the finite-difference checks."""

    def risk(h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(h @ a + c)

    return risk


# --------------------------------------------------------------------------
# 5.1 safe candidate is untouched
# --------------------------------------------------------------------------


def test_safe_candidate_is_bitwise_unchanged() -> None:
    torch.manual_seed(0)
    b, d = 3, 5
    w = torch.randn(b, d)
    bias = torch.full((b,), -10.0)
    h = torch.randn(b, d)
    h_safe, info = project(h, linear_risk(w, bias), ProjCfg(tau=0.5))
    assert torch.equal(h_safe, h)
    assert not bool(info.active.any())
    assert info.k_used.sum().item() == 0
    assert torch.equal(info.r_after, info.r_cand)
    assert torch.equal(info.r_pred, info.r_cand)
    assert float(info.corr_rel.max()) == 0.0


# --------------------------------------------------------------------------
# 5.2 closed form on a linear risk (exact)
# --------------------------------------------------------------------------


def test_closed_form_matches_linear_risk() -> None:
    torch.manual_seed(1)
    b, d = 4, 6
    w = torch.randn(b, d)
    h = torch.randn(b, d)
    tau = 0.5
    # Choose the bias so every example sits exactly 1.0 above the threshold.
    bias = (tau + 1.0) - (h * w).sum(-1)
    cfg = ProjCfg(tau=tau, K=1)
    h_safe, info = project(h, linear_risk(w, bias), cfg)

    gn2 = w.square().sum(-1, keepdim=True)
    expected = h - w / gn2
    assert torch.allclose(h_safe, expected, rtol=1e-4, atol=10 * EXACT)
    assert torch.allclose(info.r_after, torch.full((b,), tau), atol=10 * EXACT)
    assert bool(info.active.all()) and bool(info.usable.all())
    assert not bool(info.unusable.any())
    assert info.k_used.tolist() == [1] * b
    # relu(excess)/||g||^2 * ||g|| = excess/||g||
    expected_corr = (1.0 / w.square().sum(-1).sqrt()) / h.norm(dim=-1)
    assert torch.allclose(info.corr_rel, expected_corr, rtol=1e-3, atol=1e-5)


def test_linear_risk_in_logit_space_targets_logit_of_setpoint() -> None:
    torch.manual_seed(2)
    b, d = 2, 4
    w = torch.ones(b, d)
    h = torch.randn(b, d)
    tau = 0.4
    cfg = ProjCfg(tau=tau, space="logit")
    # The risk *is* the logit here; make every example violate the constraint.
    logits = (h * w).sum(-1)
    bias = (threshold_in_space(cfg) + 2.0) - logits
    h_safe, info = project(h, linear_risk(w, bias), cfg)
    assert math.isclose(threshold_in_space(cfg), math.log(tau / (1 - tau)), rel_tol=1e-9)
    assert torch.allclose(info.r_after, torch.full((b,), threshold_in_space(cfg)), atol=1e-4)
    assert torch.allclose(h_safe, h, atol=1.0)  # a real but bounded correction


# --------------------------------------------------------------------------
# 5.3 convex risk must not increase; idempotent on safe outputs
# --------------------------------------------------------------------------


def test_convex_quadratic_risk_decreases_and_stays_finite() -> None:
    torch.manual_seed(3)
    b, d = 5, 7
    centre = torch.randn(b, d)
    h = centre + 2.0 * torch.randn(b, d)
    cfg = ProjCfg(tau=0.5, K=1)
    risk = quadratic_risk(centre, k=1.0)
    r_before = risk(h)
    h_safe, info = project(h, risk, cfg)
    assert torch.isfinite(h_safe).all()
    active = r_before > 0.5
    assert bool(info.active[active].all())
    assert bool((info.r_after[active] <= r_before[active] + EXACT).all())
    assert bool(info.usable[active].all())


def test_idempotent_on_already_safe_outputs() -> None:
    torch.manual_seed(4)
    b, d = 3, 5
    w = torch.randn(b, d)
    h = torch.randn(b, d)
    tau = 0.5
    cfg = ProjCfg(tau=tau)

    # (a) A candidate strictly inside the feasible set is returned untouched.
    inside = torch.randn(b, d)
    bias_inside = (tau - 0.05) - (inside * w).sum(-1)
    h_safe, info = project(inside, linear_risk(w, bias_inside), cfg)
    assert torch.equal(h_safe, inside)
    assert not bool(info.active.any())

    # (b) Projecting the output of a projection is a no-op up to float32 noise.
    risk = linear_risk(w, torch.full((b,), 5.0))
    h_once, _ = project(h, risk, cfg)
    h_twice, info_twice = project(h_once, risk, cfg)
    assert torch.allclose(h_once, h_twice, atol=1e-5)
    assert float(info_twice.corr_rel.max()) < 1e-4


# --------------------------------------------------------------------------
# 5.4/5.5 unusable gradients are flagged, never silently skipped
# --------------------------------------------------------------------------


def test_tiny_gradient_is_flagged_unusable_and_activation_unchanged() -> None:
    torch.manual_seed(5)
    h = torch.randn(3, 4)
    cfg = ProjCfg(tau=0.5, eta=1e-8, fallback=True)

    def risk(hh: torch.Tensor) -> torch.Tensor:
        return 1.0 + 1e-7 * hh.sum(-1)

    h_safe, info = project(h, risk, cfg)
    assert torch.equal(h_safe, h)
    assert bool(info.unusable.all())
    assert not bool(info.usable.any())
    assert bool(info.active.all())
    assert info.k_used.sum().item() == 0
    assert (info.fallback_code == FallbackCode.F1_BASE_ACTION).all()
    assert torch.equal(info.r_after, info.r_cand)


def test_unusable_with_fallback_disabled_still_recorded() -> None:
    h = torch.randn(2, 4)
    cfg = ProjCfg(tau=0.5, fallback=False)

    def risk(hh: torch.Tensor) -> torch.Tensor:
        return 1.0 + 1e-9 * hh.sum(-1)

    h_safe, info = project(h, risk, cfg)
    assert torch.equal(h_safe, h)
    assert bool(info.unusable.all())
    # No ladder is consulted, but the condition is still visible in the log.
    assert (info.fallback_code == FallbackCode.NONE).all()


def test_risk_ignoring_the_activation_is_unusable_not_a_crash() -> None:
    h = torch.randn(2, 3)
    cfg = ProjCfg(tau=0.5)

    def risk(hh: torch.Tensor) -> torch.Tensor:
        return torch.full((hh.shape[0],), 0.9)

    h_safe, info = project(h, risk, cfg)
    assert torch.equal(h_safe, h)
    assert bool(info.unusable.all())
    assert info.trace and info.trace[0]["grad_unavailable"] == 1.0


def test_non_finite_risk_is_unusable() -> None:
    torch.manual_seed(21)
    h = torch.randn(2, 3)
    cfg = ProjCfg(tau=0.5)

    def risk(hh: torch.Tensor) -> torch.Tensor:
        out = hh.sum(-1) + 10.0
        return torch.where(torch.arange(hh.shape[0]) == 0, torch.full_like(out, float("nan")), out)

    h_safe, info = project(h, risk, cfg)
    assert torch.isfinite(h_safe).all()
    # An unverifiable risk must be flagged for the fallback ladder, never passed.
    assert bool(info.unusable[0]) and bool(info.active[0])
    assert bool(info.usable[1]) and not bool(info.unusable[1])
    assert int(info.fallback_code[0].item()) == FallbackCode.F1_BASE_ACTION


# --------------------------------------------------------------------------
# 5.6 per-example independence
# --------------------------------------------------------------------------


def test_per_example_independence_under_permutation() -> None:
    torch.manual_seed(6)
    b, d = 6, 5
    w = torch.randn(b, d)
    bias = torch.full((b,), 3.0)
    h = torch.randn(b, d)
    cfg = ProjCfg(tau=0.5)
    risk = linear_risk(w, bias)
    h_safe, _ = project(h, risk, cfg)

    perm = torch.randperm(b)
    h_perm, _ = project(h[perm], linear_risk(w[perm], bias[perm]), cfg)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(b)
    assert torch.allclose(h_safe, h_perm[inv], atol=MATCH)


def test_batched_result_equals_looping_over_examples() -> None:
    torch.manual_seed(7)
    b, d = 4, 5
    w = torch.randn(b, d)
    bias = torch.full((b,), 3.0)
    h = torch.randn(b, d)
    cfg = ProjCfg(tau=0.5)
    batched, _ = project(h, linear_risk(w, bias), cfg)
    for i in range(b):
        single, _ = project(h[i : i + 1], linear_risk(w[i : i + 1], bias[i : i + 1]), cfg)
        assert torch.allclose(batched[i : i + 1], single, atol=MATCH)


# --------------------------------------------------------------------------
# 5.7 gradient bookkeeping
# --------------------------------------------------------------------------


def test_critic_receives_no_gradient_but_upstream_leaf_does() -> None:
    torch.manual_seed(8)
    d = 4
    critic = torch.nn.Sequential(torch.nn.Linear(d, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    with torch.no_grad():
        critic[-1].bias.fill_(5.0)  # ensure risk is above tau for a wide range of inputs

    def risk(h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(critic(h).squeeze(-1))

    upstream = torch.randn(d, requires_grad=True)
    h_c = (upstream * 1.5).unsqueeze(0) + 1.0
    cfg = ProjCfg(tau=0.5, backward="projected")
    h_safe, info = project(h_c, risk, cfg)
    assert bool(info.active.all())
    assert all(p.grad is None for p in critic.parameters())
    h_safe.sum().backward()
    assert upstream.grad is not None and torch.isfinite(upstream.grad).all()


def test_all_backward_modes_keep_upstream_gradients_finite() -> None:
    torch.manual_seed(9)
    d = 3
    for mode in ("ste", "projected", "second_order"):
        h0 = torch.randn(d, requires_grad=True)
        cfg = ProjCfg(tau=0.3, backward=mode)  # type: ignore[arg-type]

        def risk(hh: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(hh.sum(-1) + 2.0)

        h_safe, info = project(h0.unsqueeze(0), risk, cfg)
        assert bool(info.active.all())
        h_safe.sum().backward()
        assert h0.grad is not None and torch.isfinite(h0.grad).all()


# --------------------------------------------------------------------------
# 5.8 backward Jacobian: I - g gᵀ/||g||² (projected) and I (ste)
# --------------------------------------------------------------------------


def _projected_map(risk, cfg: ProjCfg, d: int):
    def f(x: torch.Tensor) -> torch.Tensor:
        h_safe, _ = project(x.reshape(1, d), risk, cfg)
        return h_safe.reshape(-1)

    return f


def _autograd_jacobian(risk, cfg: ProjCfg, d: int, h0: torch.Tensor) -> torch.Tensor:
    """``d h_safe / d h_c`` by autograd — the Jacobian the training pass sees.

    Autograd, not finite differences: ``detach()`` inside the operator is
    *value-transparent*, so a finite-difference probe of the forward map reports
    the same tangent-projection Jacobian for every backward mode and cannot
    distinguish them (`07` §3). The backward pass is what differs.
    """
    x = h0.reshape(-1).detach().clone().requires_grad_(True)
    return torch.autograd.functional.jacobian(_projected_map(risk, cfg, d), x)


def test_projected_mode_jacobian_is_tangent_projection() -> None:
    from riskproj.projection import projected_jacobian, risk_gradient

    torch.manual_seed(10)
    d = 6
    a = torch.randn(d, dtype=torch.float64)
    h0 = torch.randn(1, d, dtype=torch.float64)
    # Put the sigmoid argument exactly at 0 so the gradient is informative
    # (neither saturated nor degenerate) and the step is far from the eta floor.
    c = -(h0 @ a).squeeze()
    risk = sigmoid_risk(a, c)
    cfg = ProjCfg(tau=0.3, K=1, backward="projected")
    r0 = risk(h0)
    assert float(r0.item()) > 0.3, "test setup must produce an active example"

    g0 = risk_gradient(risk, h0).reshape(-1)
    j_expected = projected_jacobian(g0)
    assert torch.allclose(_autograd_jacobian(risk, cfg, d, h0), j_expected, rtol=1e-6, atol=1e-6)

    # A forward-level finite-difference probe must agree too, once the step is
    # large enough that float32 quantisation of the step coefficient (the
    # projection math stays fp32, `07` §6) is negligible next to the difference.
    from riskproj.projection import finite_difference_jacobian

    j_fd = finite_difference_jacobian(_projected_map(risk, cfg, d), h0.reshape(-1), eps=1e-3)
    assert torch.allclose(j_fd, j_expected, rtol=1e-4, atol=1e-4)


def test_ste_mode_jacobian_is_identity() -> None:
    from riskproj.projection import finite_difference_jacobian

    torch.manual_seed(11)
    d = 5
    a = torch.randn(d, dtype=torch.float64)
    h0 = torch.randn(1, d, dtype=torch.float64)
    c = -(h0 @ a).squeeze()
    risk = sigmoid_risk(a, c)
    cfg = ProjCfg(tau=0.3, K=1, backward="ste")

    # Backward pass: straight-through, so the task gradient is untouched.
    j_backward = _autograd_jacobian(risk, cfg, d, h0)
    assert torch.allclose(j_backward, torch.eye(d, dtype=torch.float64), rtol=0.0, atol=1e-9)

    # Forward pass: the correction is still applied (detach only stops gradients),
    # so a finite-difference probe sees the tangent projection, not the identity.
    j_fd = finite_difference_jacobian(_projected_map(risk, cfg, d), h0.reshape(-1), eps=1e-3)
    assert not torch.allclose(j_fd, torch.eye(d, dtype=torch.float64), atol=1e-3), (
        "STE must be straight-through in the backward pass only: the forward map must "
        "still be corrected, otherwise the mode is a no-op"
    )


def test_second_order_mode_is_available_and_bounded() -> None:
    torch.manual_seed(12)
    d = 4
    a = torch.randn(d, dtype=torch.float64)
    h0 = torch.randn(1, d, dtype=torch.float64)
    c = -(h0 @ a).squeeze()
    risk = sigmoid_risk(a, c)
    h_safe, info = project(h0, risk, ProjCfg(tau=0.3, K=2, backward="second_order"))
    assert torch.isfinite(h_safe).all()
    assert bool(info.usable.any())


# --------------------------------------------------------------------------
# 5.9/5.10 precision and determinism
# --------------------------------------------------------------------------


def test_finite_under_bfloat16_autocast() -> None:
    torch.manual_seed(13)
    b, d = 3, 5
    w = torch.randn(b, d)
    bias = torch.full((b,), 3.0)
    h = torch.randn(b, d)
    cfg = ProjCfg(tau=0.5)
    risk = linear_risk(w, bias)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        h_safe, info = project(h, risk, cfg)
    assert h_safe.dtype == h.dtype, "the returned activation must keep the input dtype"
    assert torch.isfinite(h_safe).all()
    assert bool(info.usable.all())


def test_fixed_seed_gives_identical_output() -> None:
    torch.manual_seed(14)
    b, d = 4, 5
    w = torch.randn(b, d)
    h = torch.randn(b, d)
    bias = torch.full((b,), 3.0)
    cfg = ProjCfg(tau=0.5, K=2)
    risk = linear_risk(w, bias)
    a, ia = project(h, risk, cfg)
    b_, ib = project(h.clone(), risk, cfg)
    assert torch.equal(a, b_)
    assert torch.equal(ia.r_after, ib.r_after)
    assert ia.trace == ib.trace


# --------------------------------------------------------------------------
# 5.13 K-iteration trace: nonlinear risk next to the linearized prediction
# --------------------------------------------------------------------------


def test_trace_records_nonlinear_risk_against_linearized_prediction() -> None:
    torch.manual_seed(15)
    b, d = 3, 4
    h = 5.0 * torch.ones(b, d)
    centre = torch.zeros(b, d)
    risk = quadratic_risk(centre, k=1.0)  # r = 50, far above tau, one step cannot reach it
    cfg = ProjCfg(tau=0.1, K=3, collect_trace=True)
    _h_safe, info = project(h, risk, cfg)
    assert len(info.trace) == 3, "a curved risk needs every K iteration; the trace must show them"
    for entry in info.trace:
        assert "mean_r" in entry and "mean_r_new" in entry and "mean_r_pred" in entry
        assert "mean_gn2" in entry and "n_active" in entry and "mean_step_rel" in entry
        assert entry["n_active"] > 0
    means_new = [e["mean_r_new"] for e in info.trace]
    assert means_new == sorted(means_new, reverse=True), "a convex risk must not increase"
    assert info.r_after is not None and info.r_pred is not None
    assert torch.isfinite(info.r_after).all() and torch.isfinite(info.r_pred).all()
    assert float(info.r_after.mean()) < float(info.r_cand.mean())


def test_trace_is_empty_when_collection_disabled() -> None:
    torch.manual_seed(16)
    h = torch.randn(2, 4)
    w = torch.randn(2, 4)
    cfg = ProjCfg(tau=0.5, collect_trace=False)
    _, info = project(h, linear_risk(w, torch.full((2,), 5.0)), cfg)
    assert info.trace == []


# --------------------------------------------------------------------------
# variants: trust region, per-token, project_fn, thresholds, flow-step gating
# --------------------------------------------------------------------------


def test_trust_region_bounds_the_step() -> None:
    torch.manual_seed(17)
    b, d = 3, 6
    centre = torch.randn(b, d)
    h = centre + 5.0 * torch.ones(b, d)
    cfg = ProjCfg(tau=0.01, max_rel_step=0.01)
    risk = quadratic_risk(centre, k=1.0)
    r_before = risk(h)
    _h_safe, info = project(h, risk, cfg)
    assert float(info.corr_rel.max()) <= 0.01 + 1e-6
    assert bool((info.r_after <= r_before + EXACT).all())
    assert bool(info.usable.all())


def test_per_token_mode_corrects_only_tokens_with_a_gradient() -> None:
    torch.manual_seed(18)
    b, t, d = 1, 3, 4
    h = torch.randn(b, t, d)
    w = torch.zeros(b, t, d)
    w[:, 0] = torch.randn(d)

    def risk(hh: torch.Tensor) -> torch.Tensor:
        return (hh * w).sum(-1).sum(-1) + 5.0

    cfg = ProjCfg(tau=0.5, per_token=True)
    h_safe, info = project(h, risk, cfg)
    assert h_safe.shape == h.shape
    assert torch.equal(h_safe[:, 1:], h[:, 1:]), "tokens without a gradient must be untouched"
    assert not torch.equal(h_safe[:, 0], h[:, 0])
    # At least one token group was usable, so the example is not flagged unusable.
    assert bool(info.usable.all())


def test_project_fn_binds_risk_and_config() -> None:
    torch.manual_seed(19)
    h = torch.randn(2, 4)
    w = torch.randn(2, 4)
    cfg = ProjCfg(tau=0.5)
    fn = project_fn(linear_risk(w, torch.full((2,), 4.0)), cfg)
    h_safe, info = fn(h)
    assert torch.allclose(h_safe, project(h, linear_risk(w, torch.full((2,), 4.0)), cfg)[0])
    assert info.r_cand is not None


def test_tau_conventions() -> None:
    cfg = ProjCfg(tau=0.5, margin=0.1)
    assert math.isclose(setpoint(cfg), 0.4, rel_tol=1e-12)
    assert math.isclose(threshold_in_space(cfg), 0.4, rel_tol=1e-12)
    assert math.isclose(
        threshold_in_space(ProjCfg(tau=0.5, margin=0.1, space="logit")),
        math.log(0.4 / 0.6),
        rel_tol=1e-12,
    )


def test_active_at_flow_step_uses_last_steps_and_tau_ceiling() -> None:
    cfg = ProjCfg(steps_active=3, tau_proj=0.3)
    taus = [1.0 - i / 10.0 for i in range(10)]  # tau: 1.0 -> 0.1, tau=1 is noise
    active = [bool(active_at_flow_step(tau, i, 10, cfg).item()) for i, tau in enumerate(taus)]
    assert active == [False] * 7 + [True, True, True]
    with pytest.raises(ValueError):
        active_at_flow_step(0.1, 10, 10, cfg)


# --------------------------------------------------------------------------
# 5.11 flow convention (owned by the policies worker; skipped until it lands)
# --------------------------------------------------------------------------


def test_terminal_estimate_recovers_a0_when_velocity_is_exact() -> None:
    flow = pytest.importorskip(
        "riskproj.policies.flow",
        reason="flow helpers live in riskproj.policies.flow (another worker's slice)",
    )
    if not hasattr(flow, "terminal_estimate"):
        pytest.skip("riskproj.policies.flow has no terminal_estimate yet")
    torch.manual_seed(20)
    a0 = torch.randn(2, 5, 4)
    eps = torch.randn(2, 5, 4)
    tau = torch.tensor([0.7, 0.2])
    x_tau = (1.0 - tau).view(-1, 1, 1) * a0 + tau.view(-1, 1, 1) * eps  # tau=1 is noise
    v_star = eps - a0  # the flow target
    recovered = flow.terminal_estimate(x_tau, v_star, tau)
    assert torch.allclose(recovered, a0, atol=EXACT)
