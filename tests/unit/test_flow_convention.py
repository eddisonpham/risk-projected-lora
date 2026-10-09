"""The flow convention is pinned by construction, not by reading the code.

``05`` §4: ``tau = 1`` is noise, ``a_tau = (1 - tau) a0 + tau eps``, ``v* = eps - a0``,
integration ``tau: 1 -> 0``. If any of these tests fails, every "last m denoising
steps" decision and every critic input is suspect — fix the convention, do not
adjust the tolerance.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tests.tolerances import FLOW  # noqa: E402

from riskproj.policies.flow import (  # noqa: E402
    FLOW_CONVENTION,
    assert_convention,
    euler_taus,
    flow_interpolate,
    flow_target,
    integrate_euler,
    projection_active,
    tau_schedule,
    terminal_estimate,
)

pytestmark = pytest.mark.unit

#: Analytic low-noise mass of the two schedules (FACT [V]): Beta(1.5, 1.0) gives
#: 0.2 ** 1.5 = 0.0894, and tau = sigmoid(z - 1.5) with z ~ N(0,1) gives
#: Phi(1.5 + logit(0.2)) = Phi(0.11371) = 0.5453.
BETA_P_BELOW_0P2 = 0.2**1.5
LOGIT_NORMAL_P_BELOW_0P2 = 0.5 * (1.0 + math.erf(0.113706 / math.sqrt(2.0)))


def test_convention_is_declared():
    assert FLOW_CONVENTION == "tau1_is_noise"


def test_endpoints_are_exact():
    torch.manual_seed(0)
    a0 = torch.randn(4, 8, 3)
    eps = torch.randn(4, 8, 3)
    assert torch.equal(flow_interpolate(a0, eps, torch.ones(4)), eps)
    assert torch.equal(flow_interpolate(a0, eps, torch.zeros(4)), a0)
    # scalar tau broadcasts the same way
    assert torch.equal(flow_interpolate(a0, eps, 1.0), eps)


def test_terminal_estimate_recovers_a0_exactly():
    """Spec 07 §5.12: with v == v* the terminal estimate is a0, for any tau."""
    torch.manual_seed(1)
    a0 = torch.randn(5, 10, 4)
    eps = torch.randn(5, 10, 4)
    for tau_value in (1.0, 0.75, 0.5, 0.3, 0.1, 0.0):
        tau = torch.full((5,), tau_value)
        x_tau = flow_interpolate(a0, eps, tau)
        a0_hat = terminal_estimate(x_tau, flow_target(a0, eps), tau)
        assert torch.allclose(a0_hat, a0, atol=FLOW), f"tau={tau_value}"


def test_first_denoising_step_starts_from_noise():
    """At the first Euler step tau=1, so x == eps (the convention trap)."""
    torch.manual_seed(2)
    eps = torch.randn(3, 6, 2)
    taus = euler_taus(10)
    assert float(taus[0]) == 1.0
    assert torch.equal(flow_interpolate(torch.zeros_like(eps), eps, taus[0]), eps)


def test_euler_taus_shape_and_order():
    taus = euler_taus(10)
    assert taus.numel() == 11
    assert float(taus[0]) == 1.0 and float(taus[-1]) == 0.0
    assert torch.all(taus[:-1] > taus[1:])


def test_euler_integration_recovers_a0_with_an_oracle_velocity():
    torch.manual_seed(3)
    a0 = torch.randn(2, 5, 3)
    eps = torch.randn(2, 5, 3)
    target = flow_target(a0, eps)

    def predict_v(x_tau: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return target  # a perfect predictor is constant in x_tau and tau

    out = integrate_euler(predict_v, eps, num_steps=10)
    assert torch.allclose(out, a0, atol=1e-5)
    # and the intermediate state obeys the interpolant at every step
    record: list[tuple[float, torch.Tensor]] = []
    integrate_euler(predict_v, eps, num_steps=10, record=record)
    assert len(record) == 11
    for tau_value, x_tau in record:
        assert torch.allclose(x_tau, flow_interpolate(a0, eps, tau_value), atol=1e-5)


def test_assert_convention_accepts_and_rejects():
    torch.manual_seed(4)
    a0 = torch.randn(3, 4, 2)
    eps = torch.randn(3, 4, 2)
    tau = torch.tensor([0.7, 0.3, 0.9])
    x_tau = flow_interpolate(a0, eps, tau)
    assert_convention(tau, x_tau, a0, eps)
    with pytest.raises(AssertionError, match="flow convention violated"):
        assert_convention(tau, x_tau, a0, eps + 1.0)


def test_tau_range_is_enforced():
    a0 = torch.zeros(2, 3)
    eps = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        flow_interpolate(a0, eps, torch.tensor([1.5, 0.5]))
    with pytest.raises(ValueError):
        flow_interpolate(a0, eps, torch.tensor([-0.2, 0.5]))
    with pytest.raises(ValueError):
        flow_interpolate(a0, eps, torch.tensor([float("nan"), 0.5]))


def test_shape_mismatch_fails_loud():
    with pytest.raises(ValueError):
        flow_interpolate(torch.zeros(2, 3), torch.zeros(2, 4), 0.5)
    with pytest.raises(ValueError):
        flow_target(torch.zeros(2, 3), torch.zeros(2, 4))


def test_schedules_stay_in_range_and_are_seeded():
    for name in ("beta", "logit_normal", "uniform"):
        gen = torch.Generator().manual_seed(0)
        tau = tau_schedule(name, (4096,), generator=gen)
        assert tau.shape == (4096,)
        assert float(tau.min()) >= 0.0 and float(tau.max()) <= 1.0
        gen2 = torch.Generator().manual_seed(0)
        assert torch.equal(tau, tau_schedule(name, (4096,), generator=gen2))
        gen3 = torch.Generator().manual_seed(1)
        assert not torch.allclose(tau, tau_schedule(name, (4096,), generator=gen3))
    with pytest.raises(ValueError, match="unknown schedule"):
        tau_schedule("nope", (4,))


def test_logit_normal_puts_about_six_times_more_mass_below_0p2():
    """The FACT confound, asserted quantitatively (this is the B1-LN control)."""
    n = 100_000
    beta = tau_schedule("beta", (n,), generator=torch.Generator().manual_seed(0))
    ln = tau_schedule("logit_normal", (n,), generator=torch.Generator().manual_seed(0))
    p_beta = float((beta < 0.2).float().mean())
    p_ln = float((ln < 0.2).float().mean())
    assert abs(p_beta - BETA_P_BELOW_0P2) < 0.01, f"beta schedule mass {p_beta:.4f}"
    assert abs(p_ln - LOGIT_NORMAL_P_BELOW_0P2) < 0.015, f"logit-normal mass {p_ln:.4f}"
    assert p_ln > 3.0 * p_beta


def test_beta_general_parameters_refuse_rather_than_guess():
    with pytest.raises(ValueError, match="not implemented"):
        tau_schedule("beta", (8,), beta_a=2.0, beta_b=3.0)


def test_warmup_shifts_to_the_high_noise_half():
    tau = tau_schedule("beta", (2048,), generator=torch.Generator().manual_seed(0), warmup=True)
    assert float(tau.min()) >= 0.5


def test_projection_active_window():
    tau = torch.tensor([1.0, 0.5, 0.3, 0.29, 0.0])
    active = projection_active(tau, 0.3)
    assert active.tolist() == [False, False, True, True, True]
    with pytest.raises(ValueError):
        projection_active(tau, 0.0)
