"""Unit tests for the backward-pass analysis (``agents/07_projection_operator.md`` §3).

The claim under test: with the risk in the graph and ``g``/``||g||²`` detached, the
projected step has Jacobian ``I - g gᵀ/||g||²``; with ``ste`` it is the identity.
Everything here is checked against finite differences, not against a re-derivation.
"""

from __future__ import annotations

import pytest
import torch

from riskproj.projection import (
    finite_difference_jacobian,
    finite_difference_jacobian_apply,
    linearization_error,
    projected_jacobian,
    projected_jacobian_apply,
    risk_gradient,
)

try:  # tolerances live in tests/, which pytest puts on sys.path via conftest
    from tolerances import EXACT, FD_REL
except ImportError:  # pragma: no cover
    from tests.tolerances import EXACT, FD_REL

pytestmark = pytest.mark.unit


def _quadratic(centre: torch.Tensor, k: float = 1.0):
    def risk(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * k * (h - centre).square().sum(-1)

    return risk


# --------------------------------------------------------------------------
# the projected Jacobian itself
# --------------------------------------------------------------------------


def test_projected_jacobian_properties() -> None:
    torch.manual_seed(0)
    g = torch.randn(6, dtype=torch.float64)
    p = projected_jacobian(g)
    assert torch.allclose(p, p.T, atol=EXACT), "a projection matrix is symmetric"
    assert torch.allclose(p @ p, p, atol=1e-8), "projections are idempotent"
    assert torch.allclose(p @ g, torch.zeros_like(g), atol=1e-8), "it annihilates the gradient"
    # trace = n - 1 is the rank, without relying on a rank tolerance.
    assert torch.isclose(torch.trace(p), torch.tensor(5.0, dtype=torch.float64), atol=1e-8)


def test_projected_jacobian_apply_matches_matrix_form() -> None:
    torch.manual_seed(1)
    g = torch.randn(5, dtype=torch.float64)
    v = torch.randn(5, dtype=torch.float64)
    assert torch.allclose(projected_jacobian_apply(g, v), projected_jacobian(g) @ v, atol=1e-10)


def test_zero_gradient_gives_identity() -> None:
    g = torch.zeros(4, dtype=torch.float64)
    p = projected_jacobian(g)
    assert torch.allclose(p, torch.eye(4, dtype=torch.float64), atol=EXACT)
    v = torch.randn(4, dtype=torch.float64)
    assert torch.allclose(projected_jacobian_apply(g, v), v, atol=EXACT)


# --------------------------------------------------------------------------
# finite differences against the analytic gradient
# --------------------------------------------------------------------------


def test_finite_difference_jacobian_of_a_linear_map_is_its_matrix() -> None:
    torch.manual_seed(2)
    a = torch.randn(3, 4, dtype=torch.float64)
    h = torch.randn(4, dtype=torch.float64)
    j = finite_difference_jacobian(lambda x: a @ x, h)
    assert j.shape == (3, 4)
    assert torch.allclose(j, a, rtol=FD_REL, atol=FD_REL)


def test_finite_difference_jacobian_apply_matches_full_jacobian() -> None:
    torch.manual_seed(3)
    a = torch.randn(4, 4, dtype=torch.float64)
    h = torch.randn(4, dtype=torch.float64)
    v = torch.randn(4, dtype=torch.float64)
    f = lambda x: torch.tanh(a @ x)  # noqa: E731 - a local test double
    assert torch.allclose(
        finite_difference_jacobian_apply(f, h, v),
        finite_difference_jacobian(f, h) @ v,
        atol=FD_REL,
    )


def test_risk_gradient_matches_finite_differences() -> None:
    torch.manual_seed(4)
    d = 5
    centre = torch.randn(d, dtype=torch.float64)
    risk = _quadratic(centre, k=0.7)
    h = torch.randn(1, d, dtype=torch.float64)
    g = risk_gradient(risk, h).reshape(-1)
    g_fd = finite_difference_jacobian(risk, h.reshape(-1)).reshape(-1)
    assert torch.allclose(g, g_fd, rtol=FD_REL, atol=FD_REL)


# --------------------------------------------------------------------------
# linearization error: the quantity E2 measures
# --------------------------------------------------------------------------


def test_linearization_error_vanishes_for_a_linear_risk() -> None:
    torch.manual_seed(5)
    b, d = 3, 6
    w = torch.randn(b, d)
    bias = torch.randn(b)

    def risk(h: torch.Tensor) -> torch.Tensor:
        return (h * w).sum(-1) + bias

    h = torch.randn(b, d)
    u = 0.05 * torch.randn(b, d)
    err = linearization_error(risk, h, u)
    assert torch.allclose(err, torch.zeros(b), atol=1e-4)


def test_linearization_error_of_a_quadratic_equals_half_k_norm_u_squared() -> None:
    torch.manual_seed(6)
    b, d = 4, 5
    k = 0.5
    centre = torch.randn(b, d)
    risk = _quadratic(centre, k=k)
    h = torch.randn(b, d)
    u = 0.1 * torch.randn(b, d)
    err = linearization_error(risk, h, u)
    expected = 0.5 * k * u.square().sum(-1)
    assert torch.allclose(err, expected, rtol=1e-3, atol=1e-4)


def test_linearization_error_grows_with_step_size() -> None:
    torch.manual_seed(7)
    b, d = 2, 4
    centre = torch.randn(b, d)
    risk = _quadratic(centre, k=1.0)
    h = centre + 3.0
    small = linearization_error(risk, h, 0.01 * torch.ones(b, d))
    large = linearization_error(risk, h, 0.5 * torch.ones(b, d))
    assert bool((large > small).all()), "a first-order model must degrade with the step"


def test_risk_gradient_raises_when_the_risk_ignores_the_activation() -> None:
    h = torch.randn(2, 3)
    with pytest.raises(ValueError):
        risk_gradient(lambda x: torch.full((x.shape[0],), 0.5), h)
    with pytest.raises(ValueError):
        linearization_error(lambda x: torch.full((x.shape[0],), 0.5), h, torch.ones_like(h))
