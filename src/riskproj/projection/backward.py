"""Gradient/Jacobian tools behind the projection operator (``07`` §3).

The derived result this module pins down: with the risk ``r`` kept in the graph
and both ``g = ∇_h r`` and ``||g||²`` **detached**, the projected backward step

    h_safe = h - relu(r(h) - τ)/||g||² · g

has Jacobian ``∂h_safe/∂h = I - g gᵀ/||g||²``. So the *old* tangent-plane
projection does not disappear from the method — it reappears in the backward
pass, where it removes the risk-normal component of the task gradient at the
activation. Ablating the backward modes (A15) isolates whether any training-time
benefit comes from exactly this effect.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from riskproj.protocols import RiskFn

__all__ = [
    "FD_EPS",
    "finite_difference_jacobian",
    "finite_difference_jacobian_apply",
    "linearization_error",
    "projected_jacobian",
    "projected_jacobian_apply",
    "risk_gradient",
]

#: Central-difference step. float64 in the tests, so the O(eps²) truncation error
#: and the O(1/eps) round-off meet comfortably inside tests/tolerances.FD_REL.
FD_EPS = 1e-5


def _grad_or_none(total: Tensor, h: Tensor, *, create_graph: bool = False) -> Tensor | None:
    """``autograd.grad`` that maps ``RuntimeError`` (constant output, unwired input)
    onto ``None`` so callers can distinguish *no gradient exists* from a bug."""
    try:
        (g,) = torch.autograd.grad(
            total, h, retain_graph=True, create_graph=create_graph, allow_unused=True
        )
    except RuntimeError:
        return None
    return g


def risk_gradient(risk_fn: RiskFn, h: Tensor, *, create_graph: bool = False) -> Tensor:
    """``∇_h r`` for a batched risk functional.

    Uses ``autograd.grad`` on ``r.sum()`` (valid because examples are
    independent) and returns the gradient with the shape of ``h``. Raises
    ``ValueError`` when the risk functional does not depend on the activation at
    all — that is a broken critic/risk wiring, not an unusable gradient.
    """
    with torch.enable_grad():
        hk = h if h.requires_grad else h.detach().requires_grad_(True)
        r = risk_fn(hk)
        if not isinstance(r, Tensor):
            r = torch.as_tensor(r, device=hk.device)
        r = r.reshape(-1).to(torch.float32)
        g = _grad_or_none(r.sum(), hk, create_graph=create_graph)
    if g is None:
        raise ValueError("risk_fn does not depend on the activation: no gradient exists")
    return g


def projected_jacobian(g: Tensor) -> Tensor:
    """``I - g gᵀ/||g||²`` for a flattened gradient ``g`` (``[n] -> [n, n]``).

    Materialised form, for tests and analysis only: it is ``n x n`` and must never
    be built inside a training loop. Use :func:`projected_jacobian_apply` there.
    """
    v = g.reshape(-1)
    gn2 = v.to(torch.float64).square().sum()
    if float(gn2) <= 0.0:
        return torch.eye(v.numel(), device=v.device, dtype=v.dtype)
    eye = torch.eye(v.numel(), device=v.device, dtype=v.dtype)
    return eye - torch.outer(v, v) / gn2.to(v.dtype)


def projected_jacobian_apply(g: Tensor, v: Tensor) -> Tensor:
    """``(I - g gᵀ/||g||²) v`` without materialising the matrix."""
    gv = g.reshape(-1).to(torch.float64)
    vv = v.reshape(-1).to(torch.float64)
    gn2 = gv.square().sum()
    if float(gn2) <= 0.0:
        return vv.to(v.dtype)
    return (vv - gv * (gv @ vv) / gn2).to(v.dtype)


def finite_difference_jacobian(
    f: Callable[[Tensor], Tensor],
    h: Tensor,
    *,
    eps: float = FD_EPS,
) -> Tensor:
    """Central-difference Jacobian of ``f: [n] -> [m]`` at ``h``; returns ``[m, n]``.

    Evaluated in float64 and without a graph; ``f`` must accept float64 input and
    return a tensor. Row ``i`` of the result is ``d f_i / d h``.
    """
    x = h.reshape(-1).to(torch.float64).detach()
    n = x.numel()
    cols: list[Tensor] = []
    with torch.no_grad():
        for i in range(n):
            e = torch.zeros_like(x)
            e[i] = eps
            f_plus = torch.as_tensor(f(x + e)).reshape(-1).to(torch.float64)
            f_minus = torch.as_tensor(f(x - e)).reshape(-1).to(torch.float64)
            cols.append((f_plus - f_minus) / (2.0 * eps))
    return torch.stack(cols, dim=1)


def finite_difference_jacobian_apply(
    f: Callable[[Tensor], Tensor],
    h: Tensor,
    v: Tensor,
    *,
    eps: float = FD_EPS,
) -> Tensor:
    """Directional derivative ``J v`` by central differences (one extra evaluation pair)."""
    x = h.reshape(-1).to(torch.float64).detach()
    d = v.reshape(-1).to(torch.float64).detach()
    with torch.no_grad():
        f_plus = torch.as_tensor(f(x + eps * d)).reshape(-1).to(torch.float64)
        f_minus = torch.as_tensor(f(x - eps * d)).reshape(-1).to(torch.float64)
    return ((f_plus - f_minus) / (2.0 * eps)).to(v.dtype)


def linearization_error(risk_fn: RiskFn, h: Tensor, u: Tensor) -> Tensor:
    """``r(h + u) - (r(h) + gᵀ u)``, per example — the quantity E2 measures.

    Both terms are computed without a graph, so this is a *measurement* of how
    good the first-order model is over a step of size ``‖u‖``; it is deliberately
    not differentiable and must not be used to optimize anything.
    """
    with torch.enable_grad():
        hk = h.detach().requires_grad_(True)
        r = risk_fn(hk)
        if not isinstance(r, Tensor):
            r = torch.as_tensor(r, device=hk.device)
        r = r.reshape(-1).to(torch.float32)
        g = _grad_or_none(r.sum(), hk)
    if g is None:
        raise ValueError("risk_fn does not depend on the activation: linearization is undefined")
    with torch.no_grad():
        r_new = risk_fn(h.detach() + u.detach())
        if not isinstance(r_new, Tensor):
            r_new = torch.as_tensor(r_new, device=h.device)
        r_new = r_new.reshape(-1).to(torch.float32)
        linear = r.detach() + (g.to(torch.float32) * u.detach().to(torch.float32)).flatten(1).sum(
            -1
        )
    return r_new - linear
