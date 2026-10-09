"""Flow-matching convention: the single most dangerous detail in the project.

``05`` §4 — **convention trap.** FACT states the convention as
``a_tau = (1 - tau) * a0 + tau * eps`` with **tau = 1 meaning noise**, velocity
target ``v* = eps - a0`` and integration ``tau: 1 -> 0`` [V]. The SmolVLA paper
writes the interpolant the other way round [V], and the LeRobot code convention
is *likely* openpi-style (tau = 1 is noise) [R]. Getting this backwards silently
inverts the "last m denoising steps" rule and makes the critic look at pure noise.

Therefore the convention lives here and nowhere else. Everything downstream
(``f_act``, the projection operator's "last m steps", the noise-schedule control)
calls these functions instead of writing the algebra again.

Two schedules are both exposed on purpose:

* ``"beta"`` — the SmolVLA/LeRobot default sampler. ``tau ~ Beta(1.5, 1.0)``
  gives ``P(tau < 0.2) = 0.2 ** 1.5 = 0.0894``, which reproduces FACT's measured
  "only 8.9% of the gradient mass at tau < 0.2" [V].
* ``"logit_normal"`` — FACT's recommended fix, ``tau = sigmoid(s * z - m)`` with
  ``z ~ N(0, 1)``, ``m = 1.5``, ``s = 1`` [V]. It gives
  ``P(tau < 0.2) = P(z < 1.5 + logit(0.2)) = P(z < 0.1137) ~ 0.545``, i.e. about
  6x more low-noise mass — the B1-LN control that guards the
  noise-schedule confound (``08`` §2, ``10`` A5).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = [
    "FLOW_CONVENTION",
    "assert_convention",
    "euler_taus",
    "flow_interpolate",
    "flow_target",
    "integrate_euler",
    "projection_active",
    "tau_schedule",
    "terminal_estimate",
]

#: Machine-readable statement of the convention this module implements.
FLOW_CONVENTION = "tau1_is_noise"

_TOL = 1e-6


# --------------------------------------------------------------------------
# interpolation / targets
# --------------------------------------------------------------------------


def _broadcast_tau(tau: Tensor | float, like: Tensor) -> Tensor:
    """Broadcast a flow time to the trailing dims of ``like`` (``[B] -> [B,1,1]``)."""
    t = tau if isinstance(tau, Tensor) else torch.tensor(float(tau))
    t = t.to(dtype=like.dtype, device=like.device)
    if t.dim() > like.dim():
        raise ValueError(f"tau has more dims ({t.dim()}) than the tensor it scales ({like.dim()})")
    while t.dim() < like.dim():
        t = t.unsqueeze(-1)
    return t


def check_tau_range(tau: Tensor, *, tol: float = _TOL) -> None:
    """Fail loud if a flow time escapes ``[0, 1]`` (a sign error upstream)."""
    if not torch.isfinite(tau).all():
        raise ValueError("tau contains non-finite values")
    if float(tau.min()) < -tol or float(tau.max()) > 1.0 + tol:
        raise ValueError(
            f"tau must lie in [0, 1] with tau=1 meaning noise; got range [{float(tau.min()):.4f}, {float(tau.max()):.4f}]"
        )


def flow_interpolate(a0: Tensor, eps: Tensor, tau: Tensor | float) -> Tensor:
    """``a_tau = (1 - tau) * a0 + tau * eps``; ``tau=1`` returns ``eps`` exactly."""
    if a0.shape != eps.shape:
        raise ValueError(
            f"a0 and eps must share a shape; got {tuple(a0.shape)} vs {tuple(eps.shape)}"
        )
    t = _broadcast_tau(tau, a0)
    check_tau_range(t)
    return (1.0 - t) * a0 + t * eps


def flow_target(a0: Tensor, eps: Tensor) -> Tensor:
    """The velocity regression target ``v* = eps - a0`` (``d a_tau / d tau``)."""
    if a0.shape != eps.shape:
        raise ValueError(
            f"a0 and eps must share a shape; got {tuple(a0.shape)} vs {tuple(eps.shape)}"
        )
    return eps - a0


def terminal_estimate(x_tau: Tensor, v: Tensor, tau: Tensor | float) -> Tensor:
    """One-step terminal action estimate ``a0_hat = x_tau - tau * v``.

    Exact when ``v == v*``. This is the *definition* of what a flow head's
    ``f_act`` returns, and therefore what the risk critic sees (``06`` §1: the
    candidate chunk is the first ``H_c`` actions of the terminal estimate).
    """
    t = _broadcast_tau(tau, x_tau)
    return x_tau - t * v


def euler_taus(num_steps: int) -> Tensor:
    """Integration times ``[1.0, ..., 0.0]`` (length ``num_steps + 1``)."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    return torch.linspace(1.0, 0.0, num_steps + 1)


def integrate_euler(
    predict_v: Callable[[Tensor, Tensor], Tensor],
    eps: Tensor,
    *,
    num_steps: int = 10,
    schedule: str = "beta",
    record: list[tuple[float, Tensor]] | None = None,
) -> Tensor:
    """Integrate ``tau: 1 -> 0`` with explicit Euler, returning the action chunk.

    ``predict_v(x_tau, tau)`` returns the *velocity* in the convention above, so
    a perfect predictor recovers ``a0`` at the end. ``record`` collects
    ``(tau, x_tau)`` per step for diagnostics.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if schedule not in ("beta", "logit_normal", "uniform"):
        raise ValueError(f"unknown schedule {schedule!r}")
    dt = 1.0 / float(num_steps)
    x = eps
    for i in range(num_steps):
        tau = 1.0 - i * dt
        if record is not None:
            record.append((tau, x))
        v = predict_v(x, torch.full((eps.shape[0],), tau, dtype=eps.dtype, device=eps.device))
        x = x - dt * v
    if record is not None:
        record.append((0.0, x))
    return x


def projection_active(tau: Tensor, tau_proj: float) -> Tensor:
    """Boolean mask: is the projection active at this flow time?

    ``05`` §4: project only at the last ``m`` denoising steps, i.e. where the
    terminal estimate is trustworthy (``tau <= tau_proj``, default 0.3). At large
    ``tau`` the estimate is noise-dominated and the critic is meaningless.
    """
    if not (0.0 < tau_proj <= 1.0):
        raise ValueError(f"tau_proj must be in (0, 1]; got {tau_proj}")
    return tau <= tau_proj


# --------------------------------------------------------------------------
# noise schedules
# --------------------------------------------------------------------------


def _uniform(
    shape: tuple[int, ...],
    generator: torch.Generator | None,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> Tensor:
    # Draw on CPU then move: a CPU generator cannot sample on CUDA directly, and
    # common-random-number pairing requires the *same* generator to work on both.
    return torch.rand(shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )


def _normal(
    shape: tuple[int, ...],
    generator: torch.Generator | None,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )


def tau_schedule(
    name: str,
    shape: torch.Size | tuple[int, ...],
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    ln_m: float = 1.5,
    ln_s: float = 1.0,
    beta_a: float = 1.5,
    beta_b: float = 1.0,
    warmup: bool = False,
) -> Tensor:
    """Sample flow times ``tau`` in ``[0, 1]`` with ``tau = 1`` meaning noise.

    ``"beta"`` is LeRobot/SmolVLA's default sampler; ``"logit_normal"`` is the
    FACT control (B1-LN); ``"uniform"`` exists for ablations. ``shape`` is the
    *sample* shape (usually ``[B]``).

    The general ``Beta(a, b)`` case is deliberately **not** implemented: only
    ``b = 1`` and ``a = 1`` have an inverse-CDF form that a seeded
    :class:`torch.Generator` reproduces exactly, and a silently-wrong sampler is
    precisely the confound FACT identified. Add a correct general sampler (with a
    test against ``torch.distributions.Beta``) before using other parameters.
    """
    shape = tuple(shape)
    dtype = torch.float32 if dtype is None else dtype
    if name == "uniform":
        tau = _uniform(shape, generator, dtype, device)
    elif name == "beta":
        u = _uniform(shape, generator, dtype, device)
        if beta_b == 1.0:
            tau = u.pow(1.0 / float(beta_a))  # CDF: F(t) = t ** a
        elif beta_a == 1.0:
            tau = 1.0 - (1.0 - u).pow(1.0 / float(beta_b))
        else:
            raise ValueError(
                "beta schedule with a != 1 and b != 1 is not implemented; see the docstring. "
                f"got a={beta_a}, b={beta_b}"
            )
    elif name == "logit_normal":
        z = _normal(shape, generator, dtype, device)
        tau = torch.sigmoid(float(ln_s) * z - float(ln_m))
    else:
        raise ValueError(f"unknown schedule {name!r}; expected 'beta', 'logit_normal' or 'uniform'")
    if warmup:
        # Training warm-up: keep the high-noise half only. Documented as [H]; the
        # upstream trainer exposes an equivalent knob [R].
        tau = 0.5 + 0.5 * tau
    return tau.clamp(0.0, 1.0).to(device=device, dtype=dtype)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def assert_convention(
    tau: Tensor, x_tau: Tensor, claimed_a0: Tensor, eps: Tensor, *, atol: float = 1e-5
) -> None:
    """Verify that a claimed ``(x_tau, a0, eps, tau)`` quadruple obeys the convention.

    Raises :class:`AssertionError` with the offending numbers, so a convention bug
    surfaces at the point of the wrong assumption instead of as a bad result.
    """
    t = _broadcast_tau(tau, x_tau)
    expected = (1.0 - t) * claimed_a0 + t * eps
    err = (x_tau - expected).abs().max()
    if float(err) > atol:
        raise AssertionError(
            "flow convention violated: x_tau != (1-tau)*a0 + tau*eps "
            f"(max abs err {float(err):.3e} > atol {atol:.3e}). Check whether the code "
            "you are calling treats tau=1 as noise; see riskproj.policies.flow."
        )
