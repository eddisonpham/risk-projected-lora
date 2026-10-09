"""Risk-feasibility projection of a site activation.

Public entry point: :func:`project`. See ``agents/07_projection_operator.md`` for
the operator, its variants and its unit tests, and ``agents/01_research_spec.md``
§3 for the half-space step that replaces the original tangent-plane idea.

The correction is a *first-order local* step, never an exact nonlinear
projection: the nonlinear risk is re-evaluated after the step and logged next to
the linearized prediction so the linearization error (experiment E2) is a
first-class number rather than an assumption.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

from riskproj.protocols import RiskFn
from riskproj.types import FallbackCode, ProjectionInfo
from riskproj.utils.config import ProjCfg

__all__ = [
    "EPS",
    "active_at_flow_step",
    "project",
    "project_fn",
    "setpoint",
    "threshold_in_space",
]

#: Guard for divisions whose denominator is a norm that can legitimately be zero.
EPS = 1e-12


# --------------------------------------------------------------------------
# tau conventions — the only place that knows about prob vs logit space
# --------------------------------------------------------------------------


def setpoint(cfg: ProjCfg) -> float:
    """Risk level the operator projects onto: ``tau - margin`` (``07`` §1).

    A margin makes the constraint slightly conservative, which matters because
    the step is a linearization and the nonlinear risk can land above ``tau``.
    """
    return cfg.tau - cfg.margin


def threshold_in_space(cfg: ProjCfg) -> float:
    """The setpoint expressed in the space ``risk_fn`` reports (``01`` §6 item 3).

    In ``"logit"`` space the constraint is ``z <= logit(setpoint)``, which avoids
    the sigmoid-saturation regime where ``d sigma/dz`` — and therefore the whole
    gradient — vanishes.
    """
    s = setpoint(cfg)
    if not 0.0 < s < 1.0:
        raise ValueError(f"setpoint must be in (0,1); got {s}")
    if cfg.space == "logit":
        return math.log(s / (1.0 - s))
    if cfg.space == "prob":
        return s
    raise ValueError(f"unknown constraint space {cfg.space!r}")


def active_at_flow_step(
    tau: Tensor | float,
    flow_step_index: int,
    num_flow_steps: int,
    cfg: ProjCfg,
) -> Tensor:
    """Whether projection acts at denoising step ``flow_step_index`` of ``num_flow_steps``.

    Convention (``05`` §4, ``01`` §6 item 2): ``tau = 1`` is **noise** and
    ``a_tau = (1 - tau) a0 + tau * eps``, so denoising runs ``tau: 1 -> 0`` and the
    last step has ``tau = 0``. Projection is active only when *both* hold:

    * the step is among the last ``cfg.steps_active`` steps, and
    * ``tau <= cfg.tau_proj`` — at large ``tau`` the terminal estimate is
      noise-dominated and the critic's risk is meaningless.

    Returns a boolean tensor shaped like ``tau``. The two conditions are kept
    deliberately redundant: the index condition bounds cost, the ``tau``
    condition bounds meaning, and the upstream code is expected to assert they
    agree (``tests/unit/test_flow_convention.py``).
    """
    if not 0 <= flow_step_index < num_flow_steps:
        raise ValueError(f"flow_step_index {flow_step_index} outside [0, {num_flow_steps})")
    if num_flow_steps < 1:
        raise ValueError("num_flow_steps must be >= 1")
    t = torch.as_tensor(tau, dtype=torch.float32)
    late = flow_step_index >= max(0, num_flow_steps - cfg.steps_active)
    mask = (
        torch.zeros_like(t, dtype=torch.bool) if not late else torch.ones_like(t, dtype=torch.bool)
    )
    return mask & (t <= cfg.tau_proj)


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _eval_risk(risk_fn: RiskFn, h: Tensor, n: int) -> Tensor:
    """Evaluate ``risk_fn`` and normalise to a finite-agnostic ``[B]`` float32 tensor."""
    raw = risk_fn(h)
    if not isinstance(raw, Tensor):
        raw = torch.as_tensor(raw, device=h.device)
    r = raw.reshape(-1)
    if r.shape[0] != n:
        raise ValueError(
            f"risk_fn must return one risk per example: expected {n}, got {r.shape[0]} "
            f"(raw shape {tuple(raw.shape)})"
        )
    return r.to(torch.float32)


def _detached_risk(risk_fn: RiskFn, h: Tensor, n: int) -> Tensor:
    """Nonlinear risk at ``h`` with no graph retained — pure measurement."""
    with torch.no_grad():
        return _eval_risk(risk_fn, h.detach(), n)


def _risk_grad(total: Tensor, h: Tensor, *, create_graph: bool) -> Tensor | None:
    """``d total / d h``.

    Returns ``None`` (rather than raising) when ``h`` is not used by ``total``:
    a risk functional that ignores the activation is a *finding* to be logged as
    an unusable gradient, not a crash to be swallowed by a caller's try/except.
    """
    try:
        (g,) = torch.autograd.grad(
            total,
            h,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
    except RuntimeError:
        return None
    return g


def _group_view(x: Tensor, per_token: bool) -> Tensor:
    """Flatten trailing dims into one group per example (or one group per token).

    ``per_token=False``: ``[B, ...] -> [B, N]`` — all tokens of an example are
    corrected jointly (the default; ``07`` §5.5).
    ``per_token=True``: ``[B, T, D] -> [B, T, D]`` — each token gets its own norm.
    """
    if per_token and x.dim() >= 3:
        return x.flatten(2)
    return x.flatten(1)


def _expand_mask(mask: Tensor, like: Tensor) -> Tensor:
    """Broadcast an example-level ``[B]`` mask to the trailing dims of ``like``."""
    return mask.view(*mask.shape, *([1] * (like.dim() - mask.dim())))


def _ctx_extras(ctx: Any) -> dict[str, float]:
    """Numeric entries of caller telemetry, copied into every trace record.

    Lets the projection events be joined back to the episode/decision log
    (episode id, step index) without the operator having to know about them.
    """
    if not isinstance(ctx, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in ctx.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[f"ctx_{key}"] = float(value)
    return out


# --------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------


def project(
    h_c: Tensor,
    risk_fn: RiskFn,
    cfg: ProjCfg,
    *,
    ctx: Any = None,
) -> tuple[Tensor, ProjectionInfo]:
    """Project a site activation onto the risk-feasible half-space (public entry point).

    Autograd is required *inside* this call — the risk value and its gradient are
    the whole point — so the operator re-enables it for its body. Without this,
    calling ``project`` from inside ``torch.no_grad()`` (or any eval loop that
    wraps inference in ``no_grad``/``inference_mode``) would find `r`
    disconnected from the activation and would mark **every** example
    "unusable gradient", silently disabling the method. ``05`` §6 requires
    autograd around the tail for exactly this reason. The full contract is in
    :func:`_project_impl`.
    """
    with torch.enable_grad():
        return _project_impl(h_c, risk_fn, cfg, ctx=ctx)


def _project_impl(
    h_c: Tensor,
    risk_fn: RiskFn,
    cfg: ProjCfg,
    *,
    ctx: Any = None,
) -> tuple[Tensor, ProjectionInfo]:
    """Project a site activation onto the risk-feasible half-space.

    Shapes: ``h_c`` is ``[B, ...]``; ``risk_fn(h) -> [B]`` in the space named by
    ``cfg.space``. Returns ``(h_safe, info)``.

    Per example, with ``thr = threshold_in_space(cfg)`` and ``r = risk_fn(h)``:

    * ``r <= thr`` — the candidate is already feasible: it is returned
      **bitwise unchanged** and no gradient is taken (no cost, no drift).
    * ``r > thr`` and ``||g||^2 >= eta`` — ``h <- h - (r - thr)/||g||^2 * g``.
    * ``||g||^2 < eta`` (or the gradient is unavailable, or the risk is not
      finite) — the example is flagged **unusable**, its activation is left
      unchanged, and ``fallback_code`` is set to ``F1`` so the caller consults the
      fallback ladder. It is never reported as corrected.

    ``cfg.backward`` selects what the backward pass does (``07`` §3): ``"ste"``
    gives an identity Jacobian, ``"projected"`` gives ``I - g gᵀ/||g||²`` (the
    numerator's gradient ``g`` and its squared norm are detached while the risk
    stays in the graph), ``"second_order"`` additionally differentiates ``g``
    through the graph at extra cost.

    Gradient bookkeeping: critic parameters never receive gradients from this
    call (only ``h`` is passed to ``autograd.grad``), while any tensor upstream of
    ``h_c`` does, provided ``h_c`` carries a graph.

    ``ProjectionInfo.r_pred`` is the analytic first-order prediction
    ``r(h_c) + g0 · (h_safe - h_c)`` using the gradient at the *initial*
    activation, which is exactly ``linearization_error`` with the sign flipped —
    so ``r_after - r_pred`` is the E2 quantity, comparable across K.
    ``ctx`` is optional caller telemetry: its numeric entries are copied into
    every trace record (as ``ctx_<key>``) so a decision can be joined back to the
    episode log; the operator is otherwise pure.
    """
    if h_c.dim() < 2:
        raise ValueError(f"h_c must be [B, ...]; got {tuple(h_c.shape)}")
    if not torch.is_floating_point(h_c):
        raise TypeError(f"h_c must be a floating-point tensor; got {h_c.dtype}")
    n = h_c.shape[0]
    if n == 0:
        raise ValueError("h_c has an empty batch dimension")
    device = h_c.device
    thr = threshold_in_space(cfg)

    active = torch.zeros(n, dtype=torch.bool, device=device)
    usable = torch.zeros(n, dtype=torch.bool, device=device)
    unusable = torch.zeros(n, dtype=torch.bool, device=device)
    k_used = torch.zeros(n, dtype=torch.long, device=device)
    trace: list[dict[str, float]] = []

    h = h_c
    r_cand: Tensor | None = None
    g0: Tensor | None = None
    trace_extra = _ctx_extras(ctx)

    for k in range(cfg.K):
        hk = h if h.requires_grad else h.detach().requires_grad_(True)
        r = _eval_risk(risk_fn, hk, n)
        if k == 0:
            r_cand = r.detach().clone()
        # A non-finite risk cannot be certified feasible and cannot be corrected,
        # so it is treated as active-but-unusable and handed to the fallback
        # ladder. Silently passing it would be exactly the failure mode that
        # CLAUDE.md non-negotiable #4 bans.
        finite = torch.isfinite(r)
        r_ok = torch.where(finite, r, torch.zeros_like(r))
        excess = r_ok - thr
        active_k = (excess > 0) | ~finite
        active |= active_k
        if not bool(active_k.any()):
            break

        g = _risk_grad(r_ok.sum(), hk, create_graph=(cfg.backward == "second_order"))
        if g is None:
            unusable |= active_k
            trace.append(
                {
                    "k": float(k),
                    "n_active": float(active_k.sum()),
                    "n_usable": 0.0,
                    "grad_unavailable": 1.0,
                    **trace_extra,
                }
            )
            break
        if k == 0:
            g0 = g.detach().clone()

        hv = _group_view(h, cfg.per_token)
        gv = _group_view(g, cfg.per_token)
        gn2 = gv.to(torch.float32).square().sum(-1)  # [B] or [B, T]
        gn2_ok = torch.isfinite(gn2) & (gn2 >= cfg.eta)

        # active_k is [B]; per-token mode lets individual token groups be usable.
        active_g = _expand_mask(active_k, gn2) if gn2.dim() == 1 else active_k.unsqueeze(-1)
        finite_g = finite if gn2.dim() == 1 else _expand_mask(finite, gn2)
        usable_g = active_g & gn2_ok & finite_g
        usable_b = usable_g.any(-1) if usable_g.dim() > 1 else usable_g
        usable |= usable_b
        unusable |= active_k & ~usable_b

        # Numerator: the risk stays in the graph except in "ste" (identity
        # Jacobian). "second_order" keeps both the numerator and the gradient
        # direction in the graph, so it is a strict superset of "projected": the
        # tangent term I - g gᵀ/||g||² *plus* the dg/dh terms (07 §3).
        # The denominator is always detached — that is what makes the "projected"
        # Jacobian exactly I - g gᵀ/||g||² rather than something unstated.
        ex = excess.detach() if cfg.backward == "ste" else excess
        # ``gn2`` is [B] for a joint projection and [B, T] in per-token mode, so the
        # coefficient must be shaped like it. A [B] coefficient broadcast against a
        # per-token gradient is a hard shape error for any T > 1 -- and would be
        # invisible at T == 1, which is why the per-token test uses T = 3.
        coef = ex.to(torch.float32).clamp_min(0.0).reshape(
            -1, *([1] * (gn2.dim() - 1))
        ) / gn2.detach().clamp_min(cfg.eta)
        gd = g if cfg.backward == "second_order" else g.detach()
        gdv = _group_view(gd, cfg.per_token)
        u = coef.unsqueeze(-1) * gdv

        if cfg.max_rel_step is not None:
            u_norm = u.norm(dim=-1)
            h_norm = hv.detach().norm(dim=-1)
            scale = (cfg.max_rel_step * h_norm / (u_norm + EPS)).clamp(max=1.0)
            u = u * scale.unsqueeze(-1)

        u = torch.where(_expand_mask(usable_g, u), u, torch.zeros_like(u))
        h = (hv - u.to(hv.dtype)).reshape(h.shape)
        k_used = torch.where(
            usable_b if usable_b.dim() == 1 else usable_b.reshape(n),
            torch.full_like(k_used, k + 1),
            k_used,
        )

        if cfg.collect_trace:
            with torch.no_grad():
                r_new = _detached_risk(risk_fn, h, n)
            pred = r_ok.detach() - torch.where(
                usable_b if usable_b.dim() == 1 else usable_b.reshape(n),
                excess.detach(),
                torch.zeros_like(excess),
            )
            trace.append(
                {
                    "k": float(k),
                    "n_active": float(active_k.sum()),
                    "n_usable": float(usable_b.sum()),
                    "mean_r": float(r_ok.detach().mean()),
                    "mean_r_new": float(r_new.mean()),
                    "mean_r_pred": float(pred.mean()),
                    "mean_gn2": float(gn2.detach().to(torch.float32).mean()),
                    "mean_step_rel": float(
                        (
                            u.detach().flatten(1).norm(dim=1)
                            / hv.detach().flatten(1).norm(dim=1).clamp_min(EPS)
                        ).mean()
                    ),
                    **trace_extra,
                }
            )

    h_safe = h
    r_after = _detached_risk(risk_fn, h_safe, n)
    r_pred = (
        r_cand.clone()
        if (r_cand is None or g0 is None)
        else r_cand
        + (
            g0.to(torch.float32).flatten(1)
            * (h_safe.detach() - h_c.detach()).to(torch.float32).flatten(1)
        ).sum(-1)
    )
    u_total = (h_safe.detach() - h_c.detach()).to(torch.float32)
    corr_rel = u_total.flatten(1).norm(dim=1) / h_c.detach().to(torch.float32).flatten(1).norm(
        dim=1
    ).clamp_min(EPS)
    if r_cand is None:  # pragma: no cover - K >= 1 always evaluates once
        raise RuntimeError("projection loop never evaluated the risk")

    fallback_code = torch.where(
        unusable & torch.tensor(bool(cfg.fallback), device=device),
        torch.full((n,), FallbackCode.F1_BASE_ACTION, dtype=torch.long, device=device),
        torch.full((n,), FallbackCode.NONE, dtype=torch.long, device=device),
    )

    info = ProjectionInfo(
        r_cand=r_cand.detach(),
        r_after=r_after,  # nonlinear, measured after the last step
        r_pred=r_pred.detach(),  # first-order, from the initial gradient
        gn2=(g0.detach().to(torch.float32).flatten(1).square().sum(-1) if g0 is not None else None),
        active=active,
        usable=usable,
        unusable=unusable,
        k_used=k_used,
        corr_rel=corr_rel,
        ens_std=None,  # filled by the caller from the critic ensemble, when present
        fallback_code=fallback_code,
        trace=trace,
    )
    return h_safe, info


def project_fn(risk_fn: RiskFn, cfg: ProjCfg) -> Callable[[Tensor], tuple[Tensor, ProjectionInfo]]:
    """Bind a risk functional and config into a one-argument ``h -> (h_safe, info)``.

    Lets an inference wrapper (``policies/``) hand the operator a site activation
    without re-deriving the config-driven pieces at every denoising step.
    """

    def _project(h: Tensor) -> tuple[Tensor, ProjectionInfo]:
        return project(h, risk_fn, cfg)

    return _project
