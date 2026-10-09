"""Adapter plumbing shared by every policy adapter.

Three jobs live here:

1. :class:`AdapterContext` — the object ``f_act`` receives. It carries the frozen
   VLM key-values (computed once per chunk), the current flow step, the current
   noisy chunk, and the batch. The projection operator and the eval code build it
   instead of guessing at adapter internals.
2. :class:`PolicyAdapterBase` — site registry, flatten/unflatten round-trip,
   activation capture, LoRA bookkeeping and the "only LoRA is trainable" assertion.
3. :class:`FlowPolicyMixin` — the *one* implementation of the terminal-estimate
   convention for flow heads (``05`` §4): subclasses only implement the tail
   velocity ``tail_velocity(ctx, h, site)`` and the mixin turns it into a chunk.
   Deterministic heads (ACT) override ``f_act`` directly.

LoRA is implemented here as a leaf-module swap rather than through ``peft``: a
``peft`` wrap replaces the model object and would invalidate the module references
the site registry holds (``05`` §2 requires per-site hooks), so adapters swap
matching ``nn.Linear`` leaves in place with :func:`swap_module`. ``lora_B`` is
zero-initialised, which makes "zero LoRA == base policy" hold exactly — the first
clause of the adapter contract suite.
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from riskproj.policies.flow import terminal_estimate
from riskproj.utils.config import LoraCfg

__all__ = [
    "AdapterContext",
    "FlowPolicyMixin",
    "LoRALinear",
    "PolicyAdapterBase",
    "block_index",
    "discover_linear_targets",
    "replace_module",
    "swap_module",
]


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------


@dataclass
class AdapterContext:
    """Everything ``f_act`` needs that is *not* the activation being projected.

    ``tau``/``x_tau`` are required only by flow heads. ``prefix`` holds the frozen
    VLM key-values so the tail after a site can be recomputed without paying for
    the vision-language prefix again (``07`` §6 cost control).
    """

    site: str = "S2"
    tau: Tensor | None = None
    x_tau: Tensor | None = None
    prefix: Any = None
    batch: dict[str, Any] = field(default_factory=dict)
    chunk_horizon: int = 50
    action_dim: int = 4
    state: Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    #: True when the frozen prefix is reused; the operator logs it with the cost.
    prefix_cached: bool = False

    def flow(self) -> tuple[Tensor, Tensor]:
        """Return ``(x_tau, tau)`` or fail loud — never silently invent a flow step."""
        if self.x_tau is None or self.tau is None:
            raise ValueError(
                "this adapter context has no flow state (x_tau/tau); it was built for a "
                "deterministic head or before the first denoising step"
            )
        return self.x_tau, self.tau

    def with_flow(self, x_tau: Tensor, tau: Tensor) -> AdapterContext:
        return AdapterContext(
            site=self.site,
            tau=tau,
            x_tau=x_tau,
            prefix=self.prefix,
            batch=self.batch,
            chunk_horizon=self.chunk_horizon,
            action_dim=self.action_dim,
            state=self.state,
            extras=dict(self.extras),
            prefix_cached=self.prefix_cached,
        )


# --------------------------------------------------------------------------
# LoRA
# --------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """``y = W x + s * B A x`` with ``s = alpha / r`` and ``B`` initialised to zero.

    The base layer's parameters are frozen here, and each adapter's ``attach_lora``
    finishes with :meth:`PolicyAdapterBase.freeze_base`, so
    :meth:`PolicyAdapterBase.assert_no_gradients_outside_lora` holds after attachment.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: int,
        dropout: float = 0.0,
        *,
        lora_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive; got {r}")
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = float(alpha) / float(r)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # zero-init: identity map at step 0
        # Device/dtype must match the base: a bf16 expert receiving fp32 LoRA output
        # (or a CUDA model with CPU adapters) fails at the next Linear. Only the
        # adapters are moved — never the frozen base. `lora_dtype` lets Stage B keep
        # fp32 master adapters over a reduced-precision base, which then *requires*
        # autocast; the default mirrors the base exactly.
        adapt_dtype = lora_dtype or base.weight.dtype
        self.lora_A.to(device=base.weight.device, dtype=adapt_dtype)
        self.lora_B.to(device=base.weight.device, dtype=adapt_dtype)
        for param in self.base.parameters():
            param.requires_grad_(False)

    # nn.Module plumbing -------------------------------------------------
    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(self.lora_dropout(x)))

    # helpers ------------------------------------------------------------
    def delta_weight(self) -> Tensor:
        """The effective low-rank weight update ``s * B A``."""
        return self.scale * (self.lora_B.weight @ self.lora_A.weight)

    def merged_weight(self) -> Tensor:
        """``W + s B A`` — used to check that merging is exact."""
        return self.base.weight + self.delta_weight()

    def merged(self) -> nn.Linear:
        """Return an equivalent plain ``nn.Linear`` (no behavioural change)."""
        out = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            out.weight.copy_(self.merged_weight())
            if self.base.bias is not None:
                out.bias.copy_(self.base.bias)
        return out


def swap_module(root: nn.Module, name: str, new: nn.Module) -> None:
    """Replace ``root.get_submodule(name)`` with ``new`` in place (keeps the tree)."""
    if not name:
        raise ValueError("cannot swap the root module")
    parent_name, _, leaf = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if not hasattr(parent, leaf):
        raise KeyError(
            f"{name!r}: parent module {parent_name or '<root>'!r} has no attribute {leaf!r}"
        )
    setattr(parent, leaf, new)


def replace_module(root: nn.Module, name: str, factory: Any) -> nn.Module:
    """Swap the module at ``name`` for ``factory(old_module)`` and return the new one."""
    old = root.get_submodule(name)
    new = factory(old)
    swap_module(root, name, new)
    return new


_BLOCK_RE = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")


def block_index(module_name: str) -> int | None:
    """Last integer component of a module path (``...layers.3.mlp.down_proj`` -> 3)."""
    found = _BLOCK_RE.findall(module_name)
    return int(found[-1]) if found else None


def discover_linear_targets(
    root: nn.Module,
    suffixes: Sequence[str],
    *,
    last_n_blocks: int = -1,
    require_block_index: bool = False,
) -> list[str]:
    """Names of ``nn.Linear`` leaves whose path ends with one of ``suffixes``.

    ``last_n_blocks > 0`` keeps only modules in the last N numbered blocks — the
    default LoRA placement is ``{q,v,o,down}`` of the last 4 expert blocks (``05``
    §3). A name with no block index is kept only when ``last_n_blocks <= 0``, or
    dropped, unless ``require_block_index`` is False and no indexed module matched
    at all (then the un-numbered matches are returned rather than nothing).
    """
    suffix_tuple = tuple(s.strip() for s in suffixes if s.strip())
    if not suffix_tuple:
        raise ValueError("no LoRA target suffixes given")
    matches = [
        name
        for name, module in root.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] in suffix_tuple and name
    ]
    if not matches or last_n_blocks <= 0:
        return sorted(matches)
    indexed = {name: block_index(name) for name in matches}
    known = {i for i in indexed.values() if i is not None}
    if not known:
        return [] if require_block_index else sorted(matches)
    cutoff = max(known) - int(last_n_blocks) + 1
    kept = [name for name, idx in indexed.items() if idx is not None and idx >= cutoff]
    return sorted(kept if kept else matches)


# --------------------------------------------------------------------------
# adapter base
# --------------------------------------------------------------------------


class PolicyAdapterBase:
    """Site registry, activation capture and LoRA bookkeeping.

    Concrete adapters are expected to be :class:`torch.nn.Module` subclasses as
    well (the parameter helpers rely on that); the base class itself is not one so
    it can be mixed into either an ``nn.Module`` or a thin wrapper.
    """

    name: str = "base"
    #: Site used when the caller does not name one (overridden from ``cfg.site.default``).
    default_site: str = "S2"

    def __init__(
        self, *, chunk_horizon: int, action_dim: int, state_dim: int | None = None
    ) -> None:
        if chunk_horizon < 1:
            raise ValueError(f"chunk_horizon must be >= 1; got {chunk_horizon}")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1; got {action_dim}")
        self._chunk_horizon = int(chunk_horizon)
        self._action_dim = int(action_dim)
        self._state_dim = None if state_dim is None else int(state_dim)
        self._sites: dict[str, Any] = {}
        self._lora: dict[str, LoRALinear] = {}
        self._lora_cfg: LoraCfg | None = None

    # -- geometry --------------------------------------------------------
    @property
    def chunk_horizon(self) -> int:
        return self._chunk_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int | None:
        return self._state_dim

    # -- sites -----------------------------------------------------------
    def sites(self) -> dict[str, Any]:
        """Registered sites: site id -> module whose *output* is projected."""
        return dict(self._sites)

    def register_site(self, site_id: str, module: Any, *, override: bool = False) -> None:
        if site_id in self._sites and not override:
            raise KeyError(f"site {site_id!r} already registered")
        self._sites[site_id] = module

    def site_module(self, site: str) -> Any:
        if site not in self._sites:
            raise KeyError(
                f"site {site!r} is not registered on adapter {self.name!r}; "
                f"available: {sorted(self._sites)} (S0/S0f act on the action chunk, not a module)"
            )
        return self._sites[site]

    def has_site(self, site: str) -> bool:
        return site in self._sites

    def flatten_site(self, h: Tensor) -> Tensor:
        """``[B, ...] -> [B, N]``; the projection always works per example."""
        if h.dim() < 2:
            raise ValueError(
                f"site activation must have a batch dim and at least one feature dim; got {tuple(h.shape)}"
            )
        return h.reshape(h.shape[0], -1)

    def unflatten_site(self, flat: Tensor, like: Tensor) -> Tensor:
        """Inverse of :meth:`flatten_site`; raises rather than reshaping wrongly."""
        if flat.dim() != 2:
            raise ValueError(f"expected a [B, N] tensor; got {tuple(flat.shape)}")
        if flat.shape[0] != like.shape[0]:
            raise ValueError(f"batch mismatch: flat {flat.shape[0]} vs reference {like.shape[0]}")
        if flat.numel() != like.numel():
            raise ValueError(
                f"element mismatch: flat has {flat.numel()} values, reference has {like.numel()}"
            )
        return flat.reshape(like.shape)

    @contextlib.contextmanager
    def forward_hook_at(self, site: str) -> Iterator[list[Tensor]]:
        """Capture the module output at ``site``; yields the capture list."""
        module = self.site_module(site)
        if not isinstance(module, nn.Module):
            raise TypeError(f"site {site!r} is not an nn.Module; cannot hook it")
        captured: list[Tensor] = []

        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, Tensor):
                captured.append(output)
            elif isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
                captured.append(output[0])

        handle = module.register_forward_hook(_hook)
        try:
            yield captured
        finally:
            handle.remove()

    def capture_site_output(self, site: str, run_fn: Callable[[], Any]) -> tuple[Tensor, Any]:
        """Run ``run_fn`` with a hook on ``site``; return ``(activation, raw_output)``.

        ``raw_output`` matters because a site module may wrap its output (e.g.
        ``(hidden, cache)``); :meth:`site_output_override` needs that shape to
        substitute the activation without breaking downstream code.
        """
        module = self.site_module(site)
        if not isinstance(module, nn.Module):
            raise TypeError(f"site {site!r} is not an nn.Module; cannot hook it")
        captured: list[Any] = []

        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            captured.append(output)

        handle = module.register_forward_hook(_hook)
        try:
            run_fn()
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError(
                f"the site module for {site!r} did not run during the pass; it is not on the "
                "active path (confirm the resolved module with scripts/inspect_model.py)"
            )
        raw = captured[-1]
        if isinstance(raw, Tensor):
            return raw, raw
        if isinstance(raw, (tuple, list)) and raw and isinstance(raw[0], Tensor):
            return raw[0], raw
        raise TypeError(
            f"site {site!r} produced {type(raw).__name__}, expected a Tensor or a tuple "
            "starting with one; name the tensor field you mean and record it in LOG.md"
        )

    @contextlib.contextmanager
    def site_output_override(self, site: str, h: Tensor, packaging: Any = None) -> Iterator[None]:
        """Force the site module to return ``h`` for the duration of the block.

        The model-agnostic way to recompute only the *tail* after a site: substitute
        the site's output and let the ordinary forward run on. ``packaging`` mirrors
        the module's own output structure (from :meth:`capture_site_output`).
        """
        module = self.site_module(site)
        if not isinstance(module, nn.Module):
            raise TypeError(f"site {site!r} is not an nn.Module; cannot substitute its output")
        original_forward = module.forward

        def _fake_forward(*_args: Any, **_kwargs: Any) -> Any:
            if isinstance(packaging, tuple):
                return (h, *packaging[1:]) if packaging else h
            return h

        module.forward = _fake_forward  # type: ignore[method-assign]
        try:
            yield
        finally:
            module.forward = original_forward  # type: ignore[method-assign]

    # -- LoRA ------------------------------------------------------------
    @property
    def lora_cfg(self) -> LoraCfg | None:
        return self._lora_cfg

    def lora_modules(self) -> dict[str, LoRALinear]:
        return dict(self._lora)

    def register_lora(self, name: str, module: LoRALinear, cfg: LoraCfg) -> None:
        self._lora[name] = module
        self._lora_cfg = cfg

    def _named_params(self) -> Iterator[tuple[str, nn.Parameter]]:
        if not isinstance(self, nn.Module):
            raise TypeError(
                f"adapter {type(self).__name__} is not an nn.Module; parameter helpers unavailable"
            )
        yield from nn.Module.named_parameters(self)

    def lora_parameter_ids(self) -> set[int]:
        return {id(p) for module in self._lora.values() for p in module.parameters()}

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for _, p in self._named_params() if p.requires_grad]

    def n_trainable(self) -> int:
        return int(sum(p.numel() for p in self.trainable_parameters()))

    def n_total(self) -> int:
        return int(sum(p.numel() for _, p in self._named_params()))

    def assert_no_gradients_outside_lora(self) -> None:
        """Fail loud if a base parameter is still trainable (PEFT leak, ``08`` §2)."""
        if not self._lora:
            raise AssertionError(
                "no LoRA modules registered; call attach_lora() first or this check is meaningless"
            )
        allowed = self.lora_parameter_ids()
        leaked = [n for n, p in self._named_params() if p.requires_grad and id(p) not in allowed]
        if leaked:
            raise AssertionError(
                f"{len(leaked)} non-LoRA parameters are trainable, e.g. {leaked[:5]}; "
                "Stage B must adapt LoRA only (05 §5)"
            )

    def freeze_base(self) -> None:
        """Freeze every parameter that is not part of a registered LoRA module."""
        allowed = self.lora_parameter_ids()
        for _, param in self._named_params():
            if id(param) not in allowed:
                param.requires_grad_(False)

    # -- hooks -----------------------------------------------------------
    @staticmethod
    def apply_hook(h: Tensor, hook: Any = None) -> Tensor:
        """Run an optional intervention hook: ``hook(h)`` may return a replacement ``h``."""
        if hook is None:
            return h
        out = hook(h)
        return h if out is None else out

    def make_context(self, batch: Mapping[str, Any], site: str | None = None) -> AdapterContext:
        """Build the :class:`AdapterContext` for ``batch`` (overridden where the model caches more)."""
        return AdapterContext(
            site=site or self.default_site,
            tau=batch.get("tau"),
            x_tau=batch.get("x_tau"),
            batch=dict(batch),
            chunk_horizon=self._chunk_horizon,
            action_dim=self._action_dim,
            state=batch.get("state"),
        )


# --------------------------------------------------------------------------
# flow mixin
# --------------------------------------------------------------------------


class FlowPolicyMixin:
    """Turns a tail velocity into a terminal action estimate — the convention, once.

    Subclasses implement:

    * ``tail_velocity(ctx, h, site)`` — the downstream map from a site activation
      to the velocity of *this* flow step (this is the differentiable tail the
      projection differentiates through);
    * optionally ``step_velocity(ctx, x_tau, tau, site)`` — the same for a full
      forward from the noisy chunk (used for F4/F5 diagnostics and for B4b).
    """

    def tail_velocity(self, ctx: AdapterContext, h: Tensor, site: str) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} must implement tail_velocity(ctx, h, site) for flow heads"
        )

    def step_velocity(self, ctx: AdapterContext, x_tau: Tensor, tau: Tensor, site: str) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} must implement step_velocity(ctx, x_tau, tau, site)"
        )

    def f_act(self, ctx: AdapterContext, h: Tensor, site: str) -> Tensor:
        """Terminal action estimate from a site activation (``05`` §4)."""
        x_tau, tau = ctx.flow()
        velocity = self.tail_velocity(ctx, h, site)
        return terminal_estimate(x_tau, velocity, tau)

    def terminal_chunk(self, ctx: AdapterContext, x_tau: Tensor, tau: Tensor, site: str) -> Tensor:
        """Terminal estimate from the noisy chunk using the full step (B4b-style)."""
        velocity = self.step_velocity(ctx, x_tau, tau, site)
        return terminal_estimate(x_tau, velocity, tau)
