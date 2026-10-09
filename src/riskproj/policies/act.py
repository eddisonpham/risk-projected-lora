"""ACT adapter (tier A2) — the deterministic-head mechanism control.

**ACT is not a VLA.** It has no vision-language backbone in the SmolVLA sense; it
is a CVAE transformer trained behaviour cloning whose latent is decoded as zero at
inference, which makes the head deterministic (``05`` §1). State it that way in any
write-up. Two method arms do not apply to it:

* **B5 (critic best-of-N)** and **B4b (in-flow guidance)** require a stochastic head
  (flow matching): with a zero latent and no noise, N samples are identical.
* The per-denoising-step projection window (``05`` §4) is meaningless; the whole
  forward is one step, so ``f_act`` from a site is simply the rest of the forward.

Its job in the project is to answer: *does the effect need flow matching at all?*
(ablation A23).
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from riskproj.policies import sites as sites_mod
from riskproj.policies.base import (
    AdapterContext,
    LoRALinear,
    PolicyAdapterBase,
    discover_linear_targets,
    swap_module,
)
from riskproj.types import PolicyOut
from riskproj.utils.config import LoraCfg, ModelCfg
from riskproj.utils.registry import register

logger = logging.getLogger(__name__)

__all__ = ["ACTAdapter"]

_SKIP_REMEDIATION = "install with: python tasks.py bootstrap-sim"
#: Deterministic heads are single-pass; recorded in the log so a reader of a run
#: directory can tell ACT's numbers came from a different regime.
NUM_FLOW_STEPS = 0


@register("policy", "act")
class ACTAdapter(PolicyAdapterBase, nn.Module):
    """Adapter over LeRobot's ``ACTPolicy`` exposing one site activation per forward."""

    name = "act"
    #: Set in __init__; class-level annotation keeps the type stable for mypy.
    _model: nn.Module

    def __init__(
        self,
        cfg: ModelCfg,
        *,
        action_dim: int = 7,
        state_dim: int | None = None,
        device: str = "auto",
        checkpoint: str | None = None,
        load: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        PolicyAdapterBase.__init__(
            self, chunk_horizon=cfg.chunk_size, action_dim=action_dim, state_dim=state_dim
        )
        if cfg.action_head != "regression":
            logger.warning(
                "ACT is a deterministic head but model.action_head=%r; forcing 'regression' "
                "semantics (no terminal estimate, no flow convention)",
                cfg.action_head,
            )
        self.cfg = cfg
        self.default_site = cfg.site.default
        self.dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[cfg.dtype]
        self.device_name = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        self.site_specs = sites_mod.resolve_sites(cfg.site, adapter=None)
        self.matched: dict[str, str] = {}
        self.site_errors: dict[str, str] = {}
        self._packaging: Any = None
        self._checkpoint = checkpoint or cfg.checkpoint
        if load:
            self._load()

    # ------------------------------------------------------------------
    @staticmethod
    def is_available() -> bool:
        """True when the LeRobot stack is importable (tests skip otherwise)."""
        return (
            importlib.util.find_spec("lerobot") is not None
            and importlib.util.find_spec("transformers") is not None
        )

    def _load(self) -> None:
        if self._checkpoint is None:
            raise ValueError(
                "ACTAdapter needs a checkpoint: set model.checkpoint (e.g. a lerobot/act_* "
                "Meta-World checkpoint) or pass checkpoint=..."
            )
        if not self.is_available():
            raise ImportError(f"ACT needs LeRobot and transformers; {_SKIP_REMEDIATION}")
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy  # type: ignore

            policy = ACTPolicy.from_pretrained(self._checkpoint)
        except Exception as exc:
            raise ImportError(
                f"could not construct ACT from checkpoint {self._checkpoint!r}: {exc!r}. "
                f"{_SKIP_REMEDIATION}, then confirm the entry point with scripts/inspect_model.py "
                "and record it in agents/research_log/LOG.md."
            ) from exc
        self._model = policy
        self._model.to(self.device_name)
        if self.dtype is not torch.float32:
            self._model.to(self.dtype)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)
        self._discover_sites()

    def _discover_sites(self) -> None:
        self._sites.clear()
        self.matched, self.site_errors = sites_mod.register_discovered_sites(
            self, self._model, self.site_specs, action_dim=self.action_dim
        )

    @staticmethod
    def applicable_arms() -> list[str]:
        """Method arms that are meaningful on a deterministic head (`08` §2)."""
        return ["B0", "B1", "B2", "B3", "B4a", "B6"]

    def match_report(self) -> dict[str, Any]:
        return {
            "matched": dict(self.matched),
            "errors": dict(self.site_errors),
            "deterministic_head": True,
            "chunk_horizon": self.chunk_horizon,
            "action_dim": self.action_dim,
        }

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def _run_batch(self, batch: dict[str, Any]) -> Tensor:
        """One full policy pass -> action chunk ``[B, H, A]``."""
        try:
            out = self._model(batch)
        except TypeError:
            # Some LeRobot policies expose a keyword-only forward; this is the second
            # documented shape and is tried explicitly rather than guessed.
            out = self._model.forward(**batch)  # type: ignore[operator]
        return _as_chunk(out, self.chunk_horizon, self.action_dim)

    def make_context(self, batch: dict[str, Any], site: str | None = None) -> AdapterContext:
        ctx = super().make_context(batch, site)
        ctx.state = batch.get("state")
        return ctx

    def capture_pass_site(self, batch: dict[str, Any], site: str) -> tuple[Tensor, Tensor]:
        """Run one pass, returning ``(site_activation, chunk)`` and learning packaging."""
        holder: dict[str, Tensor] = {}

        def _run() -> Tensor:
            holder["chunk"] = self._run_batch(batch)
            return holder["chunk"]

        activation, raw = PolicyAdapterBase.capture_site_output(self, site, _run)
        self._packaging = raw
        return activation, holder["chunk"]

    def forward(
        self,
        batch: dict[str, Any],
        site: str | None = None,
        hook: Callable[[Tensor], Tensor] | None = None,
    ) -> PolicyOut:
        """One deterministic pass; ``hook`` (if given) rewrites the site activation."""
        site = site or self.default_site
        if site not in sites_mod.SITE_CATALOG:
            raise KeyError(f"unknown site {site!r}")
        if site in ("S0f",):
            raise ValueError(
                "S0f is in-flow action-space guidance and needs a stochastic head; ACT has a "
                "deterministic head, so use S0 (post-hoc) instead (08 §2)"
            )
        if site == "S0":
            chunk = self._run_batch(batch)
            if hook is not None:
                chunk = self.apply_hook(chunk, hook)
            return PolicyOut(
                chunk=chunk,
                site=site,
                h=None,
                extras={"deterministic_head": True, "num_flow_steps": NUM_FLOW_STEPS},
            )
        activation, chunk = self.capture_pass_site(batch, site)
        h_safe = self.apply_hook(activation, hook)
        if hook is not None and h_safe is not activation:
            with self.site_output_override(site, h_safe, self._packaging):
                chunk = self._run_batch(batch)
        return PolicyOut(
            chunk=chunk,
            site=site,
            h=h_safe,
            extras={
                "deterministic_head": True,
                "num_flow_steps": NUM_FLOW_STEPS,
                "matched_sites": dict(self.matched),
            },
        )

    def f_act(self, ctx: AdapterContext, h: Tensor, site: str) -> Tensor:
        """Deterministic head: the rest of the forward from the site (``05`` §4)."""
        if site in ("S0",):
            return h
        if self._packaging is None:
            self.capture_pass_site(ctx.batch, site)
        with self.site_output_override(site, h, self._packaging):
            return self._run_batch(ctx.batch)

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    def attach_lora(self, cfg: LoraCfg) -> int:
        if cfg.r <= 0:
            return 0
        names = discover_linear_targets(
            self._model, cfg.target_suffixes, last_n_blocks=cfg.last_n_blocks
        )
        if not names:
            available = sorted(
                {
                    n.split(".")[-1]
                    for n, m in self._model.named_modules()
                    if isinstance(m, nn.Linear)
                }
            )
            raise RuntimeError(
                "no Linear matched the LoRA target suffixes "
                f"{list(cfg.target_suffixes)} (last_n_blocks={cfg.last_n_blocks}); "
                f"Linear leaf names present: {available}. Confirm names with "
                "scripts/inspect_model.py and correct configs/model/*.yaml."
            )
        adapted = 0
        for name in names:
            old = self._model.get_submodule(name)
            if isinstance(old, LoRALinear):
                continue
            lora = LoRALinear(old, cfg.r, cfg.alpha, cfg.dropout)
            swap_module(self._model, name, lora)
            self.register_lora(name, lora, cfg)
            adapted += 1
        self._discover_sites()
        self.freeze_base()
        logger.info("attached LoRA to %d modules (r=%d, alpha=%d)", adapted, cfg.r, cfg.alpha)
        return adapted


def _as_chunk(out: Any, chunk_horizon: int, action_dim: int) -> Tensor:
    """Coerce a policy output into an action chunk ``[B, H, A]``."""
    if isinstance(out, Tensor):
        chunk = out
    elif isinstance(out, dict):
        for key in ("action", "actions", "chunk", "pred"):
            if isinstance(out.get(key), Tensor):
                chunk = out[key]
                break
        else:
            raise NotImplementedError(
                f"policy output dict has no action-like key; keys={sorted(out)}"
            )
    elif isinstance(out, (tuple, list)) and out and isinstance(out[0], Tensor):
        chunk = out[0]
    else:
        raise NotImplementedError(f"policy returned {type(out).__name__}; cannot read a chunk")
    if chunk.dim() == 2:
        chunk = chunk.reshape(chunk.shape[0], chunk_horizon, -1)
    if chunk.dim() != 3:
        raise ValueError(f"action chunk must be [B, H, A]; got {tuple(chunk.shape)}")
    if chunk.shape[-1] != action_dim:
        raise ValueError(
            f"action chunk has action dim {chunk.shape[-1]} but the adapter expects {action_dim}"
        )
    return chunk
