"""Policy adapters.

Importing this package registers every adapter (`toy`, `smolvla`, `act`). Heavy
optional dependencies (lerobot, transformers, peft) are imported lazily *inside*
adapter ``__init__``/``_load``, so importing this module is cheap and always safe —
the toy adapter and the whole contract suite run with no checkpoint and no GPU.

Use :func:`build_policy` rather than constructing an adapter directly, so the model
tier stays a config decision (``skills/software_engineering.md`` §3).
"""

from __future__ import annotations

from typing import Any

from riskproj.policies import act, smolvla, toy  # noqa: F401  (import = registration)
from riskproj.protocols import PolicyAdapter
from riskproj.utils.config import ModelCfg
from riskproj.utils.registry import available, get

__all__ = ["available_policies", "build_policy", "build_policy_for_env", "is_available"]


def build_policy(cfg: ModelCfg, **kwargs: Any) -> PolicyAdapter:
    """Construct the adapter named by ``cfg.name`` (registry lookup, not a switch)."""
    cls = get("policy", cfg.name)
    try:
        return cls(cfg, **kwargs)  # type: ignore[no-any-return]
    except TypeError as exc:
        raise TypeError(
            f"could not construct policy {cfg.name!r} from the config: {exc}. "
            "Every adapter takes `cfg` first; pass model-specific geometry as keywords."
        ) from exc


def build_policy_for_env(cfg: Any, **overrides: Any) -> PolicyAdapter:
    """Build the config's policy with the **environment's** geometry (``04`` §1).

    The geometry belongs to :class:`EnvCfg`: the adapter has to match the observations
    and action vectors the task actually produces, and ``ModelCfg`` deliberately carries
    none of it. Passing it from anywhere else is how a 14-wide proprioception vector met
    a 12-wide first layer — a shape error at best, a silently wrong site at worst.

    ``overrides`` are for callers that know better (a test with a synthetic policy);
    they replace the env-derived value rather than adding to it.
    """
    kwargs: dict[str, Any] = {
        "state_dim": int(cfg.env.state_dim or 12),
        "action_dim": int(cfg.env.action_dim),
        "chunk_horizon": int(cfg.env.exec_horizon),
    }
    kwargs.update(overrides)
    return build_policy(cfg.model, **kwargs)


def available_policies() -> list[str]:
    """Names registered in the ``policy`` registry."""
    return available("policy")


def is_available(name: str) -> bool:
    """True when the adapter can actually be loaded here (deps present).

    ``toy`` is always available. The real adapters expose a static ``is_available``;
    if a config names them on a box without LeRobot, the failure is a clear
    ImportError with the bootstrap command instead of a stack trace in a test.
    """
    cls = get("policy", name)
    checker = getattr(cls, "is_available", None)
    return True if checker is None else bool(checker())
