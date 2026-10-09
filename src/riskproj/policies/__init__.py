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

__all__ = ["available_policies", "build_policy", "is_available"]


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
