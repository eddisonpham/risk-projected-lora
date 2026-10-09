"""Lightweight registries so extensions need one file + one YAML entry.

Usage::

    @register("metric", "proxy_gold")
    class ProxyGold: ...

    metric = build("metric", "proxy_gold", cfg)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, Any]] = {}

KINDS = ("policy", "task", "metric", "method", "critic", "projector", "env", "dataset")


def register(kind: str, name: str, *, override: bool = False) -> Callable[[T], T]:
    """Decorator registering ``name`` under ``kind``."""

    def deco(obj: T) -> T:
        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket and not override:
            raise KeyError(f"{kind}:{name} already registered by {bucket[name]!r}")
        bucket[name] = obj
        return obj

    return deco


def get(kind: str, name: str) -> Any:
    try:
        return _REGISTRY[kind][name]
    except KeyError as exc:  # pragma: no cover - message quality path
        known = ", ".join(sorted(_REGISTRY.get(kind, {}))) or "<none>"
        raise KeyError(f"unknown {kind} {name!r}; registered: {known}") from exc


def build(kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
    return get(kind, name)(*args, **kwargs)


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))


def summary() -> dict[str, list[str]]:
    return {kind: available(kind) for kind in sorted(_REGISTRY)}


__all__ = ["KINDS", "available", "build", "get", "register", "summary"]
