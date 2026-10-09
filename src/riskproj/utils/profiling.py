"""Profiling helpers: G2 peak VRAM and per-decision latency (``05`` §6, ``09`` §1)."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from typing import Any

MB = 2**20


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover
        return False


def reset_peak_vram(device: int | None = None) -> None:
    if not cuda_available():
        return
    import torch

    torch.cuda.reset_peak_memory_stats(device)


def peak_vram_mb(device: int | None = None) -> float:
    """Peak *allocated* VRAM since the last reset (what G2 measures)."""
    if not cuda_available():
        return 0.0
    import torch

    return torch.cuda.max_memory_allocated(device) / MB


@contextlib.contextmanager
def measure_peak_vram(
    device: int | None = None, *, reset: bool = True
) -> Iterator[dict[str, float]]:
    """``with measure_peak_vram() as m: ...`` -> ``m['peak_mb']`` after the block."""
    result = {"peak_mb": 0.0}
    if reset:
        reset_peak_vram(device)
    try:
        yield result
    finally:
        result["peak_mb"] = peak_vram_mb(device)


class CudaTimer:
    """CUDA-event timer with a CPU fallback; reports milliseconds, synchronised."""

    def __init__(self, device: int | None = None) -> None:
        self.device = device
        self._use_cuda = cuda_available()
        self._t0: Any = None
        self._t1: Any = None
        self._wall0 = 0.0

    def __enter__(self) -> CudaTimer:
        import torch

        self._wall0 = time.perf_counter()
        if self._use_cuda:
            self._t0 = torch.cuda.Event(enable_timing=True)
            self._t1 = torch.cuda.Event(enable_timing=True)
            self._t0.record()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._use_cuda and self._t1 is not None:
            self._t1.record()
            import torch

            torch.cuda.synchronize(self.device)

    @property
    def ms(self) -> float:
        if self._use_cuda and self._t0 is not None and self._t1 is not None:
            return float(self._t0.elapsed_time(self._t1))
        return (time.perf_counter() - self._wall0) * 1e3


def count_params(module: Any, *, trainable_only: bool = False) -> int:
    if module is None:
        return 0
    return int(
        sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))
    )


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def memory_report(device: int | None = None) -> dict[str, float]:
    if not cuda_available():
        return {}
    import torch

    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / MB,
        "reserved_mb": torch.cuda.memory_reserved(device) / MB,
        "peak_mb": torch.cuda.max_memory_allocated(device) / MB,
    }


__all__ = [
    "CudaTimer",
    "count_params",
    "cuda_available",
    "human_bytes",
    "measure_peak_vram",
    "memory_report",
    "peak_vram_mb",
    "reset_peak_vram",
]
