"""Systems cost: latency, peak VRAM and parameter counts (``05`` §6, ``09`` §1).

Every results table carries this (`agents/09` §5): a method that wins by 2 pp while
tripling latency inside a 7 GB budget is not a method anyone can use, and the G2
gate upstream decides whether the model choice is viable at all.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from riskproj.utils.profiling import (
    CudaTimer,
    count_params,
    cuda_available,
    measure_peak_vram,
    memory_report,
)

__all__ = [
    "SystemsCost",
    "check_budget",
    "measure_inference_cost",
    "measure_training_cost",
    "systems_table",
]

MB_PER_GB = 1024.0


@dataclass
class SystemsCost:
    """One phase's cost. ``peak_vram_mb`` is per phase, measured after a reset."""

    latency_ms_per_decision: dict[str, float] = field(default_factory=dict)
    peak_vram_mb: dict[str, float] = field(default_factory=dict)
    trainable_params: int = 0
    total_params: int = 0
    steps_per_s: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "latency_ms_per_decision": dict(self.latency_ms_per_decision),
            "peak_vram_mb": dict(self.peak_vram_mb),
            "trainable_params": int(self.trainable_params),
            "total_params": int(self.total_params),
            "steps_per_s": float(self.steps_per_s),
            **self.extras,
        }


def _percentiles(samples: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(samples), dtype=float)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "n": float(arr.size),
    }


def measure_inference_cost(
    policy: Any,
    batch: Mapping[str, Any],
    *,
    plan: Callable[[], Any] | None = None,
    n_warmup: int = 3,
    n_iters: int = 20,
    device: str = "cpu",
    label: str = "inference",
) -> SystemsCost:
    """Time a policy forward (and optionally a full planning call) in ms per decision.

    ``plan`` should sweep the whole inference path for the method under test (policy
    + critic + projection); timing only the forward pass understates the overhead the
    paper has to report.
    """
    if n_iters < 1:
        raise ValueError("n_iters must be >= 1")
    device_index = 0 if (device.startswith("cuda") and cuda_available()) else None
    with measure_peak_vram(device_index):
        # No `no_grad` here on purpose: the projection path needs autograd around the
        # tail, so wrapping the whole measurement in no_grad would understate both the
        # time and the memory of the method under test (05 §6).
        for _ in range(max(0, n_warmup)):
            _forward(policy, batch)
        forward_samples: list[float] = []
        plan_samples: list[float] = []
        for _ in range(n_iters):
            timer = CudaTimer(device=device_index)
            with timer:
                _forward(policy, batch)
            forward_samples.append(float(timer.ms))
            if plan is not None:
                timer = CudaTimer(device=device_index)
                with timer:
                    plan()
                plan_samples.append(float(timer.ms))
        peak = _peak_after_block(device_index)

    latency = {f"{label}_forward": float(np.mean(forward_samples))}
    if plan_samples:
        latency[f"{label}_plan"] = float(np.mean(plan_samples))
    return SystemsCost(
        latency_ms_per_decision=latency,
        peak_vram_mb={label: peak},
        trainable_params=count_params(policy, trainable_only=True),
        total_params=count_params(policy),
        extras={"latency_distribution": _percentiles(forward_samples)},
    )


def _forward(policy: Any, batch: Mapping[str, Any]) -> Any:
    forward = getattr(policy, "forward", None)
    if not callable(forward):
        raise TypeError(f"{type(policy).__name__} has no forward()")
    return forward(dict(batch))


def _peak_after_block(device_index: int | None) -> float:
    report = memory_report(device_index)
    return float(report.get("peak_mb", 0.0))


def measure_training_cost(
    step_fn: Callable[[], Any], *, n_iters: int = 5, label: str = "train", warmup: int = 1
) -> SystemsCost:
    """Time a training step closure (forward + backward + optimizer step)."""
    if n_iters < 1:
        raise ValueError("n_iters must be >= 1")
    device_index = 0 if cuda_available() else None
    with measure_peak_vram(device_index):
        for _ in range(max(0, warmup)):
            step_fn()
        samples: list[float] = []
        wall0 = time.perf_counter()
        for _ in range(n_iters):
            timer = CudaTimer(device=device_index)
            with timer:
                step_fn()
            samples.append(float(timer.ms))
        wall = max(time.perf_counter() - wall0, 1e-9)
        peak = _peak_after_block(device_index)
    mean_ms = float(np.mean(samples))
    return SystemsCost(
        latency_ms_per_decision={f"{label}_step": mean_ms},
        peak_vram_mb={label: peak},
        steps_per_s=float(n_iters / wall),
        extras={"latency_distribution": _percentiles(samples), "n_iters": float(n_iters)},
    )


def check_budget(cost: SystemsCost, cfg_gate: Any, *, phase: str | None = None) -> dict[str, Any]:
    """Evaluate G2 for one phase: peak VRAM plus headroom must fit the budget.

    Never resolve a failure by weakening the protocol: the reduction order in
    ``05`` §6 (batch -> horizons -> m/K -> checkpointing -> precision -> smaller
    model) is the only sanctioned response, and a precision change must be applied
    to the baseline too.
    """
    budget = float(getattr(cfg_gate, "budget_gb", 7.0))
    headroom = float(getattr(cfg_gate, "headroom_gb", 0.0))
    if phase is None:
        phases = list(cost.peak_vram_mb)
        if len(phases) != 1:
            raise ValueError(
                f"cost carries peak VRAM for {phases}; pass phase=<one of them> to check G2"
            )
        phase = phases[0]
    if phase not in cost.peak_vram_mb:
        raise KeyError(f"cost has no peak_vram_mb entry for phase {phase!r}; got {sorted(cost.peak_vram_mb)}")
    used_mb = float(cost.peak_vram_mb[phase])
    used_gb = used_mb / MB_PER_GB
    required = used_gb + headroom
    passed = required <= budget
    return {
        "gate": "G2",
        "phase": phase,
        "passed": bool(passed),
        "budget_gb": budget,
        "headroom_gb": headroom,
        "used_gb": used_gb,
        "required_gb": required,
        "margin_gb": budget - required,
        "measured_on_cuda": bool(cuda_available()),
        "reason": (
            f"{phase}: {used_gb:.2f} GB + {headroom:.2f} GB headroom <= {budget:.2f} GB"
            if passed
            else (
                f"{phase}: {used_gb:.2f} GB + {headroom:.2f} GB headroom exceeds the "
                f"{budget:.2f} GB budget. Reduce in the 05 §6 order (batch, horizons, m/K, "
                "checkpointing, precision, smaller model) — never by weakening the "
                "evaluation protocol, and apply any precision change to every arm."
            )
        ),
    }


def systems_table(costs: Mapping[str, SystemsCost]) -> dict[str, Any]:
    """Table T5: per-phase cost, with the G2 verdict attached per phase."""
    phases = sorted({p for cost in costs.values() for p in cost.peak_vram_mb})
    table: dict[str, Any] = {"per_arm": {k: v.as_dict() for k, v in costs.items()}, "phases": phases}
    if phases:
        best = {
            phase: min(
                (cost.peak_vram_mb[phase] for cost in costs.values() if phase in cost.peak_vram_mb),
                default=float("nan"),
            )
            for phase in phases
        }
        table["peak_gb_by_phase"] = {p: best[p] / MB_PER_GB for p in phases}
        table["dominant_phase"] = max(table["peak_gb_by_phase"], key=lambda p: table["peak_gb_by_phase"][p])
    table["note"] = (
        "peak VRAM is measured after a reset per phase; on a CPU-only box it is 0 and "
        "the budget check is not evidence either way"
    )
    return table
