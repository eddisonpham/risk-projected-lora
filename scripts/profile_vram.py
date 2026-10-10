#!/usr/bin/env python
"""Measure peak VRAM for the three phases that gate G2 (``05`` §6).

Phases, measured with a reset between them:

* ``critic`` — critic training with **cached features** (the VLM is never in the loop),
* ``train`` — LoRA training *with projection* (the expensive one: the tail plus the
  critic stay in the autograd graph),
* ``inference`` — inference with projection.

Every phase runs **on the GPU when there is one** — a VRAM gate that never touches the
GPU measures nothing. On a CPU-only box every peak is ``0`` and the report says
``cuda=False`` rather than pretending. The run exits non-zero when a measured phase is
over budget, and *raises* when CUDA is available but a phase measured zero (that means
nothing reached the device, so G2 was not evaluated and there is no result to report).

    python scripts/profile_vram.py --config configs/exp/toy_smoke.yaml --phase all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _budget_gb(cfg: Any) -> float:
    return float(cfg.gates.G2.budget_gb)


def _decode(model: Any, batch: Any, steps: int) -> None:
    """Drive the model so its activation memory is actually touched."""
    import torch

    forwards = max(1, int(steps))
    for _ in range(forwards):
        out = model.forward(batch, site=getattr(model, "default_site", "S2"))
        if getattr(out, "chunk", None) is None:  # pragma: no cover - defensive
            raise RuntimeError(f"{type(model).__name__}.forward returned no chunk")


def _device() -> Any:
    """The device this profiling run uses: the GPU when there is one."""
    import torch

    from riskproj.utils.profiling import cuda_available

    return torch.device("cuda" if cuda_available() else "cpu")


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    """Move every tensor of a batch dict onto ``device`` (non-tensors pass through)."""
    import torch

    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _profiler_batch(policy: Any, *, batch_size: int, device: Any) -> dict[str, Any]:
    """A batch the *adapter* can consume, or a loud failure naming what is missing.

    The toy adapter can build its own batch (``sample_batch``). A real adapter cannot: its
    batch *is* the task's observation — two camera images, the proprioceptive state and the
    language tokens for the SmolVLA checkpoint — and only the eval layer, driving a live
    env, produces it (``05`` §6). Fabricating one here (an 8x8 stand-in image, a guessed
    token count) would report a VRAM number for a workload that never runs, so this raises
    instead. Wire the eval layer's batch path in before trusting a real-model number.
    """
    sample = getattr(policy, "sample_batch", None)
    if not callable(sample):
        raise RuntimeError(
            f"{type(policy).__name__} cannot build a profiling batch: G2 on the real model "
            "needs the task's own observation batch (images + state + language tokens), which "
            "only the eval layer builds from a live env. Add that path here rather than "
            "fabricating a batch - a wrong image size or token count measures nothing."
        )
    return _to_device(sample(batch_size=batch_size, seed=0), device)


def phase_critic(cfg: Any, *, batch_size: int) -> dict[str, int]:
    """Critic training with cached features: never loads the VLM (that is the point)."""
    import torch

    from riskproj.critic.features import FeatureSpec
    from riskproj.critic.model import RiskCritic
    from riskproj.utils.profiling import measure_peak_vram

    device = _device()
    state_dim = int(cfg.env.state_dim or 12)
    spec = FeatureSpec(
        state_dim=state_dim,
        action_dim=int(cfg.env.action_dim),
        chunk_horizon=int(cfg.env.critic_horizon),
        hist_len=max(1, int(cfg.critic.use_history)),
    )
    model = RiskCritic(cfg.critic, spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = torch.randn(batch_size, spec.dim, device=device)  # FeatureSpec.dim is a property
    labels = torch.randint(0, 2, (batch_size,), device=device).float()
    with measure_peak_vram() as measured:
        for _ in range(3):
            logits = model.logits_from_features(features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    del model, optimizer, features, labels
    return {"peak_mb": measured["peak_mb"]}


def _critic_context(batch: dict[str, Any], spec: Any) -> dict[str, Any]:
    """A critic context matching ``batch``'s size, for the profiler only.

    The critic sees ``proprio``/``ee_vel``/history blocks (``04`` §5), not the policy's
    own observation dict, and its batch must match ``h``'s. The history is zeros here:
    the point of this script is peak memory, not accuracy, and a randn context would
    only add noise to the measurement.
    """
    import torch

    state = torch.as_tensor(batch["state"])
    state = state.reshape(state.shape[0], -1) if state.dim() == 2 else state.reshape(1, -1)
    b = int(state.shape[0])
    n = max(1, int(getattr(spec, "hist_len", 1) or 1))
    # The device comes from the batch itself, so the context can never be built on a
    # different device from the activation the critic has to score.
    device, dtype = state.device, state.dtype
    return {
        "proprio": state,
        "ee_vel": torch.zeros(b, 3, device=device, dtype=dtype),
        "action_hist": torch.zeros(b, n, int(spec.action_dim), device=device, dtype=dtype),
        "proprio_hist": torch.zeros(b, n, int(spec.state_dim), device=device, dtype=dtype),
        "vision_emb": torch.zeros(b, 0, device=device, dtype=dtype),
        "force": torch.zeros(b, 0, device=device, dtype=dtype),
    }


def phase_train(cfg: Any, *, batch_size: int, projection: bool) -> dict[str, Any]:
    """LoRA training with projection: the tail and the critic stay in the graph."""
    import torch

    from riskproj.critic.features import FeatureSpec
    from riskproj.critic.model import RiskCritic, make_risk_fn
    from riskproj.policies import build_policy_for_env
    from riskproj.projection import project
    from riskproj.utils.profiling import measure_peak_vram

    device = _device()
    state_dim = int(cfg.env.state_dim or 12)
    # ``build_policy_for_env`` and not a hand-built kwarg list: the env owns the geometry,
    # and passing the toy's own parameters (``hidden``, ``num_flow_steps``, the *execution*
    # horizon) made the real adapter unconstructible - so this phase silently measured the
    # toy only.
    policy = build_policy_for_env(cfg, device=str(device))
    policy.attach_lora(cfg.model.lora)
    policy.to(device)
    params = list(policy.trainable_parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4) if params else None

    spec = FeatureSpec(
        state_dim=state_dim,
        action_dim=int(cfg.env.action_dim),
        chunk_horizon=int(cfg.env.critic_horizon),
        hist_len=max(1, int(cfg.critic.use_history)),
    )
    critic = RiskCritic(cfg.critic, spec).to(device)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)

    # A flow head needs a flow state (x_tau/tau) as well as ``state``; the policy owns
    # the only constructor that guarantees a mutually consistent batch.
    batch = _profiler_batch(policy, batch_size=batch_size, device=device)
    with measure_peak_vram() as measured:
        for _ in range(3):
            out = policy.forward(batch, site=cfg.model.site.default)
            loss = (out.chunk.float() ** 2).mean()
            if projection:
                # The adapter's own context, not a bare dict: ``f_act`` calls
                # ``ctx.flow()`` (05 §4), and it must be read from the *same* batch the
                # forward pass used — exactly as the training path builds it. Building it
                # once also keeps the tail out of the per-step path.
                flow_ctx = policy.make_context(dict(batch), site=cfg.model.site.default)

                def f_act(h: torch.Tensor) -> torch.Tensor:
                    return policy.f_act(flow_ctx, h, cfg.model.site.default)

                risk_fn = make_risk_fn(critic, _critic_context(batch, spec), f_act)
                if out.h is not None:
                    # The training loss must flow *through* the projected activation,
                    # otherwise this phase would not measure the cost of keeping the
                    # tail and the critic in the autograd graph (05 §6).
                    h_safe, _info = project(out.h, risk_fn, cfg.methods[-1].projection)
                    loss = (f_act(h_safe).float() ** 2).mean()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if optimizer is not None:
                optimizer.step()
    result: dict[str, Any] = {"peak_mb": measured["peak_mb"], "projection": projection}
    return result


def phase_inference(cfg: Any, *, batch_size: int, projection: bool) -> dict[str, Any]:
    import torch

    from riskproj.critic.features import FeatureSpec
    from riskproj.critic.model import RiskCritic, make_risk_fn
    from riskproj.policies import build_policy_for_env
    from riskproj.projection import project
    from riskproj.utils.profiling import measure_peak_vram

    device = _device()
    state_dim = int(cfg.env.state_dim or 12)
    policy = build_policy_for_env(cfg, device=str(device)).to(device).eval()
    spec = FeatureSpec(
        state_dim=state_dim,
        action_dim=int(cfg.env.action_dim),
        chunk_horizon=int(cfg.env.critic_horizon),
        hist_len=max(1, int(cfg.critic.use_history)),
    )
    critic = RiskCritic(cfg.critic, spec).to(device).eval()
    # A flow head needs x_tau/tau as well as state; the policy's own batch constructor
    # guarantees a mutually consistent batch (or fails loudly, see ``_profiler_batch``).
    batch = _profiler_batch(policy, batch_size=batch_size, device=device)
    flow_ctx = policy.make_context(dict(batch), site=cfg.model.site.default)
    with measure_peak_vram() as measured:
        with torch.no_grad():
            out = policy.forward(batch, site=cfg.model.site.default)
        if projection and out.h is not None:
            def f_act(h: torch.Tensor) -> torch.Tensor:
                return policy.f_act(flow_ctx, h, cfg.model.site.default)

            risk_fn = make_risk_fn(critic, _critic_context(batch, spec), f_act)
            project(out.h, risk_fn, cfg.methods[-1].projection)
    return {"peak_mb": measured["peak_mb"], "projection": projection}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="profile_vram.py", description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase", default="all", choices=("critic", "train", "inference", "all")
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "runs" / "_smoke" / "vram.json"))
    ns = parser.parse_args(argv)

    from riskproj.utils.config import load_config
    from riskproj.utils.io import capture_provenance
    from riskproj.utils.profiling import cuda_available

    cfg = load_config(ns.config)
    budget = _budget_gb(cfg)
    phases: dict[str, dict[str, Any]] = {}

    # A phase that cannot run is recorded as *unevaluated*, not dropped: G2 asks whether
    # all three fit, so a missing phase is a non-pass and has to be visible in the
    # artifact. Only the phases that were asked for are attempted.
    requested = (
        ["critic", "train", "inference"] if ns.phase == "all" else [str(ns.phase)]
    )
    runners = {
        "critic": lambda: phase_critic(cfg, batch_size=ns.batch_size),
        "train": lambda: phase_train(cfg, batch_size=ns.batch_size, projection=True),
        "inference": lambda: phase_inference(cfg, batch_size=ns.batch_size, projection=True),
    }
    unevaluated: dict[str, str] = {}
    for name in requested:
        try:
            phases[name] = runners[name]()
        except Exception as exc:  # recorded, then reported as a failure below
            unevaluated[name] = f"{type(exc).__name__}: {exc}"
            print(f"[profile-vram] phase {name!r} could not be measured: {unevaluated[name]}", file=sys.stderr)

    for name, payload in phases.items():
        peak_gb = float(payload["peak_mb"]) / 1024.0
        payload["peak_gb"] = round(peak_gb, 3)
        payload["budget_gb"] = budget
        payload["over_budget"] = bool(peak_gb > budget)
        payload["headroom_gb"] = round(budget - peak_gb, 3)

    report = {
        "config": str(ns.config),
        "config_hash": cfg.config_hash(),
        "cuda": cuda_available(),
        "budget_gb": budget,
        "model": cfg.model.name,
        "batch_size": ns.batch_size,
        "phases": phases,
        "peak_mb": {k: v["peak_mb"] for k, v in phases.items()},
        "unevaluated": unevaluated,
        "over_budget": [k for k, v in phases.items() if v["over_budget"]],
        "note": (
            ""
            if cuda_available()
            else "no CUDA device: peaks are zero and G2 cannot be evaluated on this box"
        ),
    }
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    capture_provenance(out.parent, extra={"profile_vram": report})

    print(f"{'phase':<12} {'peak MB':>10} {'GB':>7} {'budget':>8}")
    for name, payload in phases.items():
        print(f"{name:<12} {payload['peak_mb']:>10.1f} {payload['peak_gb']:>7.3f} {budget:>8.2f}")
    print(f"[profile-vram] cuda={report['cuda']} -> {out}")

    # A zero peak with a GPU present is not a small measurement, it is *no* measurement:
    # nothing was placed on the device, so G2 was not evaluated. Reporting that as a pass
    # is exactly the silent failure the protocol forbids.
    zero_phases = [name for name, payload in phases.items() if float(payload["peak_mb"]) <= 0.0]
    if cuda_available() and zero_phases:
        raise RuntimeError(
            f"CUDA is available but {zero_phases} measured 0 MB: nothing reached the GPU, so "
            f"G2 was not evaluated (see {out}). Fix the device placement in this script; do "
            "not report this as a pass."
        )

    if unevaluated:
        print(
            f"[profile-vram] G2 NOT EVALUATED: {sorted(unevaluated)} could not be measured "
            f"({out}). A phase that never ran is not a phase that fits.",
            file=sys.stderr,
        )
        return 1
    if report["over_budget"] and cuda_available():
        print(
            f"[profile-vram] G2 FAILED: {report['over_budget']} over {budget} GB. Reduce per "
            "05 §6 (batch -> horizon -> m, K -> checkpointing -> smaller model). Never weaken "
            "the evaluation protocol to fit.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
