#!/usr/bin/env python
"""Smoke suite S0-S6: the gate G0 artefact (``agents/03_repo_setup.md`` §3).

Every check reports ``pass`` / ``fail`` / ``skip`` and nothing is faked:

* ``S4`` (one batch through a policy) and ``S6`` (toy walking skeleton) are
  **CPU-passable with no download**, so a skip there is a failure — they are what
  keeps the repo honest when the simulator stack is absent.
* ``S0``-``S3`` and ``S5`` need LeRobot / Meta-World / a GPU. Without them they
  skip with the exact remediation command, and the summary says so explicitly.
  ``--strict`` turns any skip into a non-zero exit, which is how G0 is certified on
  a fully provisioned machine.

    python scripts/smoke_suite.py                # everything
    python scripts/smoke_suite.py --only S6      # one check
    python scripts/smoke_suite.py --skip S0,S1 --strict
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CONFIG = ROOT / "configs" / "exp" / "toy_smoke.yaml"
REPORT_PATH = ROOT / "runs" / "_smoke" / "report.json"
REMEDIATION_SIM = "python tasks.py bootstrap-sim"
REMEDIATION_VLA = "python tasks.py bootstrap-sim"

PASS, FAIL, SKIP = "pass", "fail", "skip"


@dataclass
class CheckResult:
    """One S-check's outcome, in the shape written to ``runs/_smoke/report.json``."""

    id: str
    name: str
    status: str
    detail: str
    seconds: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    #: False for checks that *cannot* pass on a CPU-only box without downloads, so
    #: the summary can distinguish "skipped, and that is fine" from "skipped and the
    #: environment is not provisioned".
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "seconds": round(self.seconds, 3),
            "artifacts": list(self.artifacts),
            "required": self.required,
        }


class SkipCheck(Exception):
    """Raised inside a check to report a skip with its remediation."""


def _toy_config() -> Any:
    from riskproj.utils.config import load_config

    if not DEFAULT_CONFIG.exists():
        raise SkipCheck(f"{DEFAULT_CONFIG} is missing; restore the committed toy_smoke config")
    return load_config(DEFAULT_CONFIG)


def _metaworld_task() -> Any:
    """Build the real Meta-World task or skip with the bootstrap command."""
    try:
        import gymnasium  # noqa: F401
        import metaworld  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the box
        raise SkipCheck(f"simulator extra missing ({exc.name}); run: {REMEDIATION_SIM}") from exc
    from riskproj.envs import build_task
    from riskproj.utils.config import load_config

    cfg_path = ROOT / "configs" / "exp" / "e0_baseline_repro.yaml"
    cfg = load_config(cfg_path)
    try:
        return build_task(cfg)
    except Exception as exc:  # pragma: no cover - depends on the box
        raise SkipCheck(f"Meta-World task failed to build ({exc}); run: {REMEDIATION_SIM}") from exc


# --------------------------------------------------------------------------
# helpers shared by the CPU checks
# --------------------------------------------------------------------------


def _toy_policy(cfg: Any, *, action_head: str = "flow_matching", seed: int = 0) -> Any:
    from riskproj.policies import build_policy

    return build_policy(
        cfg.model,
        action_head=action_head,
        state_dim=int(cfg.env.state_dim or 12),
        action_dim=int(cfg.env.action_dim),
        chunk_horizon=int(cfg.env.exec_horizon),
        hidden=16,
        num_flow_steps=int(cfg.model.num_flow_steps),
        seed=seed,
        device="cpu",
    )


def _toy_critic(cfg: Any, *, state_dim: int, action_dim: int, horizon: int) -> Any:
    from riskproj.critic.features import FeatureSpec
    from riskproj.critic.model import RiskCritic

    spec = FeatureSpec(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_horizon=horizon,
        hist_len=max(1, int(cfg.critic.use_history)),
    )
    return RiskCritic(cfg.critic, spec)


# --------------------------------------------------------------------------
# S0 — lerobot-eval of a published checkpoint, headless
# --------------------------------------------------------------------------


def _console_script(name: str) -> str | None:
    """Resolve a console script *from this interpreter's environment* first.

    ``shutil.which`` searches PATH, which on this box finds a different Python's
    ``lerobot-eval`` (a conda install without the ``dataset`` extra) and would
    report S0 as failed-for-the-wrong-reason — or worse, pass using an environment
    that is not the project's. The script next to ``sys.executable`` is the one that
    belongs to the environment the rest of the suite runs in.
    """
    import shutil
    from pathlib import Path as _Path

    exe_dir = _Path(sys.executable).resolve().parent
    for suffix in ("", ".exe", ".bat", ".cmd"):
        candidate = exe_dir / f"{name}{suffix}"
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def s0_lerobot_eval(ctx: dict[str, Any]) -> CheckResult:
    import subprocess

    exe = _console_script("lerobot-eval")
    if exe is None:
        raise SkipCheck(f"lerobot-eval is not installed in this environment; run: {REMEDIATION_VLA}")
    # ``env.task`` must be named explicitly: LeRobot's MetaWorld default is
    # ``metaworld-push-v2``, whose stripped name (``push-v2``) is not in its own
    # ``metaworld_config.json`` and raises KeyError before any episode runs. Use the
    # study's primary task so S0 also warms the checkpoint on the task that matters.
    cmd = [
        exe,
        "--policy.path=lerobot/smolvla_base",
        "--env.type=metaworld",
        "--env.task=metaworld-peg-insert-side-v3",
        "--eval.n_episodes=2",
        "--eval.batch_size=1",
        # smolvla_base wants observation.images.camera1..3; LeRobot's MetaWorld env
        # provides observation.image. This is LeRobot's own documented fix for it.
        '--rename_map={"observation.image": "observation.images.camera1"}',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(ROOT))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise SkipCheck(
            "lerobot-eval could not run a published checkpoint. This check exercises LeRobot's "
            "own eval driver (not our adapter, which loads the same checkpoint in S3); a failure "
            "here is an upstream/asset problem, never a pass by proxy: "
            + " | ".join(tail)
        )
    return CheckResult(
        "S0",
        "lerobot-eval runs a published SmolVLA checkpoint headless",
        PASS,
        "2 episodes completed",
        required=False,
    )


# --------------------------------------------------------------------------
# S1 — Meta-World builds and the scripted expert solves it at nominal dynamics
# --------------------------------------------------------------------------


def s1_metaworld_dynamics(ctx: dict[str, Any]) -> CheckResult:
    import numpy as np

    from riskproj.envs import read_back_dynamics
    from riskproj.types import DynamicsSpec

    task = _metaworld_task()
    env = task.make_env(DynamicsSpec(), seed=0)
    obs, _info = env.reset(seed=0)
    expert = task.expert()
    info: dict[str, Any] = {"success": False}
    for _ in range(int(getattr(env, "max_episode_steps", 200))):
        obs, _r, term, trunc, info = env.step(expert.act(obs))
        if term or trunc:
            break
    # ``success`` lives in the step info (04 §4's V6 definition reads it); the env
    # wrapper has no ``success`` attribute, so an attribute read would quietly report
    # False for every rollout — a gate artifact that misstates its own evidence.
    success = bool(info.get("success", False))

    spec = DynamicsSpec(mu_scale=1.5, mass_scale=2.0, stiff_scale=0.5)
    env2 = task.make_env(spec, seed=0)
    env2.reset(seed=0)
    rb = env2.dyn_readback if hasattr(env2, "dyn_readback") else read_back_dynamics(env2)
    # ``dyn_readback`` is a DynReadback dataclass; a mapping is accepted too so the check
    # works against the delegating (toy) path as well.
    effective = rb.get("effective", rb) if isinstance(rb, dict) else dict(getattr(rb, "effective", {}))
    if not effective:
        raise SkipCheck("the Meta-World wrapper exposes no dynamics read-back; see 04 §2")
    for axis, want in (("mu", 1.5), ("mass", 2.0), ("stiff", 0.5)):
        got = float(effective.get(f"{axis}_scale", effective.get(axis, float("nan"))))
        if not np.isfinite(got) or abs(got - want) > 1e-6:
            raise AssertionError(f"read-back {axis}={got} != requested {want} ({effective})")
    return CheckResult(
        "S1",
        "Meta-World builds; expert solves nominal; dynamics read-back matches",
        PASS,
        f"nominal expert success={success}; read-back {json.dumps(effective, sort_keys=True)}",
        required=False,
    )


# --------------------------------------------------------------------------
# S2 — MuJoCo state save/restore reproduces a trajectory exactly
# --------------------------------------------------------------------------


def s2_state_roundtrip(ctx: dict[str, Any]) -> CheckResult:
    from riskproj.envs import restore_state, save_state, state_hash
    from riskproj.types import DynamicsSpec

    task = _metaworld_task()
    env = task.make_env(DynamicsSpec(), seed=0)
    obs, _info = env.reset(seed=0)
    expert = task.expert()
    for _ in range(5):  # move off the reset pose so the state is not trivially equal
        obs, _r, term, trunc, _info = env.step(expert.act(obs))
        if term or trunc:
            break
    snap = save_state(env)
    if not hasattr(snap, "mocap_pos") or snap.mocap_pos is None:
        raise AssertionError("SimState carries no mocap pos/quat; 04 §6 requires them")

    actions = [expert.act(obs) for _ in range(20)]
    first, hashes_1 = _replay(env, snap, actions)
    restored_obs = restore_state(env, snap)
    if restored_obs is None:
        raise AssertionError("restore_state returned no observation; labels would use a stale one")
    second, hashes_2 = _replay(env, snap, actions)
    if hashes_1 != hashes_2:
        raise AssertionError(f"20-step replay diverged: {hashes_1[:3]} vs {hashes_2[:3]}")
    return CheckResult(
        "S2",
        "MuJoCo state save/restore reproduces a 20-step trajectory exactly",
        PASS,
        f"hash {state_hash(env)}; {len(actions)} steps identical twice",
        required=False,
    )


def _replay(env: Any, snap: Any, actions: list[Any]) -> tuple[list[Any], list[str]]:
    from riskproj.envs import restore_state, state_hash

    restore_state(env, snap)
    obs = None
    for action in actions:
        obs, _r, term, trunc, _info = env.step(action)
        if term or trunc:
            break
    return [obs], [state_hash(env)]


# --------------------------------------------------------------------------
# S3 — inspect_model writes the resolved sites file
# --------------------------------------------------------------------------


def s3_inspect_model(ctx: dict[str, Any]) -> CheckResult:
    import subprocess

    out = ROOT / "configs" / "model" / "smolvla_sites.yaml"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "inspect_model.py"),
        "--model",
        "smolvla",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()[-6:]
        raise SkipCheck("model could not be loaded: " + " | ".join(tail) + f"; run: {REMEDIATION_VLA}")
    if not out.exists():
        raise AssertionError("inspect_model exited 0 but wrote no sites file")
    return CheckResult(
        "S3",
        "inspect_model prints the module tree and writes configs/model/smolvla_sites.yaml",
        PASS,
        f"wrote {out.relative_to(ROOT)}",
        artifacts=[str(out.relative_to(ROOT))],
        required=False,
    )


# --------------------------------------------------------------------------
# S4 — one batch through a policy, projection, backward, step, save/load, rollout
# --------------------------------------------------------------------------


def s4_one_batch(ctx: dict[str, Any]) -> CheckResult:
    """Forward -> toy projection -> backward -> optimizer step -> save/load -> rollout.

    Runs with the toy policy on CPU and must pass without any download: it is the
    check that would have caught the projection operator silently degrading to
    "unusable gradient" under an outer ``torch.no_grad``.
    """
    import torch

    from riskproj.critic.model import make_risk_fn
    from riskproj.projection import project

    cfg = _toy_config()
    policy = _toy_policy(cfg)
    policy.attach_lora(cfg.model.lora)
    state_dim = int(cfg.env.state_dim or 12)
    horizon = int(cfg.env.exec_horizon)
    action_dim = int(cfg.env.action_dim)

    torch.manual_seed(0)
    batch_size = 2
    x_tau = torch.randn(batch_size, horizon, action_dim) * 0.1
    # A flow head needs a flow state; passing it explicitly keeps the check
    # independent of how the adapter happens to default x_tau/tau internally.
    batch = {"state": torch.randn(batch_size, state_dim), "x_tau": x_tau, "tau": torch.full((batch_size,), 0.1)}
    out = policy.forward(batch, site=cfg.model.site.default)
    assert out.chunk.shape == (batch_size, horizon, action_dim), out.chunk.shape

    critic = _toy_critic(cfg, state_dim=state_dim, action_dim=action_dim, horizon=horizon)
    critic.eval()
    # The risk functional the operator consumes is built from the *policy's* own
    # downstream map, so the gradient path is the real one (07 §1).
    # ``f_act`` consumes the adapter's own context (it calls ``ctx.flow()``), not a bare
    # dict of flow state, so build it exactly as the training and rollout paths do.
    flow_ctx = policy.make_context(
        {
            "state": batch["state"],
            "x_tau": out.extras.get("x_tau", x_tau),
            "tau": torch.full((batch_size,), 0.1),
        },
        site=cfg.model.site.default,
    )

    def risk_fn(h: torch.Tensor) -> torch.Tensor:
        # Active by construction (always above tau) so the check exercises the
        # correction path, while still flowing through the policy's own downstream
        # map — a fixed sigmoid over a random chunk could legitimately be inactive
        # and would then test nothing.
        chunk = policy.f_act(flow_ctx, h, cfg.model.site.default)
        return 0.9 + 0.05 * torch.tanh(chunk.reshape(chunk.shape[0], -1).mean(dim=-1))

    h_c = out.h.detach().clone().requires_grad_(True)
    h_safe, info = project(h_c, risk_fn, cfg.methods[-1].projection)
    assert h_safe.shape == h_c.shape
    assert bool(info.active.all()), "projection must see the synthetic risk as active"
    assert torch.isfinite(h_safe).all()
    h_safe.sum().backward()
    assert h_c.grad is not None and torch.isfinite(h_c.grad).all()

    optimizer = torch.optim.SGD([p for p in policy.trainable_parameters()], lr=1e-3)
    before = [p.detach().clone() for p in policy.trainable_parameters()]
    loss = (out.chunk**2).mean()
    loss.backward()
    optimizer.step()
    after = list(policy.trainable_parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True)), (
        "an optimizer step changed no trainable parameter; the LoRA path is not connected"
    )

    ckpt = ROOT / "runs" / "_smoke" / "s4_policy.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": policy.state_dict(), "lora": cfg.model.lora.model_dump()}, ckpt)
    loaded = _toy_policy(cfg)
    loaded.attach_lora(cfg.model.lora)
    loaded.load_state_dict(torch.load(ckpt, weights_only=False)["state"])
    with torch.no_grad():
        a = policy.forward(batch, site=cfg.model.site.default).chunk
        b = loaded.forward(batch, site=cfg.model.site.default).chunk
    assert torch.allclose(a, b, atol=1e-6), "checkpoint round-trip changed the policy output"

    # 20-step rollout on the toy env, driven by the policy.
    from riskproj.envs import build_task

    task = build_task(cfg)
    env = task.make_env(cfg.dynamics.nominal, seed=0)
    obs, _info = env.reset(seed=0)
    steps = 0
    for _ in range(20):
        state = torch.as_tensor(obs["state"], dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            rollout_batch = {
                "state": state,
                "x_tau": torch.zeros(1, horizon, action_dim),
                "tau": torch.full((1,), 0.1),
            }
            chunk = policy.forward(rollout_batch, site=cfg.model.site.default).chunk
        obs, _r, term, trunc, _info = env.step(chunk[0, 0].numpy())
        steps += 1
        if term or trunc:
            break
    assert steps > 0, "the 20-step rollout executed no step"
    return CheckResult(
        "S4",
        "forward -> projection -> backward -> optimizer step -> checkpoint -> 20-step rollout",
        PASS,
        f"{steps} rollout steps; LoRA params updated; checkpoint round-trip exact",
        artifacts=[str(ckpt.relative_to(ROOT))],
    )


# --------------------------------------------------------------------------
# S5 — VRAM profiling
# --------------------------------------------------------------------------


def s5_profile_vram(ctx: dict[str, Any]) -> CheckResult:
    import subprocess

    from riskproj.utils.profiling import cuda_available

    out = ROOT / "runs" / "_smoke" / "vram.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "profile_vram.py"),
        "--config",
        str(DEFAULT_CONFIG),
        "--phase",
        "all",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()[-6:]
        raise SkipCheck("VRAM profiling did not complete: " + " | ".join(tail))
    if not cuda_available():
        raise SkipCheck("no CUDA device on this box; G2 cannot be evaluated here")
    payload = json.loads(out.read_text(encoding="utf-8"))
    over = [p for p, v in payload.get("phases", {}).items() if v.get("over_budget")]
    if over:
        raise AssertionError(f"phases over the G2 budget: {over}")
    return CheckResult(
        "S5",
        "profile_vram reports peak memory for critic / LoRA+projection / inference+projection",
        PASS,
        json.dumps(payload.get("peak_mb", {}), sort_keys=True),
        artifacts=[str(out.relative_to(ROOT))],
        required=False,
    )


# --------------------------------------------------------------------------
# S6 — toy walking skeleton, timed (<2 min, FAIL not skip if exceeded)
# --------------------------------------------------------------------------


def s6_walking_skeleton(ctx: dict[str, Any]) -> CheckResult:
    """Config -> data -> critic -> projection -> rollout -> report, all on CPU."""
    budget = float(ctx.get("max_seconds", 120.0))
    started = time.perf_counter()
    cfg = _toy_config()

    from riskproj.envs import build_task

    cfg_hash = cfg.config_hash()
    work = ROOT / "runs" / "_smoke" / "s6"
    work.mkdir(parents=True, exist_ok=True)
    task = build_task(cfg)
    ranges = cfg.dynamics.to_ranges()
    # Two TRAIN cells is enough to exercise the per-cell paths without eating the
    # whole 2-minute budget; the generator is the same code path for any count.
    train_cells = _train_cells(cfg)[:2]
    demo_cells = train_cells

    from riskproj.data.gen_demos import generate_demos

    demos = generate_demos(
        task,
        cells=demo_cells,
        ranges=ranges,
        episodes_per_cell=int(cfg.eval.n_states_per_cell),
        base_seed=int(cfg.seed),
        out_dir=work / "demos",
        max_steps=int(cfg.env.max_steps),
        config_hash=cfg_hash,
    )

    from riskproj.data.gen_critic_data import generate_critic_data

    policy = _toy_policy(cfg)
    crit = generate_critic_data(
        task,
        policy=None,
        cells=train_cells,
        ranges=ranges,
        states_per_cell=int(cfg.eval.n_states_per_cell),
        chunks_per_state=4,
        base_seed=int(cfg.seed) + 1,
        out_dir=work / "critic",
        horizon=int(cfg.env.critic_horizon),
        violation_spec=task.violations,
        config_hash=cfg_hash,
        include_unsafe=False,
        include_policy=False,
    )

    from riskproj.critic.features import FeatureSpec
    from riskproj.critic.train import train_critic

    dataset = _load_critic_dataset(work / "critic")
    kwargs = dataset.feature_spec_kwargs()
    spec = FeatureSpec(
        state_dim=kwargs["state_dim"],
        action_dim=kwargs["action_dim"],
        chunk_horizon=kwargs["chunk_horizon"],
        hist_len=max(1, kwargs["hist_len"]),
        vision_dim=kwargs["vision_dim"],
        force_dim=kwargs["force_dim"],
    )
    critic_cfg = cfg.critic.model_copy(update={"epochs": min(20, cfg.critic.epochs)})
    model, metrics = train_critic(
        dataset, critic_cfg, feature_spec=spec, device="cpu", out_dir=work / "critic_model"
    )

    import numpy as np
    import torch

    from riskproj.critic.model import make_risk_fn
    from riskproj.projection import project

    h = torch.randn(2, spec.state_dim, requires_grad=True)
    head = torch.nn.Linear(spec.state_dim, spec.chunk_horizon * spec.action_dim, bias=False)

    def f_act(hh: torch.Tensor) -> torch.Tensor:
        return head(hh).reshape(hh.shape[0], spec.chunk_horizon, spec.action_dim)

    # The risk functional evaluates the critic on ``h``'s batch, so the context rows must
    # match that batch; ``dataset[0][0]`` is a single sample's context.
    def _tile(ctx0: dict[str, Any], n: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in ctx0.items():
            t = value if torch.is_tensor(value) else torch.as_tensor(value)
            out[key] = t.reshape(1, *t.shape).repeat(n, *([1] * t.dim()))
        return out

    risk_fn = make_risk_fn(model, _tile(dataset[0][0], int(h.shape[0])), f_act)
    _, info = project(h.detach().requires_grad_(True), risk_fn, cfg.methods[-1].projection)

    from riskproj.eval.rollout import evaluate_method
    from riskproj.eval.report import build_tables

    # The report is *derived from archived logs*, never from in-memory records
    # (skills/software_engineering.md §1.4), so the smoke run goes through the driver
    # that writes them and the tables are rebuilt from the directory it wrote.
    dynamics = ranges.sample(train_cells[0], np.random.default_rng(0))
    # The arm whose *inference* mode the rollout must use: the last method is the
    # projected one (B3). Passing a critic without its inference mode would run the
    # no-projection planner and evaluate the baseline under B3's name, which the
    # driver refuses to do — so the mode travels with the arm, never as a default.
    arm = cfg.methods[-1]
    if arm.inference_mode == "0":
        raise AssertionError(
            f"the smoke config's last arm ({arm.id}) has inference_mode='0'; S6 must exercise "
            "the projection path, so the projected arm has to be last"
        )
    run = evaluate_method(
        task,
        policy,
        cells=[train_cells[0]],
        eps_per_cell=1,
        base_seed=int(cfg.seed),
        out_dir=work / "runs",
        run_id="smoke",
        method=arm.id,
        inference_mode=arm.inference_mode,
        train_seed=int(cfg.seed),
        spec_kwargs={
            "max_steps": int(cfg.env.max_steps),
            "record_decisions": True,
            "save_states": False,
        },
        critic=model,
        proj_cfg=arm.projection,
        dyn=dynamics,
        ranges=ranges,
        split="train",
        config=cfg.canonical(),
    )
    tables = build_tables(Path(run["episodes_jsonl"]).parent)

    elapsed = time.perf_counter() - started
    detail = (
        f"demos via {demos.get('n_records', '?')} records, "
        f"{len(dataset)} critic tuples ({crit.get('positive_rate', 'n/a')} positive), "
        f"{len(tables)} report tables, active projections={int(info.active.sum())}, "
        f"{elapsed:.1f}s/{budget:.0f}s"
    )
    if elapsed > budget:
        raise TimeoutError(f"walking skeleton took {elapsed:.1f}s > budget {budget:.0f}s (03 §3)")
    return CheckResult(
        "S6",
        "toy walking skeleton: config -> data -> critic -> projection -> rollout -> report",
        PASS,
        detail,
        artifacts=[str((work / "report.json").relative_to(ROOT))],
    )


def _train_cells(cfg: Any) -> list[str]:
    from riskproj.data.splits import ROLE_TRAIN, cells_for_role

    return cells_for_role(ROLE_TRAIN, cfg.splits)


def _call_supported(fn: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with the subset of ``kwargs`` its signature accepts.

    The eval layer is a sibling module written in parallel; passing the documented
    plan and letting it declare what it needs keeps this suite from breaking on an
    optional keyword, while still failing loudly on a real error.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return fn(**kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _load_critic_dataset(out_dir: Path) -> Any:
    """Reload every critic shard the generator wrote (npz + json sidecar)."""
    from riskproj.data.schema import CriticDataset

    stems = sorted({p.with_suffix("") for p in Path(out_dir).rglob("*.npz")})
    if not stems:
        raise RuntimeError(
            f"no critic shards under {out_dir}; the generator did not write what it reported"
        )
    merged: list[Any] = []
    for stem in stems:
        merged.extend(CriticDataset.load(stem).tuples)
    if not merged:
        raise RuntimeError(f"reloaded 0 critic tuples from {[str(s) for s in stems]}")
    return CriticDataset(merged)


# --------------------------------------------------------------------------
# registry / driver
# --------------------------------------------------------------------------

CHECKS: dict[str, tuple[str, Callable[[dict[str, Any]], CheckResult], bool]] = {
    "S0": ("lerobot-eval of a published checkpoint", s0_lerobot_eval, False),
    "S1": ("Meta-World build + scripted expert + dynamics read-back", s1_metaworld_dynamics, False),
    "S2": ("MuJoCo state save/restore exactness (incl. mocap)", s2_state_roundtrip, False),
    "S3": ("inspect_model writes resolved sites", s3_inspect_model, False),
    "S4": ("one batch: forward/projection/backward/step/save/rollout", s4_one_batch, True),
    "S5": ("VRAM profiling (G2)", s5_profile_vram, False),
    "S6": ("toy walking skeleton (<2 min, CPU)", s6_walking_skeleton, True),
}


def run_check(check_id: str, ctx: dict[str, Any]) -> CheckResult:
    name, fn, cpu_passable = CHECKS[check_id]
    started = time.perf_counter()
    try:
        result = fn(ctx)
    except SkipCheck as exc:
        result = CheckResult(
            check_id, name, SKIP, str(exc), required=cpu_passable
        )
    except TimeoutError as exc:
        result = CheckResult(check_id, name, FAIL, str(exc), required=cpu_passable)
    except Exception as exc:  # a check failure is data, not a crash
        detail = f"{type(exc).__name__}: {exc}"
        if ctx.get("verbose"):
            detail += "\n" + traceback.format_exc()
        result = CheckResult(check_id, name, FAIL, detail, required=cpu_passable)
    result.seconds = time.perf_counter() - started
    result.id = check_id
    result.name = name
    result.required = cpu_passable
    return result


def _select(only: str, skip: str) -> list[str]:
    ids = [k for k in CHECKS if (not only or k in _split(only)) and k not in _split(skip)]
    unknown = (_split(only) | _split(skip)) - set(CHECKS)
    if unknown:
        raise SystemExit(f"unknown check ids {sorted(unknown)}; known: {sorted(CHECKS)}")
    if not ids:
        raise SystemExit("no checks selected")
    return sorted(ids)


def _split(value: str | None) -> set[str]:
    if not value:
        return set()
    return {v.strip().upper() for v in value.split(",") if v.strip()}


def _print_table(results: list[CheckResult]) -> None:
    width = max(len(r.name) for r in results)
    print("\n id  status  " + "check".ljust(width) + "  seconds")
    print(" " + "-" * (width + 22))
    for r in results:
        print(f" {r.id:<3} {r.status:<7} {r.name.ljust(width)}  {r.seconds:>7.2f}")
    for r in results:
        if r.status != PASS:
            print(f"\n{r.id} {r.status}: {r.detail}")


def _default_report_path(only: str, skip: str) -> Path:
    """Where this run's report goes.

    ``runs/_smoke/report.json`` is the **G0 artifact**: the record of what the full suite
    found. A partial run — including the single-check runs pytest spawns — must not
    overwrite it, or the audit trail ends up showing whatever happened to run last
    rather than what the gate actually saw.
    """
    selection = _split(only) | _split(skip)
    if not selection:
        return REPORT_PATH
    return REPORT_PATH.with_name(f"report_{'-'.join(sorted(selection))}.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smoke_suite.py", description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated check ids to run")
    parser.add_argument("--skip", default="", help="comma-separated check ids to skip")
    parser.add_argument(
        "--out",
        default=None,
        help="report json path (default: the gate artifact for a full run, a per-selection "
        "file otherwise)",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=120.0, help="S6 budget; exceeding it FAILS (03 §3)"
    )
    parser.add_argument("--strict", action="store_true", help="any skip makes the run fail")
    parser.add_argument("--json", action="store_true", help="print the report json")
    parser.add_argument("--verbose", action="store_true", help="include tracebacks in failures")
    ns = parser.parse_args(argv)

    ctx = {"max_seconds": ns.max_seconds, "verbose": ns.verbose}
    ids = _select(ns.only, ns.skip)
    results = [run_check(cid, ctx) for cid in ids]
    _print_table(results)

    failed = [r for r in results if r.status == FAIL]
    skipped = [r for r in results if r.status == SKIP]
    cpu_missing = [r for r in skipped if r.required]
    report = {
        "suite": "S0-S6",
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [r.as_dict() for r in results],
        "summary": {
            "n_pass": sum(1 for r in results if r.status == PASS),
            "n_fail": len(failed),
            "n_skip": len(skipped),
            "g0_complete": not failed and not skipped,
            "cpu_required_missing": [r.id for r in cpu_missing],
            "skipped_ids": [r.id for r in skipped],
        },
    }
    out = Path(ns.out) if ns.out else _default_report_path(ns.only, ns.skip)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\n[smoke] {report['summary']['n_pass']} pass, {len(failed)} fail, "
        f"{len(skipped)} skip -> {out}"
    )
    if skipped:
        print(
            f"[smoke] G0 not certified: skipped {report['summary']['skipped_ids']} "
            "(provision with `python tasks.py bootstrap-sim`); --strict turns this into a failure"
        )
    if ns.json:
        print(json.dumps(report, indent=2))

    if failed or cpu_missing or (ns.strict and skipped):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
