"""The parametrised contract suite of ``skills/software_engineering.md`` §3.

Every registered implementation of a Protocol must satisfy the same behavioural
contract, so that adding a model, task, metric or method is "one file + one YAML"
rather than an integration project. ``tests/unit/test_policy_contract.py``
already covers the model-adapter specifics (zero-LoRA equivalence, seeded
determinism, ``f_act`` consistency); this file covers the *other* protocols and
the cross-cutting guarantees, and deliberately does not repeat those assertions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from riskproj.obs import TASK_OBS_KEYS, env_obs_to_step_obs
from riskproj.policies.toy import ToyPolicy
from riskproj.protocols import Critic, Method, Metric, PolicyAdapter, Projector, Task
from riskproj.types import VIOLATION_TYPES, DynamicsSpec

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
N_HEAD_COLUMNS = 1 + len(VIOLATION_TYPES)  # any-violation + V1..V5

AXIS_NAMES = ("mu", "mass", "stiff")


# --------------------------------------------------------------------------
# Projector
# --------------------------------------------------------------------------


def _sigmoid_risk(a: torch.Tensor, bias: float, dtype: torch.dtype):
    def risk(h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(h.to(dtype) @ a.to(dtype) + bias)

    return risk


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_projector_preserves_shape_dtype_device_and_finiteness(dtype: torch.dtype) -> None:
    from riskproj.projection import project
    from riskproj.utils.config import ProjCfg

    torch.manual_seed(0)
    b, d = 4, 6
    a = torch.randn(d)
    risk = _sigmoid_risk(a, 2.0, dtype)  # far above tau -> active for every example
    h = torch.randn(b, d, dtype=dtype)
    cfg = ProjCfg(tau=0.3, K=2)

    h_safe, info = project(h, risk, cfg)

    assert h_safe.shape == h.shape
    assert h_safe.dtype == h.dtype
    assert h_safe.device == h.device
    assert torch.isfinite(h_safe).all(), "a projection must never emit non-finite activations"
    assert info.active is not None and bool(info.active.any())
    assert info.r_after is not None and torch.isfinite(info.r_after).all()


def test_projector_is_exact_on_already_safe_inputs() -> None:
    """Feasible examples must come back bitwise unchanged: a no-op step that still
    perturbs the activation would silently inject noise into every rollout."""
    from riskproj.projection import project
    from riskproj.utils.config import ProjCfg

    torch.manual_seed(1)
    b, d = 3, 5
    a = torch.randn(d)
    risk = _sigmoid_risk(a, 0.0, torch.float32)
    h = -10.0 * a + 0.001 * torch.randn(b, d)  # risk ~ sigmoid(-10*||a||^2) -> 0
    assert float(risk(h).max()) < 0.01

    h_safe, info = project(h, risk, ProjCfg(tau=0.5, K=3))

    assert torch.equal(h_safe, h), "safe inputs must be bitwise identical"
    assert not bool(info.active.any())
    assert info.corr_rel is not None and float(info.corr_rel.max()) == 0.0


def test_projector_handles_token_structured_activations() -> None:
    """A site activation is ``[B, ...]``: ``[B, T, D]`` must work as well as ``[B, D]``."""
    from riskproj.projection import project
    from riskproj.utils.config import ProjCfg

    torch.manual_seed(2)
    b, t, d = 2, 3, 4
    a = torch.randn(d)
    h = torch.randn(b, t, d)

    def risk(hh: torch.Tensor) -> torch.Tensor:
        # The operator's contract is one risk per *example* (07 §1): a risk
        # functional over a [B, T, D] site must reduce the token axis itself. The
        # token structure only affects how the *gradient* is grouped.
        return torch.sigmoid((hh * a).sum(-1) + 2.0).amax(-1)

    h_safe, _ = project(h, risk, ProjCfg(tau=0.3, K=1, per_token=True))

    assert h_safe.shape == h.shape
    assert torch.isfinite(h_safe).all()


def test_projector_protocol_shape_is_satisfied_by_a_wrapper() -> None:
    """``Projector`` is an instance protocol: document the adapter shape that the
    inference wrappers are expected to use."""
    from riskproj.projection import project
    from riskproj.utils.config import ProjCfg

    class _Bound:
        def __init__(self, cfg: ProjCfg) -> None:
            self.cfg = cfg

        def __call__(self, h_c: torch.Tensor, risk_fn: Any, cfg: Any) -> Any:
            return project(h_c, risk_fn, cfg or self.cfg)

    assert isinstance(_Bound(ProjCfg()), Projector)


# --------------------------------------------------------------------------
# Critic
# --------------------------------------------------------------------------


def _critic_instances() -> list[tuple[str, Any, dict[str, Any], torch.Tensor]]:
    """Every registered critic, with a matching context and candidate chunk."""
    from riskproj.critic.features import FeatureSpec
    from riskproj.utils import registry
    from riskproj.utils.config import CriticCfg

    torch.manual_seed(0)
    state_dim, action_dim, horizon, b = 6, 4, 5, 3
    spec = FeatureSpec(state_dim=state_dim, action_dim=action_dim, chunk_horizon=horizon)
    cfg = CriticCfg(hidden=16, depth=2, ensemble=3, seed=0)
    rng = np.random.default_rng(0)
    hist = max(cfg.use_history, 1)
    ctx = {
        "proprio": rng.normal(size=(b, state_dim)).astype(np.float32),
        "ee_vel": rng.normal(size=(b, 3)).astype(np.float32),
        "action_hist": rng.normal(size=(b, hist, action_dim)).astype(np.float32),
        "proprio_hist": rng.normal(size=(b, hist, state_dim)).astype(np.float32),
        "vision_emb": np.zeros((b, 0), np.float32),
        "force": np.zeros((b, 0), np.float32),
    }
    chunk = torch.randn(b, horizon, action_dim)
    out: list[tuple[str, Any, dict[str, Any], torch.Tensor]] = []
    for name in registry.available("critic"):
        factory = registry.get("critic", name)
        out.append((name, factory(cfg, spec), ctx, chunk))
    return out


def test_critic_protocol_holds_for_every_registered_critic() -> None:
    instances = _critic_instances()
    assert instances, "no critic is registered; the G3 gate would have nothing to evaluate"
    for name, critic, ctx, chunk in instances:
        assert isinstance(critic, Critic), f"critic {name!r} does not satisfy the Critic protocol"
        logits = critic.logits(ctx, chunk)
        assert tuple(logits.shape) == (chunk.shape[0], N_HEAD_COLUMNS), (
            f"critic {name!r} returned logits {tuple(logits.shape)}; the contract is "
            f"[B, 1+5] (any-violation then V1..V5)"
        )
        assert torch.isfinite(logits).all(), f"critic {name!r} produced non-finite logits"


def test_critic_reported_risk_is_monotone_in_the_any_violation_logit() -> None:
    """``risk(..., space="prob")`` must be a strictly increasing function of the
    any-violation logit; if it were not, the projection's gradient would point the
    wrong way and the operator would push activations further into risk."""
    for name, critic, ctx, chunk in _critic_instances():
        logit = critic.risk(ctx, chunk, space="logit")
        prob = critic.risk(ctx, chunk, space="prob")
        torch.testing.assert_close(
            prob, torch.sigmoid(logit), msg=lambda m: f"critic {name}: {m}"
        )
        # monotonicity itself, on a grid of logits
        grid = torch.linspace(-8.0, 8.0, steps=33)
        mapped = torch.sigmoid(grid)
        assert bool((mapped.diff() > 0).all()), "sigmoid stopped being monotone (paranoia check)"


def test_critic_ensemble_disagreement_is_a_usable_ood_signal() -> None:
    """Ensemble std must be zero for a single member and positive for several, since
    it is the diagnostic that makes over-optimisation visible (``06`` §4)."""
    instances = _critic_instances()
    for name, critic, ctx, chunk in instances:
        std = critic.ensemble_std(ctx, chunk)
        assert std.shape == (chunk.shape[0],)
        if len(getattr(critic, "members", [1])) > 1:
            assert float(std.detach().max()) > 0.0, (
                f"critic {name}: ensemble members are identical"
            )


# --------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------


def _toy_task() -> Any:
    from riskproj.envs import build_task
    from riskproj.utils.config import load_config

    cfg = load_config(REPO / "configs" / "exp" / "toy_smoke.yaml")
    return build_task(cfg), cfg


def test_toy_task_satisfies_the_task_protocol() -> None:
    task, _ = _toy_task()
    assert isinstance(task, Task)
    assert isinstance(task.violations.horizon, int) and task.violations.horizon >= 1
    expert = task.expert()
    assert callable(getattr(expert, "act", None)), "task.expert() must expose act(obs)"


def test_toy_env_observation_contract_on_reset_and_step() -> None:
    """The ``obs.py`` key contract is what every downstream consumer relies on."""
    task, cfg = _toy_task()
    env = task.make_env(DynamicsSpec(), seed=0)
    obs, _info = env.reset(seed=0)

    for key in TASK_OBS_KEYS:
        assert key in obs, f"task observation is missing required key {key!r}"
    step_obs = env_obs_to_step_obs(obs)  # must not raise
    # Compare against the env's own declaration: the task is the authority on its
    # state width, and a mismatch with configs/env/*.yaml is a config bug to fix at
    # the source rather than to paper over here.
    declared = getattr(env, "state_dim", None)
    if declared is not None:
        assert step_obs.proprio.shape[0] == int(declared), (
            f"obs 'state' has width {step_obs.proprio.shape[0]} but the env declares "
            f"state_dim={declared}"
        )
    assert step_obs.ee_vel.shape == (3,)

    action = np.zeros(cfg.env.action_dim, dtype=np.float32)
    out = env.step(action)
    assert len(out) == 5, "step must return the gymnasium 5-tuple"
    obs2, _reward, _term, _trunc, info = out
    for key in TASK_OBS_KEYS:
        assert key in obs2
    assert env_obs_to_step_obs(obs2) is not None
    for key in ("progress", "grasped"):
        assert key in info, f"the violation monitor needs {key!r} in the step info"
    assert cfg.env.action_dim >= 1, "the toy config must declare an action width"


def test_toy_env_readback_matches_the_requested_dynamics() -> None:
    """G1's whole premise: the perturbation must actually reach the physics, and the
    read-back must prove it (a silent no-op would invalidate every robustness number)."""
    from riskproj.data.gen_demos import readback_dict

    task, _ = _toy_task()
    spec = DynamicsSpec(mu_scale=0.7, mass_scale=1.6, stiff_scale=0.4)
    env = task.make_env(spec, seed=0)
    env.reset(seed=0)

    readback = readback_dict(env)
    for axis, want in zip(AXIS_NAMES, (0.7, 1.6, 0.4), strict=True):
        node = readback.get(axis)
        assert isinstance(node, dict), f"read-back has no {axis!r} section: {readback}"
        got = next(
            (float(node[k]) for k in ("scale", "ratio", "mean", "value") if k in node), None
        )
        assert got is not None, f"read-back for {axis!r} has no recognisable scale: {node}"
        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9), (
            f"{axis}: requested scale {want} but the physics reads back {got}"
        )


# --------------------------------------------------------------------------
# protocols and cross-cutting guarantees
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "protocol", [Critic, Method, Metric, PolicyAdapter, Projector, Task], ids=lambda p: p.__name__
)
def test_protocols_are_runtime_checkable(protocol: Any) -> None:
    assert getattr(protocol, "_is_runtime_protocol", False), (
        f"{protocol.__name__} must be decorated with @runtime_checkable: the registries and "
        "the contract suite both isinstance-check implementations (skills/software_engineering.md §2)"
    )


def test_concrete_implementations_satisfy_their_protocols() -> None:
    task, _ = _toy_task()
    assert isinstance(task, Task)
    # ``Task.expert`` is a *callable* contract (``04`` §7: the demos generator calls
    # ``task.expert()(obs)``), so an object that merely exists is not enough.
    assert callable(task.expert())
    assert isinstance(
        ToyPolicy(state_dim=12, action_dim=3, hidden=16, chunk_horizon=4, num_flow_steps=5, seed=0),
        PolicyAdapter,
    )


def test_registered_methods_build_from_the_shipped_arm_configs() -> None:
    """The registry is how the runner finds an arm and ``configs/method/`` is how arms
    are declared, so an arm that cannot be *built* from its own shipped config — or that
    fails its Protocol — is broken before any experiment runs.

    Building, not looking up, is the check: the registry stores factories
    (``skills/software_engineering.md`` §3), and only the built object can be tested.
    """
    import yaml

    import riskproj.methods  # noqa: F401  (import = registration)
    from riskproj.methods.base import build_method
    from riskproj.utils import registry
    from riskproj.utils.config import MethodCfg

    specs: dict[str, dict[str, Any]] = {}
    for path in sorted((REPO / "configs" / "method").glob("*.yaml")):
        for entry in yaml.safe_load(path.read_text(encoding="utf-8")).get("methods", []):
            specs[str(entry["id"])] = entry
    assert {"B0", "B1", "B3", "C1", "C2", "C3"} <= set(specs), (
        "the shipped arm configs must cover the baseline, the method and every control"
    )
    if not specs:  # pragma: no cover - guards against a silently empty arm catalogue
        raise AssertionError("configs/method/ declares no arms at all")

    for arm_id, entry in sorted(specs.items()):
        built = build_method(MethodCfg.model_validate(entry))
        assert isinstance(built, Method), f"arm {arm_id!r} does not satisfy Method"
        assert built.method_cfg.id == arm_id, (
            f"arm {arm_id!r} built a method claiming to be {built.method_cfg.id!r}"
        )

    # And the class of bug that a build loop over shipped fragments cannot see: a config
    # anywhere in the tree naming a method id that is not registered. An arm id is
    # resolved through the registry, so one typo is a run that dies halfway through.
    known = set(registry.available("method"))
    for path in sorted(REPO.glob("configs/**/*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            continue
        for entry in doc.get("methods", []) or []:
            assert str(entry["id"]) in known, (
                f"{path.relative_to(REPO)} names method {entry['id']!r}, which is not "
                f"registered; registered ids: {sorted(known)}"
            )


def test_tolerances_are_positive_floats_and_actually_used() -> None:
    """Tolerances live in exactly one place; zeroing one would make the numerical
    tests pass vacuously, so the constants and their usage are both checked."""
    try:
        import tolerances as tol
    except ImportError:  # pragma: no cover - pytest normally puts tests/ on sys.path
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import tolerances as tol

    expected = {
        "EXACT": tol.EXACT,
        "FD_REL": tol.FD_REL,
        "MATCH": tol.MATCH,
        "STATS": tol.STATS,
        "FLOW": tol.FLOW,
        "SIM_STATE": tol.SIM_STATE,
        "DYN_READBACK": tol.DYN_READBACK,
    }
    for name, value in expected.items():
        assert isinstance(value, float), f"tolerance {name} must be a float"
        assert value > 0.0, f"tolerance {name} must be positive; got {value}"

    numeric_tests = [
        REPO / "tests" / "unit" / "test_projection.py",
        REPO / "tests" / "unit" / "test_backward.py",
    ]
    for path in numeric_tests:
        text = path.read_text(encoding="utf-8")
        assert "tolerances" in text, (
            f"{path.name} no longer imports shared tolerances; numerical magic numbers "
            "must live in tests/tolerances.py (their rationale is documented there)"
        )
