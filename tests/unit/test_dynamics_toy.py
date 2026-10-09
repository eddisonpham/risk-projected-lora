"""Dynamics application: read-back verification, idempotence and fail-loud paths."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from riskproj.envs import ToyEnv, apply_dynamics, load_task_geoms, read_back_dynamics
from riskproj.types import DynamicsSpec

pytestmark = pytest.mark.unit


class _Opaque:
    """Neither MuJoCo-backed nor delegating."""


class _BadDelegate:
    def apply_dynamics(self, spec: DynamicsSpec) -> dict[str, Any]:
        return {"changed": True}


class _NoopDelegate:
    def apply_dynamics(self, spec: DynamicsSpec) -> dict[str, Any]:
        return {
            "changed": False,
            "effective": {"mu_scale": 1.0, "mass_scale": 1.0, "stiff_scale": 1.0},
        }


class _LyingDelegate:
    def apply_dynamics(self, spec: DynamicsSpec) -> dict[str, Any]:
        return {
            "changed": True,
            "effective": {"mu_scale": 1.0, "mass_scale": 1.0, "stiff_scale": 1.0},
        }


def test_toy_apply_dynamics_roundtrip() -> None:
    env = ToyEnv(seed=0)
    spec = DynamicsSpec(mu_scale=1.5, mass_scale=2.0, stiff_scale=4.0)
    rb = apply_dynamics(env, spec)
    assert rb.changed is True
    assert rb.verified is True
    assert rb.assert_applied() is rb
    assert set(rb.mu) == {"grip_contact", "table"}
    assert set(rb.mass) == {"object"}
    assert rb.effective["mu_scale"] == pytest.approx(1.5)
    assert rb.effective["mass_scale"] == pytest.approx(2.0)
    assert rb.effective["stiff_scale"] == pytest.approx(4.0)
    # solref mirrors MuJoCo's rule: timeconst ~ 1/sqrt(stiff_scale)
    assert rb.solref["contact"][0] == pytest.approx(0.5)


def test_nominal_spec_is_a_no_op_but_valid() -> None:
    env = ToyEnv(seed=0)
    rb = apply_dynamics(env, DynamicsSpec())
    assert rb.changed is False
    assert rb.verified is True
    assert rb.assert_applied() is rb


def test_repeated_application_is_idempotent() -> None:
    env = ToyEnv(seed=0)
    for _ in range(3):
        apply_dynamics(env, DynamicsSpec(mu_scale=2.0, mass_scale=1.5))
    eff = env.effective_dynamics()["effective"]
    assert eff["mu_scale"] == pytest.approx(2.0)
    assert eff["mass_scale"] == pytest.approx(1.5)


def test_read_back_matches_requested_scales() -> None:
    env = ToyEnv(seed=0)
    apply_dynamics(env, DynamicsSpec(mu_scale=0.5, mass_scale=2.5, stiff_scale=0.25))
    rb = read_back_dynamics(env)
    assert rb.effective["mu_scale"] == pytest.approx(0.5)
    assert rb.effective["mass_scale"] == pytest.approx(2.5)
    assert rb.effective["stiff_scale"] == pytest.approx(0.25)


def test_unsupported_target_raises_type_error() -> None:
    with pytest.raises(TypeError):
        apply_dynamics(_Opaque(), DynamicsSpec(mu_scale=2.0))
    with pytest.raises(TypeError):
        apply_dynamics(_Opaque(), DynamicsSpec())


def test_delegate_must_report_effective_scales() -> None:
    with pytest.raises(ValueError, match="effective"):
        apply_dynamics(_BadDelegate(), DynamicsSpec(mu_scale=2.0))


def test_delegate_that_changes_nothing_fails_loud() -> None:
    with pytest.raises(RuntimeError):
        apply_dynamics(_NoopDelegate(), DynamicsSpec(mu_scale=2.0))


def test_delegate_that_lies_about_the_scales_fails_loud() -> None:
    with pytest.raises(RuntimeError, match="read-back"):
        apply_dynamics(_LyingDelegate(), DynamicsSpec(mu_scale=2.0))


def test_invalid_scales_are_rejected() -> None:
    env = ToyEnv(seed=0)
    with pytest.raises(ValueError):
        apply_dynamics(env, DynamicsSpec(mu_scale=0.0))
    with pytest.raises(ValueError):
        apply_dynamics(env, DynamicsSpec(stiff_scale=-1.0))


def test_load_task_geoms_missing_file_is_actionable(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError, match="inspect_contacts"):
        load_task_geoms(tmp_path / "does-not-exist.yaml")


def test_load_task_geoms_merges_groups(tmp_path: Any) -> None:
    path = tmp_path / "task_geoms.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "task": "peg-insert-side-v3",
                "geoms": ["table"],
                "groups": {"gripper": ["grip_pad_l", "grip_pad_r"], "object": ["peg"]},
                "bodies": ["peg"],
            }
        ),
        encoding="utf-8",
    )
    spec = load_task_geoms(path)
    assert spec["geoms"] == ["grip_pad_l", "grip_pad_r", "peg", "table"]
    assert spec["bodies"] == ["peg"]
    assert spec["task"] == "peg-insert-side-v3"


def test_load_task_geoms_rejects_a_file_without_geoms(tmp_path: Any) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("task: nothing\nbodies: [peg]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no geoms"):
        load_task_geoms(path)


def test_task_critical_geoms_flags_a_task_mismatch(tmp_path: Any) -> None:
    from riskproj.envs.dynamics import task_critical_geoms

    path = tmp_path / "task_geoms.yaml"
    path.write_text(
        yaml.safe_dump({"task": "peg-insert-side-v3", "geoms": ["peg"]}), encoding="utf-8"
    )

    class _Env:
        task_name = "assembly-v3"

    with pytest.raises(ValueError, match="assembly-v3"):
        task_critical_geoms(_Env(), path=path)
