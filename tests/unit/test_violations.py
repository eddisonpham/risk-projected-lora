"""Violation events V1-V5: detection, windows, taxonomy and fail-loud behaviour."""

from __future__ import annotations

from typing import Any

import pytest

from riskproj.envs.violations import (
    ViolationMonitor,
    any_violation,
    failure_class,
    violation_rate,
    violations_in_window,
)
from riskproj.types import VIOLATION_TYPES, ViolationSpec, empty_violation_counts

pytestmark = pytest.mark.unit

SPEC = ViolationSpec(
    slip_drift=0.008,
    drop_dist=0.025,
    force_max=8.0,
    penetration=0.006,
    jam_force=4.0,
    jam_progress_eps=0.001,
    jam_steps=3,
    window=5,
    horizon=10,
)


def info(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "eef_pos": [0.0, 0.0, 0.0],
        "object_pos": [0.0, 0.0, 0.0],
        "grasped": True,
        "force": 0.0,
        "penetration": 0.0,
        "progress_m": 0.0,
    }
    base.update(over)
    return base


def fired(rec: dict[str, dict[str, float]], key: str) -> bool:
    return float(rec[key]["count"]) > 0.0


def test_record_shape_matches_empty_violation_counts() -> None:
    mon = ViolationMonitor(SPEC)
    rec = mon.observe(info())
    assert set(rec) == set(VIOLATION_TYPES)
    for key, value in rec.items():
        assert set(value) == set(empty_violation_counts()[key])
        assert value["first_step"] == -1.0
        assert value["count"] == 0.0


def test_missing_keys_fail_loud() -> None:
    mon = ViolationMonitor(SPEC)
    bad = info()
    del bad["force"]
    with pytest.raises(KeyError, match="force"):
        mon.observe(bad)


def test_non_finite_inputs_fail_loud() -> None:
    mon = ViolationMonitor(SPEC)
    with pytest.raises(ValueError):
        mon.observe(info(eef_pos=[float("nan"), 0.0, 0.0]))
    with pytest.raises(ValueError):
        mon.observe(info(force=float("inf")))
    with pytest.raises(ValueError):
        mon.observe(info(eef_pos=[0.0, 0.0]))
    with pytest.raises(ValueError):
        mon.observe(info(progress_m=float("nan")))


def test_v1_slip_fires_on_gripper_frame_drift() -> None:
    mon = ViolationMonitor(SPEC)
    mon.observe(info(grip_offset=[0.0, 0.0, 0.0]))
    rec = mon.observe(info(grip_offset=[0.0, 0.0, 0.02]))
    assert fired(rec, "V1")
    assert rec["V1"]["first_step"] == 1.0


def test_v1_requires_the_object_to_be_grasped() -> None:
    mon = ViolationMonitor(SPEC)
    mon.observe(info(grasped=False, grasp_established=False, grip_offset=[0.0, 0.0, 0.0]))
    rec = mon.observe(info(grasped=False, grasp_established=False, grip_offset=[0.0, 0.0, 0.02]))
    assert not fired(rec, "V1")


def test_v1_reference_resets_after_a_new_grasp() -> None:
    mon = ViolationMonitor(SPEC)
    mon.observe(info(grip_offset=[0.0, 0.0, 0.0]))
    mon.observe(info(grasped=False, released=True))
    rec = mon.observe(info(grip_offset=[0.0, 0.0, 0.5]))
    assert not fired(rec, "V1"), "a fresh grasp must define a fresh reference offset"


def test_v2_drop_from_object_distance() -> None:
    mon = ViolationMonitor(SPEC)
    mon.observe(info())
    rec = mon.observe(info(object_pos=[0.05, 0.0, 0.0]))
    assert fired(rec, "V2")


def test_v2_drop_from_lost_grasp_without_release() -> None:
    mon = ViolationMonitor(SPEC)
    mon.observe(info())
    rec = mon.observe(info(grasped=False, grasp_established=True, released=False))
    assert fired(rec, "V2")


def test_v2_not_fired_on_intentional_release() -> None:
    mon = ViolationMonitor(SPEC)
    mon.observe(info())
    rec = mon.observe(info(grasped=False, grasp_established=True, released=True))
    assert not fired(rec, "V2")


def test_v2_drop_when_the_object_leaves_the_gripper_frame() -> None:
    """A drop is a loss of the gripper-frame offset, not a change in absolute height."""
    mon = ViolationMonitor(SPEC)
    mon.observe(info())
    rec = mon.observe(info(object_pos=[0.0, 0.0, -0.03], object_height=0.06))
    assert fired(rec, "V2")


def test_v2_not_fired_by_commanded_lowering() -> None:
    """Placing a held object means descending with it: that is not a drop.

    Regression guard: an earlier absolute-height rule fired on every nominal toy
    expert episode, because inserting the puck descends it ~25 mm.
    """
    mon = ViolationMonitor(SPEC)
    for z in (0.10, 0.08, 0.06, 0.04):
        rec = mon.observe(info(eef_pos=[0.0, 0.0, z], object_pos=[0.0, 0.0, z - 0.01]))
        assert not fired(rec, "V2"), f"lowering to z={z} was reported as a drop"


def test_v3_over_force() -> None:
    mon = ViolationMonitor(SPEC)
    assert not fired(mon.observe(info(force=7.9)), "V3")
    assert fired(mon.observe(info(force=8.1)), "V3")


def test_v3_takes_the_largest_pair_not_the_sum() -> None:
    """A sum would put these two samples on opposite sides of `force_max`.

    The per-pair maximum is what the derived bar describes: on peg-insert-side the summed
    force is ~3.2 kN at nominal and falls as the contact stiffens, so thresholding it is
    both unreachable and confounded with the stiffness axis.
    """
    mon = ViolationMonitor(SPEC)
    assert not fired(mon.observe(info(force=[4.0, 3.0])), "V3")
    assert not fired(mon.observe(info(force=[4.0, 4.5])), "V3"), "4.0 + 4.5 > 8 must not fire"
    assert fired(mon.observe(info(force=[4.0, 8.5])), "V3")


def test_v4_penetration_uses_absolute_depth() -> None:
    mon = ViolationMonitor(SPEC)
    assert not fired(mon.observe(info(penetration=0.005)), "V4")
    assert fired(mon.observe(info(penetration=0.007)), "V4")


def test_v4_takes_the_max_over_contacts() -> None:
    mon = ViolationMonitor(SPEC)
    assert not fired(mon.observe(info(penetration=[0.001, 0.002])), "V4")
    assert fired(mon.observe(info(penetration=[0.001, 0.02])), "V4")


def test_v5_jam_needs_sustained_force_without_progress() -> None:
    mon = ViolationMonitor(SPEC)
    for _ in range(SPEC.jam_steps - 1):
        rec = mon.observe(info(force=5.0, progress_m=0.0))
        assert not fired(rec, "V5"), "V5 must not fire before jam_steps consecutive steps"
    rec = mon.observe(info(force=5.0, progress_m=0.0))
    assert fired(rec, "V5")


def test_v5_does_not_fire_when_progress_continues() -> None:
    mon = ViolationMonitor(SPEC)
    rec: dict[str, dict[str, float]] = {}
    for step in range(2 * SPEC.jam_steps):
        rec = mon.observe(info(force=5.0, progress_m=0.01 * step))
    assert not fired(rec, "V5")


def test_finalize_freezes_the_record() -> None:
    mon = ViolationMonitor(SPEC)
    mon.observe(info(force=9.0))
    rec = mon.finalize()
    assert fired(rec, "V3")
    with pytest.raises(RuntimeError):
        mon.observe(info())


def test_window_slicing_uses_absolute_steps() -> None:
    mon = ViolationMonitor(SPEC)
    mon.observe(info(force=9.0))  # step 0 -> V3
    for _ in range(4):
        mon.observe(info())
    early = mon.window(0, 1)
    late = mon.window(2, 5)
    assert fired(early, "V3") and early["V3"]["first_step"] == 0.0
    assert not fired(late, "V3")


def test_window_validates_bounds() -> None:
    mon = ViolationMonitor(SPEC)
    mon.observe(info())
    with pytest.raises(ValueError):
        mon.window(3, 2)
    with pytest.raises(ValueError):
        mon.window(-1, 1)


def test_violations_in_window_rejects_unknown_types() -> None:
    with pytest.raises(KeyError):
        violations_in_window([frozenset({"V7"})], 0, 1)


def test_any_violation_and_rate() -> None:
    clean = empty_violation_counts()
    dirty = empty_violation_counts()
    dirty["V2"]["count"] = 3.0
    dirty["V2"]["first_step"] = 1.0
    assert any_violation(dirty) is True
    assert any_violation(clean) is False
    rate = violation_rate([clean, dirty])
    assert rate["V2"] == pytest.approx(0.5)
    assert rate["any"] == pytest.approx(0.5)
    assert rate["V1"] == pytest.approx(0.0)


def test_violation_rate_of_nothing_is_nan() -> None:
    rate = violation_rate([])
    assert all(value != value for value in rate.values())  # NaN


def test_failure_class_priority_and_timeout_branches() -> None:
    rec = empty_violation_counts()
    rec["V1"]["count"] = 1.0
    rec["V5"]["count"] = 2.0
    assert failure_class(rec, success=False, progress=0.9, max_steps_reached=True) == "rim_jam"
    rec2 = empty_violation_counts()
    rec2["V4"]["count"] = 1.0
    assert failure_class(rec2, success=False, progress=0.9, max_steps_reached=True) == "penetration"
    assert (
        failure_class(empty_violation_counts(), success=False, progress=0.9, max_steps_reached=True)
        == "stop_short"
    )
    assert (
        failure_class(empty_violation_counts(), success=False, progress=0.0, max_steps_reached=True)
        == "idle_timeout"
    )
    assert (
        failure_class(
            empty_violation_counts(), success=False, progress=0.0, max_steps_reached=False
        )
        == "none"
    )


def test_failure_class_is_none_for_successful_episodes() -> None:
    rec = empty_violation_counts()
    rec["V3"]["count"] = 5.0
    assert failure_class(rec, success=True, progress=1.0, max_steps_reached=False) == "none"


def test_counts_and_per_type_accessors() -> None:
    mon = ViolationMonitor(SPEC)
    mon.observe(info(force=9.0, penetration=0.01))
    mon.observe(info(force=9.0, penetration=0.01))
    assert mon.counts()["V3"] == 2
    assert mon.counts()["V4"] == 2
    assert mon.any_violation is True
    assert mon.step == 2
    assert set(mon.per_type) == set(VIOLATION_TYPES)
