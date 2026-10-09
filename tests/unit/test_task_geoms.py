"""Task-critical geom discovery: the conventions that make ``04`` §2 enforceable.

Three things are pinned here, each of which was a real failure during provisioning:

* the shipped ``configs/env/task_geoms.yaml`` is loadable and the mass axis has a body
  to scale (a geom list with no bodies makes the payload axis a silent no-op);
* Meta-World names a task by its **class** while configs use the **registered id**
  (``SawyerPegInsertionSideEnvV3`` vs ``peg-insert-side-v3``; "Insertion" vs "insert"),
  so the mismatch guard must accept that pair and still reject a different task;
* the peg aperture's collision geoms are **unnamed**, so they can only be addressed as
  ``geom<id>``; that token must resolve, and it must not be usable to smuggle a
  positional reference to a geom that has a perfectly good name.

The resolvers take ``mujoco`` as a lazily imported module, so these run on CPU with a
fake: no simulator extra and no GPU are needed.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from riskproj.envs.dynamics import (
    _task_names_match,
    load_task_geoms,
    resolve_geom_ids,
    task_critical_geoms,
)


class _FakeMujoco:
    """The two entry points the resolver uses, backed by a geom-id -> name table."""

    class _Obj:
        mjOBJ_GEOM = "geom"
        mjOBJ_BODY = "body"

    mjtObj = _Obj()

    def __init__(self, geoms: dict[int, str | None]) -> None:
        self._geoms = geoms

    def mj_name2id(self, model: object, _obj: object, name: str) -> int:
        return next((gid for gid, gname in self._geoms.items() if gname == name), -1)

    def mj_id2name(self, model: object, _obj: object, index: int) -> str | None:
        return self._geoms.get(index)


class _FakeModel:
    def __init__(self, geoms: dict[int, str | None]) -> None:
        self._geoms = geoms
        # “Count” means *one past the highest id*, exactly as in MuJoCo: a model can have
        # unnamed ids that no name lookup reaches.
        self.ngeom = max(geoms, default=-1) + 1
        self.nbody = 0


@contextmanager
def _fake_mujoco(geoms: dict[int, str | None]) -> Iterator[None]:
    """Install a fake ``mujoco`` module for the duration of a resolver call."""
    original = sys.modules.get("mujoco")
    sys.modules["mujoco"] = _FakeMujoco(geoms)  # type: ignore[assignment]
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("mujoco", None)
        else:
            sys.modules["mujoco"] = original  # type: ignore[assignment]


# --------------------------------------------------------------------------
# the shipped discovery file
# --------------------------------------------------------------------------


def test_shipped_task_geoms_file_is_loadable_and_complete(repo_root: Path) -> None:
    path = repo_root / "configs" / "env" / "task_geoms.yaml"
    if not path.exists():
        pytest.skip("configs/env/task_geoms.yaml is written by scripts/inspect_contacts.py")
    spec = load_task_geoms(path)
    assert spec["task"] == "peg-insert-side-v3"
    assert "peg" in spec["geoms"], "the manipulated object must be friction-scaled"
    assert spec["bodies"] == ["peg"], (
        "the mass axis scales body_mass of the listed bodies; the peg is the only movable "
        "body with mass, and an empty list would make the axis a silent no-op"
    )
    groups = spec["groups"]
    for name in ("gripper", "object", "fixture"):
        assert groups.get(name), f"the {name} group must be documented in the discovery file"


# --------------------------------------------------------------------------
# task-name matching
# --------------------------------------------------------------------------


def test_task_name_match_accepts_class_name_and_id() -> None:
    assert _task_names_match("SawyerPegInsertionSideEnvV3", "peg-insert-side-v3")
    assert _task_names_match("peg-insert-side-v3", "peg-insert-side-v3")


def test_task_name_match_rejects_a_different_task() -> None:
    assert not _task_names_match("SawyerNutAssemblyEnvV3", "peg-insert-side-v3")
    assert not _task_names_match("SawyerPegInsertionSideEnvV3", "hand-insert-v3")
    assert not _task_names_match("", "peg-insert-side-v3")


def test_task_critical_geoms_rejects_a_mismatched_env(tmp_path: Path) -> None:
    path = tmp_path / "task_geoms.yaml"
    path.write_text("task: assembly-v3\ngeoms: [peg]\nbodies: [peg]\n", encoding="utf-8")

    class _Env:
        task_name = "SawyerPegInsertionSideEnvV3"

    with pytest.raises(ValueError, match="written for task"):
        task_critical_geoms(_Env(), path=path)


# --------------------------------------------------------------------------
# geom<id> resolution
# --------------------------------------------------------------------------


def test_named_geoms_and_unnamed_tokens_both_resolve() -> None:
    table: dict[int, str | None] = {0: "peg", 3: None, 4: None}
    with _fake_mujoco(table):
        assert resolve_geom_ids(_FakeModel(table), ["peg", "geom3", "geom4"]) == [0, 3, 4]


def test_positional_token_for_a_named_geom_is_rejected() -> None:
    table: dict[int, str | None] = {0: "peg", 1: "table"}
    with _fake_mujoco(table), pytest.raises(KeyError, match="address it by name"):
        resolve_geom_ids(_FakeModel(table), ["geom1"])


def test_out_of_range_positional_token_is_rejected() -> None:
    table: dict[int, str | None] = {0: "peg"}
    with _fake_mujoco(table), pytest.raises(KeyError, match="out of range"):
        resolve_geom_ids(_FakeModel(table), ["geom99"])


def test_unknown_name_is_rejected_with_the_convention_in_the_message() -> None:
    table: dict[int, str | None] = {0: "peg"}
    with _fake_mujoco(table), pytest.raises(KeyError, match="geom<id>"):
        resolve_geom_ids(_FakeModel(table), ["nope"])
