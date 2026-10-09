"""Environment layer: dynamics knobs, violation events, exact state io, tasks.

Two tasks are registered:

* ``toy`` (:mod:`riskproj.envs.toy`) — deterministic CPU walking skeleton, no
  simulator dependency, used by the CI smoke path (S6) and by any test that needs
  the whole pipeline without MuJoCo;
* ``metaworld`` (:mod:`riskproj.envs.metaworld_dyn`) — the real Meta-World v3
  task, lazily importing gymnasium/metaworld/mujoco.

Threshold units differ between the two on purpose: the toy spec is in toy units
and is **not** a substitute for the Meta-World thresholds, which must be derived
from nominal expert rollouts (``04`` §4).
"""

from __future__ import annotations

from typing import Any

from riskproj.envs.dynamics import (
    BASELINE_ATTR,
    TASK_GEOMS_PATH,
    DynReadback,
    apply_dynamics,
    contact_friction_readback,
    load_task_geoms,
    read_back_dynamics,
    task_critical_geoms,
)
from riskproj.envs.metaworld_dyn import MetaWorldDynEnv, MetaWorldExpert, MetaWorldTask, probe_task
from riskproj.envs.state_io import (
    SimState,
    restore_state,
    rollout_from_state,
    save_state,
    state_hash,
    states_equal,
)
from riskproj.envs.toy import (
    TOY_INSTRUCTION,
    TOY_VIOLATION_SPEC,
    ToyEnv,
    ToyExpert,
    ToyPhysics,
    ToyReachTask,
    set_cell_id,
)
from riskproj.envs.violations import (
    ViolationMonitor,
    any_violation,
    failure_class,
    violation_rate,
    violations_in_window,
)


def build_task(cfg: Any) -> Any:
    """Build the :class:`~riskproj.protocols.Task` named by an experiment config.

    ``cfg.env.name`` selects the task. For ``"toy"`` the *toy-scale* violation
    spec is used deliberately (``configs/violations.yaml`` holds Meta-World
    thresholds and would make every event unreachable); pass
    ``violations=...`` to :class:`ToyReachTask` directly if a different toy spec
    is wanted.
    """
    env_cfg = getattr(cfg, "env", None)
    if env_cfg is None:
        raise TypeError(f"config {type(cfg).__name__} has no `env` section")
    name = str(getattr(env_cfg, "name", ""))
    if name == "toy":
        return ToyReachTask()
    if name == "metaworld":
        violation_cfg = getattr(cfg, "violations", None)
        spec = violation_cfg.to_spec() if violation_cfg is not None else None
        return MetaWorldTask(
            task=str(getattr(env_cfg, "task", "peg-insert-side-v3")),
            violations=spec,
            camera_size=tuple(getattr(env_cfg, "camera_size", (128, 128))),
            max_steps=int(getattr(env_cfg, "max_steps", 200)),
        )
    raise KeyError(f"unknown env.name {name!r}; registered tasks: 'toy', 'metaworld'")


__all__ = [
    "BASELINE_ATTR",
    "TASK_GEOMS_PATH",
    "TOY_INSTRUCTION",
    "TOY_VIOLATION_SPEC",
    "DynReadback",
    "MetaWorldDynEnv",
    "MetaWorldExpert",
    "MetaWorldTask",
    "SimState",
    "ToyEnv",
    "ToyExpert",
    "ToyPhysics",
    "ToyReachTask",
    "ViolationMonitor",
    "any_violation",
    "apply_dynamics",
    "build_task",
    "contact_friction_readback",
    "failure_class",
    "load_task_geoms",
    "probe_task",
    "read_back_dynamics",
    "restore_state",
    "rollout_from_state",
    "save_state",
    "set_cell_id",
    "state_hash",
    "states_equal",
    "task_critical_geoms",
    "violation_rate",
    "violations_in_window",
]
