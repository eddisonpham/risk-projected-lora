"""Meta-World v3 task wrapper: dynamics knobs, violation monitoring, expert (``04`` §1-§2).

Everything heavy (``gymnasium``, ``metaworld``, ``mujoco``) is imported lazily
inside functions so that the CPU/CI path never pays for it and a machine without
the simulator extra can still run ``pytest -m unit``.

Several internals are tagged ``[R]`` in the spec and are therefore *resolved at
runtime with an explicit candidate list* rather than hard-coded:

* the scripted-expert class name (``metaworld.policies.SawyerPegInsertionSideV3Policy``
  is the expected shape, but the spec says the name is recalled, not verified);
* the gym registration id (``Meta-World/MT1/<task>``, ``Meta-World/<task>``, ...);
* the end-effector site / object accessor used to compute progress.

When a name cannot be resolved we raise with the list of things we tried, so the
smoke test S1 reports an actionable failure instead of a wrong number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from riskproj.envs.dynamics import apply_dynamics, contact_friction_readback
from riskproj.envs.state_io import restore_state, save_state
from riskproj.envs.violations import ViolationMonitor
from riskproj.types import DynamicsSpec, ViolationSpec
from riskproj.utils.registry import register

#: Candidate gym ids, tried in order (``[R]`` -> resolved at runtime).
GYM_ID_TEMPLATES: tuple[str, ...] = (
    "Meta-World/MT1/{task}",
    "Meta-World/{task}",
    "metaworld/{task}",
    "{task}",
)

#: Candidate attribute chains for the end-effector position.
EEF_ATTR_CHAINS: tuple[tuple[str, ...], ...] = (
    ("tcp_center",),
    ("_get_pos_goal",),
    ("hand", "position"),
)

#: Candidate attribute chains for the manipulated-object position/height.
OBJECT_ATTR_CHAINS: tuple[tuple[str, ...], ...] = (
    ("_get_pos_objects",),
    ("obj", "position"),
)

#: Distance below which the object counts as held, together with a closed gripper.
GRASP_DISTANCE = 0.06


def _import_sim() -> tuple[Any, Any]:
    """Import gymnasium + metaworld with an actionable error message."""
    try:
        import gymnasium as gym
        import metaworld
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise ImportError(
            "Meta-World/gymnasium are not installed. Install the simulator extra with "
            '`pip install -e ".[sim]"` and make the pinned submodule importable with '
            "`git submodule update --init third_party/Metaworld` "
            "(then `pip install -e third_party/Metaworld`)."
        ) from exc
    return gym, metaworld


def resolve_gym_id(task: str, gym: Any) -> str:
    """First gym id in :data:`GYM_ID_TEMPLATES` that is registered."""
    tried: list[str] = []
    for template in GYM_ID_TEMPLATES:
        candidate = template.format(task=task)
        tried.append(candidate)
        try:
            gym.spec(candidate)
        except Exception:  # gymnasium raises NameNotFound / DeprecatedEnv
            continue
        return candidate
    raise KeyError(
        f"none of {tried} is a registered gymnasium env for task {task!r}. "
        "Run `python -c \"import metaworld, gymnasium as gym; print([k for k in gym.envs.registry if 'peg' in k])\"` "
        "to list what this install actually registers."
    )


def _expert_candidates(task: str) -> list[str]:
    base = task.replace("-v3", "").replace("-v2", "").replace("-", " ")
    camel = "".join(part.capitalize() for part in base.split())
    out = [f"Sawyer{camel}V3Policy", f"Sawyer{camel}Policy", f"{camel}V3Policy", f"{camel}Policy"]
    # dedupe, keep order
    seen: set[str] = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def resolve_expert_class(task: str) -> type:
    """Find the scripted expert class for ``task`` in ``metaworld.policies``."""
    _import_sim()
    import metaworld.policies as policies

    tried = _expert_candidates(task)
    for name in tried:
        cls = getattr(policies, name, None)
        if cls is not None:
            return cls
    # last resort: fuzzy match on the task tokens (names drift [R])
    tokens = [t for t in task.replace("-v3", "").split("-") if t]
    available = [a for a in dir(policies) if a.endswith("Policy")]
    matches = [a for a in available if all(tok[:5].lower() in a.lower() for tok in tokens)]
    if len(matches) == 1:
        return getattr(policies, matches[0])
    raise KeyError(
        f"could not resolve a scripted expert for {task!r}. Tried {tried}; fuzzy matches: {matches}. "
        f"Available policies start with: {available[:8]}..."
    )


def _resolve_chain(obj: Any, chain: Sequence[str]) -> Any | None:
    cur = obj
    for attr in chain:
        if cur is None:
            return None
        cur = getattr(cur, attr, None)
        if callable(cur):
            cur = cur()
    return cur


@dataclass
class MetaWorldExpert:
    """Adapter giving the scripted expert the same ``act(obs) -> action`` shape as ToyExpert."""

    policy: Any
    name: str = "metaworld_expert"

    def reset(self) -> None:  # parity with ToyExpert
        return None

    def act(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            if "raw_obs" not in obs:
                raise KeyError(
                    "MetaWorldExpert.act needs the flat environment observation under the "
                    "'raw_obs' key (the policy consumes the raw Meta-World observation)"
                )
            raw = obs["raw_obs"]
        else:
            raw = obs
        action = self.policy.get_action(np.asarray(raw, dtype=np.float64).reshape(-1))
        return np.asarray(action, dtype=np.float32).reshape(-1)


@register("task", "metaworld")
class MetaWorldTask:
    """``Task`` implementation for a Meta-World v3 task (``04`` §1)."""

    name = "metaworld"

    def __init__(
        self,
        task: str = "peg-insert-side-v3",
        violations: ViolationSpec | None = None,
        *,
        render: bool = False,
        camera_size: tuple[int, int] = (128, 128),
        max_steps: int | None = None,
        task_geoms: dict[str, Any] | None = None,
    ) -> None:
        self.task = task
        self.violations = violations or ViolationSpec()
        self.render = bool(render)
        self.camera_size = (int(camera_size[0]), int(camera_size[1]))
        self.max_steps = max_steps
        self.task_geoms = task_geoms
        self._env: MetaWorldDynEnv | None = None

    def make_env(self, dyn: DynamicsSpec, seed: int, **kwargs: Any) -> MetaWorldDynEnv:
        env = MetaWorldDynEnv(
            self.task,
            self.violations,
            dyn=dyn,
            seed=seed,
            render=kwargs.pop("render", self.render),
            camera_size=kwargs.pop("camera_size", self.camera_size),
            max_steps=kwargs.pop("max_steps", self.max_steps),
            task_geoms=kwargs.pop("task_geoms", self.task_geoms),
        )
        if kwargs:
            raise TypeError(f"MetaWorldTask.make_env got unexpected kwargs {sorted(kwargs)}")
        self._env = env
        return env

    def expert(self) -> MetaWorldExpert:
        cls = resolve_expert_class(self.task)
        return MetaWorldExpert(policy=cls(), name=f"{cls.__name__}")


class MetaWorldDynEnv:
    """Wraps a Meta-World v3 env: dynamics read-back, obs contract, violations."""

    def __init__(
        self,
        task: str,
        spec: ViolationSpec,
        *,
        dyn: DynamicsSpec | None = None,
        seed: int = 0,
        render: bool = False,
        camera_size: tuple[int, int] = (128, 128),
        max_steps: int | None = None,
        task_geoms: dict[str, Any] | None = None,
    ) -> None:
        self.task_name = task
        self.spec = spec
        self.task_geoms = task_geoms
        self.seed = int(seed)
        self.camera_size = camera_size
        self._render = bool(render)
        self._last_gripper_cmd = 1.0
        self._last_raw_obs: np.ndarray | None = None
        self._last_progress_m = 0.0
        self._initial_distance: float | None = None

        gym, _ = _import_sim()
        gym_id = resolve_gym_id(task, gym)
        if render:
            self._env = gym.make(gym_id, render_mode="rgb_array", camera_name="topview")
        else:
            self._env = gym.make(gym_id)
        self._env.reset(seed=self.seed)

        base = self._env.unwrapped
        self.action_dim = int(
            np.prod(getattr(base, "action_space", gym.spaces.Box(0, 0, (4,))).shape)
        )
        self.max_episode_steps = int(
            max_steps
            if max_steps is not None
            else (
                getattr(self._env, "_max_episode_steps", None)
                or getattr(base, "max_path_length", 200)
            )
        )
        self.action_space = getattr(base, "action_space", None)
        self.observation_space = getattr(base, "observation_space", None)

        self._monitor = ViolationMonitor(spec)
        self._readback = None
        self._apply(dyn or DynamicsSpec())

    # -- dynamics ---------------------------------------------------------
    def _apply(self, dyn: DynamicsSpec) -> None:
        self._readback = apply_dynamics(self._env, dyn, task_geoms=self.task_geoms)
        self._requested_dynamics = dyn

    @property
    def dyn_readback(self) -> Any:
        return self._readback

    def effective_dynamics(self) -> dict[str, Any]:
        if self._readback is None:
            raise RuntimeError("dynamics were never applied")
        return self._readback.as_dict()

    def contact_summary(self) -> list[dict[str, float]]:
        """Raw per-contact friction/penetration, for the dynamics read-back audit."""
        return contact_friction_readback(self._env.unwrapped.data)

    # -- env api ----------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, dyn: DynamicsSpec | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        if dyn is not None:
            self._apply(dyn)
        else:
            # idempotent re-application of the requested dynamics at reset (04 §1)
            self._apply(self._requested_dynamics)
        raw, info = self._env.reset(seed=self.seed)
        self._last_raw_obs = np.asarray(raw, dtype=np.float64).reshape(-1)
        self._monitor = ViolationMonitor(self.spec)
        self._last_gripper_cmd = 1.0
        self._last_progress_m = 0.0
        self._initial_distance = None
        obs = self._obs()
        return obs, self._info(info, terminated=False, truncated=False)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.action_dim:
            raise ValueError(f"expected a {self.action_dim}-D action; got {a.shape}")
        if a.shape[0] >= 4:
            self._last_gripper_cmd = float(a[3])
        raw, reward, terminated, truncated, info = self._env.step(a)
        self._last_raw_obs = np.asarray(raw, dtype=np.float64).reshape(-1)

        force, penetration = self._contact_force_and_penetration()
        eef = self._end_effector_position()
        obj = self._object_position()
        progress_m = self._progress_metres(obj)
        step_info = {
            "eef_pos": eef,
            "object_pos": obj,
            "grasped": self._is_grasped(eef, obj),
            "force": force,
            "penetration": penetration,
            "progress_m": progress_m,
            "object_height": float(obj[2]),
            "released": bool(self._last_gripper_cmd < 0),
        }
        self._monitor.observe(step_info)
        full_info = self._info(info, terminated=bool(terminated), truncated=bool(truncated))
        full_info["violations"] = self._monitor.per_type
        full_info["any_violation"] = self._monitor.any_violation
        return self._obs(), float(reward), bool(terminated), bool(truncated), full_info

    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError(
            "MetaWorld states use save_state()/restore_state() (see envs.state_io)"
        )

    def set_state(self, state: Any) -> None:  # pragma: no cover - documented indirection
        raise NotImplementedError(
            "MetaWorld states use restore_state(env, sim_state) (see envs.state_io)"
        )

    def save_state(self) -> Any:
        return save_state(self._env)

    def restore_state(self, state: Any) -> None:
        restore_state(self._env, state)

    def close(self) -> None:
        self._env.close()

    def render_frame(self) -> np.ndarray:
        if not self._render:
            raise RuntimeError("render was disabled at construction; pass render=True")
        frame = self._env.render()
        if frame is None:
            raise RuntimeError(
                "the underlying env returned no frame; check MUJOCO_GL (wgl on Windows, egl on Linux)"
            )
        return np.asarray(frame, dtype=np.uint8)

    # -- internals --------------------------------------------------------
    @property
    def _base(self) -> Any:
        return self._env.unwrapped

    def _geom_ids(self) -> set[int]:
        import mujoco

        model = self._base.model
        names = list((self.task_geoms or {}).get("geoms") or [])
        if not names:
            from riskproj.envs.dynamics import task_critical_geoms

            names = list(task_critical_geoms(self._env)["geoms"])
        ids = set()
        for name in names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(name))
            if gid < 0:
                raise KeyError(f"geom {name!r} from task_geoms.yaml is not in this model")
            ids.add(int(gid))
        return ids

    def _contact_force_and_penetration(self) -> tuple[float, float]:
        """Sum of normal forces and max penetration over task-critical contacts."""
        import mujoco

        data = self._base.data
        critical = self._geom_ids()
        force = 0.0
        penetration = 0.0
        buf = np.zeros(6, dtype=np.float64)
        for i in range(int(data.ncon)):
            c = data.contact[i]
            if critical and not ({int(c.geom1), int(c.geom2)} & critical):
                continue
            mujoco.mj_contactForce(self._base.model, data, i, buf)
            force += float(abs(buf[0]))
            if c.dist < 0:
                penetration = max(penetration, float(-c.dist))
        return force, penetration

    def _end_effector_position(self) -> np.ndarray:
        for chain in EEF_ATTR_CHAINS:
            value = _resolve_chain(self._base, chain)
            if value is not None:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                if arr.shape[0] == 3:
                    return arr
        raise AttributeError(
            "could not read the end-effector position; tried attribute chains "
            f"{EEF_ATTR_CHAINS} on {type(self._base).__name__}. Fix EEF_ATTR_CHAINS after "
            "inspecting the task internals (S1)."
        )

    def _object_position(self) -> np.ndarray:
        for chain in OBJECT_ATTR_CHAINS:
            value = _resolve_chain(self._base, chain)
            if value is not None:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                if arr.shape[0] == 3:
                    return arr
        raise AttributeError(
            "could not read the manipulated-object position; tried attribute chains "
            f"{OBJECT_ATTR_CHAINS} on {type(self._base).__name__}. Fix OBJECT_ATTR_CHAINS (S1)."
        )

    def _progress_metres(self, obj: np.ndarray) -> float:
        goal = None
        for chain in (("_get_pos_goal",), ("goal", "position")):
            value = _resolve_chain(self._base, chain)
            if value is not None:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                if arr.shape[0] == 3:
                    goal = arr
                    break
        if goal is None:
            raise AttributeError(
                "could not read the task goal position, so progress (needed by V5/V6) is "
                "undefined; add the task's goal accessor to _progress_metres (S1)."
            )
        distance = float(np.linalg.norm(obj - goal))
        if self._initial_distance is None:
            self._initial_distance = max(distance, 1e-6)
        self._last_progress_m = float(max(0.0, self._initial_distance - distance))
        return self._last_progress_m

    def _is_grasped(self, eef: np.ndarray, obj: np.ndarray) -> bool:
        """Heuristic grasp test: gripper commanded closed and object within reach.

        Documented rather than assumed ([R]): validate against the task's own
        success signal during S1 and replace if it disagrees.
        """
        return bool(
            self._last_gripper_cmd > 0 and float(np.linalg.norm(obj - eef)) < GRASP_DISTANCE
        )

    def _obs(self) -> dict[str, Any]:
        if self._last_raw_obs is None:
            raise RuntimeError("reset() must be called before reading observations")
        eef = self._end_effector_position()
        obj = self._object_position()
        images: dict[str, np.ndarray] = {}
        if self._render:
            images = {"top": self.render_frame()}
        proprio = self._last_raw_obs.astype(np.float32)
        state = np.concatenate([proprio, eef.astype(np.float32), obj.astype(np.float32)])
        return {
            "state": state,
            "eef_vel": np.zeros(3, dtype=np.float32),  # filled by the caller if it tracks velocity
            "images": images,
            "instruction": str(getattr(self._base, "task_name", self.task_name)).replace("-", " "),
            "progress": float(self._last_progress_m),
            "progress_m": float(self._last_progress_m),
            "grasped": self._is_grasped(eef, obj),
            "eef_pose": eef.astype(np.float32),
            "object_pose": obj.astype(np.float32),
            "raw_obs": self._last_raw_obs.astype(np.float32),
        }

    def _info(self, env_info: Any, *, terminated: bool, truncated: bool) -> dict[str, Any]:
        info = dict(env_info) if isinstance(env_info, dict) else {}
        progress = float(self._last_progress_m)
        if self._initial_distance:
            progress = float(min(1.0, self._last_progress_m / self._initial_distance))
        eef = self._end_effector_position()
        obj = self._object_position()
        return {
            "success": bool(info.get("success", False)),
            "progress": progress,
            "progress_m": float(self._last_progress_m),
            "grasped": self._is_grasped(eef, obj),
            "eef_pos": eef.copy(),
            "object_pos": obj.copy(),
            "step": int(getattr(self._base, "timestep", 0))
            if hasattr(self._base, "timestep")
            else 0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "raw_obs": None if self._last_raw_obs is None else self._last_raw_obs.copy(),
            "dyn_readback": self.effective_readback(),
            "env_info": info,
        }

    def effective_readback(self) -> dict[str, Any]:
        """Achieved dynamics scales (empty before the first application)."""
        return dict(self._readback.effective) if self._readback is not None else {}


def probe_task(task: str) -> dict[str, Any]:
    """Small diagnostic used by S1/``scripts/inspect_contacts.py``: what resolves?"""
    gym, _ = _import_sim()
    out: dict[str, Any] = {"task": task}
    try:
        out["gym_id"] = resolve_gym_id(task, gym)
    except KeyError as exc:
        out["gym_id_error"] = str(exc)
    try:
        out["expert_class"] = resolve_expert_class(task).__name__
    except KeyError as exc:
        out["expert_error"] = str(exc)
    return out


__all__ = [
    "EEF_ATTR_CHAINS",
    "GRASP_DISTANCE",
    "GYM_ID_TEMPLATES",
    "OBJECT_ATTR_CHAINS",
    "MetaWorldDynEnv",
    "MetaWorldExpert",
    "MetaWorldTask",
    "probe_task",
    "resolve_expert_class",
    "resolve_gym_id",
]
