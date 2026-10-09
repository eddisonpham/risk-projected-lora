"""Toy contact task: the CPU walking skeleton (``03`` §3 S6).

The purpose of this module is *pipeline*, not physics. It is a deterministic,
pure-numpy contact task whose behaviour depends on the same three axes as the
real simulator (``mu`` friction, ``mass`` payload, ``stiff`` contact stiffness),
so that data generation, critic training, projection, rollouts, logging and
reporting can all be exercised on CPU in CI in well under two minutes, with no
MuJoCo, no torch and no downloads.

**Modelling assumptions (stated, not hidden).** This is a quasi-static,
soft-contact model, not a rigid-body engine:

* **Task.** Pick a puck off the table at ``object_start``, carry it to the slot
  at ``target`` and insert it into the hole (the object's centre ends
  ``table_z - target_z`` below the table surface). Success is a distance test on
  the *object*, plus a speed test.
* **Grip (``mu`` axis, and ``mass`` through the required load).** The grip is
  static: it holds while the load it must carry,
  ``m * (g + |a|)`` with ``m = object_mass * mass_scale``, stays below
  ``grip_quality * grip_force * mu_scale``. The moment the load exceeds that,
  the grip **breaks and the object is released** where it is. This is what makes
  the friction axis consequential: a policy that cannot hold the puck can never
  finish the task, no matter how long it keeps trying.
* **Contact (``stiff`` axis).** An unheld object can only be moved by being
  pushed, and the push is a spring: force ``= k * overlap`` with
  ``k = contact_stiffness * stiff_scale``; the object moves only once the force
  exceeds the table's sliding friction ``table_friction * mu_scale``, and then
  travels with the end effector at the equilibrium overlap
  ``force_required / k``. Stiff contacts therefore produce large forces at small
  penetration (over-force, V3) and soft contacts large penetration at small
  force (penetration, V4) — the trade-off the real model has.
* **Environment limit.** An object held or pushed downwards is stopped by the
  support beneath it: the table at ``table_z``, or the slot floor at
  ``hole_floor_z`` once the object is aligned with the slot. The support is
  compliant (a contact, not a constraint), so continuing to press builds
  penetration and therefore force — this is the regime in which V3/V4/V5 become
  reachable, and it is the toy analogue of FACT's precision-failure/jam
  signature. A misaligned insertion presses the puck against the table next to
  the slot instead of letting it in, exactly like a mis-aimed peg.
* **Grip closing** is proximity-based (there is no separate gripper command in
  the toy action space) — documented so nobody mistakes it for a real gripper.

Everything is seeded from ``reset(seed=...)`` and ``step`` uses no randomness, so
a saved state replays bitwise-exactly (S2, and the basis of the E1 paired test).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from riskproj.envs.dynamics import apply_dynamics
from riskproj.envs.violations import ViolationMonitor, failure_class
from riskproj.types import DynamicsSpec, ViolationSpec
from riskproj.utils.registry import register

#: Low-64-bit mask: PCG64's state is a 128-bit integer and does not fit in a uint64.
_U64_MASK = (1 << 64) - 1

#: Instruction string handed to the policy (and logged).
TOY_INSTRUCTION = "push the puck into the target slot"

#: Toy-scale violation thresholds. **These are in toy units and are NOT the
#: Meta-World thresholds in ``configs/violations.yaml``**; they are hand-set so
#: that perturbed chunks can reach every event. Any claim about the toy task must
#: say so explicitly, and a real task must re-derive thresholds from nominal
#: expert rollouts (``04`` §4) via ``scripts/derive_violation_thresholds.py``.
TOY_VIOLATION_SPEC = ViolationSpec(
    slip_drift=0.008,
    drop_dist=0.025,
    force_max=8.0,
    #: 8 mm, not 6: descending onto the puck costs the expert one step of
    #: commanded motion (~4 mm) of blocked travel, and the reference controller
    #: must stay clean at nominal so that violations carry information.
    penetration=0.008,
    jam_force=4.0,
    jam_progress_eps=0.001,
    jam_steps=5,
    window=10,
    horizon=10,
    provenance={
        "scale": "toy units (newtons, metres); not the Meta-World thresholds",
        "derivation": (
            "hand-set to be reachable by perturbed chunks; must be replaced by nominal-expert "
            "percentiles before any quantitative claim"
        ),
    },
)

#: Steps in a nominal expert episode at the default geometry.
DEFAULT_MAX_STEPS = 100


@dataclass(frozen=True)
class ToyPhysics:
    """Nominal toy parameters. Dynamics *scales* multiply these (``04`` §2)."""

    dt: float = 0.05
    max_episode_steps: int = DEFAULT_MAX_STEPS

    # -- actuation
    eef_max_speed: float = 0.5  # m/s
    eef_max_accel: float = 6.0  # m/s^2

    # -- contact (stiffness axis)
    contact_stiffness: float = 500.0  # N/m at stiff_scale = 1
    max_penetration: float = 0.02  # m; beyond this the contact is impassable
    contact_radius: float = 0.012  # m; region within which the eef pushes the puck
    grip_radius: float = 0.014  # m; proximity at which the grip closes
    #: Contact compliance is ``1 / contact_stiffness``: the support stops the
    #: object at its surface and the *commanded* overshoot is the penetration,
    #: which converts to force through ``k``.

    # -- friction (mu axis)
    grip_force: float = 6.0  # N of normal grip force at mu_scale = 1
    grip_quality: float = 0.75  # alignment penalty on the transmittable friction
    table_friction: float = 0.6  # N of sliding resistance at mu_scale = 1
    gravity: float = 9.81  # m/s^2

    # -- mass axis
    object_mass: float = 0.2  # kg at mass_scale = 1

    # -- geometry
    home_eef: tuple[float, float, float] = (0.0, 0.0, 0.10)
    object_start: tuple[float, float, float] = (0.10, 0.0, 0.02)
    target: tuple[float, float, float] = (-0.05, 0.0, 0.005)
    table_z: float = 0.02  # object rest height on the table
    hole_floor_z: float = 0.0  # floor of the slot
    align_tol: float = 0.008  # m; xy tolerance for entering the slot
    success_radius: float = 0.015
    success_speed: float = 0.05
    reset_jitter: float = 0.01

    @property
    def max_delta(self) -> float:
        return self.eef_max_speed * self.dt

    @property
    def insertion_depth(self) -> float:
        return float(self.table_z - self.target[2])


def _vec3(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"{name} must have 3 components; got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {arr}")
    return arr


@dataclass
class _ToyState:
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    qv: np.ndarray
    grasped: bool = False
    step: int = 0
    progress_m: float = 0.0
    grip_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))


class ToyEnv:
    """Deterministic quasi-static contact task (see the module docstring)."""

    task_name = "toy-reach-insert"
    action_dim = 3
    state_dim = 14

    def __init__(
        self,
        physics: ToyPhysics | None = None,
        spec: ViolationSpec | None = None,
        *,
        seed: int = 0,
        image_size: tuple[int, int] = (8, 8),
        max_episode_steps: int | None = None,
    ) -> None:
        self.physics = physics or ToyPhysics()
        self.max_episode_steps = (
            int(max_episode_steps)
            if max_episode_steps is not None
            else int(self.physics.max_episode_steps)
        )
        self.spec = spec or TOY_VIOLATION_SPEC
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        self._scales = DynamicsSpec()
        self._mu = 1.0
        self._mass = float(self.physics.object_mass)
        self._stiffness = float(self.physics.contact_stiffness)
        self._grip_available = float(self.physics.grip_quality * self.physics.grip_force)
        self._table_friction = float(self.physics.table_friction)

        self._state = _ToyState(
            p=np.array(self.physics.home_eef, dtype=np.float64),
            v=np.zeros(3),
            q=np.array(self.physics.object_start, dtype=np.float64),
            qv=np.zeros(3),
        )
        self._monitor = ViolationMonitor(self.spec)
        self._initial_distance = 1.0
        self._grasp_history = 0
        self._last_force = 0.0
        self._last_penetration = 0.0
        self._last_blocked = False
        self._last_load_ratio = 0.0
        self._last_drop = False
        self._done = False
        self._success = False
        #: cell id is metadata only, never a model input (``04`` §5)
        self._cell_id = "M-M-M"

    # -- attributes other modules read -----------------------------------
    @property
    def instruction(self) -> str:
        return TOY_INSTRUCTION

    @property
    def target(self) -> np.ndarray:
        return np.array(self.physics.target, dtype=np.float64)

    @property
    def is_aligned(self) -> bool:
        """Is the object lined up with the slot (so it may descend into it)?"""
        q = self._state.q
        return bool(np.linalg.norm(q[:2] - self.target[:2]) <= self.physics.align_tol)

    def support_height(self, obj_xy: np.ndarray) -> float:
        """Height the object cannot pass below: slot floor when aligned, else the table."""
        aligned = bool(np.linalg.norm(obj_xy - self.target[:2]) <= self.physics.align_tol)
        return float(self.physics.hole_floor_z if aligned else self.physics.table_z)

    # -- dynamics ---------------------------------------------------------
    def apply_dynamics(self, spec: DynamicsSpec) -> dict[str, Any]:
        """Apply scales; returns read-backs in the schema :mod:`dynamics` expects."""
        if spec.mu_scale <= 0 or spec.mass_scale <= 0 or spec.stiff_scale <= 0:
            raise ValueError(f"dynamics scales must be > 0; got {spec.as_dict()}")
        previous = self._scales
        self._scales = spec
        self._mu = float(spec.mu_scale)
        self._mass = float(self.physics.object_mass * spec.mass_scale)
        self._stiffness = float(self.physics.contact_stiffness * spec.stiff_scale)
        self._grip_available = float(self.physics.grip_quality * self.physics.grip_force * self._mu)
        self._table_friction = float(self.physics.table_friction * self._mu)
        changed = any(
            abs(float(getattr(spec, k)) - float(getattr(previous, k))) > 1e-12
            for k in ("mu_scale", "mass_scale", "stiff_scale")
        )
        timeconst = 1.0 / math.sqrt(spec.stiff_scale)  # mirrors timeconst ~ 1/sqrt(k)
        return {
            "mu": {"grip_contact": self._mu, "table": self._mu},
            "mass": {"object": float(spec.mass_scale)},
            "solref": {"contact": (timeconst, 1.0)},
            "effective": {
                "mu_scale": self._mu,
                "mass_scale": float(spec.mass_scale),
                "stiff_scale": float(spec.stiff_scale),
            },
            "changed": bool(changed),
            "notes": [
                "toy env: scales are stored directly; k = contact_stiffness * stiff_scale",
                f"effective stiffness {self._stiffness:.3f} N/m, transmittable grip friction "
                f"{self._grip_available:.3f} N, carried load at rest {self._mass * self.physics.gravity:.3f} N",
            ],
        }

    def effective_dynamics(self) -> dict[str, Any]:
        """Effective physics in the read-back schema of ``envs/dynamics.py``.

        Every axis carries a ``scale`` entry (a multiplier of the nominal value) so
        a caller can compare the *achieved* scale with the requested spec without
        knowing this environment's internals -- that comparison is what makes a
        silent dynamics no-op detectable (``04`` §2). The physical quantities are
        kept alongside for inspection.
        """
        timeconst = 1.0 / math.sqrt(self._scales.stiff_scale)
        return {
            "mu": {"scale": self._mu, "grip_contact": self._mu, "table": self._mu},
            "mass": {
                "scale": float(self._scales.mass_scale),
                "object": float(self._scales.mass_scale),
            },
            "stiff": {"scale": float(self._scales.stiff_scale), "timeconst": timeconst},
            "solref": {"contact": (timeconst, 1.0)},
            "effective": self._scales.as_dict(),
            "changed": not self._scales.is_nominal(),
            "verified": True,
            "notes": [],
        }

    @property
    def dyn_readback(self) -> dict[str, Any]:
        """Read-back as ``data.gen_demos.readback_dict`` expects it (``04`` §2)."""
        return self.effective_dynamics()

    def close(self) -> None:
        """gymnasium-style no-op: the toy env owns no external resources.

        It exists because the rollout driver closes environments uniformly, and a
        missing ``close`` would otherwise surface as an ``AttributeError`` at the
        end of an otherwise successful episode.
        """
        return None

    @property
    def requested_dynamics(self) -> DynamicsSpec:
        return self._scales

    # -- reset / step ------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, dyn: DynamicsSpec | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode; ``dyn`` re-applies dynamics if given."""
        if dyn is not None:
            apply_dynamics(self, dyn)
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        jitter = self.physics.reset_jitter
        p = np.array(self.physics.home_eef, dtype=np.float64)
        q = np.array(self.physics.object_start, dtype=np.float64)
        q[:2] += self._rng.uniform(-jitter, jitter, size=2)
        q[2] = self.physics.table_z
        p += self._rng.uniform(-jitter / 2, jitter / 2, size=3)
        self._state = _ToyState(p=p, v=np.zeros(3), q=q, qv=np.zeros(3))
        self._monitor = ViolationMonitor(self.spec)
        self._initial_distance = float(max(np.linalg.norm(q - self.target), 1e-6))
        self._grasp_history = 0
        self._last_force = 0.0
        self._last_penetration = 0.0
        self._last_blocked = False
        self._last_load_ratio = 0.0
        self._last_drop = False
        self._done = False
        self._success = False
        return self._obs(), self._info(terminated=False, truncated=False)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called after the episode finished; call reset() first")
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.action_dim:
            raise ValueError(f"action must have {self.action_dim} components; got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"action contains non-finite values: {a}")
        ph = self.physics
        a = np.clip(a, -ph.max_delta, ph.max_delta)

        st = self._state
        st.step += 1

        # ---- end-effector actuation: velocity target with acceleration limit
        v_des = a / ph.dt
        dv = np.clip(v_des - st.v, -ph.eef_max_accel * ph.dt, ph.eef_max_accel * ph.dt)
        v_new = st.v + dv
        p_prop = st.p + v_new * ph.dt

        force = 0.0
        penetration = 0.0
        blocked = False
        load_ratio = 0.0
        drop = False
        grasped_now = st.grasped or bool(np.linalg.norm(p_prop - st.q) < ph.grip_radius)

        if grasped_now:
            # ---- grip: hold-or-drop against a static friction budget ---------
            required = self._mass * (ph.gravity + float(np.linalg.norm(dv)) / ph.dt)
            load_ratio = required / self._grip_available
            if load_ratio <= 1.0:
                motion = p_prop - st.p
                q_new = st.q + motion
                p_new = p_prop
                support = self.support_height(q_new[:2])
                if q_new[2] < support:
                    penetration = float(support - q_new[2])
                    q_new[2] = support
                    force = self._stiffness * min(penetration, ph.max_penetration)
                qv_new = (q_new - st.q) / ph.dt
                st.grip_offset = q_new - p_new
            else:
                # the grip breaks: the object is released where it is
                drop = True
                grasped_now = False
                q_new = st.q.copy()
                q_new[2] = max(float(st.q[2]), float(ph.table_z))  # it falls back to the table
                qv_new = np.zeros(3)
                p_new = p_prop
                st.grip_offset = np.zeros(3)
        else:
            # ---- free: only a push through contact can move the object --------
            p_new = p_prop
            q_new = st.q.copy()
            qv_new = st.qv.copy()
            u = p_new - q_new
            dist = float(np.linalg.norm(u))
            if dist < ph.contact_radius:
                u = u / max(dist, 1e-12)  # push direction: from the eef toward the puck
                overlap0 = ph.contact_radius - dist
                advance = float(np.dot(p_prop - st.p, u))
                a_needed = float(np.linalg.norm(v_new - st.qv)) / ph.dt
                required = self._table_friction + self._mass * a_needed
                overlap_required = required / self._stiffness
                final_overlap = min(overlap0 + advance, overlap_required)
                blocked = overlap_required > min(ph.max_penetration, overlap0 + advance)
                force = self._stiffness * final_overlap
                displacement = max(0.0, overlap0 + advance - final_overlap)
                q_new = st.q + u * displacement
                qv_new = (q_new - st.q) / ph.dt
                support = self.support_height(q_new[:2])
                if q_new[2] < support:
                    penetration = float(support - q_new[2])
                    q_new[2] = support
                    force = max(force, self._stiffness * min(penetration, ph.max_penetration))
                    qv_new = (q_new - st.q) / ph.dt
                p_new = p_prop + u * (final_overlap - (overlap0 + advance))

        v_eff = p_new - st.p
        st.p = p_new
        st.v = v_eff / ph.dt
        st.q = q_new
        st.qv = qv_new
        st.grasped = grasped_now
        self._grasp_history = self._grasp_history + 1 if grasped_now else 0
        self._last_force = float(force)
        self._last_penetration = float(penetration)
        self._last_blocked = bool(blocked)
        self._last_load_ratio = float(load_ratio)
        self._last_drop = bool(drop)

        distance = float(np.linalg.norm(st.q - self.target))
        st.progress_m = float(max(0.0, self._initial_distance - distance))
        speed = float(np.linalg.norm(st.qv))
        self._success = bool(distance < ph.success_radius and speed < ph.success_speed)
        terminated = self._success
        truncated = st.step >= self.max_episode_steps
        self._done = terminated or truncated

        info = self._info(terminated=terminated, truncated=truncated)
        obs = self._obs()
        # Built once and kept: the violated-chunk labeller must consume the *same*
        # mapping this monitor does, so the violation conventions (signs, windows, grip
        # bookkeeping) live in exactly one place (`04` §4/§6).
        self._last_step_info: dict[str, Any] = {
            "eef_pos": st.p,
            "object_pos": st.q,
            "grasped": st.grasped,
            "force": force,
            "penetration": penetration,
            "progress_m": st.progress_m,
            "grip_offset": st.grip_offset.copy(),
            "grasp_established": st.grasped,
            "released": False,  # the toy grip is never opened deliberately
            "object_height": float(st.q[2]),
            "progress": info["progress"],
        }
        self._monitor.observe(self._last_step_info)
        info["violations"] = self._monitor.per_type
        info["any_violation"] = self._monitor.any_violation
        return obs, -distance, terminated, truncated, info

    # -- observations ------------------------------------------------------
    def _progress(self) -> float:
        return (
            float(self._state.progress_m / self._initial_distance)
            if self._initial_distance > 0
            else 0.0
        )

    def _proprio(self) -> np.ndarray:
        st = self._state
        return np.concatenate(
            [st.p, st.v, st.q, st.qv, np.array([1.0 if st.grasped else 0.0, self._progress()])]
        ).astype(np.float32)

    def _render(self) -> np.ndarray:
        """Tiny deterministic top view (uint8, HxWx3): slot, puck, eef."""
        h, w = self.image_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        span = 0.3
        origin = np.array([-0.15, -0.15])
        st = self._state

        def to_px(xy: np.ndarray) -> tuple[int, int]:
            rel = (np.asarray(xy, dtype=np.float64)[:2] - origin) / span
            col = int(np.clip(rel[0] * (w - 1), 0, w - 1))
            row = int(np.clip((1.0 - rel[1]) * (h - 1), 0, h - 1))
            return row, col

        r, c = to_px(self.target)
        img[r, c] = (0, 0, 255)
        r, c = to_px(st.q)
        img[r, c] = (0, 255, 0)
        r, c = to_px(st.p)
        img[r, c] = (255, 0, 0)
        return img

    def _obs(self) -> dict[str, Any]:
        st = self._state
        img = self._render()
        return {
            "state": self._proprio(),
            "eef_vel": st.v.astype(np.float32),
            "images": {"top": img, "wrist": img.copy()},
            "instruction": TOY_INSTRUCTION,
            # privileged-sim input: off by default for the critic (04 §5)
            "force": np.array([self._last_force], dtype=np.float32),
            "progress": self._progress(),
            "progress_m": float(st.progress_m),
            "grasped": bool(st.grasped),
            "eef_pose": st.p.astype(np.float32),
            "object_pose": st.q.astype(np.float32),
            "contacts": {
                "pairs": [("eef", "puck")] if (self._last_force > 0 or st.grasped) else [],
                "forces": [float(self._last_force)] if self._last_force > 0 else [],
                "dists": [-float(self._last_penetration)] if self._last_penetration > 0 else [],
            },
        }

    def get_obs(self) -> dict[str, Any]:
        """Public observation getter, for the ``state_io`` restore contract.

        ``restore_state`` refreshes the observation through a *named* getter
        (``get_obs``/``_get_obs``/``observe``/``get_observation``) so that
        counterfactual labels are built from the restored state rather than a stale
        observation (``04`` §6). ``_obs`` is private, so this is the alias it looks for.
        """
        return self._obs()

    def last_step_info(self) -> dict[str, Any]:
        """The step-info mapping this env fed its own ``ViolationMonitor`` on the last
        ``step`` (``04`` §6). The violated-chunk labeller consumes this instead of
        re-deriving contact/force quantities from the observation, where the sign and
        window conventions could silently diverge from the ones the monitor was
        validated with."""
        stored = getattr(self, "_last_step_info", None)
        if stored is None:
            raise RuntimeError("last_step_info() called before any step()")
        return dict(stored)

    def _info(self, *, terminated: bool, truncated: bool) -> dict[str, Any]:
        st = self._state
        return {
            "success": self._success,
            "progress": self._progress(),
            "progress_m": float(st.progress_m),
            "grasped": bool(st.grasped),
            "eef_pos": st.p.copy(),
            "object_pos": st.q.copy(),
            "eef_vel": st.v.copy(),
            "force": float(self._last_force),
            "penetration": float(self._last_penetration),
            "blocked": bool(self._last_blocked),
            #: required grip load / transmittable friction (>= 1 means the grip broke)
            "grip_load_ratio": float(self._last_load_ratio),
            "dropped": bool(self._last_drop),
            "aligned": self.is_aligned,
            "step": int(st.step),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "dyn_readback": self.effective_dynamics()["effective"],
        }

    # -- state -------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        """Flat state dict, exact enough for bitwise replay (``04`` §6)."""
        st = self._state
        bg = self._rng.bit_generator.state
        return {
            "p": st.p.copy(),
            "v": st.v.copy(),
            "q": st.q.copy(),
            "qv": st.qv.copy(),
            "grip_offset": st.grip_offset.copy(),
            "grasped": np.asarray(1.0 if st.grasped else 0.0),
            "step": np.asarray(float(st.step)),
            "progress_m": np.asarray(float(st.progress_m)),
            "initial_distance": np.asarray(float(self._initial_distance)),
            "grasp_history": np.asarray(float(self._grasp_history)),
            "done": np.asarray(1.0 if self._done else 0.0),
            "success": np.asarray(1.0 if self._success else 0.0),
            "mu_scale": np.asarray(float(self._scales.mu_scale)),
            "mass_scale": np.asarray(float(self._scales.mass_scale)),
            "stiff_scale": np.asarray(float(self._scales.stiff_scale)),
            # PCG64's state is a 128-bit integer: it cannot be stored in one
            # uint64 (numpy raises OverflowError), and float() would silently
            # round it. Store it as two 64-bit halves and rebuild exactly.
            "rng_state_lo": np.asarray(int(bg["state"]["state"]) & _U64_MASK, dtype=np.uint64),
            "rng_state_hi": np.asarray(
                (int(bg["state"]["state"]) >> 64) & _U64_MASK, dtype=np.uint64
            ),
            "rng_inc": np.asarray(int(bg["state"]["inc"]) & _U64_MASK, dtype=np.uint64),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`get_state` (fail loud on gaps)."""
        required = {"p", "v", "q", "qv", "grasped", "step", "initial_distance"}
        missing = sorted(required - set(state))
        if missing:
            raise KeyError(f"state is missing required keys {missing}")

        def scalar(key: str, default: float = 0.0) -> float:
            return float(np.asarray(state.get(key, default)).reshape(-1)[0])

        def int_scalar(key: str, default: int = 0) -> int:
            """Exact integer read (never routed through float, which would round)."""
            return int(np.asarray(state.get(key, default), dtype=object).reshape(-1)[0])

        st = self._state
        st.p = _vec3(state["p"], name="p").copy()
        st.v = _vec3(state["v"], name="v").copy()
        st.q = _vec3(state["q"], name="q").copy()
        st.qv = _vec3(state["qv"], name="qv").copy()
        st.grip_offset = _vec3(state.get("grip_offset", np.zeros(3)), name="grip_offset").copy()
        st.grasped = bool(scalar("grasped"))
        st.step = int(scalar("step"))
        st.progress_m = scalar("progress_m")
        self._initial_distance = float(np.asarray(state["initial_distance"]).reshape(-1)[0])
        self._grasp_history = int(scalar("grasp_history"))
        self._done = bool(scalar("done"))
        self._success = bool(scalar("success"))
        self._last_load_ratio = 0.0
        self._last_drop = False
        if {"mu_scale", "mass_scale", "stiff_scale"} <= set(state):
            self.apply_dynamics(
                DynamicsSpec(
                    mu_scale=scalar("mu_scale", 1.0),
                    mass_scale=scalar("mass_scale", 1.0),
                    stiff_scale=scalar("stiff_scale", 1.0),
                )
            )
        raw_state: int | None = None
        if "rng_state_lo" in state and "rng_state_hi" in state:
            # Rebuild the 128-bit PCG64 state from its halves; int(), never
            # float(), so no precision is lost.
            raw_state = int_scalar("rng_state_lo") | (int_scalar("rng_state_hi") << 64)
        elif "rng_state" in state:
            raw_state = int_scalar("rng_state")
        if raw_state is not None:
            self._rng.bit_generator.state = {
                "bit_generator": "PCG64",
                "state": {"state": raw_state, "inc": int_scalar("rng_inc", 0)},
                "has_uint32": 0,
                "uinteger": 0,
            }

    # -- episode summary ---------------------------------------------------
    def episode_summary(self) -> dict[str, Any]:
        """Per-episode record in the shape the rollout logger expects (``09`` §3)."""
        violations = self._monitor.finalize()
        st = self._state
        return {
            "success": bool(self._success),
            "steps": int(st.step),
            "violations": violations,
            "any_violation": any(float(v["count"]) > 0.0 for v in violations.values()),
            "failure_class": failure_class(
                violations,
                success=bool(self._success),
                progress=self._progress(),
                max_steps_reached=bool(st.step >= self.max_episode_steps),
            ),
            "progress": self._progress(),
            "progress_m": float(st.progress_m),
            "cell_id": self._cell_id,
            "dyn_readback": self.effective_dynamics()["effective"],
        }


class ToyExpert:
    """Scripted controller that solves the toy task at nominal dynamics.

    Not held: hover above the puck, then descend onto it (the grip closes on
    proximity). Held: command the *object* (the grip is rigid while it holds), so
    the puck is raised clear of the table, transferred above the slot and lowered
    into it. Commanding the object rather than the end effector keeps the puck
    from being driven into the table by whatever offset the grasp happened to
    leave, which is what a position-controlled policy would get wrong.
    """

    name = "toy_expert"

    def __init__(
        self,
        physics: ToyPhysics | None = None,
        *,
        hover_height: float = 0.04,
        #: the grip closes on proximity (grip_radius = 14 mm), so the descent
        #: waypoint sits just inside it: further travel would only press the puck
        #: into the table it is already resting on.
        descend_height: float = 0.010,
        lift_height: float = 0.02,
        approach_speed: float = 0.30,
        descend_speed: float = 0.08,
        transfer_speed: float = 0.35,
        insert_speed: float = 0.15,
        waypoint_tol: float = 0.008,
    ) -> None:
        self.physics = physics or ToyPhysics()
        self.hover_height = hover_height
        self.descend_height = descend_height
        self.lift_height = lift_height
        self.approach_speed = approach_speed
        self.descend_speed = descend_speed
        self.transfer_speed = transfer_speed
        self.insert_speed = insert_speed
        self.waypoint_tol = waypoint_tol
        self.phase = "approach"

    def reset(self) -> None:
        self.phase = "approach"

    def _clip(self, delta: np.ndarray, speed: float) -> np.ndarray:
        cap = speed * self.physics.dt
        norm = float(np.linalg.norm(delta))
        if norm > cap and norm > 0:
            delta = delta * (cap / norm)
        return delta.astype(np.float32)

    def __call__(self, obs: dict[str, Any]) -> np.ndarray:
        """``Task.expert()`` returns a *callable* ``expert(obs) -> [A]`` (``04`` §7).

        The controller is an object because it carries phase state, so the callable
        form the demos generator uses is an alias for :meth:`act`.
        """
        return self.act(obs)

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        ph = self.physics
        p = np.asarray(obs["eef_pose"], dtype=np.float64)
        q = np.asarray(obs["object_pose"], dtype=np.float64)
        target = np.asarray(ph.target, dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0])

        if not bool(obs["grasped"]):
            hover = q + up * self.hover_height
            descend = q + up * self.descend_height
            # The approach test must be *geometric*: the earlier version compared
            # the distance to the hover waypoint while commanding the descend
            # waypoint, which sits 30 mm lower. The descent therefore always left
            # the 8 mm hover tolerance band, flipped the phase back to `approach`
            # and produced a limit cycle at z ~ 0.05 that never grasped.
            # Approach until we are above the puck *and* still higher than the
            # hover band; after that, commit to the descent.
            horizontally_above = float(np.linalg.norm((p - hover)[:2])) <= self.waypoint_tol
            if not horizontally_above and float(p[2]) > float(hover[2]) - self.waypoint_tol:
                self.phase = "approach"
                return self._clip(hover - p, self.approach_speed)
            self.phase = "descend"
            return self._clip(descend - p, self.descend_speed)

        above_slot = target + up * self.lift_height
        if float(np.linalg.norm(q - target)) < ph.success_radius:
            self.phase = "done"
            return self._clip(target - q, self.insert_speed)
        if float(q[2]) < float(ph.table_z) + 0.005:
            self.phase = "lift"
            return self._clip(up * self.lift_height, self.transfer_speed)
        if float(np.linalg.norm(q - above_slot)) > self.waypoint_tol:
            self.phase = "transfer"
            return self._clip(above_slot - q, self.transfer_speed)
        self.phase = "insert"
        return self._clip(target - q, self.insert_speed)

    def waypoint(self, obs: dict[str, Any]) -> tuple[np.ndarray, float]:
        """Nominal next target (diagnostics/tests): returns the object-space goal."""
        target = np.asarray(self.physics.target, dtype=np.float64)
        q = np.asarray(obs["object_pose"], dtype=np.float64)
        if not bool(obs["grasped"]):
            return q + np.array([0.0, 0.0, self.hover_height]), self.approach_speed
        if float(np.linalg.norm(q - target)) < self.physics.success_radius:
            return target, self.insert_speed
        if float(q[2]) < float(self.physics.table_z) + 0.005:
            return q + np.array([0.0, 0.0, self.lift_height]), self.transfer_speed
        return target + np.array([0.0, 0.0, self.lift_height]), self.transfer_speed


@register("task", "toy")
class ToyReachTask:
    """``Task`` implementation for the toy contact problem (``03`` §3 S6)."""

    name = "toy"

    def __init__(
        self,
        violations: ViolationSpec | None = None,
        *,
        physics: ToyPhysics | None = None,
        max_episode_steps: int | None = None,
        image_size: tuple[int, int] = (8, 8),
    ) -> None:
        self.physics = physics or ToyPhysics()
        # The toy has its own threshold units; a caller-supplied spec is honoured
        # but must be in toy units (see the module docstring).
        self.violations = violations or TOY_VIOLATION_SPEC
        self.max_episode_steps = max_episode_steps
        self.image_size = image_size

    def make_env(self, dyn: DynamicsSpec, seed: int, **kwargs: Any) -> ToyEnv:
        env = ToyEnv(
            self.physics,
            self.violations,
            seed=seed,
            image_size=kwargs.pop("image_size", self.image_size),
            max_episode_steps=kwargs.pop("max_episode_steps", self.max_episode_steps),
        )
        if kwargs:
            raise TypeError(f"ToyReachTask.make_env got unexpected kwargs {sorted(kwargs)}")
        apply_dynamics(env, dyn)
        env.reset(seed=seed)
        return env

    def expert(self) -> ToyExpert:
        return ToyExpert(self.physics)

    def label(self, cell_id: str = "M-M-M") -> str:
        return f"toy/{cell_id}"


def set_cell_id(env: ToyEnv, cell_id: str) -> ToyEnv:
    """Record the cell id as *metadata only* — never as a model input (``04`` §5)."""
    env._cell_id = str(cell_id)
    return env


__all__ = [
    "DEFAULT_MAX_STEPS",
    "TOY_INSTRUCTION",
    "TOY_VIOLATION_SPEC",
    "ToyEnv",
    "ToyExpert",
    "ToyPhysics",
    "ToyReachTask",
    "set_cell_id",
]
