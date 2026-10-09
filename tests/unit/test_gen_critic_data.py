"""D-crit generation: perturbations, replay, labels and the positivity guard.

Everything here runs on a tiny fake simulator, so the labelling logic is testable
before Meta-World exists. The fake env implements the same contracts the real task
adapter must (``obs.TASK_OBS_KEYS``, ``dyn_readback``, ``get_state``/``set_state``,
``info['success']``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from riskproj.data.gen_critic_data import (
    SimStateReplay,
    _normalize_labels,
    _pad_chunk,
    collect_states,
    generate_critic_data,
    perturb_chunk,
    unsafe_chunks,
)
from riskproj.data.gen_demos import assert_dynamics_applied, generate_demos, readback_dict
from riskproj.data.schema import read_demos
from riskproj.types import VIOLATION_TYPES, DynamicsSpec, ViolationSpec
from riskproj.utils.config import DynamicsRangesCfg

ACTION_DIM = 2
STATE_DIM = 2
STATE_EVERY = 4
MAX_STEPS = 12


class FakeEnv:
    """Deterministic 1-D toy simulator whose physics really depend on the dynamics."""

    def __init__(self, dyn: DynamicsSpec, seed: int) -> None:
        self.dyn = dyn
        self.instruction = "reach the target"
        self.success = False
        self.max_steps = MAX_STEPS
        self.dyn_readback = {
            "mu": {"scale": float(dyn.mu_scale)},
            "mass": {"scale": float(dyn.mass_scale)},
            "stiff": {"scale": float(dyn.stiff_scale)},
        }
        self._pos = 0.0
        self._vel = 0.0
        self._steps = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._seed = int(seed)

    # -- gymnasium-ish API -------------------------------------------------
    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._pos, self._vel, self._steps = 0.0, 0.0, 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.success = False
        return self._obs(), {"success": False}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        assert a.shape[0] == ACTION_DIM, a.shape
        self._last_action = a
        acc = (float(a[0]) - self.dyn.mu_scale * self._vel) / self.dyn.mass_scale
        self._vel += acc * 0.1
        self._pos += self._vel * 0.1
        self._steps += 1
        self.success = bool(self._pos >= 0.9)
        truncated = self._steps >= self.max_steps
        return self._obs(), 0.0, self.success, truncated, self._info()

    def close(self) -> None:
        pass

    # -- state io ----------------------------------------------------------
    def get_state(self) -> dict[str, float]:
        return {"pos": self._pos, "vel": self._vel, "steps": float(self._steps)}

    def set_state(self, state: Any) -> dict[str, Any]:
        # state_io round-trips values through numpy arrays, so a field may arrive as
        # a 0-d or 1-element array; read it out explicitly instead of assuming a scalar.
        def _scalar(key: str) -> float:
            return float(np.asarray(state[key]).reshape(-1)[0])

        self._pos = _scalar("pos")
        self._vel = _scalar("vel")
        self._steps = int(_scalar("steps"))
        return self._obs()

    # -- obs ---------------------------------------------------------------
    def _force(self) -> float:
        return float(np.abs(self._last_action).sum() * 10.0 * self.dyn.stiff_scale)

    def _obs(self) -> dict[str, Any]:
        return {
            "state": np.array([self._pos, self._vel], dtype=np.float32),
            "eef_vel": np.array([self._vel, 0.0, 0.0], dtype=np.float32),
            "images": {"top": np.zeros((3, 4, 4), dtype=np.float32)},
            "instruction": self.instruction,
            "progress": float(np.clip(self._pos, 0.0, 1.0)),
            "grasped": bool(self._pos > 0.2),
            "force": np.array([self._force()], dtype=np.float32),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "progress": float(np.clip(self._pos, 0.0, 1.0)),
            "force": np.array([self._force()], dtype=np.float32),
            "grasped": bool(self._pos > 0.2),
        }


class FakeTask:
    name = "toy_reach"
    violations = ViolationSpec(force_max=1.0, horizon=3)

    def __init__(self) -> None:
        self.envs: list[FakeEnv] = []

    def make_env(self, dyn: DynamicsSpec, seed: int, **kwargs: Any) -> FakeEnv:
        env = FakeEnv(dyn, seed)
        self.envs.append(env)
        return env

    def expert(self):
        def _expert(obs: dict[str, Any]) -> np.ndarray:
            return np.array([0.5, 0.0], dtype=np.float32)

        return _expert


class StubMonitor:
    """Flags V3 whenever the summed force exceeds the spec threshold."""

    def __init__(self, spec: ViolationSpec, *, never: bool = False) -> None:
        self.spec = spec
        self.never = never
        self.counts = dict.fromkeys(VIOLATION_TYPES, 0)
        self.first = dict.fromkeys(VIOLATION_TYPES, -1)

    def observe(self, info: dict[str, Any], *, action: Any = None) -> dict[str, Any]:
        """Mirrors ``ViolationMonitor.observe`` (the real monitor's API), which is what
        the labeller calls; ``action`` is accepted and unused, as in the real one."""
        if self.never:
            return dict(self.counts)
        force = float(np.asarray(info.get("force", [0.0]), dtype=np.float32).sum())
        if force > self.spec.force_max:
            self.counts["V3"] += 1
            if self.first["V3"] < 0:
                self.first["V3"] = int(info.get("step", -1))
        return {
            v: {"count": float(self.counts[v]), "first_step": float(self.first[v])}
            for v in VIOLATION_TYPES
        }

    def finalize(self) -> dict[str, Any]:
        return {
            v: {"count": float(self.counts[v]), "first_step": float(self.first[v])}
            for v in VIOLATION_TYPES
        }


def _factory(**kw):
    def _make(spec: ViolationSpec):
        return StubMonitor(spec, **kw)

    return _make


RANGES = DynamicsRangesCfg().to_ranges()
HORIZON = 3


# --------------------------------------------------------------------------
# perturbations and unsafe chunks
# --------------------------------------------------------------------------


def test_perturb_chunk_is_deterministic_and_scaled() -> None:
    chunk = np.ones((64, 4), dtype=np.float32)
    a = perturb_chunk(chunk, 0.1, np.random.default_rng(0))
    b = perturb_chunk(chunk, 0.1, np.random.default_rng(0))
    c = perturb_chunk(chunk, 0.1, np.random.default_rng(1))
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)
    rms = float(np.sqrt(np.mean((a - chunk) ** 2)))
    assert 0.5 * 0.1 < rms < 1.5 * 0.1  # sigma is a relative RMS, within sampling noise
    bigger = perturb_chunk(chunk, 0.4, np.random.default_rng(0))
    assert float(np.sqrt(np.mean((bigger - chunk) ** 2))) > rms


def test_perturb_chunk_low_frequency_also_matches_sigma() -> None:
    chunk = np.ones((32, 2), dtype=np.float32)
    white = perturb_chunk(chunk, 0.2, np.random.default_rng(3), low_frequency=False)
    smooth = perturb_chunk(chunk, 0.2, np.random.default_rng(3), low_frequency=True)
    for arr in (white, smooth):
        rms = float(np.sqrt(np.mean((arr - chunk) ** 2)))
        assert 0.5 * 0.2 < rms < 1.5 * 0.2
    # smoothed noise is more correlated along the horizon than white noise
    dw = np.diff(white[:, 0])
    ds = np.diff(smooth[:, 0])
    assert float(np.mean(ds**2)) < float(np.mean(dw**2))
    with pytest.raises(ValueError, match="sigma"):
        perturb_chunk(chunk, -1.0, np.random.default_rng(0))


def test_unsafe_chunks_overpush_and_early_release() -> None:
    chunk = np.tile(np.array([[0.4, 0.9]], dtype=np.float32), (10, 1))
    spec = {"gripper_index": 1, "gripper_open": 1.0, "overpush_range": (2.0, 2.0)}
    out = unsafe_chunks(chunk, spec, np.random.default_rng(0), n_variants=2)
    assert len(out) == 3  # two over-push + one early release
    np.testing.assert_allclose(out[0], chunk * 2.0)
    assert out[2].shape == chunk.shape
    assert out[2][-1, 1] == pytest.approx(1.0)  # gripper released
    np.testing.assert_allclose(out[2][:, 0], chunk[:, 0])  # arm command untouched
    assert unsafe_chunks(np.zeros((0, 2), np.float32), spec, np.random.default_rng(0)) == []


def test_pad_chunk_trims_and_repeats() -> None:
    chunk = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert _pad_chunk(chunk, 2).shape == (2, 2)
    padded = _pad_chunk(chunk, 5)
    assert padded.shape == (5, 2)
    np.testing.assert_allclose(padded[2], chunk[2])
    with pytest.raises(ValueError, match="empty chunk"):
        _pad_chunk(np.zeros((0, 2), np.float32), 3)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def test_normalize_labels_accepts_bool_and_count_shapes() -> None:
    a = _normalize_labels({"V3": True})
    assert a["V3"] is True and a["V1"] is False
    b = _normalize_labels(
        {"V1": {"count": 2.0, "first_step": 3.0}, "V5": {"count": 0.0, "first_step": -1.0}}
    )
    assert b["V1"] is True and b["V5"] is False
    c = _normalize_labels(None)
    assert set(c) == set(VIOLATION_TYPES) and not any(c.values())
    assert all(isinstance(v, bool) for v in c.values())


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


def test_sim_state_replay_round_trip() -> None:
    task = FakeTask()
    env = task.make_env(DynamicsSpec(), 0)
    replay = SimStateReplay(env)
    assert replay.available
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.array([0.5, 0.0], dtype=np.float32))
    snap = replay.snapshot()
    pos_after = env._pos
    env.step(np.array([0.9, 0.0], dtype=np.float32))
    assert env._pos != pos_after
    obs = replay.restore(snap)
    assert isinstance(obs, dict) and "state" in obs
    assert env._pos == pytest.approx(pos_after)


def test_dynamics_readback_guard() -> None:
    non_nominal = DynamicsSpec(mu_scale=0.5, mass_scale=2.0, stiff_scale=0.3)
    good = {"mu": {"scale": 0.5}, "mass": {"scale": 2.0}, "stiff": {"scale": 0.3}}
    assert_dynamics_applied(non_nominal, good, "M-M-M")
    assert_dynamics_applied(DynamicsSpec(), {}, "M-M-M")  # nominal needs no read-back
    with pytest.raises(RuntimeError, match="not applied"):
        assert_dynamics_applied(
            non_nominal,
            {"mu": {"scale": 1.0}, "mass": {"scale": 2.0}, "stiff": {"scale": 0.3}},
            "M-M-M",
        )
    with pytest.raises(ValueError, match="no dynamics read-back"):
        assert_dynamics_applied(non_nominal, {}, "M-M-M")
    task = FakeTask()
    env = task.make_env(non_nominal, 0)
    env.reset(seed=0)
    rb = readback_dict(env)
    assert rb["mass"]["scale"] == 2.0
    assert_dynamics_applied(non_nominal, rb, "M-M-M")


# --------------------------------------------------------------------------
# state collection
# --------------------------------------------------------------------------


def test_collect_states_snapshots_history_and_horizon_chunks() -> None:
    task = FakeTask()
    snapshots, meta, env = collect_states(
        task,
        "M-M-M",
        ranges=RANGES,
        seed=5,
        n_states=3,
        state_every=STATE_EVERY,
        max_steps=MAX_STEPS,
        horizon=HORIZON,
    )
    assert meta["n_states"] == len(snapshots) == 3
    assert meta["cell"] == "M-M-M"
    indices = [s["action_index"] for s in snapshots]
    assert indices == [0, 4, 8]
    for i, snap in enumerate(snapshots):
        assert snap["chunk"].shape == (HORIZON, ACTION_DIM), snap["chunk"].shape
        assert len(snap["hist"]) == indices[i]  # strictly *before* the pre-action obs
        assert snap["state"] is not None
        assert "state" in snap["obs"]
    # the env is handed back open so replay happens in the same instance
    assert isinstance(SimStateReplay(env).restore(snapshots[1]["state"]), dict)
    env.close()


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def test_generate_critic_data_end_to_end(tmp_path: Path) -> None:
    task = FakeTask()
    summary = generate_critic_data(
        task,
        cells=["M-M-M"],
        ranges=RANGES,
        states_per_cell=2,
        chunks_per_state=3,
        base_seed=1,
        out_dir=tmp_path / "crit",
        horizon=HORIZON,
        violation_spec=task.violations,
        config_hash="c0ffee00",
        sigma_grid=(0.02, 0.05),
        action_spec={"gripper_index": 1, "gripper_open": 1.0},
        max_steps=MAX_STEPS,
        state_every=STATE_EVERY,
        min_pos_rate=0.0,
        monitor_factory=_factory(),
        manifest_path=tmp_path / "MANIFEST.json",
        data_root=tmp_path,
    )
    assert summary["n_tuples"] == 12  # 2 states x (1 expert + 2 perturbed + 3 unsafe)
    assert summary["feature_spec"]["action_dim"] == ACTION_DIM
    assert summary["feature_spec"]["chunk_horizon"] == HORIZON
    assert summary["n_state_restores"] == 12
    assert summary["per_source_balance"]["expert"] == pytest.approx(2 / 12)
    assert summary["per_source_balance"]["unsafe"] == pytest.approx(6 / 12)
    assert set(summary["per_type_balance"]) == set(VIOLATION_TYPES)
    assert Path(summary["paths"][0]).exists()
    assert (tmp_path / "crit" / "DATA_CARD.json").exists()
    assert summary["manifest_hash"] != "no-manifest"
    card = json.loads((tmp_path / "crit" / "DATA_CARD.json").read_text())
    assert card["extra"]["horizon"] == HORIZON
    assert card["extra"]["sigma_grid"] == [0.02, 0.05]

    from riskproj.data.schema import CriticDataset

    loaded = CriticDataset.load(tmp_path / "crit" / "tuples_0000")
    assert len(loaded) == 12
    t = loaded.tuples[0]
    assert t.chunk.shape == (HORIZON, ACTION_DIM)
    assert set(t.ctx) == {"proprio", "ee_vel", "action_hist", "proprio_hist", "vision_emb", "force"}
    assert all(isinstance(v, bool) for v in t.labels.values())
    assert "V3" in t.labels


def test_generate_critic_data_uses_policy_chunks(tmp_path: Path) -> None:
    task = FakeTask()
    seen: list[dict[str, Any]] = []

    def policy(obs: dict[str, Any]) -> np.ndarray:
        seen.append(obs)
        return np.full((HORIZON, ACTION_DIM), 0.1, dtype=np.float32)

    summary = generate_critic_data(
        task,
        cells=["M-M-M"],
        ranges=RANGES,
        states_per_cell=1,
        chunks_per_state=1,
        base_seed=2,
        out_dir=tmp_path / "crit_policy",
        horizon=HORIZON,
        violation_spec=task.violations,
        config_hash="abc",
        policy=policy,
        sigma_grid=(),
        include_unsafe=False,
        max_steps=MAX_STEPS,
        state_every=STATE_EVERY,
        min_pos_rate=0.0,
        monitor_factory=_factory(),
        manifest_path=tmp_path / "MANIFEST.json",
        data_root=tmp_path,
    )
    assert seen, "policy was never queried"
    assert summary["n_tuples"] == 2  # expert + policy for the single state
    assert summary["per_source_balance"]["policy"] == pytest.approx(0.5)
    assert summary["per_source_balance"]["expert"] == pytest.approx(0.5)
    assert summary["per_source_balance"]["unsafe"] == 0.0


def test_generate_critic_data_rejects_bad_policy_chunk(tmp_path: Path) -> None:
    task = FakeTask()
    with pytest.raises(ValueError, match=r"policy chunk must be \[H, A\]"):
        generate_critic_data(
            task,
            cells=["M-M-M"],
            ranges=RANGES,
            states_per_cell=1,
            chunks_per_state=1,
            base_seed=0,
            out_dir=tmp_path / "bad",
            horizon=HORIZON,
            violation_spec=task.violations,
            config_hash="abc",
            policy=lambda obs: np.zeros(3, dtype=np.float32),
            sigma_grid=(),
            include_unsafe=False,
            max_steps=MAX_STEPS,
            state_every=STATE_EVERY,
            min_pos_rate=0.0,
            monitor_factory=_factory(),
            manifest_path=tmp_path / "MANIFEST.json",
            data_root=tmp_path,
        )


def test_positivity_guard_fires(tmp_path: Path) -> None:
    task = FakeTask()
    with pytest.raises(RuntimeError, match="positivity"):
        generate_critic_data(
            task,
            cells=["M-M-M"],
            ranges=RANGES,
            states_per_cell=1,
            chunks_per_state=2,
            base_seed=0,
            out_dir=tmp_path / "safe",
            horizon=HORIZON,
            violation_spec=task.violations,
            config_hash="abc",
            sigma_grid=(0.02,),
            max_steps=MAX_STEPS,
            state_every=STATE_EVERY,
            min_pos_rate=0.5,
            monitor_factory=_factory(never=True),  # nothing is ever a violation
            manifest_path=tmp_path / "MANIFEST.json",
            data_root=tmp_path,
        )


def test_generate_critic_data_validates_inputs(tmp_path: Path) -> None:
    task = FakeTask()
    common = {
        "ranges": RANGES,
        "states_per_cell": 1,
        "base_seed": 0,
        "out_dir": tmp_path / "x",
        "violation_spec": task.violations,
        "config_hash": "abc",
        "max_steps": MAX_STEPS,
        "state_every": STATE_EVERY,
        "min_pos_rate": 0.0,
        "monitor_factory": _factory(),
        "manifest_path": tmp_path / "MANIFEST.json",
        "data_root": tmp_path,
    }
    with pytest.raises(ValueError, match="horizon"):
        generate_critic_data(task, cells=["M-M-M"], chunks_per_state=1, horizon=0, **common)
    with pytest.raises(ValueError, match="chunks_per_state"):
        generate_critic_data(task, cells=["M-M-M"], chunks_per_state=0, horizon=HORIZON, **common)
    with pytest.raises(ValueError, match="no cells"):
        generate_critic_data(task, cells=[], chunks_per_state=1, horizon=HORIZON, **common)


# --------------------------------------------------------------------------
# demos generator (same fake task)
# --------------------------------------------------------------------------


def test_generate_demos_writes_records_and_card(tmp_path: Path) -> None:
    task = FakeTask()
    summary = generate_demos(
        task,
        cells=["M-M-M", "L-M-M"],
        ranges=RANGES,
        episodes_per_cell=2,
        base_seed=0,
        out_dir=tmp_path / "demos",
        max_steps=MAX_STEPS,
        config_hash="d00d",
        action_dim=ACTION_DIM,
        keep_failures=True,
        manifest_path=tmp_path / "MANIFEST.json",
        data_root=tmp_path,
    )
    assert summary["n_records"] == 4
    assert set(summary["per_cell"]) == {"M-M-M", "L-M-M"}
    assert 0.0 <= summary["success_rate"] <= 1.0
    records = read_demos(tmp_path / "demos")
    assert len(records) == 4
    assert all(r.actions.shape[1] == ACTION_DIM for r in records)
    assert all(r.dyn["mass_scale"] > 0 for r in records)
    card = json.loads((tmp_path / "demos" / "DATA_CARD.json").read_text())
    assert card["n_records"] == 4
    assert card["extra"]["dyn_readback"], "per-cell read-back must be recorded in the card"
    assert card["cell_histogram"]["M-M-M"] == 2
