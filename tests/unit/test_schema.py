"""Dataset schema: tuple validation, round-trip, leak guard, data cards, manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from riskproj.data.schema import (
    DATASET_VERSION,
    CriticDataset,
    CriticTuple,
    DemoRecord,
    assert_deployable_ctx,
    data_card,
    read_demos,
    register_dataset,
    write_demos,
)
from riskproj.types import VIOLATION_TYPES


def _ctx(
    state_dim: int = 2, action_dim: int = 1, hist_len: int = 2, vision: int = 0, force: int = 0
):
    return {
        "proprio": np.zeros(state_dim, dtype=np.float32),
        "ee_vel": np.zeros(3, dtype=np.float32),
        "action_hist": np.zeros(hist_len * action_dim, dtype=np.float32),
        "proprio_hist": np.zeros(hist_len * state_dim, dtype=np.float32),
        "vision_emb": np.zeros(vision, dtype=np.float32),
        "force": np.zeros(force, dtype=np.float32),
    }


def _tuple(i: int, *, cell: str = "M-M-M", source: str = "expert", labels: dict | None = None):
    return CriticTuple(
        tuple_id=f"t{i}",
        cell_id=cell,
        split="train",
        seed=i,
        ctx=_ctx(),
        chunk=np.full((4, 1), float(i), dtype=np.float32),
        labels=labels or {v: (v == "V3") for v in VIOLATION_TYPES},
        chunk_source=source,
        perturbation_sigma=0.05 if source == "perturbed" else None,
    )


def _dataset(n: int = 6) -> CriticDataset:
    cells = ["M-M-M", "L-M-M", "H-M-M"]
    sources = ["expert", "perturbed", "unsafe"]
    tuples = []
    for i in range(n):
        labels = {v: (i % 2 == 0 and v in ("V3", "V1")) for v in VIOLATION_TYPES}
        tuples.append(_tuple(i, cell=cells[i % 3], source=sources[i % 3], labels=labels))
    return CriticDataset(tuples)


# --------------------------------------------------------------------------
# CriticTuple validation
# --------------------------------------------------------------------------


def test_tuple_rejects_missing_ctx_keys() -> None:
    ctx = _ctx()
    del ctx["force"]
    with pytest.raises(ValueError, match="missing keys"):
        CriticTuple("t", "M-M-M", "train", 0, ctx, np.zeros((4, 1), np.float32), {})


def test_tuple_rejects_bad_shapes_and_unknown_keys() -> None:
    with pytest.raises(ValueError, match=r"chunk must be \[H_c, A\]"):
        CriticTuple("t", "M-M-M", "train", 0, _ctx(), np.zeros((4,), np.float32), {})
    with pytest.raises(ValueError, match="unknown label keys"):
        CriticTuple("t", "M-M-M", "train", 0, _ctx(), np.zeros((4, 1), np.float32), {"V9": True})
    with pytest.raises(ValueError, match="unknown chunk_source"):
        CriticTuple("t", "M-M-M", "train", 0, _ctx(), np.zeros((4, 1), np.float32), {}, "made_up")


def test_any_violation_is_the_or_of_types() -> None:
    safe = _tuple(0, labels=dict.fromkeys(VIOLATION_TYPES, False))
    risky = _tuple(1, labels={v: (v == "V5") for v in VIOLATION_TYPES})
    assert safe.any_violation is False
    assert risky.any_violation is True
    assert risky.label_vector().shape == (len(VIOLATION_TYPES),)


# --------------------------------------------------------------------------
# CriticDataset
# --------------------------------------------------------------------------


def test_getitem_shapes_and_unflattened_history() -> None:
    ds = _dataset(3)
    ctx, chunk, label_any, label_per_type = ds[0]
    assert isinstance(ctx, dict)
    assert set(ctx) == {"proprio", "ee_vel", "action_hist", "proprio_hist", "vision_emb", "force"}
    assert ctx["action_hist"].shape == (2, 1)
    assert ctx["proprio_hist"].shape == (2, 2)
    assert chunk.shape == (4, 1)
    assert label_any.dim() == 0
    assert label_per_type.shape == (len(VIOLATION_TYPES),)
    assert all(isinstance(v, torch.Tensor) for v in ctx.values())


def test_balances_and_histograms() -> None:
    ds = _dataset(6)
    bal = ds.class_balance()
    assert bal["n"] == 6
    assert bal["pos_rate"] == pytest.approx(0.5)
    per_type = ds.per_type_balance()
    assert set(per_type) == set(VIOLATION_TYPES)
    assert per_type["V3"] == pytest.approx(0.5)
    assert per_type["V2"] == 0.0
    assert sum(ds.cell_histogram().values()) == 6
    assert ds.split_histogram() == {"train": 6}
    assert ds.source_balance()["expert"] == pytest.approx(2 / 6)
    assert set(ds.per_cell_positives()) <= {"M-M-M", "L-M-M", "H-M-M"}


def test_feature_spec_reports_real_widths() -> None:
    spec = _dataset(3).feature_spec_kwargs()
    assert spec == {
        "state_dim": 2,
        "action_dim": 1,
        "chunk_horizon": 4,
        "hist_len": 2,
        "vision_dim": 0,
        "force_dim": 0,
    }


def test_empty_dataset_is_safe() -> None:
    ds = CriticDataset([])
    assert len(ds) == 0
    assert ds.class_balance()["n"] == 0
    assert ds.labels_per_type().shape == (0, len(VIOLATION_TYPES))
    assert ds.feature_spec_kwargs()["chunk_horizon"] == 0


def test_subset_preserves_order() -> None:
    ds = _dataset(4)
    sub = ds.subset([3, 1])
    assert len(sub) == 2
    assert [t.tuple_id for t in sub.tuples] == ["t3", "t1"]


def test_save_load_round_trip(tmp_path: Path) -> None:
    ds = _dataset(5)
    stem = tmp_path / "tuples_0000"
    npz = ds.save(stem)
    assert npz.exists() and (tmp_path / "tuples_0000.json").exists()
    loaded = CriticDataset.load(stem)
    assert len(loaded) == 5
    assert [t.tuple_id for t in loaded.tuples] == [t.tuple_id for t in ds.tuples]
    assert loaded.labels_any().tolist() == ds.labels_any().tolist()
    assert loaded.feature_spec_kwargs() == ds.feature_spec_kwargs()
    for a, b in zip(ds.tuples, loaded.tuples, strict=True):
        assert a.cell_id == b.cell_id
        assert a.chunk_source == b.chunk_source
        assert a.perturbation_sigma == b.perturbation_sigma
        assert b.ctx["proprio"].shape == (2,)
        assert b.ctx["proprio_hist"].shape == (2 * 2,)
        np.testing.assert_allclose(a.chunk, b.chunk)
    # labels survive the round trip exactly (this is what G3 consumes)
    np.testing.assert_array_equal(loaded.labels_per_type().numpy(), ds.labels_per_type().numpy())


def test_load_rejects_a_foreign_layout(tmp_path: Path) -> None:
    ds = _dataset(2)
    stem = tmp_path / "tuples_0000"
    ds.save(stem)
    meta = json.loads((tmp_path / "tuples_0000.json").read_text())
    meta["spec"]["state_dim"] = 99
    (tmp_path / "tuples_0000.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="does not match the reloaded arrays"):
        CriticDataset.load(stem)


# --------------------------------------------------------------------------
# leak guard
# --------------------------------------------------------------------------


def test_deployable_guard_blocks_non_deployable_inputs() -> None:
    assert_deployable_ctx(_ctx(), use_vision=False, privileged_force=False)
    with pytest.raises(ValueError, match="vision embedding"):
        assert_deployable_ctx(_ctx(vision=8), use_vision=False, privileged_force=False)
    with pytest.raises(ValueError, match="simulated force"):
        assert_deployable_ctx(_ctx(force=6), use_vision=False, privileged_force=False)
    # explicitly enabled inputs are allowed (ablation E-critic-force)
    assert_deployable_ctx(_ctx(vision=8, force=6), use_vision=True, privileged_force=True)


# --------------------------------------------------------------------------
# demos
# --------------------------------------------------------------------------


def _demo(i: int, *, success: bool = True) -> DemoRecord:
    t = i + 3
    return DemoRecord(
        ep_id=f"ep{i}",
        cell_id="M-M-M" if i % 2 else "L-M-M",
        split="train",
        seed=i,
        task="toy_reach",
        success=success,
        steps=t,
        actions=np.full((t, 2), 0.1 * i, dtype=np.float32),
        proprio=np.zeros((t, 4), dtype=np.float32),
        instruction="insert the peg",
        progress=np.linspace(0, 1, t, dtype=np.float32),
        dyn={"mu_scale": 1.0, "mass_scale": 1.0, "stiff_scale": 1.0},
    )


def test_demo_record_validates_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        DemoRecord("e", "M-M-M", "train", 0, "t", True, 2, np.zeros((2, 2)), np.zeros((3, 4)))


def test_demos_round_trip_with_variable_lengths(tmp_path: Path) -> None:
    records = [_demo(0), _demo(1, success=False), _demo(2)]
    write_demos(tmp_path, records)
    back = read_demos(tmp_path)
    assert len(back) == 3
    for a, b in zip(records, back, strict=True):
        assert a.ep_id == b.ep_id
        assert a.success == b.success
        assert a.cell_id == b.cell_id
        assert a.instruction == b.instruction
        assert b.actions.shape == a.actions.shape
        np.testing.assert_allclose(a.actions, b.actions)
        np.testing.assert_allclose(a.progress, b.progress)


def test_write_demos_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no records"):
        write_demos(tmp_path, [])


# --------------------------------------------------------------------------
# data cards and manifests
# --------------------------------------------------------------------------


def test_data_card_for_critic_dataset() -> None:
    card = data_card(_dataset(4), generator="gen_critic_data", config_hash="abcd1234")
    assert card["dataset_version"] == DATASET_VERSION
    assert card["kind"] == "critic_tuples"
    assert card["config_hash"] == "abcd1234"
    assert card["n_records"] == 4
    assert "class_balance" in card
    assert set(card["per_type_balance"]) == set(VIOLATION_TYPES)
    assert "cell_histogram" in card
    assert card["feature_spec"]["chunk_horizon"] == 4


def test_data_card_for_demo_records() -> None:
    card = data_card([_demo(0), _demo(1, success=False)], generator="gen_demos", config_hash="beef")
    assert card["kind"] == "demos"
    assert card["success_rate"] == pytest.approx(0.5)
    assert card["task"] == "toy_reach"
    assert sum(card["cell_histogram"].values()) == 2


def test_register_dataset_writes_manifest_entry(tmp_path: Path) -> None:
    ds = _dataset(2)
    npz = ds.save(tmp_path / "out" / "tuples_0000")
    manifest = tmp_path / "MANIFEST.json"
    entry = register_dataset(
        npz,
        generator="gen_critic_data",
        generator_version=DATASET_VERSION,
        config_hash="deadbeef",
        manifest_path=manifest,
        base=tmp_path,
        n_records=2,
    )
    assert entry["sha256"]
    assert entry["n_records"] == 2
    assert entry["path"] == "out/tuples_0000.npz"
    written = json.loads(manifest.read_text())
    assert "out/tuples_0000.npz" in written["entries"]
    assert written["entries"]["out/tuples_0000.npz"]["generator_version"] == DATASET_VERSION
