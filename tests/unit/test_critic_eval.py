"""Critic evaluation: T1 table, calibration, per-cell breakdown, proxy-gold, claims.

The synthetic critic is trained briefly then deliberately *saturated* (its output
weights are scaled up). That is the failure mode temperature scaling exists to fix,
so the calibration assertion tests the mechanism rather than hoping a fresh model
happens to be miscalibrated.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from riskproj.critic.features import FeatureSpec
from riskproj.critic.model import RiskCritic
from riskproj.critic.train import train_critic
from riskproj.eval.critic_eval import (
    MEASURED_VIOLATION_KEYS,
    ORIGINAL_VIOLATION_KEYS,
    critic_claim_guard,
    critic_features_from_dataset,
    critic_table,
    ensemble_disagreement,
    proxy_gold_curve,
    reliability_diagram_data,
)
from riskproj.utils.config import CriticCfg

pytestmark = pytest.mark.unit

STATE_DIM = 4
ACTION_DIM = 2
HORIZON = 3
N_CELLS = 4
N = 240
# context_dim = state + ee_vel + 1*action + 1*state = 4 + 3 + 2 + 4 = 13; chunk = 6
DIM = 13 + HORIZON * ACTION_DIM


def spec() -> FeatureSpec:
    return FeatureSpec(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, chunk_horizon=HORIZON, hist_len=1
    )


class FeatureDataset:
    """Indexable ``(features, any_violation, per_type)`` source with a known signal.

    Each cell holds both classes by construction, so a per-cell AUROC exists for
    every cell and a broken cell cannot hide behind the pooled number (06 §5).
    """

    def __init__(self, n: int = N, dim: int = DIM, seed: int = 0, flip: float = 0.0) -> None:
        rng = np.random.default_rng(seed)
        feats = rng.normal(scale=0.3, size=(n, dim)).astype(np.float32)
        # feature 0 carries the label; alternating blocks give every cell both signs
        sign = np.asarray([1.0 if (i // N_CELLS) % 2 == 0 else -1.0 for i in range(n)])
        feats[:, 0] = sign * (1.0 + 0.2 * rng.random(n))
        self.features = feats
        self.y_any = (sign > 0).astype(np.float32)
        self.y_type = np.stack(
            [(sign > 0).astype(np.float32) if k % 2 == 0 else (sign < 0).astype(np.float32)
             for k in range(5)],
            axis=1,
        )
        if flip:
            # Calibration needs *miscalibration*, and a perfectly separable signal has
            # none: there the BCE-optimal temperature is always "as sharp as possible"
            # (|z| ~ 1.3, AUROC 1.0). Flipping a fraction of the labels is what creates a
            # confidently-wrong critic — the regime `06` §3 calibrates for.
            if not 0.0 <= float(flip) < 1.0:
                raise ValueError(f"flip must be in [0, 1); got {flip}")
            k = int(round(float(flip) * n))
            idx = rng.permutation(n)[:k]
            self.y_any = self.y_any.copy()
            self.y_any[idx] = 1.0 - self.y_any[idx]
            self.y_type = self.y_type.copy()
            self.y_type[idx] = 1.0 - self.y_type[idx]
        self.cells = [f"cell{i % N_CELLS}" for i in range(n)]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, i: int):  # type: ignore[no-untyped-def]
        return (
            torch.from_numpy(self.features[i]),
            torch.tensor(self.y_any[i]),
            torch.from_numpy(self.y_type[i]),
        )


def make_critic(*, seed: int = 0, inflate: float = 1.0) -> RiskCritic:
    # Enough epochs that the fitted critic separates the signal cleanly. At the default
    # ``inflate`` its raw scores come out under-confident (fitted T < 1); ``inflate`` is
    # how a test asks for the saturated regime instead of assuming the fixture is in it.
    cfg = CriticCfg(hidden=16, depth=2, ensemble=2, epochs=40, batch_size=64, seed=seed)
    model, _ = train_critic(FeatureDataset(), cfg, feature_spec=spec(), device="cpu")
    model.eval()
    if inflate != 1.0:
        with torch.no_grad():
            for member in model.members:
                member.out.weight.mul_(inflate)
    return model


@pytest.fixture(scope="module")
def critic() -> RiskCritic:
    return make_critic()


def test_critic_features_from_dataset_normalises_all_three_item_shapes(critic: RiskCritic) -> None:
    batch = critic_features_from_dataset(FeatureDataset(n=8), critic)
    assert batch.features.shape == (8, DIM)
    assert batch.y_any.shape == (8,)
    assert batch.y_type.shape == (8, 5)
    assert batch.cell_ids is None, "this dataset does not expose .tuples, so ids must be passed in"


def test_critic_features_rejects_a_width_mismatch(critic: RiskCritic) -> None:
    class Wrong(FeatureDataset):
        def __init__(self) -> None:
            super().__init__(n=4, dim=DIM + 3)

    with pytest.raises(ValueError, match="regenerate D-crit"):
        critic_features_from_dataset(Wrong(), critic)


def test_ensemble_disagreement_is_nonnegative(critic: RiskCritic) -> None:
    ds = FeatureDataset(n=6)
    ctx = {
        "proprio": ds.features[:, :STATE_DIM],
        "ee_vel": ds.features[:, :3],
        "action_hist": ds.features[:, :ACTION_DIM].reshape(6, 1, ACTION_DIM),
        "proprio_hist": ds.features[:, :STATE_DIM].reshape(6, 1, STATE_DIM),
        "vision_emb": np.zeros((6, 0), np.float32),
        "force": np.zeros((6, 0), np.float32),
    }
    chunk = torch.randn(6, HORIZON, ACTION_DIM)
    out = ensemble_disagreement(critic, ctx, chunk)
    assert out.shape == (6,)
    assert bool((out >= 0).all())


# --------------------------------------------------------------------------
# T1
# --------------------------------------------------------------------------


def test_critic_table_discriminates_and_calibrates(critic: RiskCritic) -> None:
    ds = FeatureDataset()
    table = critic_table(critic, ds, tau=0.5, split="val", cells=ds.cells, n_bins=15)

    assert table["n"] == float(N)
    assert table["auroc"] > 0.9, "the critic was trained on a separable signal"
    assert 0.0 <= table["base_rate"] <= 1.0
    assert table["auprc_over_base_rate"] == pytest.approx(table["auprc"] / table["base_rate"])
    assert table["calibration_source"] == "fitted-on-this-split"
    # Which side of 1 the fitted temperature lands on depends on whether the raw scores
    # are over- or under-confident; the saturated case is pinned by the next test. What
    # must hold for *any* critic (`06` §3) is that the fit produced a usable positive
    # scalar and that calibrating did not make ECE worse.
    assert 0.0 < table["temperature"] < float("inf")
    assert abs(table["temperature"] - 1.0) > 1e-6, (
        "a fitted temperature of exactly 1 means the fit did not run"
    )
    assert table["ece_after"] <= table["ece_before"] + 1e-6
    assert table["fnr_at_tau"] <= 1.0
    assert table["brier"] >= 0.0
    assert table["gate"] is None
    assert any("G3 was not evaluated" in note for note in table["notes"])


def test_critic_table_saturation_is_what_calibration_fixes() -> None:
    # Overconfident *and* wrong on part of the data: without the label noise the critic
    # is perfect (AUROC 1.0) and softening could only hurt, so the softening direction
    # of temperature scaling is untestable on the clean fixture.
    saturated = make_critic(inflate=8.0)
    ds = FeatureDataset(flip=0.2)
    table = critic_table(saturated, ds, tau=0.5, split="val", cells=ds.cells)
    assert table["saturation_rate"] > 0.5, "the inflated critic must actually be saturated"
    assert table["temperature"] > 1.5, "an overconfident, partly-wrong critic must be softened"
    assert table["ece_after"] < table["ece_before"]


def test_critic_table_has_a_per_cell_breakdown_with_every_cell(critic: RiskCritic) -> None:
    ds = FeatureDataset()
    table = critic_table(critic, ds, tau=0.5, split="val", cells=ds.cells)
    assert set(table["per_cell"]) == set(ds.cells)
    for cell, entry in table["per_cell"].items():
        assert entry["n"] > 0
        assert math.isfinite(entry["auroc"]), f"{cell} has no cell AUROC"
    assert set(table["per_type"]) == {"V1", "V2", "V3", "V4", "V5"}


def test_critic_table_warns_when_cells_are_unavailable(critic: RiskCritic) -> None:
    table = critic_table(critic, FeatureDataset(n=80), tau=0.5, split="val")
    assert table["per_cell"] == {}
    assert any("no cell ids available" in note for note in table["notes"])


def test_critic_table_refuses_to_calibrate_on_a_sealed_split(critic: RiskCritic) -> None:
    ds = FeatureDataset()
    with pytest.raises(ValueError, match="sealed"):
        critic_table(critic, ds, tau=0.5, split="test_combo", cells=ds.cells)
    # with a frozen temperature it is allowed, and the provenance says so
    table = critic_table(
        critic, ds, tau=0.5, split="test_combo", cells=ds.cells, temperature=1.25
    )
    assert table["calibration_source"] == "frozen (supplied)"
    assert table["temperature"] == pytest.approx(1.25)


def test_critic_table_writes_a_json_artifact(critic: RiskCritic, tmp_path) -> None:
    ds = FeatureDataset(n=60)
    critic_table(critic, ds, tau=0.5, split="val", out_dir=tmp_path)
    assert (tmp_path / "critic_table.json").exists()


def test_critic_table_reports_gradient_diagnostics_for_features_only(critic: RiskCritic) -> None:
    ds = FeatureDataset(n=60)
    table = critic_table(critic, ds, tau=0.5, split="val")
    grad = table["gradient_diagnostics"]
    assert grad["scope"] == "critic_feature_input"
    assert math.isfinite(grad["mean_gn2"])
    site = table["gradient_diagnostics_site"]
    assert not math.isfinite(site["mean_gn2"])
    assert any("site-level gradient diagnostics not computed" in n for n in table["notes"])


def test_critic_table_accepts_the_ctx_chunk_item_shape(critic: RiskCritic) -> None:
    """``data.schema.CriticDataset`` yields ``(ctx, chunk, any, per_type)``."""

    class CtxDataset:
        def __init__(self, n: int = 24) -> None:
            rng = np.random.default_rng(0)
            self.proprio = rng.normal(size=(n, STATE_DIM)).astype(np.float32)
            self.chunk = rng.normal(size=(n, HORIZON, ACTION_DIM)).astype(np.float32)
            sign = np.asarray([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
            self.y_any = (sign > 0).astype(np.float32)

        def __len__(self) -> int:
            return int(self.proprio.shape[0])

        def __getitem__(self, i: int):  # type: ignore[no-untyped-def]
            ctx = {
                "proprio": self.proprio[i],
                "ee_vel": np.zeros(3, np.float32),
                "action_hist": np.zeros((1, ACTION_DIM), np.float32),
                "proprio_hist": np.zeros((1, STATE_DIM), np.float32),
                "vision_emb": np.zeros(0, np.float32),
                "force": np.zeros(0, np.float32),
            }
            return (
                ctx,
                torch.from_numpy(self.chunk[i]),
                torch.tensor(self.y_any[i]),
                torch.zeros(5),
            )

    table = critic_table(critic, CtxDataset(), tau=0.5, split="val")
    assert table["n"] == 24.0
    for key in ("auroc", "auprc", "ece_after", "fnr_at_tau", "temperature"):
        assert key in table, f"{key} missing from the table built from (ctx, chunk) items"


def test_reliability_diagram_data_bins_every_sample() -> None:
    rng = np.random.default_rng(0)
    probs = rng.random(100)
    labels = (probs > 0.5).astype(float)
    data = reliability_diagram_data(probs, labels, n_bins=10)
    assert len(data["counts"]) == 10
    assert sum(data["counts"]) == 100.0


# --------------------------------------------------------------------------
# proxy vs gold
# --------------------------------------------------------------------------


def _decision(i: int, corr: float, reduction: float, correct_it: bool) -> dict[str, object]:
    return {
        "step": i,
        "corr_rel": corr,
        "r_cand": 0.9,
        "r_after": [0.9 - reduction],
        "k_used": 1,
        "active": True,
        "usable": True,
        "original_violation": True,
        "measured_violation": (not correct_it),
    }


def test_proxy_gold_curve_bins_are_ordered_and_labelled_with_n() -> None:
    records = [
        _decision(i, corr=0.01 * (i + 1), reduction=0.1 * (i // 10 + 1), correct_it=i < 30)
        for i in range(40)
    ]
    out = proxy_gold_curve(records, n_bins=4)
    assert out["n_records"] == 40.0
    assert out["n_unpaired"] == 0.0
    assert out["gold_quantity"] == "paired_reduction"
    assert len(out["bins"]) == 4
    xs = [b["mean_x"] for b in out["bins"]]
    assert xs == sorted(xs), "bins must be ordered by the x axis"
    assert all(b["n"] > 0 for b in out["bins"])


def test_proxy_gold_curve_finds_the_divergence_point() -> None:
    # Bins 0..2: the correction removes the measured violation. Bin 3: the critic
    # still likes it while the gold quantity stops improving -> exploitation.
    records = []
    for k in range(4):
        for j in range(10):
            records.append(
                _decision(
                    i=k * 10 + j,
                    corr=0.05 * (k + 1) + 0.001 * j,
                    reduction=0.1 * (k + 1),
                    correct_it=(k < 3),
                )
            )
    out = proxy_gold_curve(records, n_bins=4)
    assert out["divergence_point"] is not None
    assert out["bins"][0]["measured_violation_reduction"] == pytest.approx(1.0)
    assert out["bins"][-1]["measured_violation_reduction"] == pytest.approx(0.0)


def test_proxy_gold_curve_without_original_labels_reports_a_rate_and_says_so() -> None:
    records = [
        {"step": i, "corr_rel": 0.01 * (i + 1), "r_cand": 0.9, "r_after": [0.5], "measured_violation": i % 3 == 0}
        for i in range(30)
    ]
    out = proxy_gold_curve(records, n_bins=3)
    assert out["n_unpaired"] == 30.0
    assert out["gold_quantity"] == "violation_rate"
    assert any("no original-chunk violation" in note for note in out["notes"])


def test_proxy_gold_curve_fails_loud_on_a_missing_measured_field() -> None:
    with pytest.raises(KeyError, match="none of"):
        proxy_gold_curve([{"step": 0, "corr_rel": 0.1, "r_cand": 0.9, "r_after": [0.5]}])


def test_proxy_gold_curve_rejects_an_unknown_axis() -> None:
    with pytest.raises(ValueError, match="x must be"):
        proxy_gold_curve([_decision(0, 0.1, 0.1, True)], x="step")


def test_proxy_gold_curve_accepts_k_used_as_the_axis() -> None:
    records = [
        {"step": i, "k_used": 1 + (i % 3), "corr_rel": 0.02, "r_cand": 0.9, "r_after": [0.6],
         "original_violation": 1.0, "measured_violation": 0.0}
        for i in range(30)
    ]
    out = proxy_gold_curve(records, x="k_used")
    labels = [b["label"] for b in out["bins"]]
    assert labels == sorted(labels)
    assert all(b["n"] > 0 for b in out["bins"])


def test_measured_and_original_key_sets_do_not_overlap() -> None:
    assert not set(MEASURED_VIOLATION_KEYS) & set(ORIGINAL_VIOLATION_KEYS)


# --------------------------------------------------------------------------
# claim guard
# --------------------------------------------------------------------------


def test_claim_guard_always_forbids_success_and_safety_claims() -> None:
    forbidden = critic_claim_guard({"ece_after": 0.01, "fnr_at_tau": 0.05, "auroc": 0.95})
    joined = " ".join(forbidden)
    assert "predicts task success" in joined
    assert "safe" in joined
    # that table has no per-cell or site-gradient evidence either
    assert any("per-cell" in f for f in forbidden)


def test_claim_guard_flags_a_table_without_calibration() -> None:
    forbidden = critic_claim_guard({"auroc": 0.95, "fnr_at_tau": 0.05})
    assert any("calibrated" in f for f in forbidden)


def test_claim_guard_flags_an_uncalibrated_or_weak_table() -> None:
    table = {
        "ece_after": 0.2,
        "fnr_at_tau": 0.4,
        "auroc": 0.6,
        "calibration_source": "fitted-on-this-split",
        "per_cell": {"L-L-M": {"auroc": float("nan")}},
        "gradient_diagnostics_site": {"mean_gn2": 1.0},
    }
    forbidden = " ".join(critic_claim_guard(table))
    assert "ECE" in forbidden
    assert "FNR" in forbidden
    assert "AUROC" in forbidden
    assert "no cell AUROC" in forbidden


def test_claim_guard_does_not_flag_a_fully_evidenced_table() -> None:
    table = {
        "ece_after": 0.02,
        "fnr_at_tau": 0.05,
        "auroc": 0.95,
        "calibration_source": "frozen (supplied)",
        "per_cell": {"L-L-M": {"auroc": 0.9}},
        "gradient_diagnostics_site": {"mean_gn2": 1.0e-3},
    }
    forbidden = " ".join(critic_claim_guard(table))
    assert "calibrated" not in forbidden
    assert "FNR" not in forbidden
    assert "valid risk proxy" not in forbidden
