"""Calibration tests: temperature scaling must soften overconfidence, and the tau
rule must respect its FNR cap rather than silently relaxing it (``06`` §3-§6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from riskproj.critic.calibrate import (
    apply_temperature,
    choose_tau,
    ece_before_after,
    fit_temperature,
    fnr_curve,
    reliability,
)
from riskproj.utils.metrics import ece

pytestmark = pytest.mark.unit


def overconfident_noise(n: int = 4000, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Confidence uncorrelated with correctness: the classic overconfidence case."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    return torch.tensor(sign * 3.0), torch.tensor(y)


# --------------------------------------------------------------------------
# temperature scaling
# --------------------------------------------------------------------------


def test_fit_temperature_softens_uncorrelated_confidence() -> None:
    logits, targets = overconfident_noise()
    t = fit_temperature(logits, targets)
    assert t > 5.0  # must push probabilities towards 0.5, i.e. T >> 1
    ece_before = ece(targets.numpy(), torch.sigmoid(logits).numpy())
    ece_after = ece(targets.numpy(), torch.sigmoid(logits / t).numpy())
    assert ece_after < ece_before
    assert ece_after <= 0.05


def test_ece_before_after_reports_both_sides() -> None:
    logits, targets = overconfident_noise(n=1500)
    t = fit_temperature(logits, targets)
    out = ece_before_after(logits, targets, t)
    assert out["ece_after"] <= out["ece_before"]
    assert out["nll_after"] <= out["nll_before"]
    assert out["n"] == 1500.0
    assert out["temperature"] == pytest.approx(t)


def test_fit_temperature_single_class_returns_one() -> None:
    logits = torch.randn(50)
    assert fit_temperature(logits, torch.ones(50)) == 1.0


def test_apply_temperature_is_monotone_and_validated() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0])
    softened = apply_temperature(logits, 2.0)
    assert torch.all(torch.argsort(softened) == torch.argsort(logits))
    assert float(softened.abs().max()) < float(logits.abs().max())
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(logits, 0.0)


def test_fit_temperature_rejects_non_binary_targets() -> None:
    with pytest.raises(ValueError, match="binary"):
        fit_temperature(torch.randn(4), torch.tensor([0.0, 1.0, 2.0, 0.0]))


def test_fit_temperature_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        fit_temperature(torch.randn(4), torch.tensor([0.0, 1.0]))


def test_reliability_has_expected_shape() -> None:
    logits, targets = overconfident_noise(n=600)
    curve = reliability(logits, targets, n_bins=7)
    assert len(curve["counts"]) == 7
    assert len(curve["bin_edges"]) == 8


# --------------------------------------------------------------------------
# tau selection (VAL only)
# --------------------------------------------------------------------------


def separable_probs(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(float)
    p = np.where(y == 1, 0.9, 0.1) + rng.normal(0.0, 0.02, size=n)
    return np.clip(p, 0.0, 1.0), y


def test_choose_tau_respects_fnr_cap_on_separable_data() -> None:
    p, y = separable_probs()
    res = choose_tau(p, y, {"fnr_cap": 0.15})
    assert res["feasible"] == 1.0
    assert res["fnr"] <= 0.15
    assert res["precision"] >= 0.99
    assert 0.1 < res["tau"] < 0.95
    assert res["predicted_positive_rate"] < 1.0  # a threshold that flags everything is useless
    assert res["n_positives"] > 0


def test_choose_tau_is_infeasible_when_thresholds_are_restricted() -> None:
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < 0.3).astype(float)
    res = choose_tau(p, y, {"fnr_cap": 0.01, "tau_lo": 0.8, "tau_hi": 0.99})
    assert res["feasible"] == 0.0
    assert res["reason"] == 2.0


def test_choose_tau_without_positives_is_infeasible() -> None:
    res = choose_tau(np.linspace(0, 1, 100), np.zeros(100))
    assert res["feasible"] == 0.0
    assert res["reason"] == 1.0
    assert res["n_positives"] == 0.0


def test_choose_tau_smaller_grid_is_still_valid() -> None:
    p, y = separable_probs(n=200)
    res = choose_tau(p, y, {"fnr_cap": 0.2, "n_grid": 21})
    assert res["feasible"] == 1.0
    assert res["n_candidates"] <= 23


def test_choose_tau_rejects_probability_scale_violations() -> None:
    p, y = separable_probs(n=50)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        choose_tau(p * 10.0, y)
    with pytest.raises(ValueError, match="differ"):
        choose_tau(p, y[:-1])
    with pytest.raises(ValueError, match="tau_lo"):
        choose_tau(p, y, {"tau_lo": 0.9, "tau_hi": 0.1})


def test_fnr_curve_lengths() -> None:
    p, y = separable_probs(n=120)
    taus = np.linspace(0.05, 0.95, 7)
    curve = fnr_curve(p, y, taus)
    assert len(curve["tau"]) == 7 == len(curve["fnr"]) == len(curve["precision"])
    assert curve["fnr"][0] <= curve["fnr"][-1]  # looser threshold detects more
