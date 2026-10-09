"""G3 gate tests: the gate must fail loudly on bad critics and never average a
broken cell away (``06`` §5, ``09`` §5)."""

from __future__ import annotations

import numpy as np
import pytest

from riskproj.critic.gates import (
    GATE_G3,
    GateReport,
    evaluate_g3,
    gate_status_report,
    load_gate_reports,
)
from riskproj.utils.config import CriticGateCfg

pytestmark = pytest.mark.unit

TAU = 0.5


def separable(n: int = 400, base: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < base).astype(float)
    p = np.where(y == 1, 0.99, 0.01)
    return p, y


def cells_for(n: int, k: int = 4) -> np.ndarray:
    return np.array([f"cell{i % k}" for i in range(n)], dtype=object)


def disagreement(n: int, val: float = 0.08, train: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    return np.full(n, val), np.full(n, train)


# --------------------------------------------------------------------------
# passing case
# --------------------------------------------------------------------------


def test_g3_passes_on_a_separable_calibrated_critic() -> None:
    p, y = separable()
    n = len(y)
    vd, td = disagreement(n)
    report = evaluate_g3(
        p,
        y,
        tau=TAU,
        cfg_gate=CriticGateCfg(),
        cells=cells_for(n),
        val_disagreement=vd,
        train_disagreement=td,
    )
    assert report.passed, report.failures
    assert report.gate == GATE_G3
    assert report.metrics["auroc"] == pytest.approx(1.0)
    assert report.metrics["ece"] <= 0.05
    assert report.metrics["fnr_at_tau"] == pytest.approx(0.0)
    assert report.metrics["val_disagreement"] > report.metrics["train_disagreement"]


def test_g3_records_low_support_as_a_note() -> None:
    # Deterministic 5/200 positives, so the note path is exercised without relying on a draw.
    n = 200
    y = np.zeros(n)
    y[:5] = 1.0
    p = np.where(y == 1, 0.99, 0.01)
    report = evaluate_g3(
        p,
        y,
        tau=TAU,
        cfg_gate=CriticGateCfg(),
        val_disagreement=np.full(n, 0.1),
        train_disagreement=np.zeros(n),
    )
    assert any("low VAL support" in note for note in report.notes)
    assert report.passed, report.failures


# --------------------------------------------------------------------------
# failing cases
# --------------------------------------------------------------------------


def test_g3_fails_on_inverted_predictions() -> None:
    p, y = separable()
    n = len(y)
    vd, td = disagreement(n)
    report = evaluate_g3(
        1.0 - p, y, tau=TAU, cfg_gate=CriticGateCfg(), val_disagreement=vd, train_disagreement=td
    )
    assert not report.passed
    assert any("AUROC" in f for f in report.failures)
    assert any("AUPRC" in f for f in report.failures)


def test_g3_fails_a_single_broken_cell_instead_of_averaging_it_away() -> None:
    p, y = separable(n=800, base=0.25)
    cells = cells_for(len(y), k=4)
    broken = p.copy()
    broken[cells == "cell2"] = 1.0 - broken[cells == "cell2"]
    vd, td = disagreement(len(y))
    report = evaluate_g3(
        broken,
        y,
        tau=TAU,
        cfg_gate=CriticGateCfg(),
        cells=cells,
        val_disagreement=vd,
        train_disagreement=td,
    )
    assert not report.passed
    assert any("cell2" in f for f in report.failures)
    assert report.metrics["cell::cell0::auroc"] == pytest.approx(1.0)
    assert report.metrics["cell::cell2::auroc"] == pytest.approx(0.0)


def test_g3_notes_cells_with_a_single_class() -> None:
    p, y = separable(n=200)
    cells = np.array(["mixed"] * 100 + ["posonly"] * 50 + ["negonly"] * 50, dtype=object)
    y_mod = y.copy()
    y_mod[cells == "posonly"] = 1.0
    y_mod[cells == "negonly"] = 0.0
    vd, td = disagreement(len(y))
    report = evaluate_g3(
        p,
        y_mod,
        tau=TAU,
        cfg_gate=CriticGateCfg(),
        cells=cells,
        val_disagreement=vd,
        train_disagreement=td,
    )
    assert any("posonly" in n for n in report.notes)
    assert any("negonly" in n for n in report.notes)
    assert np.isnan(report.metrics["cell::posonly::auroc"])


def test_g3_fails_per_type_fnr_and_notes_types_without_positives() -> None:
    p, y = separable()
    per_type_labels = {"V1": np.zeros_like(y), "V3": y}
    per_type_probs = {"V1": np.zeros_like(p), "V3": 1.0 - p}
    vd, td = disagreement(len(y))
    report = evaluate_g3(
        p,
        y,
        tau=TAU,
        cfg_gate=CriticGateCfg(),
        per_type_labels=per_type_labels,
        per_type_probs=per_type_probs,
        val_disagreement=vd,
        train_disagreement=td,
    )
    assert not report.passed
    assert any(f.startswith("V3:") and "FNR" in f for f in report.failures)
    assert any("V1: no positives" in n for n in report.notes)
    assert np.isnan(report.metrics["V1_fnr_at_tau"])


def test_g3_fails_when_disagreement_is_missing_but_required() -> None:
    p, y = separable()
    report = evaluate_g3(p, y, tau=TAU, cfg_gate=CriticGateCfg())
    assert not report.passed
    assert any("disagreement" in f for f in report.failures)


def test_g3_fails_when_val_disagreement_is_not_above_train() -> None:
    p, y = separable()
    n = len(y)
    vd, td = disagreement(n, val=0.01, train=0.09)
    report = evaluate_g3(
        p, y, tau=TAU, cfg_gate=CriticGateCfg(), val_disagreement=vd, train_disagreement=td
    )
    assert not report.passed
    assert any("no OOD signal" in f for f in report.failures)


def test_g3_disagreement_disabled_records_values_as_note() -> None:
    p, y = separable()
    n = len(y)
    vd, td = disagreement(n, val=0.01, train=0.09)
    report = evaluate_g3(
        p,
        y,
        tau=TAU,
        cfg_gate=CriticGateCfg(require_val_disagreement_gt_train=False),
        val_disagreement=vd,
        train_disagreement=td,
    )
    assert report.passed, report.failures
    assert any("disagreement check disabled" in n for n in report.notes)
    assert report.metrics["val_disagreement"] == pytest.approx(0.01)


def test_g3_fails_immediately_on_a_single_class_split() -> None:
    report = evaluate_g3(np.full(50, 0.5), np.ones(50), tau=TAU, cfg_gate=CriticGateCfg())
    assert not report.passed
    assert any("single class" in f for f in report.failures)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def test_g3_validates_inputs() -> None:
    p, y = separable(n=100)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        evaluate_g3(p * 2.0, y, tau=TAU, cfg_gate=CriticGateCfg())
    with pytest.raises(ValueError, match="differ"):
        evaluate_g3(p, y[:-5], tau=TAU, cfg_gate=CriticGateCfg())
    with pytest.raises(ValueError, match="tau"):
        evaluate_g3(p, y, tau=1.5, cfg_gate=CriticGateCfg())
    with pytest.raises(ValueError, match="cells has"):
        evaluate_g3(p, y, tau=TAU, cfg_gate=CriticGateCfg(), cells=["a", "b"])
    with pytest.raises(ValueError, match="together"):
        evaluate_g3(p, y, tau=TAU, cfg_gate=CriticGateCfg(), per_type_probs={"V1": p})
    with pytest.raises(ValueError, match="per-type keys differ"):
        evaluate_g3(
            p,
            y,
            tau=TAU,
            cfg_gate=CriticGateCfg(),
            per_type_probs={"V1": p},
            per_type_labels={"V2": y},
        )
    with pytest.raises(ValueError, match="binary"):
        evaluate_g3(p, np.full_like(y, 2.0), tau=TAU, cfg_gate=CriticGateCfg())


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_gate_status_report_round_trip(tmp_path) -> None:
    p, y = separable()
    report = evaluate_g3(
        p,
        y,
        tau=TAU,
        cfg_gate=CriticGateCfg(),
        val_disagreement=np.full(len(y), 0.1),
        train_disagreement=np.zeros(len(y)),
    )
    path = tmp_path / "gates.json"
    payload = gate_status_report([report], path)
    assert payload["all_passed"] is True
    loaded = load_gate_reports(path)
    assert len(loaded) == 1 and loaded[0].passed is True
    assert loaded[0].gate == GATE_G3
    assert loaded[0].metrics["auroc"] == pytest.approx(1.0)


def test_gate_status_report_flags_failure(tmp_path) -> None:
    failing = GateReport(gate=GATE_G3, passed=False, failures=["nope"])
    payload = gate_status_report([failing], tmp_path / "gates.json")
    assert payload["all_passed"] is False


def test_load_gate_reports_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gate_reports(tmp_path / "absent.json")
