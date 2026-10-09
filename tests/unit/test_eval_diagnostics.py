"""Diagnostics, anti-gaming, systems and reporting — all from synthetic archives.

No simulator, no model, no GPU: these modules are pure functions of the archived
logs, and testing them that way is what proves the logs are the source of truth
(``skills/software_engineering.md`` §1.4).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from riskproj.eval import (
    DEFAULT_MPCD,
    SystemsCost,
    build_tables,
    check_budget,
    claim_ladder_rung,
    decision_frame,
    depth_sweep_table,
    fallback_conditioned_success,
    linearization_error_table,
    load_run,
    per_cell_heatmap,
    suppression_report,
    systems_table,
    write_report,
)
from riskproj.eval.anti_gaming import action_statistics, success_violation_pareto
from riskproj.types import FallbackCode, empty_violation_counts
from riskproj.utils.io import write_jsonl

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# fixtures built by hand (so the expected numbers are verifiable by eye)
# --------------------------------------------------------------------------


def decision(
    *,
    active: bool,
    usable: bool,
    corr_rel: float,
    r_cand: float,
    r_after: float,
    r_pred: float,
    k_used: int = 1,
    fallback_code: int = FallbackCode.NONE,
    site: str = "S2",
) -> dict:
    return {
        "ep_id": "run:method:cell:0",
        "step": 0,
        "flow_step": 7,
        "tau": 0.1,
        "r_cand": r_cand,
        "gn2": 1e-3,
        "active": active,
        "usable": usable,
        "k_used": k_used,
        "corr_rel": corr_rel,
        "r_after": [r_after],
        "r_pred": [r_pred],
        "ens_std": 0.01,
        "fallback_code": fallback_code,
        "true_violation_next_H": None,
        "extra": {"site": site, "mode": "latent"},
    }


def episode(
    *,
    method: str = "B3",
    split: str = "val",
    cell: str = "L-L-M",
    success: bool = True,
    train_seed: int = 0,
    fallback: dict[str, int] | None = None,
    violations: dict | None = None,
    site: str = "S2",
    latent_ms: float = 2.0,
    action_norm_mean: float = 1.0,
    failure_class: str = "none",
) -> dict:
    return {
        "run_id": "run",
        "method": method,
        "train_seed": train_seed,
        "env_seed": 0,
        "cell_id": cell,
        "split": split,
        "success": success,
        "steps": 20,
        "t_success": 20.0 if success else None,
        "dyn_readback": {"mu_scale": 1.0, "mass_scale": 1.0, "stiff_scale": 1.0},
        "violations": violations or empty_violation_counts(),
        "failure_class": failure_class,
        "fallback": fallback or {"F1": 0, "F2": 0, "F3": 0},
        "stall_steps": 0,
        "action_norm_mean": action_norm_mean,
        "jerk": 0.1,
        "proj": {"n_dec": 3.0},
        "latency_ms": {"policy": 1.0, "critic": 0.5, "proj": latent_ms},
        "peak_vram_mb": 1024.0,
        "extra": {"site": site, "planner": "latent"},
    }


def violated(kind: str = "V4", count: float = 2.0, first_step: float = 3.0) -> dict:
    rec = empty_violation_counts()
    rec[kind] = {"first_step": first_step, "count": count}
    return rec


# --------------------------------------------------------------------------
# decision_frame
# --------------------------------------------------------------------------


def test_decision_frame_separates_active_usable_and_unusable() -> None:
    records = [
        decision(active=False, usable=False, corr_rel=0.0, r_cand=0.1, r_after=0.1, r_pred=0.1),
        decision(active=True, usable=True, corr_rel=0.2, r_cand=0.9, r_after=0.4, r_pred=0.45),
        decision(
            active=True,
            usable=False,
            corr_rel=0.0,
            r_cand=0.9,
            r_after=0.9,
            r_pred=0.9,
            fallback_code=FallbackCode.F1_BASE_ACTION,
        ),
    ]
    frame = decision_frame(records)
    assert frame["n_decisions"] == 3
    assert frame["n_active"] == 2  # the safe candidate is not "active"
    assert frame["n_unusable"] == 1
    assert frame["correction_rate"] == pytest.approx(2 / 3)
    assert frame["unusable_rate_of_active"] == pytest.approx(0.5)
    assert frame["mean_corr_rel"] == pytest.approx(0.2), "only corrected rows count toward the norm"
    assert frame["mean_linearization_error"] == pytest.approx(-0.05, abs=1e-9)
    assert frame["fallback_rates"]["F1"] == pytest.approx(1 / 3)
    assert frame["fallback_rates"]["F2"] == 0.0


def test_decision_frame_is_empty_safe() -> None:
    frame = decision_frame([])
    assert frame["n_decisions"] == 0
    assert np.isnan(frame["correction_rate"])


def test_decision_frame_accepts_dataclass_records_too() -> None:
    from riskproj.types import DecisionRecord

    rec = DecisionRecord(ep_id="e", step=0, r_cand=0.7, active=True, usable=True, corr_rel=0.1)
    frame = decision_frame([rec])
    assert frame["n_decisions"] == 1
    assert frame["correction_rate"] == 1.0


# --------------------------------------------------------------------------
# linearization error (figure F1 / experiment E2)
# --------------------------------------------------------------------------


def test_linearization_table_bins_by_step_size_and_keeps_the_sign() -> None:
    records = [
        decision(active=True, usable=True, corr_rel=0.01, r_cand=0.9, r_after=0.5, r_pred=0.4),
        decision(active=True, usable=True, corr_rel=0.02, r_cand=0.9, r_after=0.6, r_pred=0.4),
        decision(active=True, usable=True, corr_rel=0.5, r_cand=0.9, r_after=0.9, r_pred=0.4),
    ]
    table = linearization_error_table(records, bins=2)
    assert table["n_samples"] == 3
    assert len(table["bins"]) == 2
    small, large = table["bins"]
    assert small["median_step"] < large["median_step"]
    # Signed error is r_after - r_pred. For a risk that curves upward in the step
    # direction the first-order model *under*-predicts the risk, so the error grows
    # positive with step size; it cannot be assumed negative for small steps. The
    # checkable claims are the sign at a large step and monotone growth in |error|.
    assert large["median_signed_error"] > 0
    assert abs(small["median_signed_error"]) < abs(large["median_signed_error"])
    assert "negative means" in table["sign_convention"]
    assert "positive means" in table["sign_convention"]


def test_linearization_table_skips_records_without_both_risks() -> None:
    row = decision(active=True, usable=True, corr_rel=0.1, r_cand=0.9, r_after=0.5, r_pred=0.4)
    row.pop("r_pred")
    table = linearization_error_table([row])
    assert table["n_samples"] == 0
    assert "r_after and r_pred" in table["note"]


# --------------------------------------------------------------------------
# per-cell heatmap and depth sweep
# --------------------------------------------------------------------------


def test_heatmap_carries_the_split_of_every_cell() -> None:
    rows = [
        episode(cell="L-L-M", split="val", success=True),
        episode(cell="L-L-M", split="val", success=False),
        episode(cell="xL-M-M", split="test_range", success=False),
    ]
    heat = per_cell_heatmap(rows, metric="success")
    assert heat["table"]["B3"]["L-L-M"] == pytest.approx(0.5)
    assert heat["cell_split"]["xL-M-M"] == "test_range"
    assert heat["cell_split"]["L-L-M"] == "val"
    assert heat["nested"]["test_range"] == ["xL-M-M"]
    assert "never pooled" in heat["guard"]


def test_heatmap_derives_the_split_when_the_record_omits_it() -> None:
    row = episode(cell="L-H-M", split="")
    row["split"] = ""
    heat = per_cell_heatmap([row], metric="violation")
    assert heat["cell_split"]["L-H-M"] == "test_combo", "sign product -1 is a TEST-combo cell"


def test_depth_sweep_raises_rather_than_reporting_an_empty_table() -> None:
    rows = [episode(site="S2"), episode(site="S3")]
    for row in rows:
        row["extra"].pop("site")
    with pytest.raises(KeyError, match="site"):
        depth_sweep_table(rows)


def test_depth_sweep_groups_by_site() -> None:
    rows = [episode(site="S2", success=True), episode(site="S3", success=False)]
    table = depth_sweep_table(rows)
    assert set(table["per_site"]) == {"S2", "S3"}
    assert table["per_site"]["S2"]["success_rate"] == pytest.approx(1.0)
    assert table["n_records_without_site"] == 0


# --------------------------------------------------------------------------
# anti-gaming
# --------------------------------------------------------------------------


def test_action_statistics_measures_suppression_and_stalls() -> None:
    stall = [np.array([0.0, 0.0, 0.0])] * 5
    stats = action_statistics(stall)
    assert stats["mean_norm"] == 0.0
    assert stats["stall_fraction"] == pytest.approx(1.0)
    moving = [np.array([float(i), 0.0, 0.0]) for i in range(5)]
    ms = action_statistics(moving)
    assert ms["mean_norm"] > 0
    assert ms["stall_fraction"] == 0.0
    assert ms["path_length"] == pytest.approx(4.0)


def test_suppression_report_flags_a_shrunken_method() -> None:
    rows = [
        episode(method="B1", action_norm_mean=1.0),
        episode(method="B3", action_norm_mean=1.0),
        episode(method="C2", action_norm_mean=0.5),
    ]
    report = suppression_report(rows, baseline="B1")
    assert report["per_method"]["B3"]["relative_change"] == pytest.approx(0.0)
    assert report["suppression_flag"] == ["C2"]
    assert "not a safety gain" in report["guard"]


def test_suppression_report_is_explicit_when_the_baseline_is_absent() -> None:
    report = suppression_report([episode(method="B3")], baseline="B1")
    assert "error" in report and report["available"] == ["B3"]


def test_fallback_conditioned_success_labels_fallback_driven_gains() -> None:
    rows = [episode(method="B3", fallback={"F1": 0, "F2": 8, "F3": 2}) for _ in range(4)]
    rows += [episode(method="B1") for _ in range(4)]
    out = fallback_conditioned_success(rows, max_fallback_rate=0.3)
    assert out["labelled"]["B3"] == "not_an_adaptation_gain"
    assert out["labelled"]["B1"] == "ok"
    assert out["per_method"]["B1"]["success_given_no_fallback"] == pytest.approx(1.0)


def test_pareto_keeps_splits_separate() -> None:
    rows = [
        episode(method="B3", split="test_combo", success=True, violations=violated()),
        episode(method="B3", split="test_range", success=False),
    ]
    pareto = success_violation_pareto(rows)
    assert set(pareto["table"]["B3"]) == {"test_combo", "test_range"}
    assert pareto["table"]["B3"]["test_combo"]["violation_rate"] == pytest.approx(1.0)
    assert pareto["table"]["B3"]["test_range"]["violation_rate"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# systems / G2
# --------------------------------------------------------------------------


class FakeGate:
    budget_gb = 7.0
    headroom_gb = 0.25


def test_check_budget_passes_and_fails_with_the_reason() -> None:
    ok = SystemsCost(peak_vram_mb={"inference": 4096.0})
    verdict = check_budget(ok, FakeGate(), phase="inference")
    assert verdict["passed"] is True
    assert verdict["used_gb"] == pytest.approx(4.0)

    too_big = SystemsCost(peak_vram_mb={"train": 7168.0})
    bad = check_budget(too_big, FakeGate(), phase="train")
    assert bad["passed"] is False
    assert "05 §6" in bad["reason"], "the failure must name the sanctioned reduction order"


def test_check_budget_requires_an_explicit_phase_when_ambiguous() -> None:
    cost = SystemsCost(peak_vram_mb={"a": 1.0, "b": 2.0})
    with pytest.raises(ValueError, match="phase"):
        check_budget(cost, FakeGate())


def test_systems_table_reports_the_dominant_phase() -> None:
    costs = {
        "critic_train": SystemsCost(peak_vram_mb={"critic_train": 2048.0}, trainable_params=10),
        "inference": SystemsCost(peak_vram_mb={"inference": 512.0}, trainable_params=10),
    }
    table = systems_table(costs)
    assert table["dominant_phase"] == "critic_train"
    assert table["peak_gb_by_phase"]["critic_train"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# claim ladder
# --------------------------------------------------------------------------


def test_claim_ladder_rung_climbs_only_with_the_required_artefacts() -> None:
    main = {"table": {"B3": {"val": {"success": {"rate": 0.5}}}}}
    assert claim_ladder_rung({"T2_main": main}) == "L0"

    main["n_train_seeds"] = {"B3": 3}
    assert claim_ladder_rung({"T2_main": main}) == "L1"
    assert claim_ladder_rung({"T2_main": main, "T4_controls": {"available": False}}) == "L1"

    main["table"]["B3"]["test_combo"] = {"success": {"rate": 0.4}}
    with_controls = {"T2_main": main, "T4_controls": {"available": True}}
    assert claim_ladder_rung(with_controls) == "L3"
    main["table"]["B3"].pop("test_combo")
    assert claim_ladder_rung(with_controls) == "L2"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _write_run(root: Path, rows: list[dict], decisions: list[dict]) -> Path:
    folder = root / "run-abc12345" / "B3" / "0"
    folder.mkdir(parents=True, exist_ok=True)
    write_jsonl(folder / "episodes.jsonl", rows)
    write_jsonl(folder / "decisions.jsonl", decisions)
    return root / "run-abc12345"


def test_load_run_raises_when_there_are_no_raw_logs(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="episodes.jsonl"):
        load_run(empty)


def test_report_is_derived_and_marks_unavailable_tables(tmp_path: Path) -> None:
    rows = [
        episode(method="B3", success=True, train_seed=0),
        episode(method="B3", success=False, train_seed=1),
        episode(method="B1", success=True, train_seed=0),
        episode(method="C1", success=True, train_seed=0),
    ]
    decisions = [
        decision(active=True, usable=True, corr_rel=0.05, r_cand=0.8, r_after=0.4, r_pred=0.45)
    ]
    run_root = _write_run(tmp_path, rows, decisions)

    tables = build_tables(run_root)
    assert tables["n_episodes"] == 4
    assert tables["T2_main"]["table"]["B3"]["val"]["success"]["n"] == 2.0
    assert tables["proj_diagnostics"]["correction_rate"] == pytest.approx(1.0)
    # T1 needs simulated labels; absent labels must be reported, not faked.
    assert tables["T1_critic"]["available"] is False
    assert "true_violation_next_H" in tables["T1_critic"]["reason"]
    assert tables["T4_controls"]["available"] is True
    assert tables["F2_counterfactual"]["available"] is False
    assert tables["claim_rung"] in ("L0", "L1", "L2", "L3")

    path = write_report(run_root)
    text = path.read_text(encoding="utf-8")
    assert "**derived artifact**" in text
    assert "claim-ladder rung" in text
    assert "we observed" in text and "guarantees" in text  # allowed vs forbidden language
    payload = json.loads((run_root / "tables.json").read_text(encoding="utf-8"))
    assert payload["n_episodes"] == 4
    assert payload["mpcd"] == pytest.approx(DEFAULT_MPCD)
    assert payload["sources"], "the report must list its source logs with hashes"


def test_report_mpcd_is_configurable_and_recorded(tmp_path: Path) -> None:
    run_root = _write_run(tmp_path, [episode()], [])
    write_report(run_root, mpcd=0.1)
    payload = json.loads((run_root / "tables.json").read_text(encoding="utf-8"))
    assert payload["mpcd"] == pytest.approx(0.1)


def test_report_t1_scores_labelled_decisions(tmp_path: Path) -> None:
    rows = [episode()]
    decisions = []
    for i in range(20):
        risky = i % 2 == 0
        decisions.append(
            decision(
                active=True,
                usable=True,
                corr_rel=0.01,
                r_cand=0.9 if risky else 0.1,
                r_after=0.2,
                r_pred=0.2,
            )
        )
        decisions[-1]["true_violation_next_H"] = risky
    run_root = _write_run(tmp_path, rows, decisions)
    tables = build_tables(run_root)
    assert tables["T1_critic"]["available"] is True
    assert tables["T1_critic"]["auroc"] == pytest.approx(1.0)
    assert tables["T1_critic"]["base_rate"] == pytest.approx(0.5)
    assert tables["F8_reliability"]["available"] is True
