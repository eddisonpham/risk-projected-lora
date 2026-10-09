"""Statistics layer: CIs, paired tests, multiplicity, power and the log summary.

These tests pin the *rules* from ``agents/09`` §2, not just the arithmetic:
pre-registered comparisons alone get corrected, TEST splits may not be pooled, and
the replication unit is the training seed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskproj.eval.stats import (
    ALL_KEY,
    MPCD_DEFAULT,
    PRIMARY_COMPARISONS,
    REPLICATION_UNIT,
    detectable_difference,
    holm_primary,
    iqm_with_ci,
    fisher_unpaired,
    mcnemar_paired,
    never_pool_test_splits,
    normal_quantile,
    paired_episode_delta,
    performance_profile,
    power_for_n,
    probability_of_improvement,
    required_n_for_mpcd,
    stratified_bootstrap_ci,
    summarise_records,
)
from riskproj.utils.metrics import bootstrap_ci, mcnemar_exact, wilson_interval

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def test_normal_quantile_is_the_inverse_of_the_cdf() -> None:
    assert normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert normal_quantile(0.5) == pytest.approx(0.0, abs=1e-6)
    assert normal_quantile(0.8) == pytest.approx(0.8416212, abs=1e-4)


def test_replication_unit_is_the_training_seed() -> None:
    # 08 §4: an episode bootstrap over one trained policy answers a different
    # question from a run-level interval, so the constant is part of the contract.
    assert REPLICATION_UNIT == "train_seed"


# --------------------------------------------------------------------------
# stratified bootstrap
# --------------------------------------------------------------------------


def test_stratified_ci_covers_the_point_and_widens_with_fewer_samples() -> None:
    # Both strata must *vary*, otherwise the bootstrap of the mean has zero width and
    # the width comparison tests nothing (a stratum of identical values cannot
    # produce an interval to compare).
    few = {"M-M-M": [1.0, 0.0, 1.0, 0.0], "H-H-M": [1.0, 0.0, 0.0, 1.0]}
    many = {k: v * 5 for k, v in few.items()}
    wide = stratified_bootstrap_ci(many, n_boot=2000, seed=0)
    thin = stratified_bootstrap_ci(few, n_boot=2000, seed=0)
    for key in ("M-M-M", "H-H-M", ALL_KEY):
        assert wide[key]["lo"] <= wide[key]["point"] <= wide[key]["hi"]
        assert wide[key]["n"] > 0
    width_many = wide["M-M-M"]["hi"] - wide["M-M-M"]["lo"]
    width_few = thin["M-M-M"]["hi"] - thin["M-M-M"]["lo"]
    assert width_few > width_many, "fewer samples must widen the interval"
    assert wide[ALL_KEY]["n"] == pytest.approx(40.0)


def test_stratified_pooled_estimate_is_the_sample_mean() -> None:
    strata = {"a": [0.0, 1.0], "b": [1.0, 1.0, 0.0, 1.0]}
    out = stratified_bootstrap_ci(strata, n_boot=500, seed=1)
    assert out[ALL_KEY]["point"] == pytest.approx(np.mean([0.0, 1.0, 1.0, 1.0, 0.0, 1.0]))


def test_stratified_ci_rejects_an_empty_stratum() -> None:
    with pytest.raises(ValueError, match="empty"):
        stratified_bootstrap_ci({"a": []}, n_boot=10)


def test_iqm_with_ci_matches_the_iqm_and_reports_provenance() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    out = iqm_with_ci(values, n_boot=1000, seed=0)
    assert out["n"] == 6.0
    assert out["lo"] <= out["point"] <= out["hi"]
    # 0.0/1.0 flags which implementation produced the interval, so a report can say.
    assert out["used_rliable"] in (0.0, 1.0)


def test_performance_profile_is_monotone_in_the_threshold() -> None:
    prof = performance_profile({"B1": [0.1, 0.4, 0.6, 0.9], "B3": [0.5, 0.6, 0.7, 0.95]})
    assert prof["n_runs"]["B1"] == 4.0
    assert len(prof["thresholds"]) == len(prof["profiles"]["B1"])
    # fraction above the threshold must not increase as the threshold rises
    assert all(
        a >= b - 1e-12 for a, b in zip(prof["profiles"]["B1"][:-1], prof["profiles"]["B1"][1:])
    )


def test_probability_of_improvement_counts_ties_half() -> None:
    assert probability_of_improvement([1.0, 1.0], [0.0, 0.0]) == pytest.approx(1.0)
    assert probability_of_improvement([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="paired"):
        probability_of_improvement([1.0], [1.0, 0.0])


# --------------------------------------------------------------------------
# paired tests
# --------------------------------------------------------------------------


def test_paired_episode_delta_matches_a_hand_computed_interval() -> None:
    a = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    b = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    out = paired_episode_delta(a, b, n_boot=4000, seed=3)
    diff = a - b
    expected = bootstrap_ci(diff, n_boot=4000, seed=3)
    assert out["point"] == pytest.approx(expected["point"])
    assert out["lo"] == pytest.approx(expected["lo"])
    assert out["hi"] == pytest.approx(expected["hi"])
    # b_only: a succeeded where b failed (2 cases); c_only: the reverse (0)
    assert out["b_a_only"] == pytest.approx(2.0)
    assert out["c_b_only"] == pytest.approx(0.0)
    assert out["n_discordant"] == pytest.approx(2.0)
    # 8 pairs with 2 discordant ones cannot exclude zero; the flag must simply be
    # consistent with the interval it reports (a stronger, checkable claim than
    # asserting the outcome of one small sample).
    assert out["excludes_zero"] == float(out["lo"] > 0.0 or out["hi"] < 0.0)


def test_paired_delta_reports_a_length_mismatch_instead_of_faking_a_pairing() -> None:
    out = paired_episode_delta([1.0, 1.0, 1.0], [0.0, 0.0], n_boot=100, seed=0)
    assert "pairing lengths differ" in out["note"]
    assert out["n"] == 2.0


def test_mcnemar_paired_matches_the_primitive() -> None:
    a = [True, True, False, True, False, True]
    b = [True, False, False, False, True, True]
    out = mcnemar_paired(a, b)
    ref = mcnemar_exact(int(out["b_a_only"]), int(out["c_b_only"]))
    assert out["p_value"] == pytest.approx(ref["p_value"])
    assert out["n_discordant"] == pytest.approx(3.0)


def test_mcnemar_paired_requires_equal_lengths_and_binary_values() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        mcnemar_paired([1, 0], [1, 0, 1])
    with pytest.raises(ValueError, match="binary"):
        mcnemar_paired([1, 2], [1, 0])


def test_fisher_unpaired_handles_a_clear_difference() -> None:
    out = fisher_unpaired(30, 40, 5, 40)
    assert out["p_value"] < 1e-6
    assert out["rate_a"] == pytest.approx(0.75)


# --------------------------------------------------------------------------
# multiplicity: primaries only
# --------------------------------------------------------------------------


def test_holm_primary_adjusts_only_the_preregistered_comparisons() -> None:
    p = {"B3_vs_B1": 0.01, "B3_vs_B2": 0.04, "B3_vs_B4a": 0.2, "B3_vs_B5": 0.001}
    out = holm_primary(p, alpha=0.05)
    assert set(PRIMARY_COMPARISONS) <= set(out)
    for name in PRIMARY_COMPARISONS:
        assert out[name]["exploratory"] == 0.0
        assert math.isfinite(out[name]["threshold"])
    # step-down over three primaries: the smallest p is tested at alpha/3
    assert out["B3_vs_B1"]["threshold"] == pytest.approx(0.05 / 3)
    # an exploratory comparison is left alone and flagged
    assert out["B3_vs_B5"]["exploratory"] == 1.0
    assert math.isnan(out["B3_vs_B5"]["threshold"])
    assert out["B3_vs_B5"]["p"] == pytest.approx(0.001)


def test_holm_primary_is_monotone_step_down() -> None:
    p = {"B3_vs_B1": 0.001, "B3_vs_B2": 0.02, "B3_vs_B4a": 0.03}
    out = holm_primary(p, alpha=0.05)
    rejects = [out[n]["reject"] for n in ("B3_vs_B1", "B3_vs_B2", "B3_vs_B4a")]
    # once a comparison fails to reject, no weaker one may reject
    assert rejects == sorted(rejects, reverse=True)


# --------------------------------------------------------------------------
# power / sample size
# --------------------------------------------------------------------------


def test_required_n_matches_the_closed_form_and_grows_with_discordance() -> None:
    out = required_n_for_mpcd(0.2, mpcd=0.1, alpha=0.05, power=0.8)
    z_a, z_b = normal_quantile(0.975), normal_quantile(0.8)
    expected = math.ceil(0.2 * (z_a + z_b) ** 2 / 0.1**2)
    assert out["n_per_arm"] == pytest.approx(float(expected))
    assert out["n_discordant_needed"] == pytest.approx(math.ceil(expected * 0.2))
    assert out["mpcd"] == 0.1 and out["power"] == 0.8

    # Discordance is variance, not signal for a paired test, so n grows with it.
    lo = required_n_for_mpcd(0.1, mpcd=0.1)
    hi = required_n_for_mpcd(0.3, mpcd=0.1)
    assert hi["n_per_arm"] > lo["n_per_arm"]
    assert hi["n_discordant_needed"] > lo["n_discordant_needed"]


def test_required_n_rejects_an_impossible_effect() -> None:
    with pytest.raises(ValueError, match="exceeds the discordant rate"):
        required_n_for_mpcd(0.05, mpcd=0.2)


def test_power_and_detectable_difference_are_consistent() -> None:
    n = int(required_n_for_mpcd(0.25, mpcd=MPCD_DEFAULT, power=0.8)["n_per_arm"])
    assert power_for_n(n, 0.25, effect=MPCD_DEFAULT) == pytest.approx(0.8, abs=0.02)
    dd = detectable_difference(n, 0.25, power=0.8)
    assert dd == pytest.approx(MPCD_DEFAULT, rel=0.05)
    # more episodes can only detect smaller differences
    assert detectable_difference(n * 4, 0.25, power=0.8) < dd


# --------------------------------------------------------------------------
# the log summary and its guards
# --------------------------------------------------------------------------


def _record(
    method: str = "B1",
    split: str = "val",
    cell: str = "L-L-M",
    success: bool = True,
    violation: str | None = None,
    fallback: dict[str, int] | None = None,
    train_seed: int = 0,
) -> dict[str, object]:
    viol = {
        v: {"first_step": (3.0 if v == violation else -1.0), "count": (2.0 if v == violation else 0.0)}
        for v in ("V1", "V2", "V3", "V4", "V5")
    }
    return {
        "run_id": "r1",
        "method": method,
        "train_seed": train_seed,
        "env_seed": 0,
        "cell_id": cell,
        "split": split,
        "success": success,
        "steps": 40,
        "violations": viol,
        "fallback": fallback or {"F1": 0, "F2": 0, "F3": 0},
        "stall_steps": 0,
        "latency_ms": {"policy": 12.0, "critic": 3.0, "proj": 4.0},
        "peak_vram_mb": 1024.0,
    }


def test_summarise_records_reports_wilson_cis_and_per_type_rates() -> None:
    records = [
        _record(success=True),
        _record(success=False, violation="V3"),
        _record(success=True, violation="V3"),
        _record(success=False),
    ]
    out = summarise_records(records)
    group = out["groups"]["B1|val"]
    assert group["n"] == 4.0
    assert group["success_rate"] == pytest.approx(0.5)
    lo, hi = wilson_interval(2, 4)
    assert group["success_ci"]["lo"] == pytest.approx(lo)
    assert group["success_ci"]["hi"] == pytest.approx(hi)
    assert group["per_type_violation_rate"]["V3"] == pytest.approx(0.5)
    assert group["any_violation_rate"] == pytest.approx(0.5)
    assert out["replication_unit"] == "train_seed"


def test_summarise_records_surfaces_fallback_contaminated_success() -> None:
    records = [
        _record(success=True, fallback={"F1": 3, "F2": 0, "F3": 0}),
        _record(success=True, fallback={"F1": 1, "F2": 0, "F3": 0}),
        _record(success=False),
        _record(success=False),
    ]
    out = summarise_records(records)
    group = out["groups"]["B1|val"]
    assert group["fallback_rate"]["F1"] == pytest.approx(0.5)
    assert group["fallback_conditioned_success"]["with_fallback"] == pytest.approx(1.0)
    assert group["fallback_conditioned_success"]["without_fallback"] == pytest.approx(0.0)
    # successful episodes are exactly the ones that fell back: that is contamination
    assert any("contaminated by fallback" in w for w in out["warnings"])


def test_summarise_records_flags_f3_termination() -> None:
    records = [_record(success=False, fallback={"F1": 0, "F2": 0, "F3": 2}) for _ in range(12)]
    out = summarise_records(records)
    assert any("F3 termination" in w for w in out["warnings"])
    assert out["groups"]["B1|val"]["fallback_rate"]["F3"] == pytest.approx(1.0)


def test_never_pool_test_splits_refuses_to_merge_sealed_splits() -> None:
    ok = never_pool_test_splits({"B3": ["test_combo"]})
    assert ok["B3"] == ["test_combo"]
    with pytest.raises(ValueError, match="must be reported separately"):
        never_pool_test_splits({"B3": ["test_combo", "test_range"]})


def test_summarise_records_refuses_a_grouping_that_merges_the_sealed_splits() -> None:
    records = [_record(split="test_combo"), _record(split="test_range")]
    with pytest.raises(ValueError, match="must be reported separately"):
        summarise_records(records, group_by=("method",))


def test_summarise_records_warns_on_small_groups() -> None:
    out = summarise_records([_record(), _record(success=False)])
    assert any("only 2 episodes" in w for w in out["warnings"])


def test_summarise_records_unknown_metric_and_missing_keys_fail_loud() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        summarise_records([_record()], metric="profit")
    bad = _record()
    bad.pop("split")
    with pytest.raises(KeyError, match="grouping keys"):
        summarise_records([bad])
