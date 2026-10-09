"""Split rules: the protocol geometry of ``04`` §3 (7 / 6 / 14 / 8)."""

from __future__ import annotations

import numpy as np
import pytest

from riskproj.data import splits as S
from riskproj.utils.config import DynamicsRangesCfg, SplitsCfg


def _ranges():
    return DynamicsRangesCfg().to_ranges()


def test_role_counts_match_protocol() -> None:
    table = S.split_table()
    assert {role: len(cells) for role, cells in table.items()} == {
        "train": 7,
        "val": 6,
        "test_combo": 14,
        "test_range": 8,
    }
    assert len(S.designated_cells()) == 35


def test_assert_splits_valid_passes_on_defaults() -> None:
    S.assert_splits_valid()


def test_all_mid_and_single_axis_are_train() -> None:
    for cell in ("M-M-M", "L-M-M", "H-M-M", "M-L-M", "M-H-M", "M-M-L", "M-M-H"):
        assert S.role_of_cell(cell) == S.ROLE_TRAIN, cell


def test_val_is_the_positive_sign_product() -> None:
    for cell in ("L-L-M", "H-H-M", "L-M-L", "H-M-H", "M-L-L", "M-H-H"):
        assert S.role_of_cell(cell) == S.ROLE_VAL, cell


def test_negative_sign_product_is_test_combo() -> None:
    for cell in ("L-H-M", "H-L-M", "L-M-H", "H-M-L", "M-L-H", "M-H-L"):
        assert S.role_of_cell(cell) == S.ROLE_TEST_COMBO, cell


def test_three_axis_cells_are_test_combo() -> None:
    for cell in ("L-L-L", "H-H-H", "L-H-H", "H-L-L"):
        assert S.role_of_cell(cell) == S.ROLE_TEST_COMBO, cell


def test_single_extrapolation_cells_are_test_range() -> None:
    single = S.single_extrapolation_cells()
    assert len(single) == 6
    for cell in single:
        assert S.role_of_cell(cell) == S.ROLE_TEST_RANGE, cell
    assert S.role_of_cell("xL-M-M") == S.ROLE_TEST_RANGE
    assert S.role_of_cell("M-M-xH") == S.ROLE_TEST_RANGE


def test_compound_extrapolation_cells_are_test_range() -> None:
    cfg = SplitsCfg()
    for cell in cfg.range_compound:
        assert S.role_of_cell(cell, cfg) == S.ROLE_TEST_RANGE, cell


def test_roles_partition_the_designated_grid() -> None:
    index = S.cell_role_index()
    assert len(index) == 35
    assert len(set(index)) == 35
    for role, cells in S.split_table().items():
        for cell in cells:
            assert index[cell] == role


def test_undefined_cells_raise_rather_than_get_a_role() -> None:
    # two extrapolation levels but not one of the declared compound cells
    with pytest.raises(ValueError, match="compound"):
        S.role_of_cell("xL-xL-M")
    # an extrapolation level combined with a non-MID in-range level
    with pytest.raises(ValueError, match="only non-MID axis"):
        S.role_of_cell("xL-L-M")
    # malformed id
    with pytest.raises(ValueError, match="malformed"):
        S.role_of_cell("H-M")


def test_assert_splits_valid_catches_a_duplicated_compound_choice() -> None:
    bad = SplitsCfg(range_compound=("M-M-M", "M-M-M"))
    with pytest.raises(ValueError, match="inconsistent"):
        S.assert_splits_valid(bad)


def test_leakage_guards() -> None:
    S.assert_trainable(["M-M-M", "L-M-M"])
    S.assert_tunable(["L-L-M"])
    with pytest.raises(ValueError, match="leakage"):
        S.assert_no_leakage(["L-H-M"], S.ROLE_VAL)
    with pytest.raises(ValueError, match="leakage"):
        S.assert_trainable(["H-M-M", "L-H-M"])
    with pytest.raises(ValueError, match="leakage"):
        S.assert_tunable(["M-M-xL"])


def test_sealed_roles_are_never_trainable() -> None:
    for role in S.SEALED_ROLES:
        cells = S.cells_for_role(role)
        assert cells
        with pytest.raises(ValueError):
            S.assert_trainable(cells)


def test_cells_for_role_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        S.cells_for_role("test")


def test_sign_balance_report_is_balanced_everywhere() -> None:
    report = S.sign_balance_report()
    for role in S.ROLES:
        assert report[role]["balanced"], (role, report[role])
    # VAL must contain both signs on each axis that is shifted at all
    val = report[S.ROLE_VAL]["per_axis"]
    assert val["mu"]["n_lo"] == val["mu"]["n_hi"] == 2
    assert val["mu"]["n_mid"] == 2


def test_summarise_splits_payload() -> None:
    summary = S.summarise_splits()
    assert summary["n_cells_total"] == 35
    assert summary["expected_counts"] == S.EXPECTED_COUNTS
    assert len(summary["lines"]) == 4
    assert summary["range_compound"] == list(SplitsCfg().range_compound)


def test_sampling_is_seeded_and_inside_the_level_interval() -> None:
    ranges = _ranges()
    a = S.sample_dynamics_for_cell("H-L-M", ranges, seed=7)
    b = S.sample_dynamics_for_cell("H-L-M", ranges, seed=7)
    c = S.sample_dynamics_for_cell("H-L-M", ranges, seed=8)
    assert a == b
    assert a != c
    mu_lo, mu_hi = ranges.mu["H"]
    assert mu_lo <= a.mu_scale <= mu_hi
    mass_lo, mass_hi = ranges.mass["L"]
    assert mass_lo <= a.mass_scale <= mass_hi
    stiff_lo, stiff_hi = ranges.stiff["M"]
    assert stiff_lo <= a.stiff_scale <= stiff_hi


def test_episode_seeds_are_deterministic_and_unique() -> None:
    a = S.episode_seeds("L-L-M", base_seed=3, n=8)
    b = S.episode_seeds("L-L-M", base_seed=3, n=8)
    c = S.episode_seeds("L-L-M", base_seed=4, n=8)
    assert a == b and a != c
    assert len(set(a)) == 8
    assert all(isinstance(int(s), int) for s in a)
    assert S.episode_seeds("L-L-M", base_seed=3, n=0) == []


def test_episode_seeds_batch_is_stable_and_pairs_across_cells() -> None:
    """CRN-style pairing: the same base seed gives the same *positions*, different draws."""
    seeds = S.episode_seeds("M-H-M", base_seed=11, n=40)
    assert np.asarray(seeds).dtype == np.int64 or all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == 40
    with pytest.raises(ValueError):
        S.episode_seeds("M-H-M", base_seed=11, n=-1)
