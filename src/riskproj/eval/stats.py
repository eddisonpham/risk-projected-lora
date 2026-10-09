"""Aggregate statistics for the evaluation suite (``agents/09_evaluation_suite.md`` §2).

Three rules from the spec are enforced here rather than left to discipline:

1. **The replication unit is the training seed** (``08`` §4), never the episode.
   Episode-level CIs are reported, but aggregate intervals resample *runs*; an
   episode bootstrap over one trained policy answers a different question.
2. **Only pre-registered primary comparisons get multiplicity correction**
   (:func:`holm_primary`). Everything else comes back marked exploratory and
   unadjusted, so an exploratory p-value cannot drift into a headline number.
3. **TEST-combo and TEST-range are never pooled** (:func:`never_pool_test_splits`,
   ``09`` §2). ``summarise_records`` calls that guard, so a grouping that tries to
   merge them raises instead of quietly reporting an average.

The estimators themselves come from :mod:`riskproj.utils.metrics` (AUROC, AUPRC,
ECE, Wilson, McNemar, Fisher, Holm, IQM, bootstrap) and are re-exported so an
analyser needs one import.

``rliable`` is used when this installation actually exposes an entry point we can
call; otherwise the auditable numpy implementation runs. Either way the result
records which one produced the number (``"used_rliable"``), because "which library
computed this CI" is part of the provenance of a reported figure.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from riskproj.data.splits import (
    ROLE_TEST_COMBO,
    ROLE_TEST_RANGE,
    ROLE_TRAIN,
    ROLE_VAL,
)
from riskproj.types import VIOLATION_TYPES
from riskproj.utils.metrics import (
    auprc,
    auroc,
    bootstrap_ci,
    ece,
    fisher_exact,
    fnr_at_threshold,
    holm_bonferroni,
    iqm,
    mcnemar_exact,
    precision_recall_at_threshold,
    reliability_curve,
    two_proportion_ci_diff,
    wilson_interval,
)

#: Minimum practically meaningful difference, fixed after the pilot and before
#: TEST (``09`` §2, ``11`` G7). A placeholder until the prereg tag is cut.
MPCD_DEFAULT: float = 0.05
ALPHA_DEFAULT: float = 0.05

#: The unit of replication for every aggregate interval (``08`` §4).
REPLICATION_UNIT: str = "train_seed"

#: Pre-registered primary comparisons; only these get multiplicity correction.
PRIMARY_COMPARISONS: tuple[str, ...] = ("B3_vs_B1", "B3_vs_B2", "B3_vs_B4a")

#: Splits that must be reported separately and never merged into one number.
POOLED_FORBIDDEN_SPLITS: tuple[str, ...] = (ROLE_TEST_COMBO, ROLE_TEST_RANGE)
#: Every split group a report keeps distinct (``09`` §2): the tuning split, the
#: in-distribution split, and the two sealed TEST splits each answer a different
#: question, and averaging them lets one hide behind another.
SEPARATE_SPLIT_GROUPS: tuple[str, ...] = (
    ROLE_TRAIN,
    ROLE_VAL,
    ROLE_TEST_COMBO,
    ROLE_TEST_RANGE,
)

#: Key holding the pooled (all-strata) estimate in :func:`stratified_bootstrap_ci`.
ALL_KEY = "__all__"

#: ``wilson`` is the name the analysis scripts use; same function.
wilson = wilson_interval


# --------------------------------------------------------------------------
# small numerical helpers
# --------------------------------------------------------------------------


def _arr(values: Any, *, name: str = "values") -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return arr.reshape(-1)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_quantile(p: float) -> float:
    """Standard-normal quantile by bisection on :func:`_norm_cdf`.

    Bisection rather than a closed-form approximation: it is monotone, has no
    approximation error to justify in a write-up, and costs microseconds.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1); got {p}")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _rliable_entry_point(name: str) -> Callable[..., Any] | None:
    """Return a callable from ``rliable`` if this install exposes it, else ``None``.

    Probed defensively: the numpy path below is the one the test suite exercises, so
    an absent, renamed or differently-shaped rliable entry point degrades to an
    auditable implementation instead of failing an analysis at report time.
    """
    try:
        from rliable import library as rl_library  # type: ignore[import-not-found]
        from rliable import metrics as rl_metrics  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - depends on the installed extras
        return None
    for module in (rl_library, rl_metrics):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn  # type: ignore[no-any-return]
    return None


def rliable_available() -> bool:
    """Whether an rliable entry point we can call is importable."""
    return _rliable_entry_point("get_interval_estimates") is not None


# --------------------------------------------------------------------------
# stratified / aggregate intervals
# --------------------------------------------------------------------------


def stratified_bootstrap_ci(
    values: Mapping[str, Sequence[float]],
    *,
    n_boot: int = 10_000,
    alpha: float = ALPHA_DEFAULT,
    seed: int = 0,
    statistic: str = "mean",
) -> dict[str, dict[str, float]]:
    """Bootstrap CIs per stratum (cell) plus a pooled estimate over all strata.

    Strata are cells, and resampling happens *within* each stratum, so a pooled
    interval reflects cell heterogeneity instead of pretending the episodes were
    drawn from one population (``09`` §2: report per-cell, then aggregate).

    Returns ``{cell: {point, lo, hi, n}}`` plus an :data:`ALL_KEY` entry.
    """
    if not values:
        raise ValueError("no strata supplied")
    fn = np.mean if statistic == "mean" else np.median
    rng = np.random.default_rng(seed)

    out: dict[str, dict[str, float]] = {}
    pooled_parts: list[np.ndarray] = []
    pooled_values: list[np.ndarray] = []
    for name, raw in values.items():
        arr = _arr(raw, name=f"values[{name}]")
        if arr.size == 0:
            raise ValueError(f"stratum {name!r} is empty")
        idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
        samples = fn(arr[idx], axis=1)
        lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[str(name)] = {
            "point": float(fn(arr)),
            "lo": float(lo),
            "hi": float(hi),
            "n": float(arr.size),
        }
        pooled_parts.append(idx)
        pooled_values.append(arr)

    # Column blocks of the concatenated index matrix correspond to strata, so
    # concatenating per-stratum draws is *exactly* stratified resampling and needs
    # no python loop over bootstrap replicates.
    pooled_idx = np.concatenate(pooled_parts, axis=1)
    pooled = np.concatenate(pooled_values)
    if pooled_idx.shape[1] != pooled.size:
        raise RuntimeError("internal error: pooled index width does not match the pooled sample")
    pooled_samples = fn(pooled[pooled_idx], axis=1)
    lo, hi = np.percentile(pooled_samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    out[ALL_KEY] = {
        "point": float(fn(pooled)),
        "lo": float(lo),
        "hi": float(hi),
        "n": float(pooled.size),
    }
    return out


def iqm_with_ci(
    per_run_values: Sequence[float],
    *,
    n_boot: int = 10_000,
    alpha: float = ALPHA_DEFAULT,
    seed: int = 0,
) -> dict[str, float]:
    """Interquartile mean over runs, with a bootstrap CI (``09`` §2, rliable).

    ``per_run_values`` are per-*training-seed* aggregates (one number per run), which
    is the replication unit aggregate ``rliable`` recommends reporting instead of a
    mean over episodes.
    """
    arr = _arr(per_run_values, name="per_run_values")
    if arr.size == 0:
        raise ValueError("no run values supplied")
    ci = bootstrap_ci(arr, n_boot=n_boot, alpha=alpha, statistic="mean", seed=seed)
    implementation = "numpy"
    point = float(iqm(arr))
    lo, hi = float(ci["lo"]), float(ci["hi"])

    entry = _rliable_entry_point("get_interval_estimates")
    if entry is not None:  # pragma: no cover - exercised only with the stats extra
        try:
            estimates = entry({"run": arr.reshape(1, -1)}, iqm, reps=n_boot)
            candidate_point = float(np.asarray(estimates[0]).reshape(-1)[0])
            interval = np.asarray(estimates[1], dtype=float).reshape(-1)
            if np.isfinite(candidate_point) and interval.size == 2:
                point = candidate_point
                lo, hi = float(interval[0]), float(interval[1])
                implementation = "rliable"
        except Exception:  # pragma: no cover - any API mismatch falls back
            implementation = "numpy"

    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "n": float(arr.size),
        "alpha": float(alpha),
        # 1.0 = rliable produced this interval, 0.0 = the numpy path did. Kept float
        # so the whole result serialises; a report should state which one it was.
        "used_rliable": 1.0 if implementation == "rliable" else 0.0,
    }


def performance_profile(
    per_run_values: Mapping[str, Sequence[float]],
    *,
    thresholds: Sequence[float] | None = None,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Empirical performance profiles: fraction of runs above each threshold.

    Implemented in numpy: the profile is a step function of the run values, and the
    definition (``P(value >= threshold)`` over runs, averaged over cells when a
    method has several cells) is short enough to audit directly. ``rliable`` is used
    for the IQM aggregate in :func:`iqm_with_ci`, which is where its CI machinery
    actually adds something.
    """
    if not per_run_values:
        raise ValueError("no methods supplied")
    flat: dict[str, np.ndarray] = {str(k): _arr(v, name=f"{k}") for k, v in per_run_values.items()}
    if thresholds is None:
        allv = np.concatenate(list(flat.values())) if flat else np.zeros(0)
        thresholds = np.linspace(float(allv.min()), float(allv.max()), 25)
    th = np.asarray(thresholds, dtype=float).reshape(-1)
    profiles = {
        name: [float(np.mean(v >= t if higher_is_better else v <= t)) for t in th]
        for name, v in flat.items()
    }
    return {
        "thresholds": [float(t) for t in th],
        "profiles": profiles,
        "n_runs": {name: float(v.size) for name, v in flat.items()},
        "higher_is_better": higher_is_better,
        "implementation": "numpy (performance profiles are a step function of run values)",
        "used_rliable": rliable_available(),
    }


def probability_of_improvement(
    a: Sequence[float], b: Sequence[float], *, n_boot: int = 10_000, seed: int = 0
) -> float:
    """``P(a > b)`` with ties counted as half, by bootstrap over paired runs."""
    aa, bb = _arr(a, name="a"), _arr(b, name="b")
    if aa.size != bb.size:
        raise ValueError(
            f"probability_of_improvement needs paired runs: {aa.size} vs {bb.size} "
            "(the replication unit is the training seed, 08 §4)"
        )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, aa.size, size=(n_boot, aa.size))
    diff = aa[idx] - bb[idx]
    return float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))


# --------------------------------------------------------------------------
# paired tests
# --------------------------------------------------------------------------


def _binary(x: np.ndarray, *, name: str) -> np.ndarray:
    vals = np.unique(x)
    if not np.all(np.isin(vals, (0.0, 1.0))):
        raise ValueError(f"{name} must be binary 0/1 for a paired test; got {vals[:5]}")
    return (x == 1.0).astype(float)


def paired_episode_delta(
    a: Sequence[float],
    b: Sequence[float],
    *,
    metric: str = "success",
    n_boot: int = 10_000,
    alpha: float = ALPHA_DEFAULT,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap CI of ``a - b`` plus discordant-pair counts.

    Pairing is only meaningful when the two arms saw the same states and the same
    flow noise (CRN, ``08`` §4); a length mismatch is therefore returned as an
    explicit ``note`` rather than quietly truncated into a fake pairing.
    """
    aa, bb = _arr(a, name="a"), _arr(b, name="b")
    note = ""
    if aa.size != bb.size:
        n = min(aa.size, bb.size)
        note = (
            f"pairing lengths differ ({aa.size} vs {bb.size}); paired on the first {n} "
            "entries — valid only if those entries are CRN-aligned (08 §4)"
        )
        aa, bb = aa[:n], bb[:n]
    if aa.size == 0:
        raise ValueError("no paired observations")
    diff = aa - bb
    ci = bootstrap_ci(diff, n_boot=n_boot, alpha=alpha, seed=seed)
    out: dict[str, Any] = {
        "point": float(ci["point"]),
        "lo": float(ci["lo"]),
        "hi": float(ci["hi"]),
        "n": float(ci["n"]),
        "metric": metric,
        "alpha": float(alpha),
        "excludes_zero": float(bool(ci["lo"] > 0.0 or ci["hi"] < 0.0)),
        "note": note,
    }
    if metric in ("success", "violation", "any_violation", "failure"):
        ba, bbin = _binary(aa, name="a"), _binary(bb, name="b")
        b_only = int(np.sum((ba == 1.0) & (bbin == 0.0)))
        c_only = int(np.sum((ba == 0.0) & (bbin == 1.0)))
        out["b_a_only"] = float(b_only)
        out["c_b_only"] = float(c_only)
        out["n_discordant"] = float(b_only + c_only)
        out["discordant_rate"] = float((b_only + c_only) / ba.size)
        out["mcnemar_p"] = float(mcnemar_exact(b_only, c_only)["p_value"])
    return out


def mcnemar_paired(a_success: Sequence[Any], b_success: Sequence[Any]) -> dict[str, float]:
    """Exact McNemar test on two paired binary outcome vectors (``09`` §2)."""
    aa = _binary(_arr(a_success, name="a_success"), name="a_success")
    bb = _binary(_arr(b_success, name="b_success"), name="b_success")
    if aa.size != bb.size:
        raise ValueError(f"paired test needs equal lengths; got {aa.size} and {bb.size}")
    b_only = int(np.sum((aa == 1.0) & (bb == 0.0)))
    c_only = int(np.sum((aa == 0.0) & (bb == 1.0)))
    res = mcnemar_exact(b_only, c_only)
    return {
        "b_a_only": float(b_only),
        "c_b_only": float(c_only),
        "n_discordant": float(res["n_discordant"]),
        "p_value": float(res["p_value"]),
        "odds_ratio": float(res["odds_ratio"]),
    }


def fisher_unpaired(
    a_success: int, a_n: int, b_success: int, b_n: int
) -> dict[str, float]:
    """Two-sided Fisher exact test for two *unpaired* arms (FACT's choice, ``09`` §2)."""
    for name, val in (("a_n", a_n), ("b_n", b_n)):
        if val < 0:
            raise ValueError(f"{name} must be >= 0")
    if a_success > a_n or b_success > b_n or min(a_success, b_success) < 0:
        raise ValueError("successes must lie in [0, n] for both arms")
    res = fisher_exact(a_success, a_n - a_success, b_success, b_n - b_success)
    return {
        "odds_ratio": float(res["odds_ratio"]),
        "p_value": float(res["p_value"]),
        "rate_a": float(a_success / a_n) if a_n else float("nan"),
        "rate_b": float(b_success / b_n) if b_n else float("nan"),
    }


# --------------------------------------------------------------------------
# multiplicity
# --------------------------------------------------------------------------


def holm_primary(
    p_values: Mapping[str, float],
    *,
    alpha: float = ALPHA_DEFAULT,
    primary: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Holm-Bonferroni over the pre-registered primaries only (``09`` §2).

    Entries outside ``primary`` are returned **unadjusted** and flagged with
    ``exploratory = 1.0`` (the return type is float-valued so the dict serialises
    straight to JSON; 1.0 means exploratory, 0.0 means primary). A caller that wants
    a corrected p-value for an exploratory comparison must say so explicitly, and
    that decision belongs in the log.
    """
    names = set(primary) if primary is not None else set(PRIMARY_COMPARISONS)
    primaries = {k: float(v) for k, v in p_values.items() if k in names}
    adjusted = holm_bonferroni(primaries, alpha=alpha) if primaries else {}
    out: dict[str, dict[str, float]] = {}
    for name, p in p_values.items():
        if name in adjusted:
            out[name] = {**adjusted[name], "exploratory": 0.0}
        else:
            out[name] = {
                "p": float(p),
                "threshold": float("nan"),
                "reject": float("nan"),
                "exploratory": 1.0,
            }
    return out


# --------------------------------------------------------------------------
# power / sample size (pre-registration, 09 §2)
# --------------------------------------------------------------------------


def required_n_for_mpcd(
    discordant_rate: float,
    mpcd: float = MPCD_DEFAULT,
    *,
    alpha: float = ALPHA_DEFAULT,
    power: float = 0.8,
) -> dict[str, float]:
    """Pairs per arm needed to detect ``mpcd`` in a paired binary comparison.

    McNemar form (Connor 1987): with discordant rate ``p_d`` and effect
    ``delta = p_b - p_c``, the test statistic is ``(b-c)/sqrt(b+c)``, so

        power = Phi( delta * sqrt(n / p_d) - z_{1-alpha/2} )
        n      = p_d * (z_{1-alpha/2} + z_{1-power})^{2} / delta^{2}

    Note the direction, which is not the intuitive one: **more discordance means
    more noise**, so n grows with ``p_d`` (and so does the number of discordant
    pairs ``n * p_d``). Discordance is variance, not signal — which is why the
    pilot's measured discordant rate feeds straight into this formula rather than
    being treated as a power bonus.
    """
    if not 0.0 < discordant_rate <= 1.0:
        raise ValueError(f"discordant_rate must be in (0,1]; got {discordant_rate}")
    if mpcd <= 0.0:
        raise ValueError(f"mpcd must be > 0; got {mpcd}")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must lie in (0,1)")
    if mpcd > discordant_rate:
        # delta cannot exceed the discordant rate: |p_b - p_c| <= p_b + p_c = p_d.
        raise ValueError(
            f"mpcd {mpcd} exceeds the discordant rate {discordant_rate}; a paired design "
            "cannot show a difference larger than its discordance"
        )
    z_alpha = normal_quantile(1.0 - alpha / 2.0)
    z_power = normal_quantile(power)
    n_per_arm = discordant_rate * (z_alpha + z_power) ** 2 / mpcd**2
    n_pairs = int(math.ceil(n_per_arm))
    return {
        "n_per_arm": float(n_pairs),
        "n_discordant_needed": float(math.ceil(n_pairs * discordant_rate)),
        "discordant_rate": float(discordant_rate),
        "mpcd": float(mpcd),
        "alpha": float(alpha),
        "power": float(power),
        "z_alpha": float(z_alpha),
        "z_power": float(z_power),
        "formula": 1.0,  # marker: McNemar/Connor, paired binary outcome
    }


def power_for_n(
    n: int,
    discordant_rate: float,
    *,
    effect: float = MPCD_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
) -> float:
    """Power of a paired design of ``n`` pairs to detect ``effect``."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if not 0.0 < discordant_rate <= 1.0:
        raise ValueError(f"discordant_rate must be in (0,1]; got {discordant_rate}")
    z_alpha = normal_quantile(1.0 - alpha / 2.0)
    return float(_norm_cdf(effect * math.sqrt(n / discordant_rate) - z_alpha))


def detectable_difference(
    n: int,
    discordant_rate: float,
    *,
    alpha: float = ALPHA_DEFAULT,
    power: float = 0.8,
) -> float:
    """Smallest effect a paired design of ``n`` pairs can detect at ``power``."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if not 0.0 < discordant_rate <= 1.0:
        raise ValueError(f"discordant_rate must be in (0,1]; got {discordant_rate}")
    z_alpha = normal_quantile(1.0 - alpha / 2.0)
    z_power = normal_quantile(power)
    return float(math.sqrt(discordant_rate / n) * (z_alpha + z_power))


# --------------------------------------------------------------------------
# episode-log summary
# --------------------------------------------------------------------------


def _violation_fired(record: Mapping[str, Any], vtype: str) -> bool:
    viol = record.get("violations") or {}
    if not isinstance(viol, Mapping):
        raise TypeError(f"record['violations'] must be a mapping; got {type(viol).__name__}")
    entry = viol.get(vtype)
    if isinstance(entry, Mapping):
        return float(entry.get("count", 0.0) or 0.0) > 0.0
    if isinstance(entry, bool):
        return entry
    return bool(entry)


def any_violation(record: Mapping[str, Any]) -> bool:
    return any(_violation_fired(record, v) for v in VIOLATION_TYPES)


def _episode_value(record: Mapping[str, Any], metric: str) -> float:
    if metric == "success":
        return 1.0 if record.get("success") else 0.0
    if metric in ("violation", "any_violation"):
        return 1.0 if any_violation(record) else 0.0
    if metric == "failure":
        return 0.0 if record.get("success") else 1.0
    raise ValueError(f"unknown metric {metric!r}; expected success | any_violation | failure")


def _group_key(record: Mapping[str, Any], group_by: Sequence[str]) -> str:
    missing = [k for k in group_by if k not in record]
    if missing:
        raise KeyError(
            f"record is missing grouping keys {missing}; got {sorted(record)} "
            "(the schema is agents/09 §3)"
        )
    return "|".join(str(record[k]) for k in group_by)


def _splits_of(value: Any) -> set[str]:
    """Best-effort extraction of the split names a group covers."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        for key in ("splits", "split"):
            if key in value:
                inner = value[key]
                return {str(inner)} if isinstance(inner, str) else {str(s) for s in inner}
        return set()
    out: set[str] = set()
    for item in value:
        if isinstance(item, Mapping):
            if "split" in item:
                out.add(str(item["split"]))
        else:
            out.add(str(item))
    return out


def never_pool_test_splits(groups: Mapping[str, Any]) -> dict[str, list[str]]:
    """Report the splits each group covers, raising if sealed splits are merged.

    ``09`` §2: TEST-combo and TEST-range are reported separately and never pooled,
    because they answer different questions (compositional generalisation versus
    extrapolation) and averaging them lets one hide behind the other.

    Returns ``{group: [splits]}`` so the caller can assert the separation happened;
    :data:`SEPARATE_SPLIT_GROUPS` lists the groups that must stay distinct.
    """
    out: dict[str, list[str]] = {}
    for name, value in groups.items():
        splits = sorted(_splits_of(value))
        out[str(name)] = splits
        sealed = [s for s in splits if s in POOLED_FORBIDDEN_SPLITS]
        if len(sealed) > 1:
            raise ValueError(
                f"group {name!r} pools {sealed}; {ROLE_TEST_COMBO} and {ROLE_TEST_RANGE} "
                "must be reported separately (09 §2)"
            )
    return out


def summarise_records(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str = "success",
    group_by: Sequence[str] = ("method", "split"),
) -> dict[str, Any]:
    """Summarise archived episode records (``09`` §3) per group, with CIs and guards.

    Always reports success **alongside** violations, fallback rates and stall rates
    (``09`` §5): a success number without its fallback rate is not interpretable,
    because F2/F3 fallback inflates success by making the policy idle rather than
    competent.

    ``fallback_conditioned_success`` splits the success rate by whether any fallback
    fired, which is the quickest way to see the failure mode the spec warns about.
    Problems are returned in ``"warnings"`` rather than raised, so the table is still
    produced and the analyst sees the numbers next to the caveat; the one thing that
    *does* raise is an attempt to pool the sealed splits, which
    :func:`never_pool_test_splits` refuses.
    """
    if not records:
        raise ValueError("no records to summarise")
    group_by = tuple(group_by)
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        buckets.setdefault(_group_key(rec, group_by), []).append(rec)

    # Refuse a grouping that merges the sealed splits: with group_by=("method",) a
    # group would cover val + test_combo + test_range, which 09 §2 forbids.
    never_pool_test_splits(
        {
            name: sorted({str(r.get("split", "unknown")) for r in recs})
            for name, recs in buckets.items()
        }
    )

    warnings: list[str] = []
    groups: dict[str, Any] = {}
    for name, recs in sorted(buckets.items()):
        n = len(recs)
        values = np.asarray([_episode_value(r, metric) for r in recs], dtype=float)
        successes = int(np.sum([1.0 if r.get("success") else 0.0 for r in recs]))
        ws = wilson_interval(successes, n)
        falls = [r.get("fallback") or {} for r in recs]
        fallback_rates = {
            code: float(np.mean([1.0 if float(f.get(code, 0) or 0) > 0 else 0.0 for f in falls]))
            for code in ("F1", "F2", "F3")
        }
        any_fb = np.asarray(
            [1.0 if any(float(f.get(c, 0) or 0) > 0 for c in ("F1", "F2", "F3")) else 0.0 for f in falls]
        )
        success_flag = np.asarray([1.0 if r.get("success") else 0.0 for r in recs], dtype=float)
        fallback_conditioned = {
            "with_fallback": float(success_flag[any_fb == 1.0].mean())
            if np.any(any_fb == 1.0)
            else float("nan"),
            "without_fallback": float(success_flag[any_fb == 0.0].mean())
            if np.any(any_fb == 0.0)
            else float("nan"),
            "n_with_fallback": float(np.sum(any_fb == 1.0)),
        }
        per_type = {
            v: float(np.mean([1.0 if _violation_fired(r, v) else 0.0 for r in recs]))
            for v in VIOLATION_TYPES
        }
        stall = float(np.mean([1.0 if float(r.get("stall_steps", 0) or 0) > 0 else 0.0 for r in recs]))
        group = {
            "n": float(n),
            "metric": metric,
            "metric_rate": float(values.mean()),
            "success_rate": float(successes / n),
            "success_ci": {"lo": float(ws[0]), "hi": float(ws[1])},
            "any_violation_rate": float(
                np.mean([1.0 if any_violation(r) else 0.0 for r in recs])
            ),
            "per_type_violation_rate": per_type,
            "fallback_rate": fallback_rates,
            "fallback_conditioned_success": fallback_conditioned,
            "stall_rate": stall,
            "mean_latency_ms": float(
                np.mean([float((r.get("latency_ms") or {}).get("policy", 0.0) or 0.0) for r in recs])
            ),
            "mean_peak_vram_mb": float(
                np.mean([float(r.get("peak_vram_mb", 0.0) or 0.0) for r in recs])
            ),
        }
        groups[name] = group

        if fallback_rates["F3"] > 0.0:
            warnings.append(
                f"{name}: F3 termination in {fallback_rates['F3']:.1%} of episodes — F3 is a "
                "failure, never a safe outcome, and it must be reported next to success (09 §5)"
            )
        if fallback_rates["F1"] + fallback_rates["F2"] > 0.30:
            warnings.append(
                f"{name}: fallback rate {fallback_rates['F1'] + fallback_rates['F2']:.1%} exceeds "
                "the G6 budget of 30%; gains under this condition are not adaptation gains"
            )
        wf, wo = fallback_conditioned["with_fallback"], fallback_conditioned["without_fallback"]
        if math.isfinite(wf) and math.isfinite(wo) and wf >= wo and fallback_conditioned["n_with_fallback"] > 0:
            warnings.append(
                f"{name}: episodes that fell back succeeded at {wf:.1%} versus {wo:.1%} without — "
                "the success signal is contaminated by fallback (09 §5)"
            )
        if stall > 0.25:
            warnings.append(
                f"{name}: {stall:.1%} of episodes stalled; a stalled policy can show low violation "
                "rates for the wrong reason (do-nothing check, 09 §1 anti_gaming)"
            )
        if n < 10:
            warnings.append(f"{name}: only {n} episodes — report the CI, not the point estimate")

    return {
        "metric": metric,
        "group_by": list(group_by),
        "replication_unit": REPLICATION_UNIT,
        "n_records": len(records),
        "n_groups": len(groups),
        "groups": groups,
        "warnings": warnings,
    }


__all__ = [
    "ALPHA_DEFAULT",
    "ALL_KEY",
    "MPCD_DEFAULT",
    "POOLED_FORBIDDEN_SPLITS",
    "PRIMARY_COMPARISONS",
    "REPLICATION_UNIT",
    "SEPARATE_SPLIT_GROUPS",
    "any_violation",
    "auprc",
    "auroc",
    "bootstrap_ci",
    "detectable_difference",
    "ece",
    "fisher_exact",
    "fisher_unpaired",
    "fnr_at_threshold",
    "holm_bonferroni",
    "holm_primary",
    "iqm",
    "iqm_with_ci",
    "mcnemar_exact",
    "mcnemar_paired",
    "never_pool_test_splits",
    "normal_quantile",
    "paired_episode_delta",
    "performance_profile",
    "power_for_n",
    "precision_recall_at_threshold",
    "probability_of_improvement",
    "reliability_curve",
    "required_n_for_mpcd",
    "rliable_available",
    "stratified_bootstrap_ci",
    "summarise_records",
    "two_proportion_ci_diff",
    "wilson",
    "wilson_interval",
]
