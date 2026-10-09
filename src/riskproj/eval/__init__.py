"""Evaluation layer: rollouts, counterfactuals, diagnostics, systems, reports.

Everything here consumes **archived logs**, never live models, so any table can be
recomputed from a run directory. ``stats`` and ``critic_eval`` are imported
defensively because they are written in parallel with the rest of this package: a
missing module must not make the diagnostics, the anti-gaming checks or the report
writer unavailable, and it must not be silently faked either — a table that needs a
missing statistic reports ``available: false`` with the reason.
"""

from __future__ import annotations

from riskproj.eval.anti_gaming import (
    action_statistics,
    do_nothing_baseline,
    fallback_conditioned_success,
    suppression_report,
    success_violation_pareto,
)
from riskproj.eval.counterfactual import (
    CounterfactualRecord,
    paired_counterfactual,
    pairwise_summary,
)
from riskproj.eval.diagnostics import (
    decision_frame,
    depth_sweep_table,
    linearization_error_table,
    per_cell_heatmap,
    pooled_violation_rates,
    split_of_cell,
)
from riskproj.eval.report import (
    DEFAULT_MPCD,
    build_tables,
    claim_ladder_rung,
    load_run,
    write_report,
)
from riskproj.eval.rollout import (
    BestOfNPlanner,
    ChunkPlanner,
    LatencyTracker,
    NoProjectionPlanner,
    PlannedChunk,
    RolloutOutput,
    RolloutSpec,
    build_planner,
    cell_episode_plan,
    evaluate_method,
    latency_tracker,
    run_episode,
)
from riskproj.eval.systems import (
    SystemsCost,
    check_budget,
    measure_inference_cost,
    measure_training_cost,
    systems_table,
)

#: Metric identifiers every evaluator must be able to emit (``09`` §1).
METRIC_IDS: tuple[str, ...] = (
    "success",
    "violation",
    "fallback",
    "steps",
    "correction_rate",
    "unusable_rate",
    "linearization_error",
    "latency_ms",
    "peak_vram_mb",
)

_STATS_NAMES = (
    "MPCD_DEFAULT",
    "holm_primary",
    "iqm_with_ci",
    "mcnemar_paired",
    "never_pool_test_splits",
    "paired_episode_delta",
    "stratified_bootstrap_ci",
    "summarise_records",
    "wilson",
)
_CRITIC_EVAL_NAMES = ("critic_table", "proxy_gold_curve", "reliability_diagram_data")

__all__ = [
    "DEFAULT_MPCD",
    "METRIC_IDS",
    "BestOfNPlanner",
    "ChunkPlanner",
    "CounterfactualRecord",
    "LatencyTracker",
    "NoProjectionPlanner",
    "PlannedChunk",
    "RolloutOutput",
    "RolloutSpec",
    "SystemsCost",
    "action_statistics",
    "build_planner",
    "build_tables",
    "cell_episode_plan",
    "check_budget",
    "claim_ladder_rung",
    "decision_frame",
    "depth_sweep_table",
    "do_nothing_baseline",
    "evaluate_method",
    "fallback_conditioned_success",
    "latency_tracker",
    "linearization_error_table",
    "load_run",
    "measure_inference_cost",
    "measure_training_cost",
    "pairwise_summary",
    "paired_counterfactual",
    "per_cell_heatmap",
    "pooled_violation_rates",
    "run_episode",
    "split_of_cell",
    "success_violation_pareto",
    "suppression_report",
    "systems_table",
    "write_report",
]


def _statistics_available() -> dict[str, list[str]]:
    """Which parallel-authored names are importable right now (for tests/diagnostics)."""
    available: dict[str, list[str]] = {"stats": [], "critic_eval": []}
    try:
        import riskproj.eval.stats as stats

        available["stats"] = sorted(n for n in _STATS_NAMES if hasattr(stats, n))
    except Exception:  # pragma: no cover - module written in parallel
        pass
    try:
        import riskproj.eval.critic_eval as critic_eval

        available["critic_eval"] = sorted(n for n in _CRITIC_EVAL_NAMES if hasattr(critic_eval, n))
    except Exception:  # pragma: no cover
        pass
    return available
