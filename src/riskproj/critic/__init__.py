"""Risk critic: deployable-input ensemble, training, calibration, G3 gate.

The critic is a **proxy**. It is never a success metric and never evidence of
physical validity (``06`` §7); simulator outcomes decide.
"""

from __future__ import annotations

from riskproj.critic.calibrate import (
    apply_temperature,
    choose_tau,
    ece_before_after,
    fit_temperature,
    fnr_curve,
    reliability,
)
from riskproj.critic.features import (
    FEATURE_LAYOUT_VERSION,
    CriticFeatureBuilder,
    FeatureSpec,
    flatten_context_torch,
)
from riskproj.critic.gates import (
    GATE_G3,
    GateReport,
    evaluate_g3,
    gate_status_report,
    load_gate_reports,
)
from riskproj.critic.model import (
    AGGREGATORS,
    ANY_VIOLATION_COL,
    N_HEADS,
    RiskCritic,
    aggregate_members,
    make_risk_fn,
)
from riskproj.critic.train import (
    CriticSample,
    CriticSampleSource,
    CriticTrainConfig,
    assert_frozen,
    freeze,
    gradient_diagnostics,
    labels_from_records,
    train_critic,
)
from riskproj.utils.config import CriticCfg
from riskproj.utils.registry import register


@register("critic", "mlp_ensemble")
def build_mlp_ensemble(cfg: CriticCfg, spec: FeatureSpec) -> RiskCritic:
    """Registry entry point: ``build("critic", "mlp_ensemble", cfg, spec)``."""
    return RiskCritic(cfg, spec)


__all__ = [
    "AGGREGATORS",
    "ANY_VIOLATION_COL",
    "FEATURE_LAYOUT_VERSION",
    "GATE_G3",
    "N_HEADS",
    "CriticFeatureBuilder",
    "CriticSample",
    "CriticSampleSource",
    "CriticTrainConfig",
    "FeatureSpec",
    "GateReport",
    "RiskCritic",
    "aggregate_members",
    "apply_temperature",
    "assert_frozen",
    "build_mlp_ensemble",
    "choose_tau",
    "ece_before_after",
    "evaluate_g3",
    "fit_temperature",
    "flatten_context_torch",
    "fnr_curve",
    "freeze",
    "gate_status_report",
    "gradient_diagnostics",
    "labels_from_records",
    "load_gate_reports",
    "make_risk_fn",
    "reliability",
    "train_critic",
]
