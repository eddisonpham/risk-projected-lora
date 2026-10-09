"""Projection package: the operator, its backward-pass analysis and its fallback ladder."""

from __future__ import annotations

from riskproj.projection.backward import (
    finite_difference_jacobian,
    finite_difference_jacobian_apply,
    linearization_error,
    projected_jacobian,
    projected_jacobian_apply,
    risk_gradient,
)
from riskproj.projection.fallback import FallbackDecision, FallbackLadder
from riskproj.projection.operator import (
    EPS,
    active_at_flow_step,
    project,
    project_fn,
    setpoint,
    threshold_in_space,
)
from riskproj.types import FallbackCode, ProjectionInfo
from riskproj.utils.config import ProjCfg

__all__ = [
    "EPS",
    "FallbackCode",
    "FallbackDecision",
    "FallbackLadder",
    "ProjCfg",
    "ProjectionInfo",
    "active_at_flow_step",
    "finite_difference_jacobian",
    "finite_difference_jacobian_apply",
    "linearization_error",
    "project",
    "project_fn",
    "projected_jacobian",
    "projected_jacobian_apply",
    "risk_gradient",
    "setpoint",
    "threshold_in_space",
]
