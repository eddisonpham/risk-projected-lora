"""Numerical tolerances in one place, so property tests stay honest.

Every constant here has a reason; raising one to make a test pass is a protocol
change and must be logged in ``agents/research_log/LOG.md``.
"""

from __future__ import annotations

#: float32 round-off scale for exact closed-form comparisons.
EXACT = 1e-6
#: finite-difference vs analytic Jacobian (relative), with h=1e-5 in float64.
FD_REL = 5e-4
#: two implementations that must agree "to numerical precision".
MATCH = 1e-5
#: statistics: bootstrap CI width sanity, AUROC/ECE checks.
STATS = 1e-3
#: closed-form a0 recovery from (x_tau, v) in flow matching.
FLOW = 1e-6
#: MuJoCo state save/restore exactness (S2): bitwise for qpos/qvel, tight for mocap.
SIM_STATE = 1e-9
#: read-back of applied dynamics must be exact, not approximate.
DYN_READBACK = 1e-9
