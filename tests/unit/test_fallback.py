"""Unit tests for the fallback ladder (``01`` §3, ``07`` §4, CLAUDE.md #4).

The ladder exists so that an unusable gradient is *visible*: F1 re-checks the base
action, F2 holds briefly, F3 terminates and counts as a failure. Nothing may pass
through uncorrected and unlogged.
"""

from __future__ import annotations

import pytest
import torch

from riskproj.projection import FallbackCode, FallbackLadder

pytestmark = pytest.mark.unit


def _actions(b: int = 2, h: int = 3, a: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(b, h, a), torch.randn(b, h, a)


# --------------------------------------------------------------------------
# no fallback needed
# --------------------------------------------------------------------------


def test_no_fallback_when_every_gradient_is_usable() -> None:
    cand, base = _actions()
    ladder = FallbackLadder(window=3)
    decision = ladder.update(
        usable=torch.ones(2, dtype=torch.bool),
        action_candidate=cand,
        base_action=base,
        base_action_risk=torch.full((2,), 0.9),
        threshold=0.5,
    )
    assert decision.code == FallbackCode.NONE
    assert torch.equal(decision.action, cand)
    assert ladder.counts() == {"none": 2, "F1": 0, "F2": 0, "F3": 0}
    assert len(ladder.events()) == 1
    assert "no fallback" in ladder.events()[0]["reason"]


# --------------------------------------------------------------------------
# F1: the base action is acceptable
# --------------------------------------------------------------------------


def test_f1_uses_the_base_action_when_the_critic_passes_it() -> None:
    cand, base = _actions()
    ladder = FallbackLadder(window=3)
    decision = ladder.update(
        usable=torch.tensor([False, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=torch.tensor([0.1, 0.2]),
        threshold=0.5,
    )
    assert decision.code == FallbackCode.F1_BASE_ACTION
    assert torch.equal(decision.action, base)
    assert ladder.counts()["F1"] == 2
    assert (decision.per_example == FallbackCode.F1_BASE_ACTION).all()


def test_non_finite_base_risk_is_not_accepted_as_f1() -> None:
    cand, base = _actions()
    ladder = FallbackLadder(window=1)
    decision = ladder.update(
        usable=torch.tensor([False, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=torch.tensor([float("nan"), float("inf")]),
        threshold=0.5,
    )
    assert decision.code in (FallbackCode.F2_HOLD, FallbackCode.F3_TERMINATE)


# --------------------------------------------------------------------------
# F2 then F3: the hold budget is finite
# --------------------------------------------------------------------------


def test_f2_holds_the_last_action_then_escalates_to_f3() -> None:
    cand, base = _actions()
    ladder = FallbackLadder(window=2)
    worst_risk = torch.full((2,), 0.99)
    ladder.note_action(base * 10.0)

    d1 = ladder.update(
        usable=torch.tensor([False, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=worst_risk,
        threshold=0.5,
    )
    assert d1.code == FallbackCode.F2_HOLD
    assert torch.equal(d1.action, base * 10.0), "F2 must replay the last executed action"
    assert d1.steps_held == 1

    d2 = ladder.update(
        usable=torch.tensor([False, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=worst_risk,
        threshold=0.5,
    )
    assert d2.code == FallbackCode.F2_HOLD and d2.steps_held == 2

    d3 = ladder.update(
        usable=torch.tensor([False, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=worst_risk,
        threshold=0.5,
    )
    assert d3.code == FallbackCode.F3_TERMINATE, "F2 may not hold forever"
    # counts are per example: two rows held twice, then two rows terminated once
    assert ladder.counts()["F2"] == 4 and ladder.counts()["F3"] == 2
    assert any("F3" in e["reason"] for e in ladder.events())


def test_f2_without_a_previous_action_holds_the_base_action_not_the_candidate() -> None:
    cand, base = _actions()
    ladder = FallbackLadder(window=3)
    decision = ladder.update(
        usable=torch.tensor([False, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=torch.full((2,), 0.9),
        threshold=0.5,
    )
    assert decision.code == FallbackCode.F2_HOLD
    assert torch.equal(decision.action, base), "the risky candidate must not be re-issued"


def test_missing_base_risk_makes_f1_unavailable_but_the_event_says_so() -> None:
    cand, base = _actions()
    ladder = FallbackLadder(window=3)
    decision = ladder.update(
        usable=torch.tensor([False, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=None,
        threshold=0.5,
    )
    assert decision.code == FallbackCode.F2_HOLD
    assert "F1 unavailable" in ladder.events()[-1]["reason"]


# --------------------------------------------------------------------------
# mixed batches, disabled ladder, reporting
# --------------------------------------------------------------------------


def test_mixed_batch_keeps_per_example_codes() -> None:
    # A three-row batch: row 0 usable, row 1 passable by the critic (F1), row 2 not (F2).
    cand, base = _actions(b=3)
    ladder = FallbackLadder(window=3)
    decision = ladder.update(
        usable=torch.tensor([True, False, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=torch.tensor([0.9, 0.1, 0.9]),
        threshold=0.5,
    )
    assert decision.per_example.tolist() == [
        FallbackCode.NONE,
        FallbackCode.F1_BASE_ACTION,
        FallbackCode.F2_HOLD,
    ]
    assert decision.code == FallbackCode.F2_HOLD
    assert decision.details["n_needing"] == 2
    assert ladder.counts() == {"none": 1, "F1": 1, "F2": 1, "F3": 0}
    # only the F1 row is replaced by the base action
    assert torch.equal(decision.action[0], cand[0])
    assert torch.equal(decision.action[1], base[1])


def test_disabled_ladder_is_loud_not_silent() -> None:
    cand, base = _actions()
    ladder = FallbackLadder(window=3, enabled=False)
    decision = ladder.update(
        usable=torch.tensor([False, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=torch.full((2,), 0.1),
        threshold=0.5,
    )
    assert decision.code == FallbackCode.NONE
    assert ladder.counts()["none"] == 2
    event = ladder.events()[-1]
    assert "disabled" in event["reason"] and event["n_examples"] == 2


def test_summary_reports_rates_and_counts() -> None:
    cand, base = _actions()
    ladder = FallbackLadder(window=2)
    ladder.update(
        usable=torch.tensor([True, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=torch.tensor([0.9, 0.1]),
        threshold=0.5,
    )
    s = ladder.summary()
    assert s["fallback_F1"] == 1.0 and s["fallback_none"] == 1.0
    assert s["fallback_rate_F1"] == pytest.approx(0.5)
    assert s["fallback_n_consultations"] == 1.0


def test_reset_clears_hold_state_but_not_the_log() -> None:
    cand, base = _actions()
    ladder = FallbackLadder(window=2)
    ladder.update(
        usable=torch.tensor([False, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=torch.full((2,), 0.9),
        threshold=0.5,
    )
    n_events = len(ladder.events())
    ladder.reset()
    d = ladder.update(
        usable=torch.tensor([False, False]),
        action_candidate=cand,
        base_action=base,
        base_action_risk=torch.full((2,), 0.9),
        threshold=0.5,
    )
    assert d.code == FallbackCode.F2_HOLD and d.steps_held == 1
    assert len(ladder.events()) == n_events + 1


# --------------------------------------------------------------------------
# misuse
# --------------------------------------------------------------------------


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError):
        FallbackLadder(window=0)


def test_batch_mismatch_raises() -> None:
    cand, base = _actions(b=3)
    ladder = FallbackLadder(window=2)
    with pytest.raises(ValueError):
        ladder.update(
            usable=torch.zeros(2, dtype=torch.bool),
            action_candidate=cand,
            base_action=base,
            base_action_risk=torch.zeros(2),
            threshold=0.5,
        )


def test_usable_must_be_one_dimensional() -> None:
    cand, base = _actions()
    ladder = FallbackLadder(window=2)
    with pytest.raises(ValueError):
        ladder.update(
            usable=torch.zeros(2, 1, dtype=torch.bool),
            action_candidate=cand,
            base_action=base,
            base_action_risk=torch.zeros(2),
            threshold=0.5,
        )
