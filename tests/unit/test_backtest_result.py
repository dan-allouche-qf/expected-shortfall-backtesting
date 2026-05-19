"""Unit tests for the BacktestResult dataclass."""

from __future__ import annotations

import pytest

from esback.backtests import BacktestResult


def test_decision_rejects_below_level() -> None:
    r = BacktestResult(name="X", statistic=2.0, p_value=0.01, alpha=0.05, reference="paper")
    assert r.decision(level=0.05) == "REJECT"


def test_decision_does_not_reject_above_level() -> None:
    r = BacktestResult(name="X", statistic=0.5, p_value=0.20, alpha=0.05, reference="paper")
    assert r.decision(level=0.05) == "Fail to reject"


def test_decision_at_level_does_not_reject() -> None:
    """The convention is strict inequality: p == level should not reject."""
    r = BacktestResult(name="X", statistic=1.0, p_value=0.05, alpha=0.05, reference="paper")
    assert r.decision(level=0.05) == "Fail to reject"


def test_decision_returns_na_for_nan_pvalue() -> None:
    """A NaN p-value means the test was not conclusive (e.g. AS_Z1 / McNeil_Frey
    when no parametric simulator was supplied); decision() must surface that
    rather than silently classifying it as 'Fail to reject'.
    """
    result = BacktestResult(
        name="dummy",
        statistic=1.23,
        p_value=float("nan"),
        alpha=0.05,
        reference="test",
    )
    assert result.decision() == "N/A"
    assert result.decision(level=0.01) == "N/A"


def test_extra_defaults_to_empty_dict() -> None:
    r = BacktestResult(name="X", statistic=1.0, p_value=0.5, alpha=0.05, reference="paper")
    assert r.extra == {}


def test_result_is_frozen() -> None:
    r = BacktestResult(name="X", statistic=1.0, p_value=0.5, alpha=0.05, reference="paper")
    with pytest.raises(AttributeError):
        r.statistic = 2.0  # type: ignore[misc]
