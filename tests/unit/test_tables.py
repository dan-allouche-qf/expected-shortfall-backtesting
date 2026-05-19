"""Unit tests for esback.reporting.tables."""

from __future__ import annotations

from esback.backtests.base import BacktestResult
from esback.reporting.tables import (
    battery_summary,
    battery_to_dataframe,
    comparison_table,
)


def _result(name: str, p: float) -> BacktestResult:
    return BacktestResult(
        name=name,
        statistic=1.0,
        p_value=p,
        alpha=0.10,
        reference="ref",
    )


def test_battery_to_dataframe_columns_and_index() -> None:
    battery = {"U_ES": _result("U_ES", 0.01), "C_ES(5)": _result("C_ES(5)", 0.20)}
    df = battery_to_dataframe(battery)
    assert df.index.name == "name"
    assert set(df.columns) == {"statistic", "p_value", "alpha", "decision", "reference"}
    assert list(df.index) == ["U_ES", "C_ES(5)"]


def test_decision_column_reflects_p_value() -> None:
    battery = {"R": _result("R", 0.01), "F": _result("F", 0.50)}
    df = battery_to_dataframe(battery, decision_level=0.05)
    assert df.loc["R", "decision"] == "REJECT"
    assert df.loc["F", "decision"] == "Fail to reject"


def test_decision_level_threshold_is_strict() -> None:
    """p == level should not reject (matches BacktestResult.decision)."""
    battery = {"X": _result("X", 0.05)}
    df = battery_to_dataframe(battery, decision_level=0.05)
    assert df.loc["X", "decision"] == "Fail to reject"


def test_battery_summary_omits_reference() -> None:
    battery = {"X": _result("X", 0.01)}
    df = battery_summary(battery)
    assert "reference" not in df.columns
    assert "decision" in df.columns


def test_comparison_table_multiindex_columns() -> None:
    b1 = {"U_ES": _result("U_ES", 0.01), "C_ES(5)": _result("C_ES(5)", 0.30)}
    b2 = {"U_ES": _result("U_ES", 0.50), "C_ES(5)": _result("C_ES(5)", 0.04)}
    df = comparison_table({"P1": b1, "P2": b2})
    assert ("P1", "p_value") in df.columns
    assert ("P2", "p_value") in df.columns
    assert df.shape == (2, 8)
