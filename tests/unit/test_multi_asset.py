"""Unit tests for the multi-asset grid evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esback.multi_asset import (
    AssetSpec,
    PeriodSpec,
    _is_window,
    pivot_p_values,
    reject_rate_per_period,
    reject_rate_per_test,
    run_one,
)


def _toy_data(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.05, 0.08, 0.90
    e = rng.standard_t(df=8.0, size=n) * np.sqrt((8 - 2) / 8)
    r = np.zeros(n)
    s2 = np.full(n, 1.0)
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = 0.05 + 0.05 * r[t - 1] + np.sqrt(s2[t]) * e[t]
    return pd.DataFrame({"Return": r}, index=pd.date_range("2018-01-01", periods=n, freq="B"))


def test_is_window_subtracts_years_correctly() -> None:
    is_start, is_end = _is_window("2020-01-01", in_sample_years=5)
    assert is_start == "2015-01-01"
    assert is_end == "2019-12-31"


def test_run_one_produces_battery_and_n_oos() -> None:
    df = _toy_data(n=1500)
    is_start = str(df.index[0].date())
    is_end = str(df.index[1000].date())
    oos_start = str(df.index[1001].date())
    oos_end = str(df.index[-1].date())

    out = run_one(
        df,
        is_start=is_start,
        is_end=is_end,
        oos_start=oos_start,
        oos_end=oos_end,
        alpha_es=0.10,
        alpha_var=0.05,
        m=5,
        n_sim=100,
        n_bootstrap=100,
    )
    assert "battery" in out
    assert out["n_oos"] > 0
    assert "U_ES" in out["battery"]
    assert "Kupiec_POF" in out["battery"]
    assert "AS_Z2" in out["battery"]


def test_pivot_p_values_shape() -> None:
    grid = pd.DataFrame(
        {
            "asset": ["A", "A", "B", "B"],
            "period": ["P1", "P1", "P1", "P1"],
            "test": ["t1", "t2", "t1", "t2"],
            "p_value": [0.01, 0.20, 0.30, 0.04],
            "statistic": [0.0] * 4,
            "alpha": [0.05] * 4,
            "decision": ["REJECT", "Fail to reject", "Fail to reject", "REJECT"],
            "n_oos": [100] * 4,
        }
    )
    p = pivot_p_values(grid, "P1")
    assert p.shape == (2, 2)
    assert set(p.columns) == {"A", "B"}
    assert set(p.index) == {"t1", "t2"}


def test_reject_rate_per_test_returns_proportions() -> None:
    grid = pd.DataFrame(
        {
            "asset": ["A", "A", "B", "B"],
            "period": ["P1", "P1", "P1", "P1"],
            "test": ["t1", "t1", "t1", "t1"],
            "p_value": [0.01, 0.10, 0.04, 0.30],
            "statistic": [0.0] * 4,
            "alpha": [0.05] * 4,
            "decision": [""] * 4,
            "n_oos": [100] * 4,
        }
    )
    rates = reject_rate_per_test(grid, level=0.05)
    assert rates.loc["t1", "reject_rate"] == pytest.approx(0.5)


def test_reject_rate_per_period_returns_matrix() -> None:
    grid = pd.DataFrame(
        {
            "asset": ["A", "B", "A", "B"],
            "period": ["P1", "P1", "P2", "P2"],
            "test": ["t1", "t1", "t1", "t1"],
            "p_value": [0.01, 0.10, 0.04, 0.50],
            "statistic": [0.0] * 4,
            "alpha": [0.05] * 4,
            "decision": [""] * 4,
            "n_oos": [100] * 4,
        }
    )
    out = reject_rate_per_period(grid, level=0.05)
    assert out.loc["t1", "P1"] == pytest.approx(0.5)
    assert out.loc["t1", "P2"] == pytest.approx(0.5)


def test_specs_are_frozen_dataclasses() -> None:
    a = AssetSpec(name="X", ticker="^X")
    p = PeriodSpec(name="P", oos_start="2020-01-01", oos_end="2020-12-31")
    with pytest.raises(AttributeError):
        a.name = "Y"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        p.name = "Q"  # type: ignore[misc]
