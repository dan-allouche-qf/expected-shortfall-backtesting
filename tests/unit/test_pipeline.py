"""Unit tests for esback.pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from esback.pipeline import PeriodArtifacts, run_period


def _toy_data(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    omega, alpha, beta = 0.05, 0.08, 0.90
    e = rng.standard_t(df=8.0, size=n) * np.sqrt((8 - 2) / 8)
    r = np.zeros(n)
    s2 = np.full(n, 1.0)
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = 0.05 + 0.05 * r[t - 1] + np.sqrt(s2[t]) * e[t]
    return pd.DataFrame({"Return": r}, index=pd.date_range("2018-01-01", periods=n, freq="B"))


def test_run_period_returns_period_artifacts() -> None:
    df = _toy_data()
    out = run_period(
        df,
        is_start=str(df.index[0].date()),
        is_end=str(df.index[400].date()),
        oos_end=str(df.index[-1].date()),
        label="Toy",
    )
    assert isinstance(out, PeriodArtifacts)


def test_run_period_dimensions_consistent() -> None:
    df = _toy_data()
    out = run_period(
        df,
        is_start=str(df.index[0].date()),
        is_end=str(df.index[400].date()),
        oos_end=str(df.index[-1].date()),
        label="Toy",
    )
    n_oos = len(out.u_t)
    assert out.var_t.shape == (n_oos,)
    assert out.es_t.shape == (n_oos,)
    assert out.h_t.shape == (n_oos,)
    assert out.H_t.shape == (n_oos,)
    assert len(out.forecast.oos_index) == n_oos


def test_run_period_battery_keys() -> None:
    df = _toy_data()
    out = run_period(
        df,
        is_start=str(df.index[0].date()),
        is_end=str(df.index[400].date()),
        oos_end=str(df.index[-1].date()),
        label="Toy",
    )
    assert set(out.battery.keys()) == {"U_ES", "C_ES(5)", "U_VaR", "C_VaR(5)"}
    assert set(out.robust.keys()) == {"MU_ES", "MC_ES(5)"}


def test_pit_residuals_in_unit_interval() -> None:
    df = _toy_data()
    out = run_period(
        df,
        is_start=str(df.index[0].date()),
        is_end=str(df.index[400].date()),
        oos_end=str(df.index[-1].date()),
        label="Toy",
    )
    assert out.u_t.min() >= 0.0
    assert out.u_t.max() <= 1.0
