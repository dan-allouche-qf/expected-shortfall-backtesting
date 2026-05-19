"""Numerical regression tests against committed reference outputs.

The reference outputs under ``tests/data/`` pin the deterministic part
of the pipeline at the fitted parameters of the AR(1)-GARCH(1,1)-t
model on the 2008 and COVID-19 reference periods. Replaying the
package at those parameters must recover the PIT residuals, the VaR
and ES paths, the violation arrays, and every backtest p-value to
within 1e-10 -- 1e-12. Pinning the parameters isolates these tests
from the optimiser, which would otherwise drift across arch versions.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from esback.backtests.du_escanciano import (
    conditional_es,
    conditional_var,
    es_severity,
    robust_conditional_es,
    robust_unconditional_es,
    unconditional_es,
    unconditional_var,
    var_violations,
)
from esback.config import TESTS_DATA_DIR
from esback.models import ar1_garch_t
from esback.models.base import ModelFit
from esback.pit import compute_pit_t_std, pit_from_theta
from esback.risk_measures import es_t, var_t


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    df = pd.read_parquet(TESTS_DATA_DIR / "sp500_returns_19970101_20241231.parquet")
    df.index = pd.to_datetime(df.index)
    return df


@pytest.fixture(scope="module")
def reference_summary() -> dict[str, Any]:
    return json.loads((TESTS_DATA_DIR / "reference_summary.json").read_text())


@pytest.fixture(scope="module")
def reference_days() -> dict[str, pd.DataFrame]:
    return {
        "p1": pd.read_parquet(TESTS_DATA_DIR / "reference_p1_days.parquet"),
        "p2": pd.read_parquet(TESTS_DATA_DIR / "reference_p2_days.parquet"),
    }


def _model_fit_from_summary(period_cfg: dict[str, Any]) -> ModelFit:
    return ModelFit(
        params=dict(period_cfg["params"]),
        param_names=list(period_cfg["param_names"]),
        param_cov=np.asarray(period_cfg["param_cov"], dtype=float),
        in_sample_n=int(period_cfg["is_end_idx"]),
    )


@pytest.mark.parametrize("period_key", ["p1", "p2"])
def test_pit_residuals_reproduce_reference(
    returns: pd.DataFrame,
    reference_summary: dict[str, Any],
    reference_days: dict[str, pd.DataFrame],
    period_key: str,
) -> None:
    cfg = reference_summary[period_key]
    expected = reference_days[period_key]

    mask_all = (returns.index >= cfg["is_start"]) & (returns.index <= cfg["oos_end"])
    returns_all = returns.loc[mask_all, "Return"]

    forecast = ar1_garch_t.forecast_oos(
        cfg["params"], returns_all, cfg["is_end_idx"], cfg["oos_end_idx"]
    )
    u_t = compute_pit_t_std(forecast.eps_oos, forecast.nu)

    np.testing.assert_allclose(u_t, expected["u_t"].to_numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        forecast.mu_oos, expected["mu_oos"].to_numpy(), rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        forecast.eps_oos, expected["eps_oos"].to_numpy(), rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        forecast.sigma_oos, expected["oos_sigma"].to_numpy(), rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("period_key", ["p1", "p2"])
def test_var_es_paths_reproduce_reference(
    returns: pd.DataFrame,
    reference_summary: dict[str, Any],
    reference_days: dict[str, pd.DataFrame],
    period_key: str,
) -> None:
    cfg = reference_summary[period_key]
    expected = reference_days[period_key]

    mask_all = (returns.index >= cfg["is_start"]) & (returns.index <= cfg["oos_end"])
    returns_all = returns.loc[mask_all, "Return"]
    forecast = ar1_garch_t.forecast_oos(
        cfg["params"], returns_all, cfg["is_end_idx"], cfg["oos_end_idx"]
    )

    var_path = var_t(forecast.mu_oos, forecast.sigma_oos, 0.05, forecast.nu)
    es_path = es_t(forecast.mu_oos, forecast.sigma_oos, 0.10, forecast.nu)

    np.testing.assert_allclose(var_path, expected["var_t"].to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(es_path, expected["es_t"].to_numpy(), rtol=1e-10)


@pytest.mark.parametrize("period_key", ["p1", "p2"])
def test_violations_reproduce_reference(
    returns: pd.DataFrame,
    reference_summary: dict[str, Any],
    reference_days: dict[str, pd.DataFrame],
    period_key: str,
) -> None:
    cfg = reference_summary[period_key]
    expected = reference_days[period_key]

    mask_all = (returns.index >= cfg["is_start"]) & (returns.index <= cfg["oos_end"])
    returns_all = returns.loc[mask_all, "Return"]
    forecast = ar1_garch_t.forecast_oos(
        cfg["params"], returns_all, cfg["is_end_idx"], cfg["oos_end_idx"]
    )
    u_t = compute_pit_t_std(forecast.eps_oos, forecast.nu)

    np.testing.assert_array_equal(var_violations(u_t, 0.05), expected["h_t"].to_numpy())
    np.testing.assert_allclose(es_severity(u_t, 0.10), expected["H_t"].to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("period_key", ["p1", "p2"])
def test_basic_backtests_reproduce_reference(
    returns: pd.DataFrame,
    reference_summary: dict[str, Any],
    period_key: str,
) -> None:
    cfg = reference_summary[period_key]
    basic = cfg["basic_010_005"]

    mask_all = (returns.index >= cfg["is_start"]) & (returns.index <= cfg["oos_end"])
    returns_all = returns.loc[mask_all, "Return"]
    forecast = ar1_garch_t.forecast_oos(
        cfg["params"], returns_all, cfg["is_end_idx"], cfg["oos_end_idx"]
    )
    u_t = compute_pit_t_std(forecast.eps_oos, forecast.nu)

    res_u_es = unconditional_es(u_t, 0.10)
    assert res_u_es.statistic == pytest.approx(basic["U_ES"], rel=1e-10)
    assert res_u_es.p_value == pytest.approx(basic["p_U_ES"], rel=1e-10, abs=1e-15)

    res_c_es = conditional_es(u_t, 0.10, m=5)
    assert res_c_es.statistic == pytest.approx(basic["C_ES"], rel=1e-10)
    assert res_c_es.p_value == pytest.approx(basic["p_C_ES"], rel=1e-10, abs=1e-15)

    res_u_var = unconditional_var(u_t, 0.05)
    assert res_u_var.statistic == pytest.approx(basic["U_VaR"], rel=1e-10)
    assert res_u_var.p_value == pytest.approx(basic["p_U_VaR"], rel=1e-10, abs=1e-15)

    res_c_var = conditional_var(u_t, 0.05, m=5)
    assert res_c_var.statistic == pytest.approx(basic["C_VaR"], rel=1e-10)
    assert res_c_var.p_value == pytest.approx(basic["p_C_VaR"], rel=1e-10, abs=1e-15)


@pytest.mark.parametrize("period_key", ["p1", "p2"])
def test_robust_backtests_reproduce_reference(
    returns: pd.DataFrame,
    reference_summary: dict[str, Any],
    period_key: str,
) -> None:
    cfg = reference_summary[period_key]

    mask_all = (returns.index >= cfg["is_start"]) & (returns.index <= cfg["oos_end"])
    returns_all = returns.loc[mask_all, "Return"]
    forecast = ar1_garch_t.forecast_oos(
        cfg["params"], returns_all, cfg["is_end_idx"], cfg["oos_end_idx"]
    )
    u_t = compute_pit_t_std(forecast.eps_oos, forecast.nu)

    fit = _model_fit_from_summary(cfg)
    mu_es = robust_unconditional_es(
        u_t, returns_all, fit, cfg["is_end_idx"], cfg["oos_end_idx"], 0.10
    )
    mc_es = robust_conditional_es(
        u_t, returns_all, fit, cfg["is_end_idx"], cfg["oos_end_idx"], 0.10, m=5
    )

    assert mu_es.statistic == pytest.approx(cfg["robust"]["MU_ES"], rel=1e-8)
    assert mu_es.p_value == pytest.approx(cfg["robust"]["p_MU_ES"], rel=1e-8, abs=1e-15)
    assert mc_es.statistic == pytest.approx(cfg["robust"]["MC_ES"], rel=1e-8)
    assert mc_es.p_value == pytest.approx(cfg["robust"]["p_MC_ES"], rel=1e-8, abs=1e-15)


@pytest.mark.parametrize("period_key", ["p1", "p2"])
def test_pit_from_theta_matches_forecast_oos_at_fitted_point(
    returns: pd.DataFrame,
    reference_summary: dict[str, Any],
    period_key: str,
) -> None:
    cfg = reference_summary[period_key]
    mask_all = (returns.index >= cfg["is_start"]) & (returns.index <= cfg["oos_end"])
    returns_all = returns.loc[mask_all, "Return"]

    theta_hat = np.array([cfg["params"][name] for name in cfg["param_names"]])
    u_via_theta = pit_from_theta(
        theta_hat, cfg["param_names"], returns_all, cfg["is_end_idx"], cfg["oos_end_idx"]
    )

    forecast = ar1_garch_t.forecast_oos(
        cfg["params"], returns_all, cfg["is_end_idx"], cfg["oos_end_idx"]
    )
    u_via_forecast = compute_pit_t_std(forecast.eps_oos, forecast.nu)

    np.testing.assert_allclose(u_via_theta, u_via_forecast, rtol=1e-12, atol=1e-15)
