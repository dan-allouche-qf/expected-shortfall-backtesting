"""Unit tests for the unified backtest battery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from esback.backtests.battery import battery_dataframe, run_full_battery
from esback.backtests.es_baselines import StandardizedTSimulator
from esback.models.base import ModelFit
from esback.pit import compute_pit_t_std
from esback.risk_measures import es_t as _compute_es_t
from esback.risk_measures import var_t as _compute_var_t


def _toy_artifacts(n: int = 500, nu: float = 8.0, alpha: float = 0.05, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    mu = np.full(n, 0.05)
    sigma = np.full(n, 1.5)
    eps = rng.standard_t(df=nu, size=n) * np.sqrt((nu - 2) / nu)
    returns = mu + sigma * eps
    var_t = _compute_var_t(mu, sigma, alpha=alpha, nu=nu)
    es_t = _compute_es_t(mu, sigma, alpha=alpha, nu=nu)
    u_t = compute_pit_t_std(eps, nu)
    h_t = (-returns > var_t).astype(float)
    return {
        "u_t": u_t,
        "oos_returns": returns,
        "var_t": var_t,
        "es_t": es_t,
        "sigma_oos": sigma,
        "mu_oos": mu,
        "nu": nu,
        "h_t": h_t,
        "alpha": alpha,
    }


def test_battery_minimal_inputs_runs() -> None:
    """With just the PIT-side inputs, the battery still produces every test
    that does not depend on optional kwargs."""
    d = _toy_artifacts()
    results = run_full_battery(
        u_t=d["u_t"],
        oos_returns=d["oos_returns"],
        var_t=d["var_t"],
        es_t=d["es_t"],
        sigma_oos=d["sigma_oos"],
        h_t=d["h_t"],
        alpha_es=d["alpha"],
        alpha_var=d["alpha"],
    )
    expected_min_keys = {
        "U_ES",
        "C_ES(5)",
        "U_VaR",
        "C_VaR(5)",
        "Kupiec_POF",
        "Christoffersen_CCI",
        "Christoffersen_CC",
        "KLM_multinomial(4)",
        "AS_Z1",
        "AS_Z2",
        "McNeil_Frey",
    }
    assert expected_min_keys.issubset(results.keys())


def test_battery_with_simulator_yields_finite_as_p_values() -> None:
    d = _toy_artifacts()
    sim = StandardizedTSimulator(d["mu_oos"], d["sigma_oos"], d["nu"])
    results = run_full_battery(
        u_t=d["u_t"],
        oos_returns=d["oos_returns"],
        var_t=d["var_t"],
        es_t=d["es_t"],
        sigma_oos=d["sigma_oos"],
        h_t=d["h_t"],
        alpha_es=d["alpha"],
        alpha_var=d["alpha"],
        es_simulator=sim,
        n_sim=500,
    )
    assert np.isfinite(results["AS_Z1"].p_value)
    assert np.isfinite(results["AS_Z2"].p_value)


def test_battery_with_robust_inputs_includes_robust_tests() -> None:
    d = _toy_artifacts()
    fit = ModelFit(
        params={
            "Const": 0.05,
            "Return[1]": 0.0,
            "omega": 0.05,
            "alpha[1]": 0.08,
            "beta[1]": 0.90,
            "nu": d["nu"],
        },
        param_names=["Const", "Return[1]", "omega", "alpha[1]", "beta[1]", "nu"],
        param_cov=np.eye(6) * 1e-4,
        in_sample_n=300,
    )
    returns_all = pd.Series(
        np.concatenate([np.zeros(300), d["oos_returns"]]),
        index=pd.date_range("2020-01-01", periods=800, freq="B"),
    )
    results = run_full_battery(
        u_t=d["u_t"],
        oos_returns=d["oos_returns"],
        var_t=d["var_t"],
        es_t=d["es_t"],
        sigma_oos=d["sigma_oos"],
        h_t=d["h_t"],
        alpha_es=d["alpha"],
        alpha_var=d["alpha"],
        fit=fit,
        returns_all=returns_all,
        is_end_idx=300,
        oos_end_idx=799,
    )
    assert "MU_ES" in results
    assert "MC_ES(5)" in results


def test_battery_dataframe_returns_one_row_per_test() -> None:
    d = _toy_artifacts()
    results = run_full_battery(
        u_t=d["u_t"],
        oos_returns=d["oos_returns"],
        var_t=d["var_t"],
        es_t=d["es_t"],
        sigma_oos=d["sigma_oos"],
        h_t=d["h_t"],
        alpha_es=d["alpha"],
        alpha_var=d["alpha"],
    )
    df = battery_dataframe(results)
    assert len(df) == len(results)
    assert "decision" in df.columns
    assert "p_value" in df.columns


def test_battery_passes_klm_levels_through() -> None:
    d = _toy_artifacts()
    custom_levels = (0.01, 0.05, 0.10)
    results = run_full_battery(
        u_t=d["u_t"],
        oos_returns=d["oos_returns"],
        var_t=d["var_t"],
        es_t=d["es_t"],
        sigma_oos=d["sigma_oos"],
        h_t=d["h_t"],
        alpha_es=d["alpha"],
        alpha_var=d["alpha"],
        klm_levels=custom_levels,
    )
    assert "KLM_multinomial(3)" in results
