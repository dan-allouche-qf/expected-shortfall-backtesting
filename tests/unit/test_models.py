"""Unit tests for esback.models.ar1_garch_t."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esback.models import ar1_garch_t
from esback.models.base import ModelFit


def _toy_params() -> dict[str, float]:
    """Persistent GARCH(1,1) close to typical equity calibration."""
    return {
        "Const": 0.05,
        "Return[1]": 0.10,
        "omega": 0.02,
        "alpha[1]": 0.08,
        "beta[1]": 0.90,
        "nu": 8.0,
    }


def test_volatility_path_seed_at_unconditional_variance() -> None:
    p = _toy_params()
    returns = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    sigma = ar1_garch_t.volatility_path(p, returns)
    expected_sigma2_0 = p["omega"] / (1 - p["alpha[1]"] - p["beta[1]"])
    assert sigma[0] == pytest.approx(np.sqrt(expected_sigma2_0))


def test_volatility_path_recursion_one_step() -> None:
    """Hand-checked GARCH(1,1) recursion for a 2-step series of zero returns."""
    p = _toy_params()
    returns = np.array([0.0, 0.0])
    sigma = ar1_garch_t.volatility_path(p, returns)

    sigma2_0 = p["omega"] / (1 - p["alpha[1]"] - p["beta[1]"])
    # at t=1: prev_lag = 0 (since t<2), mu_prev = const, eps_prev = -const
    eps_prev = -p["Const"]
    sigma2_1 = p["omega"] + p["alpha[1]"] * eps_prev**2 + p["beta[1]"] * sigma2_0
    assert sigma[1] == pytest.approx(np.sqrt(sigma2_1))


def test_volatility_path_length_matches_input() -> None:
    p = _toy_params()
    returns = np.linspace(-1, 1, 50)
    sigma = ar1_garch_t.volatility_path(p, returns)
    assert sigma.shape == returns.shape
    assert np.all(sigma > 0)


def test_forecast_oos_dimensions() -> None:
    p = _toy_params()
    rng = np.random.default_rng(42)
    n_total = 100
    returns_all = pd.Series(
        rng.standard_normal(n_total),
        index=pd.date_range("2020-01-01", periods=n_total, freq="B"),
    )
    is_end_idx = 70
    oos_end_idx = 99
    artifacts = ar1_garch_t.forecast_oos(p, returns_all, is_end_idx, oos_end_idx)

    n_oos = oos_end_idx - is_end_idx + 1
    assert artifacts.mu_oos.shape == (n_oos,)
    assert artifacts.sigma_oos.shape == (n_oos,)
    assert artifacts.eps_oos.shape == (n_oos,)
    assert artifacts.oos_returns.shape == (n_oos,)
    assert len(artifacts.oos_index) == n_oos


def test_forecast_oos_mu_uses_lag_one_returns() -> None:
    """mu_oos[k] should equal const + a0 * returns[is_end_idx - 1 + k]."""
    p = _toy_params()
    n_total = 30
    returns_vals = np.arange(n_total, dtype=float)
    returns_all = pd.Series(
        returns_vals, index=pd.date_range("2024-01-01", periods=n_total, freq="B")
    )
    is_end_idx = 20
    oos_end_idx = 29
    artifacts = ar1_garch_t.forecast_oos(p, returns_all, is_end_idx, oos_end_idx)

    expected_mu = p["Const"] + p["Return[1]"] * returns_vals[is_end_idx - 1 : oos_end_idx]
    np.testing.assert_allclose(artifacts.mu_oos, expected_mu, rtol=1e-12)


def test_forecast_oos_returns_match_slice() -> None:
    p = _toy_params()
    n_total = 40
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(n_total)
    returns_all = pd.Series(vals, index=pd.date_range("2020-01-01", periods=n_total, freq="B"))
    artifacts = ar1_garch_t.forecast_oos(p, returns_all, 25, 39)
    np.testing.assert_array_equal(artifacts.oos_returns, vals[25:40])


def test_fit_returns_model_fit_with_required_params() -> None:
    """End-to-end fit on a small simulated AR(1)-GARCH(1,1)-t series."""
    rng = np.random.default_rng(123)
    n = 800
    e = rng.standard_t(df=8.0, size=n) * np.sqrt((8.0 - 2) / 8.0)
    returns = np.zeros(n)
    sigma2 = np.full(n, 1.0)
    omega, alpha, beta = 0.05, 0.08, 0.90
    for t in range(1, n):
        sigma2[t] = omega + alpha * (returns[t - 1]) ** 2 + beta * sigma2[t - 1]
        returns[t] = 0.05 + 0.05 * returns[t - 1] + np.sqrt(sigma2[t]) * e[t]
    series = pd.Series(returns, index=pd.date_range("2020-01-01", periods=n, freq="B"))

    fit = ar1_garch_t.fit(series)
    assert isinstance(fit, ModelFit)
    for required in ("omega", "alpha[1]", "beta[1]", "nu"):
        assert required in fit.params
    assert fit.in_sample_n == n
    assert fit.param_cov.shape == (len(fit.param_names), len(fit.param_names))
