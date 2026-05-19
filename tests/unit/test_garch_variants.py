"""Unit tests for GARCH variants and multi-model dispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esback.models import garch_variants


def _ar1_garch_series(n: int = 800, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.05, 0.08, 0.90
    e = rng.standard_t(df=8.0, size=n) * np.sqrt((8 - 2) / 8)
    r = np.zeros(n)
    s2 = np.full(n, 1.0)
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = 0.05 + 0.05 * r[t - 1] + np.sqrt(s2[t]) * e[t]
    return pd.Series(r, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_fit_variant_garch_t() -> None:
    s = _ar1_garch_series()
    fit = garch_variants.fit_variant(s, vol="GARCH", dist="t")
    for k in ("omega", "alpha[1]", "beta[1]", "nu"):
        assert k in fit.params
    assert fit.in_sample_n == len(s)


def test_fit_variant_gjr_t_introduces_gamma() -> None:
    s = _ar1_garch_series()
    fit = garch_variants.fit_variant(s, vol="GJR", dist="t")
    assert "gamma[1]" in fit.params


def test_fit_variant_garch_normal_has_no_nu() -> None:
    s = _ar1_garch_series()
    fit = garch_variants.fit_variant(s, vol="GARCH", dist="normal")
    assert "nu" not in fit.params


def test_fit_variant_unknown_vol_raises() -> None:
    s = _ar1_garch_series()
    with pytest.raises(ValueError, match="Unsupported volatility"):
        garch_variants.fit_variant(s, vol="FIGARCH", dist="t")  # type: ignore[arg-type]


def test_fit_variant_unknown_dist_raises() -> None:
    s = _ar1_garch_series()
    with pytest.raises(ValueError, match="Unsupported innovation"):
        garch_variants.fit_variant(s, vol="GARCH", dist="ged")  # type: ignore[arg-type]


def test_volatility_path_gjr_matches_garch_when_no_leverage() -> None:
    """With ``gamma = 0`` the GJR recursion collapses to plain GARCH."""
    params_garch = {
        "Const": 0.0,
        "Return[1]": 0.0,
        "omega": 0.05,
        "alpha[1]": 0.08,
        "beta[1]": 0.90,
    }
    params_gjr = {**params_garch, "gamma[1]": 0.0}
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(200)
    sigma_garch = garch_variants.volatility_path_garch(params_garch, returns)
    sigma_gjr = garch_variants.volatility_path_gjr(params_gjr, returns)
    np.testing.assert_allclose(sigma_garch, sigma_gjr, rtol=1e-12)


def test_volatility_path_gjr_diverges_when_leverage_positive() -> None:
    """With ``gamma > 0`` and a negative shock, GJR sigma grows faster."""
    params_garch = {
        "Const": 0.0,
        "Return[1]": 0.0,
        "omega": 0.05,
        "alpha[1]": 0.05,
        "beta[1]": 0.90,
    }
    params_gjr = {**params_garch, "gamma[1]": 0.10}
    returns = np.array([0.0, -2.0, 0.0, 0.0, 0.0])
    sigma_garch = garch_variants.volatility_path_garch(params_garch, returns)
    sigma_gjr = garch_variants.volatility_path_gjr(params_gjr, returns)
    # After the negative shock, GJR sigma is strictly above GARCH sigma.
    assert sigma_gjr[2] > sigma_garch[2]


def test_forecast_oos_variant_garch_t() -> None:
    s = _ar1_garch_series(n=600)
    fit = garch_variants.fit_variant(s.iloc[:400], vol="GARCH", dist="t")
    variant = garch_variants.forecast_oos_variant(
        fit.params, s, is_end_idx=400, oos_end_idx=599, vol="GARCH", dist="t"
    )
    assert variant.vol == "GARCH"
    assert variant.dist == "t"
    assert np.isfinite(variant.forecast.nu)
    assert variant.forecast.mu_oos.shape == (200,)


def test_forecast_oos_variant_normal_has_nan_nu() -> None:
    s = _ar1_garch_series(n=400)
    fit = garch_variants.fit_variant(s.iloc[:300], vol="GARCH", dist="normal")
    variant = garch_variants.forecast_oos_variant(
        fit.params, s, is_end_idx=300, oos_end_idx=399, vol="GARCH", dist="normal"
    )
    assert np.isnan(variant.forecast.nu)
