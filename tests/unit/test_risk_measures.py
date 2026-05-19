"""Unit tests for esback.risk_measures."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from esback.risk_measures import (
    es_multiplier_t_std,
    es_t,
    var_quantile_t_std,
    var_t,
)


def test_var_quantile_negative_for_lower_tail() -> None:
    """The 5% quantile of a zero-mean distribution is negative."""
    q = var_quantile_t_std(0.05, nu=8.0)
    assert q < 0


def test_es_multiplier_more_extreme_than_quantile() -> None:
    """E[e | e <= q] should be lower than q itself."""
    alpha = 0.05
    nu = 6.0
    q = var_quantile_t_std(alpha, nu)
    m = es_multiplier_t_std(alpha, nu)
    assert m < q


def test_var_t_loss_positive() -> None:
    mu = np.zeros(5)
    sigma = np.ones(5)
    v = var_t(mu, sigma, alpha=0.05, nu=8.0)
    assert np.all(v > 0)


def test_es_t_more_severe_than_var_t_at_same_alpha() -> None:
    mu = np.array([0.0])
    sigma = np.array([1.0])
    v = var_t(mu, sigma, alpha=0.05, nu=8.0)
    e = es_t(mu, sigma, alpha=0.05, nu=8.0)
    assert e[0] > v[0]


def test_var_t_scales_linearly_with_sigma() -> None:
    """At zero mean, VaR is proportional to sigma."""
    sigma = np.array([1.0, 2.0, 4.0])
    mu = np.zeros_like(sigma)
    v = var_t(mu, sigma, alpha=0.05, nu=10.0)
    np.testing.assert_allclose(v[1] / v[0], 2.0, rtol=1e-12)
    np.testing.assert_allclose(v[2] / v[0], 4.0, rtol=1e-12)


def test_var_t_shifts_with_mu() -> None:
    """A constant shift in mu shifts VaR by the same amount in the opposite direction."""
    sigma = np.array([1.0, 1.0])
    mu_zero = np.array([0.0, 0.0])
    mu_pos = np.array([0.5, 0.5])
    v0 = var_t(mu_zero, sigma, alpha=0.05, nu=10.0)
    v1 = var_t(mu_pos, sigma, alpha=0.05, nu=10.0)
    np.testing.assert_allclose(v0 - v1, 0.5, rtol=1e-12)


def test_es_multiplier_converges_to_normal_limit() -> None:
    """As nu -> infinity, the standardized-t becomes a standard normal,
    so the ES multiplier converges to -phi(z_alpha) / alpha."""
    alpha = 0.05
    z_alpha = float(stats.norm.ppf(alpha))
    expected = -float(stats.norm.pdf(z_alpha)) / alpha
    actual = es_multiplier_t_std(alpha, nu=500.0)
    assert actual == pytest.approx(expected, rel=1e-3)


def test_var_quantile_converges_to_normal_limit() -> None:
    alpha = 0.05
    expected = float(stats.norm.ppf(alpha))
    actual = var_quantile_t_std(alpha, nu=500.0)
    assert actual == pytest.approx(expected, rel=1e-3)


def test_var_quantile_caches_repeated_calls() -> None:
    var_quantile_t_std.cache_clear()
    var_quantile_t_std(0.05, 8.0)
    var_quantile_t_std(0.05, 8.0)
    info = var_quantile_t_std.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_es_multiplier_caches_repeated_calls() -> None:
    es_multiplier_t_std.cache_clear()
    es_multiplier_t_std(0.10, 8.0)
    es_multiplier_t_std(0.10, 8.0)
    info = es_multiplier_t_std.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_es_multiplier_decreases_with_alpha() -> None:
    """Smaller alpha pushes the conditional expectation deeper into the tail."""
    nu = 8.0
    m_small = es_multiplier_t_std(0.01, nu)
    m_large = es_multiplier_t_std(0.10, nu)
    assert m_small < m_large
