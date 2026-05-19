"""Unit tests for esback.pit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from esback.pit import compute_pit_t_std, ks_test_uniform, pit_from_theta


def test_pit_at_zero_is_half() -> None:
    """A symmetric distribution evaluated at zero has CDF = 0.5."""
    u = compute_pit_t_std(np.array([0.0]), nu=8.0)
    np.testing.assert_allclose(u, 0.5, rtol=1e-12)


def test_pit_monotone_increasing() -> None:
    eps = np.linspace(-3, 3, 100)
    u = compute_pit_t_std(eps, nu=6.0)
    assert np.all(np.diff(u) > 0)


def test_pit_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    nu = 5.0
    eps = rng.standard_t(df=nu, size=10_000) * np.sqrt((nu - 2) / nu)
    u = compute_pit_t_std(eps, nu)
    assert u.min() >= 0.0
    assert u.max() <= 1.0


def test_pit_distribution_is_approximately_uniform() -> None:
    """If eps follows a standardized-t(nu), then u = PIT(eps) ~ U(0, 1)."""
    rng = np.random.default_rng(42)
    nu = 7.0
    n = 100_000
    eps = rng.standard_t(df=nu, size=n) * np.sqrt((nu - 2) / nu)
    u = compute_pit_t_std(eps, nu)
    assert abs(u.mean() - 0.5) < 0.005
    assert abs(u.var() - 1 / 12) < 0.005


def test_pit_symmetry_around_half() -> None:
    """For a symmetric innovation, F(-x) + F(x) = 1."""
    nu = 9.0
    eps = np.array([0.5, 1.0, 2.5, 3.0])
    u_pos = compute_pit_t_std(eps, nu)
    u_neg = compute_pit_t_std(-eps, nu)
    np.testing.assert_allclose(u_pos + u_neg, 1.0, rtol=1e-12)


def test_ks_test_returns_floats() -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(size=2000)
    stat, pval = ks_test_uniform(u)
    assert isinstance(stat, float)
    assert isinstance(pval, float)
    assert 0.0 <= stat <= 1.0
    assert 0.0 <= pval <= 1.0


def test_ks_test_does_not_reject_uniform() -> None:
    """A truly U(0,1) sample should not be rejected at 5%."""
    rng = np.random.default_rng(7)
    u = rng.uniform(size=5000)
    _, pval = ks_test_uniform(u)
    assert pval > 0.05


def test_ks_test_rejects_non_uniform() -> None:
    """A sample heavily concentrated near 0 should reject."""
    rng = np.random.default_rng(11)
    u = rng.beta(0.5, 5, size=2000)
    _, pval = ks_test_uniform(u)
    assert pval < 0.01


def test_pit_from_theta_returns_correct_length() -> None:
    """pit_from_theta should produce one value per OOS observation."""
    n_total = 100
    rng = np.random.default_rng(0)
    returns = pd.Series(
        rng.standard_normal(n_total),
        index=pd.date_range("2024-01-01", periods=n_total, freq="B"),
    )
    param_names = ["Const", "Return[1]", "omega", "alpha[1]", "beta[1]", "nu"]
    theta = np.array([0.05, 0.1, 0.02, 0.08, 0.9, 8.0])
    u = pit_from_theta(theta, param_names, returns, is_end_idx=70, oos_end_idx=99)
    assert u.shape == (30,)
    assert u.min() >= 0.0
    assert u.max() <= 1.0


def test_pit_from_theta_clamps_nu() -> None:
    """nu close to 2 must be clamped above 2 to keep sqrt(nu/(nu-2)) finite."""
    n_total = 50
    rng = np.random.default_rng(1)
    returns = pd.Series(
        rng.standard_normal(n_total),
        index=pd.date_range("2024-01-01", periods=n_total, freq="B"),
    )
    param_names = ["Const", "Return[1]", "omega", "alpha[1]", "beta[1]", "nu"]
    theta = np.array([0.0, 0.0, 0.05, 0.05, 0.9, 2.0])
    u = pit_from_theta(theta, param_names, returns, is_end_idx=30, oos_end_idx=49)
    assert np.all(np.isfinite(u))
