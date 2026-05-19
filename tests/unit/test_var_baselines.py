"""Unit tests for VaR baselines."""

from __future__ import annotations

import numpy as np
import pytest

from esback.backtests.var_baselines import (
    christoffersen_cc,
    christoffersen_cci,
    kratz_lok_mcneil,
    kupiec_pof,
)


def test_kupiec_pof_zero_at_exact_violation_rate() -> None:
    """When pi_hat = alpha exactly, the LR statistic vanishes."""
    n = 1000
    alpha = 0.05
    h_t = np.zeros(n)
    h_t[: round(n * alpha)] = 1
    res = kupiec_pof(h_t, alpha)
    assert res.statistic == pytest.approx(0.0, abs=1e-10)


def test_kupiec_pof_size_under_h0() -> None:
    """Under Bernoulli(alpha), LR_uc is approximately chi2(1) and rejects at ~5%."""
    rng = np.random.default_rng(0)
    n_trials = 800
    n = 500
    alpha = 0.05
    rejects = 0
    for _ in range(n_trials):
        h = (rng.random(n) < alpha).astype(float)
        if not np.isnan(kupiec_pof(h, alpha).p_value) and kupiec_pof(h, alpha).p_value < 0.05:
            rejects += 1
    rate = rejects / n_trials
    assert 0.020 < rate < 0.085


def test_kupiec_pof_rejects_inflated_violations() -> None:
    """A 20% violation rate when alpha=5% must reject sharply."""
    n = 500
    alpha = 0.05
    h_t = (np.random.default_rng(1).random(n) < 0.20).astype(float)
    res = kupiec_pof(h_t, alpha)
    assert res.p_value < 1e-6


def test_kupiec_pof_returns_nan_for_zero_violations() -> None:
    res = kupiec_pof(np.zeros(100), alpha=0.05)
    assert np.isnan(res.statistic)
    assert np.isnan(res.p_value)


def test_christoffersen_cci_does_not_reject_iid_violations() -> None:
    """Independent Bernoulli violations should not be flagged."""
    rng = np.random.default_rng(2)
    h = (rng.random(2000) < 0.05).astype(float)
    res = christoffersen_cci(h, alpha=0.05)
    assert res.p_value > 0.05


def test_christoffersen_cci_rejects_clustered_violations() -> None:
    """Strongly clustered violations should reject the independence null."""
    n = 500
    h = np.zeros(n)
    # alternating clusters of size 5
    for k in range(10):
        h[10 * k : 10 * k + 5] = 1
    res = christoffersen_cci(h, alpha=0.05)
    assert res.p_value < 0.01


def test_christoffersen_cc_combines_uc_and_ind() -> None:
    """CC stat must equal UC stat + IND stat (numerically)."""
    rng = np.random.default_rng(3)
    h = (rng.random(1000) < 0.05).astype(float)
    uc = kupiec_pof(h, 0.05)
    ind = christoffersen_cci(h, 0.05)
    cc = christoffersen_cc(h, 0.05)
    assert cc.statistic == pytest.approx(uc.statistic + ind.statistic, rel=1e-12)


def test_kratz_lok_mcneil_size_under_h0() -> None:
    """Under U(0,1) PIT, KLM rejects at approximately 5%."""
    rng = np.random.default_rng(4)
    rejects = 0
    n_trials = 600
    n = 1000
    for _ in range(n_trials):
        u = rng.uniform(size=n)
        if kratz_lok_mcneil(u).p_value < 0.05:
            rejects += 1
    rate = rejects / n_trials
    assert 0.020 < rate < 0.085


def test_kratz_lok_mcneil_rejects_concentrated_lower_tail() -> None:
    """A PIT concentrated near 0 must reject."""
    rng = np.random.default_rng(5)
    u = rng.beta(0.5, 5, size=2000)
    res = kratz_lok_mcneil(u)
    assert res.p_value < 1e-6


def test_kratz_lok_mcneil_observed_counts_sum_to_n() -> None:
    rng = np.random.default_rng(6)
    u = rng.uniform(size=500)
    res = kratz_lok_mcneil(u)
    assert sum(res.extra["observed"]) == 500


def test_kratz_lok_mcneil_default_levels_have_5_bins() -> None:
    res = kratz_lok_mcneil(np.full(100, 0.5))
    assert len(res.extra["observed"]) == 5
    assert res.extra["df"] == 4
