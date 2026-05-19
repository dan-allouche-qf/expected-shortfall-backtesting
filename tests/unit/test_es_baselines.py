"""Unit tests for ES baselines."""

from __future__ import annotations

import numpy as np
import pytest

from esback.backtests.es_baselines import (
    StandardizedTSimulator,
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    mcneil_frey,
)
from esback.risk_measures import es_t as _compute_es_t
from esback.risk_measures import var_t as _compute_var_t


def _toy_oos(n: int = 500, nu: float = 8.0, seed: int = 0, alpha: float = 0.05) -> dict:
    """Generate OOS data with the correct VaR/ES under standardised-t(nu).

    Both VaR and ES are evaluated at the same ``alpha`` because the
    Acerbi-Szekely and McNeil-Frey tests assume that convention; using
    different alpha levels would introduce a deterministic non-zero
    bias that would break the size interpretation under H0.
    """
    rng = np.random.default_rng(seed)
    mu = np.full(n, 0.05)
    sigma = np.full(n, 1.5)
    eps = rng.standard_t(df=nu, size=n) * np.sqrt((nu - 2) / nu)
    returns = mu + sigma * eps
    var_t = _compute_var_t(mu, sigma, alpha=alpha, nu=nu)
    es_t = _compute_es_t(mu, sigma, alpha=alpha, nu=nu)
    return {
        "returns": returns,
        "var_t": var_t,
        "es_t": es_t,
        "mu": mu,
        "sigma": sigma,
        "nu": nu,
        "alpha": alpha,
    }


def test_simulator_returns_correct_shape() -> None:
    sim = StandardizedTSimulator(mu_oos=np.zeros(100), sigma_oos=np.ones(100), nu=8.0)
    rng = np.random.default_rng(0)
    out = sim.simulate_returns(50, rng)
    assert out.shape == (50, 100)


def test_simulator_reproducible_with_seed() -> None:
    sim = StandardizedTSimulator(np.zeros(50), np.ones(50), nu=8.0)
    a = sim.simulate_returns(10, np.random.default_rng(1))
    b = sim.simulate_returns(10, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


def test_z1_returns_nan_with_no_violations() -> None:
    n = 500
    res = acerbi_szekely_z1(
        oos_returns=np.full(n, 0.0),
        var_t=np.full(n, 100.0),  # huge VaR -> no violations
        es_t=np.full(n, 100.0),
    )
    assert np.isnan(res.statistic)
    assert np.isnan(res.p_value)


def test_z2_finite_even_without_violations() -> None:
    n = 500
    res = acerbi_szekely_z2(
        oos_returns=np.full(n, 0.0),
        var_t=np.full(n, 100.0),
        es_t=np.full(n, 100.0),
        alpha=0.05,
    )
    assert np.isfinite(res.statistic)


def test_z1_size_under_h0_with_simulator() -> None:
    """When the simulator matches the data DGP, Z1's MC p-value should be
    approximately uniform under H0 -- so rejection at 5% is near 5%."""
    rng = np.random.default_rng(123)
    n_trials = 200
    rejects = 0
    for trial in range(n_trials):
        d = _toy_oos(n=300, nu=8.0, seed=trial)
        sim = StandardizedTSimulator(d["mu"], d["sigma"], d["nu"])
        res = acerbi_szekely_z1(
            d["returns"],
            d["var_t"],
            d["es_t"],
            simulator=sim,
            n_sim=400,
            seed=int(rng.integers(0, 10**9)),
        )
        if not np.isnan(res.p_value) and res.p_value < 0.05:
            rejects += 1
    rate = rejects / n_trials
    assert 0.005 < rate < 0.15


def test_z2_size_under_h0_with_simulator() -> None:
    rng = np.random.default_rng(456)
    n_trials = 200
    rejects = 0
    for trial in range(n_trials):
        d = _toy_oos(n=300, nu=8.0, seed=trial)
        sim = StandardizedTSimulator(d["mu"], d["sigma"], d["nu"])
        res = acerbi_szekely_z2(
            d["returns"],
            d["var_t"],
            d["es_t"],
            alpha=0.05,
            simulator=sim,
            n_sim=400,
            seed=int(rng.integers(0, 10**9)),
        )
        if res.p_value < 0.05:
            rejects += 1
    rate = rejects / n_trials
    assert 0.005 < rate < 0.15


def test_z1_rejects_when_tails_heavier_than_claimed() -> None:
    """Real heavy-tailed data fitted by a light-tailed model: Z1 should reject.

    With ``nu_real = 3.5`` and ``nu_claimed = 100`` the conditional ES
    under the heavy-tailed reality exceeds the light-tailed claim by
    enough that, with n = 3000 OOS observations, Z1 sits comfortably
    above zero and the right-tail Monte-Carlo p-value is well below 5%.
    """
    rng = np.random.default_rng(7)
    n = 3000
    nu_real, nu_claimed = 3.5, 100.0
    mu = np.full(n, 0.0)
    sigma = np.full(n, 1.0)
    eps_real = rng.standard_t(df=nu_real, size=n) * np.sqrt((nu_real - 2) / nu_real)
    real_returns = mu + sigma * eps_real

    var_claimed = _compute_var_t(mu, sigma, alpha=0.05, nu=nu_claimed)
    es_claimed = _compute_es_t(mu, sigma, alpha=0.05, nu=nu_claimed)

    sim = StandardizedTSimulator(mu, sigma, nu_claimed)
    res = acerbi_szekely_z1(real_returns, var_claimed, es_claimed, simulator=sim, n_sim=2000)
    assert res.statistic > 0
    assert res.p_value < 0.05


def test_mcneil_frey_size_under_h0() -> None:
    rng = np.random.default_rng(789)
    n_trials = 300
    rejects = 0
    for trial in range(n_trials):
        d = _toy_oos(n=400, seed=trial)
        res = mcneil_frey(
            d["returns"],
            d["var_t"],
            d["es_t"],
            d["sigma"],
            n_bootstrap=300,
            seed=int(rng.integers(0, 10**9)),
        )
        if not np.isnan(res.p_value) and res.p_value < 0.05:
            rejects += 1
    rate = rejects / n_trials
    assert 0.005 < rate < 0.15


def test_mcneil_frey_few_violations_returns_nan() -> None:
    n = 200
    out = mcneil_frey(
        oos_returns=np.full(n, 0.0),
        var_t=np.full(n, 1000.0),
        es_t=np.full(n, 1000.0),
        sigma_oos=np.ones(n),
    )
    assert np.isnan(out.statistic)
    assert out.extra["n_violations"] == 0


def test_mcneil_frey_rejects_systematic_underestimation() -> None:
    """When the claimed ES is materially below the realised conditional
    expectation of the loss given a violation, the bootstrap statistic
    is pushed to the right tail and the test rejects.
    """
    d = _toy_oos(n=600, seed=11)
    res = mcneil_frey(
        d["returns"],
        d["var_t"],
        d["es_t"] / 3,
        d["sigma"],
        n_bootstrap=1000,
    )
    assert res.statistic > 0
    assert res.p_value < 0.05


def test_z1_z2_p_values_unchanged_when_seed_repeats() -> None:
    d = _toy_oos(n=300, seed=21)
    sim = StandardizedTSimulator(d["mu"], d["sigma"], d["nu"])
    a = acerbi_szekely_z2(
        d["returns"], d["var_t"], d["es_t"], 0.05, simulator=sim, n_sim=500, seed=42
    )
    b = acerbi_szekely_z2(
        d["returns"], d["var_t"], d["es_t"], 0.05, simulator=sim, n_sim=500, seed=42
    )
    assert a.statistic == pytest.approx(b.statistic, rel=1e-12)
    assert a.p_value == pytest.approx(b.p_value, rel=1e-12)
