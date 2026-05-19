"""Smoke tests for the MDN-LSTM head and its risk-measure helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import stats

from esback.backtests.es_baselines import MDNSimulator
from esback.models.dl.lstm_mdn import (
    LSTMMDNForecast,
    es_from_mdn,
    fit_and_forecast_mdn,
    nll_loss_studentt,
    pit_from_mdn,
    var_from_mdn,
)


def test_lstm_mdn_forward_returns_three_finite_tensors() -> None:
    torch.manual_seed(0)
    model = LSTMMDNForecast(seq_len=10, hidden_size=8)
    x = torch.randn(4, 10, 1)
    loc, scale, df = model(x)
    assert loc.shape == (4,)
    assert scale.shape == (4,)
    assert df.shape == (4,)
    assert torch.isfinite(loc).all()
    assert (scale > 0).all()
    assert (df > 2.0).all()


def test_nll_loss_finite_on_typical_inputs() -> None:
    loc = torch.tensor([0.0, 0.1])
    scale = torch.tensor([1.0, 0.8])
    df = torch.tensor([6.0, 8.0])
    target = torch.tensor([-2.0, 1.5])
    loss = nll_loss_studentt(loc, scale, df, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0  # NLL is non-negative for these parameters


def test_fit_and_forecast_smoke() -> None:
    rng = np.random.default_rng(0)
    n_is, n_oos = 300, 50
    returns = rng.standard_t(df=6.0, size=n_is + n_oos) * 1.2

    params, history = fit_and_forecast_mdn(
        returns_full=returns,
        n_is=n_is,
        seq_len=10,
        hidden_size=8,
        epochs=3,
        batch_size=32,
        seed=0,
        device=torch.device("cpu"),
    )
    assert set(params) == {"loc", "scale", "df"}
    assert params["loc"].shape == (n_oos,)
    assert (params["scale"] > 0).all()
    assert (params["df"] > 2).all()
    assert len(history.nll) == 3
    assert all(np.isfinite(history.nll))


def test_pit_uniform_on_correct_specification() -> None:
    """If returns are drawn from the same Student-t the MDN parameterises,
    the PIT residual is uniform on (0,1)."""
    rng = np.random.default_rng(1)
    n = 2000
    # Stationary parameters so the PIT is i.i.d. under correct spec
    loc = np.full(n, 0.05)
    scale = np.full(n, 1.4)
    df = np.full(n, 7.0)
    # Realisation drawn from Student-t(df, loc, scale)
    oos_returns = stats.t.rvs(df=7.0, loc=0.05, scale=1.4, size=n, random_state=rng)
    u_t = pit_from_mdn(loc, scale, df, oos_returns)
    _, ks_pval = stats.kstest(u_t, "uniform")
    assert ks_pval > 0.01, (
        f"PIT residuals depart from Uniform(0,1) under correct spec: p = {ks_pval}"
    )


def test_var_es_ordering_and_sign() -> None:
    """At alpha = 0.05 in the left tail, ES >= VaR holds and both are positive
    when the conditional location is roughly zero."""
    loc = np.array([0.0, 0.05])
    scale = np.array([1.0, 1.5])
    df = np.array([6.0, 6.0])
    var_t = var_from_mdn(loc, scale, df, alpha=0.05)
    es_t = es_from_mdn(loc, scale, df, alpha=0.05)
    assert (var_t > 0).all()
    assert (es_t >= var_t).all()


def test_mdn_simulator_shape_and_finite() -> None:
    rng = np.random.default_rng(2)
    n_oos = 30
    loc = np.full(n_oos, 0.02)
    scale = np.full(n_oos, 1.1)
    df = np.full(n_oos, 8.0)
    sim = MDNSimulator(loc_oos=loc, scale_oos=scale, df_oos=df)
    paths = sim.simulate_returns(n_sim=100, rng=rng)
    assert paths.shape == (100, n_oos)
    assert np.isfinite(paths).all()


def test_mdn_simulator_reproducible_with_seed() -> None:
    sim = MDNSimulator(
        loc_oos=np.full(50, 0.02),
        scale_oos=np.full(50, 1.1),
        df_oos=np.full(50, 8.0),
    )
    a = sim.simulate_returns(10, np.random.default_rng(1))
    b = sim.simulate_returns(10, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_es_from_mdn_matches_numerical_integration(alpha: float) -> None:
    """The closed-form ES matches a Monte-Carlo estimate from the same
    Student-t distribution to within 3 %."""
    rng = np.random.default_rng(3)
    n = 500_000
    loc, scale, df = 0.03, 1.2, 5.5
    samples = stats.t.rvs(df=df, loc=loc, scale=scale, size=n, random_state=rng)
    var_mc = -np.quantile(samples, alpha)
    es_mc = -samples[samples < -var_mc].mean()

    var_cf = var_from_mdn(np.array([loc]), np.array([scale]), np.array([df]), alpha=alpha)[0]
    es_cf = es_from_mdn(np.array([loc]), np.array([scale]), np.array([df]), alpha=alpha)[0]

    assert abs(var_cf - var_mc) / var_mc < 0.03
    assert abs(es_cf - es_mc) / es_mc < 0.03
