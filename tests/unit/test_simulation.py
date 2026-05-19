"""Unit tests for simulation DGPs and the power study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esback.simulation.dgp import AR1GarchDGP, GjrGarchDGP
from esback.simulation.power_study import Scenario, run_power_study, summarize_power


def test_ar1_garch_dgp_returns_series_of_length_n() -> None:
    dgp = AR1GarchDGP()
    s = dgp.simulate(100, np.random.default_rng(0))
    assert isinstance(s, pd.Series)
    assert len(s) == 100


def test_ar1_garch_dgp_normal_distribution() -> None:
    dgp = AR1GarchDGP(dist="normal")
    s = dgp.simulate(2000, np.random.default_rng(0))
    assert s.std() > 0


def test_ar1_garch_dgp_unsupported_dist_raises() -> None:
    dgp = AR1GarchDGP(dist="ged")
    with pytest.raises(ValueError, match="Unsupported"):
        dgp.simulate(10, np.random.default_rng(0))


def test_gjr_garch_dgp_returns_series_of_length_n() -> None:
    dgp = GjrGarchDGP()
    s = dgp.simulate(100, np.random.default_rng(1))
    assert len(s) == 100


def test_gjr_dgp_seeded_reproducibility() -> None:
    dgp = GjrGarchDGP()
    a = dgp.simulate(50, np.random.default_rng(2))
    b = dgp.simulate(50, np.random.default_rng(2))
    pd.testing.assert_series_equal(a, b)


def test_gjr_dgp_more_volatile_after_negative_shocks_than_garch() -> None:
    """A leverage-positive GJR DGP should produce a heavier left tail than a
    matched symmetric GARCH DGP. Empirical lower-tail mean check."""
    rng_g = np.random.default_rng(3)
    rng_gjr = np.random.default_rng(3)
    n = 5000
    gar = AR1GarchDGP(alpha1=0.10, beta1=0.85, nu=8.0)
    gjr = GjrGarchDGP(alpha1=0.05, beta1=0.85, gamma=0.10, nu=8.0)
    sg = gar.simulate(n, rng_g).to_numpy()
    sj = gjr.simulate(n, rng_gjr).to_numpy()
    # The 5th percentile (the lower tail) is more negative for GJR.
    assert np.percentile(sj, 5) < np.percentile(sg, 5)


def test_power_study_smoke() -> None:
    """Tiny power study: 2 scenarios x 1 sample size x 3 reps. End-to-end."""
    scenarios = [
        Scenario(name="H0_garch_t", dgp=AR1GarchDGP()),
        Scenario(name="H1_gjr_t", dgp=GjrGarchDGP()),
    ]
    out = run_power_study(
        scenarios,
        sample_sizes=[300],
        n_replications=3,
        in_sample_fraction=0.6,
        n_bootstrap=50,
        seed=42,
    )
    assert not out.empty
    assert {"scenario", "n", "rep", "test", "p_value", "rejected"} <= set(out.columns)


def test_summarize_power_aggregates_correctly() -> None:
    df = pd.DataFrame(
        {
            "scenario": ["A", "A", "A", "A"],
            "n": [100, 100, 100, 100],
            "rep": [0, 1, 2, 3],
            "test": ["t1"] * 4,
            "p_value": [0.01, 0.10, 0.04, 0.30],
        }
    )
    summary = summarize_power(df, level=0.05)
    assert summary.loc[0, "rejection_rate"] == pytest.approx(0.5)


def test_summarize_power_handles_empty() -> None:
    summary = summarize_power(pd.DataFrame())
    assert summary.empty
    assert list(summary.columns) == ["scenario", "n", "test", "rejection_rate"]


def test_summarize_power_treats_nan_as_not_rejected() -> None:
    df = pd.DataFrame(
        {
            "scenario": ["A"] * 4,
            "n": [100] * 4,
            "rep": [0, 1, 2, 3],
            "test": ["t1"] * 4,
            "p_value": [np.nan, np.nan, 0.01, 0.10],
        }
    )
    summary = summarize_power(df, level=0.05)
    assert summary.loc[0, "rejection_rate"] == pytest.approx(0.25)
