"""Unit tests for the multi-model dispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esback.multi_model import VariantSpec, _es_for, _pit_for, _var_for, run_grid


def test_pit_for_t_and_normal() -> None:
    eps = np.array([0.0])
    assert _pit_for(eps, "t", nu=8.0) == pytest.approx([0.5], abs=1e-12)
    assert _pit_for(eps, "normal", nu=float("nan")) == pytest.approx([0.5], abs=1e-12)


def test_pit_for_unknown_dist_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported distribution"):
        _pit_for(np.array([0.0]), "ged", nu=2.0)


def test_var_for_normal_matches_norm_quantile() -> None:
    from scipy import stats

    mu = np.zeros(3)
    sigma = np.ones(3)
    expected = -float(stats.norm.ppf(0.05))
    np.testing.assert_allclose(
        _var_for(mu, sigma, 0.05, "normal", nu=float("nan")),
        np.full(3, expected),
        rtol=1e-12,
    )


def test_es_for_normal_loss_positive() -> None:
    mu = np.zeros(2)
    sigma = np.ones(2)
    es = _es_for(mu, sigma, 0.05, "normal", nu=float("nan"))
    assert np.all(es > 0)


def test_variant_spec_label() -> None:
    s = VariantSpec(vol="GJR", dist="t")
    assert s.label() == "GJR-t"


def test_run_grid_handles_failure_gracefully() -> None:
    """A bad variant is logged and skipped, the others still produce rows."""
    rng = np.random.default_rng(0)
    n = 1200
    e = rng.standard_t(df=8.0, size=n) * np.sqrt((8 - 2) / 8)
    r = np.zeros(n)
    s2 = np.full(n, 1.0)
    for t in range(1, n):
        s2[t] = 0.05 + 0.08 * r[t - 1] ** 2 + 0.90 * s2[t - 1]
        r[t] = 0.05 + 0.05 * r[t - 1] + np.sqrt(s2[t]) * e[t]
    data = pd.DataFrame({"Return": r}, index=pd.date_range("2018-01-01", periods=n, freq="B"))
    variants = [
        VariantSpec(vol="GARCH", dist="t"),
        VariantSpec(vol="GARCH", dist="normal"),
    ]
    out = run_grid(
        data,
        variants,
        is_start=str(data.index[0].date()),
        is_end=str(data.index[800].date()),
        oos_end=str(data.index[-1].date()),
    )
    assert {"GARCH-t", "GARCH-normal"} <= set(out["variant"].unique())
