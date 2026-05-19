"""Unit tests for Du-Escanciano backtests."""

from __future__ import annotations

import numpy as np
import pytest

from esback.backtests.du_escanciano import (
    conditional_es,
    conditional_var,
    es_severity,
    run_battery,
    unconditional_es,
    unconditional_var,
    var_violations,
)


def test_var_violations_binary_indicator() -> None:
    u_t = np.array([0.01, 0.05, 0.06, 0.10, 0.99])
    h = var_violations(u_t, alpha=0.05)
    np.testing.assert_array_equal(h, [1.0, 1.0, 0.0, 0.0, 0.0])


def test_es_severity_zero_when_no_violation() -> None:
    """If u_t > alpha, severity is zero."""
    u_t = np.array([0.20, 0.50, 0.99])
    H = es_severity(u_t, alpha=0.10)
    np.testing.assert_array_equal(H, [0.0, 0.0, 0.0])


def test_es_severity_value_at_violation() -> None:
    """For u_t = 0.04 and alpha = 0.10: H_t = (1/0.10) * (0.10 - 0.04) = 0.6."""
    u_t = np.array([0.04])
    H = es_severity(u_t, alpha=0.10)
    np.testing.assert_allclose(H, [0.6], rtol=1e-12)


def test_es_severity_max_at_zero() -> None:
    """As u_t -> 0, severity approaches its maximum 1."""
    u_t = np.array([1e-12])
    H = es_severity(u_t, alpha=0.10)
    np.testing.assert_allclose(H, [0.99999999999], rtol=1e-9)


def test_unconditional_es_zero_under_h0() -> None:
    """U_ES vanishes when H_bar = alpha/2 exactly.

    Solving H_t = alpha/2 for u_t at a violation gives
    u_t = alpha - alpha**2 / 2. With this value on every day,
    H_bar = alpha/2 and the U_ES statistic is exactly zero.
    """
    n = 1000
    alpha = 0.10
    u_t = np.full(n, alpha - alpha**2 / 2)
    res = unconditional_es(u_t, alpha)
    assert res.extra["H_bar"] == pytest.approx(alpha / 2, abs=1e-12)
    assert abs(res.statistic) < 1e-9


def test_unconditional_es_size_under_h0() -> None:
    """Under U(0, 1) PIT, U_ES rejection rate at 5% should be near 5%."""
    rng = np.random.default_rng(0)
    n_trials = 800
    n = 500
    alpha = 0.10
    rejects = 0
    for _ in range(n_trials):
        u = rng.uniform(size=n)
        if unconditional_es(u, alpha).p_value < 0.05:
            rejects += 1
    rate = rejects / n_trials
    assert 0.025 < rate < 0.085


def test_unconditional_var_size_under_h0() -> None:
    rng = np.random.default_rng(1)
    n_trials = 800
    n = 500
    alpha = 0.05
    rejects = 0
    for _ in range(n_trials):
        u = rng.uniform(size=n)
        if unconditional_var(u, alpha).p_value < 0.05:
            rejects += 1
    rate = rejects / n_trials
    assert 0.020 < rate < 0.085


def test_conditional_es_size_under_h0() -> None:
    """C_ES(5) on i.i.d. uniform should reject at approx 5%."""
    rng = np.random.default_rng(2)
    n_trials = 600
    n = 1000
    alpha = 0.10
    rejects = 0
    for _ in range(n_trials):
        u = rng.uniform(size=n)
        if conditional_es(u, alpha, m=5).p_value < 0.05:
            rejects += 1
    rate = rejects / n_trials
    assert 0.020 < rate < 0.090


def test_unconditional_es_rejects_inflated_severity() -> None:
    """Half the days violating with severity 1 must reject sharply."""
    n = 500
    alpha = 0.10
    u_t = np.concatenate([np.full(n // 2, 0.5), np.full(n // 2, 1e-12)])
    res = unconditional_es(u_t, alpha)
    assert res.p_value < 1e-6


def test_run_battery_returns_four_results() -> None:
    rng = np.random.default_rng(3)
    u = rng.uniform(size=500)
    battery = run_battery(u, alpha_es=0.10, alpha_var=0.05, m=5)
    assert set(battery.keys()) == {"U_ES", "C_ES(5)", "U_VaR", "C_VaR(5)"}


def test_decision_method_on_battery() -> None:
    rng = np.random.default_rng(4)
    u = rng.uniform(size=500)
    battery = run_battery(u)
    for name, res in battery.items():
        assert res.decision() in ("REJECT", "Fail to reject")
        assert res.name == name


def test_extra_payload_contains_intermediate_quantities() -> None:
    rng = np.random.default_rng(5)
    u = rng.uniform(size=300)
    res = unconditional_es(u, 0.10)
    assert "H_bar" in res.extra
    assert "H_t" in res.extra
    assert res.extra["n"] == 300


def test_conditional_var_extra_holds_autocorrs() -> None:
    rng = np.random.default_rng(6)
    u = rng.uniform(size=300)
    res = conditional_var(u, 0.05, m=5)
    assert res.extra["m"] == 5
    assert len(res.extra["rho"]) == 5
