"""Probability-integral-transform residuals.

The PIT residuals ``u_t = F_t(Y_t)`` are the cornerstone of every
ES/VaR backtest in this project: under correct conditional specification
they are i.i.d. ``Uniform(0, 1)``. This module provides the closed-form
PIT for the standardized Student-t innovation distribution, a recomputation
helper for parameter perturbations (used by the robust Du-Escanciano
backtests for numerical differentiation), and a Kolmogorov-Smirnov test
for the diagnostic figure.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy import stats


def compute_pit_t_std(eps: np.ndarray, nu: float) -> np.ndarray:
    """PIT residuals for innovations following a standardized Student-t.

    The arch package parameterises innovations so that ``Var(e_t) = 1``,
    which means ``e_t = z_t * sqrt((nu - 2) / nu)`` with ``z_t ~ t(nu)``.
    The CDF of ``e_t`` is therefore ``F_nu(x * sqrt(nu / (nu - 2)))``.
    """
    scale = float(np.sqrt(nu / (nu - 2)))
    return cast(np.ndarray, stats.t.cdf(eps * scale, df=nu))


def pit_from_theta(
    theta_vec: np.ndarray,
    param_names: list[str],
    returns_all: pd.Series,
    is_end_idx: int,
    oos_end_idx: int,
) -> np.ndarray:
    """Recompute the OOS PIT residuals at a perturbed parameter vector.

    Used by :mod:`esback.backtests.du_escanciano` to numerically
    differentiate ``H_bar`` and the autocorrelations with respect to the
    fitted parameters, which is how MU_ES and MC_ES correct for parameter
    estimation uncertainty. We clamp the persistence ``1 - alpha1 - beta1``
    and the conditional variance to keep the recursion finite when a
    perturbation pushes the parameters close to the unit-root boundary;
    we also clamp ``nu`` away from 2 so that ``sqrt(nu / (nu - 2))`` stays
    finite.
    """
    params = {k: float(v) for k, v in zip(param_names, theta_vec, strict=True)}
    const = params.get("Const", 0.0)
    a0 = params.get("Return[1]", params.get("Y.L1", params.get("y[1]", 0.0)))
    omega = params["omega"]
    alpha1 = params["alpha[1]"]
    beta1 = params["beta[1]"]
    nu = max(params["nu"], 2.01)

    returns = returns_all.to_numpy()
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / max(1 - alpha1 - beta1, 0.001)
    for t in range(1, n):
        prev_lag = returns[t - 2] if t >= 2 else 0.0
        mu_prev = const + a0 * prev_lag
        eps_prev = returns[t - 1] - mu_prev
        sigma2[t] = omega + alpha1 * eps_prev**2 + beta1 * sigma2[t - 1]
    sigma = np.sqrt(np.maximum(sigma2, 1e-10))

    sl = slice(is_end_idx, oos_end_idx + 1)
    oos_returns = returns[sl]
    oos_sigma = sigma[sl]
    mu_oos = const + a0 * returns[is_end_idx - 1 : oos_end_idx]
    eps_oos = (oos_returns - mu_oos) / oos_sigma
    return compute_pit_t_std(eps_oos, nu)


def compute_pit_normal(eps: np.ndarray) -> np.ndarray:
    """PIT residuals for unit-variance normal innovations.

    The standardised normal coincides with the ordinary standard normal
    because the latter already has unit variance, so the PIT is just
    ``Phi(eps)``.
    """
    return cast(np.ndarray, stats.norm.cdf(eps))


def ks_test_uniform(u_t: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov test that ``u_t`` is uniformly distributed.

    Returns the statistic and the asymptotic p-value.
    """
    result = stats.kstest(u_t, "uniform")
    return float(result.statistic), float(result.pvalue)
