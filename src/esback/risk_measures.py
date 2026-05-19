"""Risk measure helpers under the standardized Student-t innovation
distribution.

``var_quantile_t_std`` is closed-form (inverse CDF, scaled so that the
innovation has unit variance: ``q_alpha = t_nu^{-1}(alpha) / sqrt(nu /
(nu - 2))``). ``es_multiplier_t_std`` uses numerical integration
(:func:`scipy.integrate.quad`) of the unit-variance t density truncated
below ``q_alpha``, divided by ``alpha``. Both are pure functions of
``(alpha, nu)`` and benefit from :func:`functools.lru_cache` to
amortize repeated calls in walk-forward backtesting loops.

The cost of ``quad`` is microseconds per uncached call and sub-microsecond
when the cache hits; the lru_cache makes the choice between numerical
integration and the Acerbi-Tasche closed form (semi-analytical, density
evaluated at the quantile point) practically irrelevant for the workloads
this package targets.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy import stats
from scipy.integrate import quad


@functools.lru_cache(maxsize=256)
def var_quantile_t_std(alpha: float, nu: float) -> float:
    """alpha-quantile of the unit-variance Student-t with ``nu`` degrees of freedom."""
    scale = float(np.sqrt(nu / (nu - 2)))
    return float(stats.t.ppf(alpha, df=nu) / scale)


@functools.lru_cache(maxsize=256)
def es_multiplier_t_std(alpha: float, nu: float) -> float:
    """ES multiplier for the unit-variance Student-t.

    Returns ``m(alpha) = (1/alpha) * E[e | e <= F^{-1}(alpha)]`` where
    ``e`` has unit variance. The conditional ES at level ``alpha`` for a
    location-scale model with mean ``mu`` and volatility ``sigma`` is
    then ``-(mu + sigma * m(alpha))`` (loss-positive convention).
    """
    scale = float(np.sqrt(nu / (nu - 2)))
    q = float(stats.t.ppf(alpha, df=nu) / scale)

    def integrand(x: float) -> float:
        return float(x * stats.t.pdf(x * scale, df=nu) * scale)

    val, _ = quad(integrand, -np.inf, q)
    return float(val / alpha)


def var_t(mu: np.ndarray, sigma: np.ndarray, alpha: float, nu: float) -> np.ndarray:
    """Conditional Value-at-Risk at level ``alpha`` (loss-positive convention).

    For each day, ``VaR_t = -(mu_t + sigma_t * q_alpha)`` where ``q_alpha``
    is the unit-variance Student-t quantile. Positive numbers are losses.
    """
    q = var_quantile_t_std(alpha, nu)
    return -(mu + sigma * q)


def es_t(mu: np.ndarray, sigma: np.ndarray, alpha: float, nu: float) -> np.ndarray:
    """Conditional Expected Shortfall at level ``alpha`` (loss-positive)."""
    m = es_multiplier_t_std(alpha, nu)
    return -(mu + sigma * m)


@functools.lru_cache(maxsize=64)
def var_quantile_normal(alpha: float) -> float:
    """``alpha``-quantile of a standard normal (zero mean, unit variance)."""
    return float(stats.norm.ppf(alpha))


@functools.lru_cache(maxsize=64)
def es_multiplier_normal(alpha: float) -> float:
    """ES multiplier for the standard normal: ``-phi(z_alpha) / alpha``."""
    z = float(stats.norm.ppf(alpha))
    return float(-stats.norm.pdf(z) / alpha)


def var_normal(mu: np.ndarray, sigma: np.ndarray, alpha: float) -> np.ndarray:
    """Conditional VaR for AR(1)-GARCH(1,1)-Normal (loss-positive)."""
    q = var_quantile_normal(alpha)
    return -(mu + sigma * q)


def es_normal(mu: np.ndarray, sigma: np.ndarray, alpha: float) -> np.ndarray:
    """Conditional ES for AR(1)-GARCH(1,1)-Normal (loss-positive)."""
    m = es_multiplier_normal(alpha)
    return -(mu + sigma * m)
