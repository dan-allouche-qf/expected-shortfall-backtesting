"""AR(1)-GARCH(1,1) with standardized Student-t innovations.

This is the baseline model used in Du-Escanciano (2017): an AR(1) for the
conditional mean, a GARCH(1,1) for the conditional variance, and a
Student-t for the innovations standardised to unit variance. We delegate
the maximum-likelihood estimation to the ``arch`` package and keep this
module focused on the deterministic recursions (volatility and mean
paths, OOS residual extraction) so they can be reused both in the basic
backtests and in the robust MU_ES / MC_ES tests, where they need to be
re-evaluated at perturbed parameter vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

from esback.models.base import ModelFit


@dataclass(frozen=True)
class ForecastArtifacts:
    """OOS conditional moments and standardized residuals.

    Attributes
    ----------
    mu_oos
        Conditional mean over the OOS window, length ``n_oos``.
    sigma_oos
        Conditional volatility over the OOS window, length ``n_oos``.
    eps_oos
        Standardised residuals ``(Y_t - mu_t) / sigma_t``.
    oos_returns
        Realised returns over the OOS window.
    oos_index
        DatetimeIndex of the OOS window.
    nu
        Estimated degrees of freedom of the innovation distribution.
    """

    mu_oos: np.ndarray
    sigma_oos: np.ndarray
    eps_oos: np.ndarray
    oos_returns: np.ndarray
    oos_index: pd.Index
    nu: float


def _unpack(params: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    """Extract ``(const, a0, omega, alpha1, beta1, nu)`` from arch's parameter dict.

    arch labels the AR(1) coefficient differently across versions
    (``Return[1]``, ``Y.L1``, ``y[1]``). We probe all known aliases.
    ``nu`` defaults to NaN when the innovation distribution is Normal,
    so this helper is shared between the Student-t baseline and the
    Normal variants in :mod:`esback.models.garch_variants`.
    """
    const = params.get("Const", 0.0)
    a0 = params.get("Return[1]", params.get("Y.L1", params.get("y[1]", 0.0)))
    return (
        const,
        a0,
        params["omega"],
        params["alpha[1]"],
        params["beta[1]"],
        params.get("nu", float("nan")),
    )


def _garch_recursion(
    params: dict[str, float],
    returns: np.ndarray,
    *,
    gamma: float = 0.0,
) -> np.ndarray:
    """Run the GARCH(1,1) or GJR(1,1,1) recursion over the full series.

    ``gamma=0`` recovers the symmetric GARCH; ``gamma>0`` adds the
    Glosten-Jagannathan-Runkle leverage term ``gamma * 1{eps_{t-1} < 0}``.
    Seeded at the unconditional variance ``omega / (1 - alpha - gamma/2 - beta)``;
    falls back to 1.0 if the parameter set is non-stationary (persistence <= 0).
    A final ``np.clip(..., 0.0, None)`` guards against numerically negative
    ``sigma2`` produced by ill-conditioned MLE fits (e.g. very short OOS
    windows on illiquid assets) — the original implementation emitted an
    ``invalid value encountered in sqrt`` warning here.
    """
    const, a0, omega, alpha1, beta1, _ = _unpack(params)
    persistence = 1.0 - alpha1 - gamma / 2.0 - beta1
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / persistence if persistence > 0 else 1.0
    for t in range(1, n):
        prev_lag = returns[t - 2] if t >= 2 else 0.0
        mu_prev = const + a0 * prev_lag
        eps_prev = returns[t - 1] - mu_prev
        leverage = gamma if eps_prev < 0 else 0.0
        sigma2[t] = omega + (alpha1 + leverage) * eps_prev**2 + beta1 * sigma2[t - 1]
    return np.sqrt(np.clip(sigma2, 0.0, None))


def fit(returns_is: pd.Series) -> ModelFit:
    """Estimate AR(1)-GARCH(1,1)-t by conditional maximum likelihood."""
    am = arch_model(returns_is, mean="ARX", lags=1, vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off", options={"maxiter": 1000})
    return ModelFit(
        params={k: float(v) for k, v in res.params.items()},
        param_names=list(res.params.index),
        param_cov=res.param_cov.values,
        in_sample_n=len(returns_is),
        log_likelihood=float(res.loglikelihood) if hasattr(res, "loglikelihood") else None,
    )


def volatility_path(params: dict[str, float], returns: np.ndarray) -> np.ndarray:
    """Reconstruct ``sigma_t`` over the full series via the GARCH(1,1) recursion."""
    return _garch_recursion(params, returns, gamma=0.0)


def forecast_oos(
    params: dict[str, float],
    returns_all: pd.Series,
    is_end_idx: int,
    oos_end_idx: int,
) -> ForecastArtifacts:
    """Evaluate the conditional moments and residuals on the OOS window.

    The OOS slice is ``[is_end_idx, oos_end_idx]`` inclusive on both ends.
    The conditional mean uses the AR(1) form ``mu_t = const + a0 * Y_{t-1}``,
    and the volatility recursion is run on the full history so the OOS
    residuals are properly conditioned on the in-sample-fitted dynamics
    rolling forward without re-estimation.
    """
    const, a0, _, _, _, nu = _unpack(params)
    returns = returns_all.to_numpy()
    sigma_full = volatility_path(params, returns)

    sl = slice(is_end_idx, oos_end_idx + 1)
    oos_returns = returns[sl]
    sigma_oos = sigma_full[sl]
    oos_index = returns_all.index[sl]

    mu_oos = const + a0 * returns[is_end_idx - 1 : oos_end_idx]
    eps_oos = (oos_returns - mu_oos) / sigma_oos

    return ForecastArtifacts(
        mu_oos=mu_oos,
        sigma_oos=sigma_oos,
        eps_oos=eps_oos,
        oos_returns=oos_returns,
        oos_index=oos_index,
        nu=nu,
    )
