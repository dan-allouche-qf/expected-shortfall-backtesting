"""GARCH variants and innovation distributions.

Adds the asymmetric GJR-GARCH(1,1,1) volatility recursion and the
Normal innovation distribution alongside the AR(1)-GARCH(1,1)-t in
:mod:`esback.models.ar1_garch_t`, giving four ``(vol, dist)``
combinations driven by :func:`fit_variant`: ``(GARCH, t)``,
``(GARCH, normal)``, ``(GJR, t)``, ``(GJR, normal)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from arch import arch_model

from esback.models.ar1_garch_t import ForecastArtifacts, _garch_recursion, _unpack
from esback.models.base import ModelFit

VolName = Literal["GARCH", "GJR"]
DistName = Literal["t", "normal"]


_VOL_ARCH_KWARGS: dict[str, dict[str, Any]] = {
    "GARCH": {"vol": "GARCH", "p": 1, "q": 1, "o": 0},
    "GJR": {"vol": "GARCH", "p": 1, "q": 1, "o": 1},
}

_DIST_ARCH_NAME: dict[str, str] = {
    "t": "t",
    "normal": "normal",
}


def fit_variant(
    returns_is: pd.Series,
    *,
    vol: VolName = "GARCH",
    dist: DistName = "t",
) -> ModelFit:
    """Estimate AR(1)-{GARCH or GJR}-(1,1)-{t or normal}."""
    if vol not in _VOL_ARCH_KWARGS:
        raise ValueError(f"Unsupported volatility model: {vol!r}")
    if dist not in _DIST_ARCH_NAME:
        raise ValueError(f"Unsupported innovation distribution: {dist!r}")

    am = arch_model(
        returns_is,
        mean="ARX",
        lags=1,
        dist=_DIST_ARCH_NAME[dist],  # type: ignore[arg-type]
        **_VOL_ARCH_KWARGS[vol],
    )
    res = am.fit(disp="off", options={"maxiter": 1000})
    return ModelFit(
        params={k: float(v) for k, v in res.params.items()},
        param_names=list(res.params.index),
        param_cov=res.param_cov.values,
        in_sample_n=len(returns_is),
        log_likelihood=float(res.loglikelihood) if hasattr(res, "loglikelihood") else None,
    )


def volatility_path_gjr(params: dict[str, float], returns: np.ndarray) -> np.ndarray:
    """GJR-GARCH(1,1,1) recursion over the full series.

    ``sigma2_t = omega + (alpha + gamma * 1[eps_{t-1} < 0]) * eps_{t-1}^2 + beta * sigma2_{t-1}``.
    Thin wrapper over :func:`esback.models.ar1_garch_t._garch_recursion`.
    """
    gamma = params.get("gamma[1]", 0.0)
    return _garch_recursion(params, returns, gamma=gamma)


def volatility_path_garch(params: dict[str, float], returns: np.ndarray) -> np.ndarray:
    """Symmetric GARCH(1,1) recursion. Equivalent to ``volatility_path`` in
    :mod:`esback.models.ar1_garch_t`; re-exported for symmetry with
    :func:`volatility_path_gjr`."""
    return _garch_recursion(params, returns, gamma=0.0)


@dataclass(frozen=True)
class VariantArtifacts:
    """OOS conditional moments and residuals for a non-baseline variant.

    Mirrors :class:`ForecastArtifacts` but carries the variant identifier
    so downstream code can dispatch on the innovation distribution.
    """

    forecast: ForecastArtifacts
    vol: str
    dist: str


def forecast_oos_variant(
    params: dict[str, float],
    returns_all: pd.Series,
    is_end_idx: int,
    oos_end_idx: int,
    *,
    vol: VolName = "GARCH",
    dist: DistName = "t",
) -> VariantArtifacts:
    """Reconstruct OOS conditional moments for the chosen ``(vol, dist)`` pair.

    For ``dist == "normal"`` the returned ``forecast.nu`` is set to NaN
    because the Normal does not have a degrees-of-freedom parameter.
    """
    const, a0, _, _, _, _ = _unpack(params)
    returns = returns_all.to_numpy()
    if vol == "GARCH":
        sigma_full = volatility_path_garch(params, returns)
    elif vol == "GJR":
        sigma_full = volatility_path_gjr(params, returns)
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"Unsupported volatility model: {vol!r}")

    sl = slice(is_end_idx, oos_end_idx + 1)
    oos_returns = returns[sl]
    sigma_oos = sigma_full[sl]
    oos_index = returns_all.index[sl]
    mu_oos = const + a0 * returns[is_end_idx - 1 : oos_end_idx]
    eps_oos = (oos_returns - mu_oos) / sigma_oos

    nu = float(params.get("nu", float("nan"))) if dist == "t" else float("nan")
    forecast = ForecastArtifacts(
        mu_oos=mu_oos,
        sigma_oos=sigma_oos,
        eps_oos=eps_oos,
        oos_returns=oos_returns,
        oos_index=oos_index,
        nu=nu,
    )
    return VariantArtifacts(forecast=forecast, vol=vol, dist=dist)
