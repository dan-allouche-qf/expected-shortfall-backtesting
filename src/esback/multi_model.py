"""Multi-model evaluation grid.

For each ``(vol, dist)`` combination, fits the model on the IS window,
reconstructs OOS conditional moments, computes PIT/VaR/ES with the
matching innovation distribution, and runs the unified Du-Escanciano
+ baselines battery. The output is a long-form DataFrame keyed by
``(vol, dist, test)`` with statistic, p-value, alpha, and decision
columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from esback._logging import get_logger
from esback.backtests.battery import run_full_battery
from esback.backtests.du_escanciano import var_violations
from esback.backtests.es_baselines import StandardizedTSimulator
from esback.models import garch_variants
from esback.pit import compute_pit_normal, compute_pit_t_std
from esback.risk_measures import es_normal, es_t, var_normal, var_t


@dataclass(frozen=True)
class VariantSpec:
    vol: str
    dist: str

    def label(self) -> str:
        return f"{self.vol}-{self.dist}"


def _pit_for(eps: np.ndarray, dist: str, nu: float) -> np.ndarray:
    if dist == "t":
        return compute_pit_t_std(eps, nu)
    if dist == "normal":
        return compute_pit_normal(eps)
    raise ValueError(f"Unsupported distribution: {dist!r}")


def _var_for(mu: np.ndarray, sigma: np.ndarray, alpha: float, dist: str, nu: float) -> np.ndarray:
    if dist == "t":
        return var_t(mu, sigma, alpha, nu)
    if dist == "normal":
        return var_normal(mu, sigma, alpha)
    raise ValueError(f"Unsupported distribution: {dist!r}")


def _es_for(mu: np.ndarray, sigma: np.ndarray, alpha: float, dist: str, nu: float) -> np.ndarray:
    if dist == "t":
        return es_t(mu, sigma, alpha, nu)
    if dist == "normal":
        return es_normal(mu, sigma, alpha)
    raise ValueError(f"Unsupported distribution: {dist!r}")


def run_one_variant(
    data: pd.DataFrame,
    is_start: str,
    is_end: str,
    oos_end: str,
    *,
    vol: str,
    dist: str,
    alpha_es: float,
    alpha_var: float,
    m: int,
    n_sim: int = 5_000,
    n_bootstrap: int = 5_000,
) -> dict[str, Any]:
    """Fit ``(vol, dist)`` on the IS window and run the full battery on the OOS one."""
    mask_is = (data.index >= is_start) & (data.index <= is_end)
    mask_all = (data.index >= is_start) & (data.index <= oos_end)
    returns_is = data.loc[mask_is, "Return"]
    returns_all = data.loc[mask_all, "Return"]
    is_end_idx = int(mask_is.sum())
    oos_end_idx = len(returns_all) - 1

    fit = garch_variants.fit_variant(returns_is, vol=vol, dist=dist)  # type: ignore[arg-type]
    variant = garch_variants.forecast_oos_variant(
        fit.params,
        returns_all,
        is_end_idx,
        oos_end_idx,
        vol=vol,  # type: ignore[arg-type]
        dist=dist,  # type: ignore[arg-type]
    )
    forecast = variant.forecast
    nu = forecast.nu

    u_t = _pit_for(forecast.eps_oos, dist, nu)
    var_at_es = _var_for(forecast.mu_oos, forecast.sigma_oos, alpha_es, dist, nu)
    es_at_es = _es_for(forecast.mu_oos, forecast.sigma_oos, alpha_es, dist, nu)
    var_at_var = _var_for(forecast.mu_oos, forecast.sigma_oos, alpha_var, dist, nu)
    es_at_var = _es_for(forecast.mu_oos, forecast.sigma_oos, alpha_var, dist, nu)
    h_t = var_violations(u_t, alpha_var)

    # Acerbi-Szekely simulator only meaningful for the t baseline; for
    # Normal we skip the parametric MC p-value (statistic still computed).
    sim = (
        StandardizedTSimulator(forecast.mu_oos, forecast.sigma_oos, nu)
        if dist == "t" and np.isfinite(nu)
        else None
    )

    battery = run_full_battery(
        u_t=u_t,
        oos_returns=forecast.oos_returns,
        var_t=var_at_var,
        es_t=es_at_var,
        sigma_oos=forecast.sigma_oos,
        h_t=h_t,
        alpha_es=alpha_es,
        alpha_var=alpha_var,
        m=m,
        es_simulator=sim,
        n_sim=n_sim,
        n_bootstrap=n_bootstrap,
    )

    return {
        "fit": fit,
        "variant": variant,
        "u_t": u_t,
        "var_t_at_es": var_at_es,
        "es_t_at_es": es_at_es,
        "h_t": h_t,
        "battery": battery,
        "n_oos": oos_end_idx - is_end_idx + 1,
    }


def run_grid(
    data: pd.DataFrame,
    variants: list[VariantSpec],
    *,
    is_start: str,
    is_end: str,
    oos_end: str,
    alpha_es: float = 0.10,
    alpha_var: float = 0.05,
    m: int = 5,
) -> pd.DataFrame:
    """Run the battery on every (vol, dist) variant and pivot to long form."""
    log = get_logger("esback.multi_model")
    rows: list[dict[str, Any]] = []
    for spec in variants:
        log.info("Fitting %s", spec.label())
        try:
            outcome = run_one_variant(
                data,
                is_start=is_start,
                is_end=is_end,
                oos_end=oos_end,
                vol=spec.vol,
                dist=spec.dist,
                alpha_es=alpha_es,
                alpha_var=alpha_var,
                m=m,
            )
        except Exception as exc:
            log.warning("  %s failed: %s", spec.label(), exc)
            continue
        for test_name, result in outcome["battery"].items():
            rows.append(
                {
                    "variant": spec.label(),
                    "vol": spec.vol,
                    "dist": spec.dist,
                    "test": test_name,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "alpha": result.alpha,
                    "decision": result.decision(),
                    "log_likelihood": outcome["fit"].log_likelihood,
                    "n_oos": outcome["n_oos"],
                }
            )
    return pd.DataFrame(rows)
