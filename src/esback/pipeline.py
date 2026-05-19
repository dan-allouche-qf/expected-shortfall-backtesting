"""End-to-end pipeline for one in-sample / out-of-sample split.

Composes the data loader, the AR(1)-GARCH(1,1)-t model, the PIT
transformation, the analytical VaR and ES formulas, and the
Du-Escanciano backtest battery into a single :func:`run_period` entry
point. The CLI and the replication notebook both consume it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from esback.backtests.base import BacktestResult
from esback.backtests.du_escanciano import (
    es_severity,
    run_battery,
    run_robust_battery,
    var_violations,
)
from esback.models import ar1_garch_t
from esback.models.base import ModelFit
from esback.pit import compute_pit_t_std
from esback.risk_measures import es_t, var_t


@dataclass(frozen=True)
class PeriodArtifacts:
    """All outputs of running the full pipeline on one period.

    Carries everything the CLI and the figure helpers need to produce
    the replication report: the model fit, the OOS PIT residuals, the
    risk paths, and the basic plus robust backtest results.
    """

    label: str
    fit: ModelFit
    is_end_idx: int
    oos_end_idx: int
    returns_all: pd.Series
    forecast: ar1_garch_t.ForecastArtifacts
    u_t: np.ndarray
    var_t: np.ndarray
    es_t: np.ndarray
    h_t: np.ndarray
    H_t: np.ndarray
    battery: dict[str, BacktestResult]
    robust: dict[str, BacktestResult]


def run_period(
    data: pd.DataFrame,
    is_start: str,
    is_end: str,
    oos_end: str,
    label: str,
    alpha_es: float = 0.10,
    alpha_var: float = 0.05,
    m: int = 5,
) -> PeriodArtifacts:
    """Run the full backtest pipeline for one ``(in-sample, out-of-sample)`` split.

    ``data`` must have a ``Return`` column and a date index. Slicing is
    inclusive on both ends. The fit happens on the in-sample window; the
    PIT, VaR, ES and backtests are evaluated on the OOS window.
    """
    mask_is = (data.index >= is_start) & (data.index <= is_end)
    mask_all = (data.index >= is_start) & (data.index <= oos_end)
    returns_is = data.loc[mask_is, "Return"]
    returns_all = data.loc[mask_all, "Return"]
    is_end_idx = int(mask_is.sum())
    oos_end_idx = len(returns_all) - 1

    fit = ar1_garch_t.fit(returns_is)
    forecast = ar1_garch_t.forecast_oos(fit.params, returns_all, is_end_idx, oos_end_idx)
    u_t = compute_pit_t_std(forecast.eps_oos, forecast.nu)
    var_path = var_t(forecast.mu_oos, forecast.sigma_oos, alpha_var, forecast.nu)
    es_path = es_t(forecast.mu_oos, forecast.sigma_oos, alpha_es, forecast.nu)
    H_t = es_severity(u_t, alpha_es)
    h_t = var_violations(u_t, alpha_var)

    battery = run_battery(u_t, alpha_es=alpha_es, alpha_var=alpha_var, m=m)
    robust = run_robust_battery(
        u_t, returns_all, fit, is_end_idx, oos_end_idx, alpha_es=alpha_es, m=m
    )

    return PeriodArtifacts(
        label=label,
        fit=fit,
        is_end_idx=is_end_idx,
        oos_end_idx=oos_end_idx,
        returns_all=returns_all,
        forecast=forecast,
        u_t=u_t,
        var_t=var_path,
        es_t=es_path,
        h_t=h_t,
        H_t=H_t,
        battery=battery,
        robust=robust,
    )
