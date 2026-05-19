"""Multi-asset, multi-crisis evaluation.

For each ``(asset, period)`` configuration, fits AR(1)-GARCH(1,1)-t on
an in-sample window ending just before the crisis OOS slice and runs
the unified backtest battery. The result is a DataFrame indexed by
``(asset, period, test_name)`` with statistic, p-value, alpha, and
decision columns -- the matrix that powers the multi-asset heatmaps.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from esback._logging import get_logger
from esback.backtests.battery import run_full_battery
from esback.backtests.es_baselines import StandardizedTSimulator
from esback.data.loaders import load_yfinance
from esback.pipeline import run_period
from esback.risk_measures import es_t as _compute_es_t
from esback.risk_measures import var_t as _compute_var_t


@dataclass(frozen=True)
class AssetSpec:
    name: str
    ticker: str


@dataclass(frozen=True)
class PeriodSpec:
    name: str
    oos_start: str
    oos_end: str


def _is_window(oos_start: str, in_sample_years: int) -> tuple[str, str]:
    """Return the in-sample ``(start, end)`` ending one day before ``oos_start``."""
    oos_ts = pd.Timestamp(oos_start)
    is_end = (oos_ts - pd.Timedelta(days=1)).date().isoformat()
    is_start = (oos_ts - pd.DateOffset(years=in_sample_years)).date().isoformat()
    return is_start, is_end


def run_one(
    data: pd.DataFrame,
    is_start: str,
    is_end: str,
    oos_start: str,
    oos_end: str,
    *,
    alpha_es: float,
    alpha_var: float,
    m: int,
    n_sim: int = 10_000,
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    """Fit on the in-sample window and run the full battery on the OOS one."""
    artifacts = run_period(
        data,
        is_start=is_start,
        is_end=is_end,
        oos_end=oos_end,
        label="",
        alpha_es=alpha_es,
        alpha_var=alpha_var,
        m=m,
    )
    # The OOS slice may extend beyond ``oos_end`` of the period if the
    # data range is wider; trim it to the requested OOS window.
    mask_oos = (artifacts.forecast.oos_index >= oos_start) & (
        artifacts.forecast.oos_index <= oos_end
    )
    if not mask_oos.any():
        raise ValueError(
            f"No OOS observations between {oos_start} and {oos_end} "
            f"for in-sample [{is_start}, {is_end}]."
        )

    var_at_var = _compute_var_t(
        artifacts.forecast.mu_oos[mask_oos],
        artifacts.forecast.sigma_oos[mask_oos],
        alpha=alpha_var,
        nu=artifacts.forecast.nu,
    )
    es_at_var = _compute_es_t(
        artifacts.forecast.mu_oos[mask_oos],
        artifacts.forecast.sigma_oos[mask_oos],
        alpha=alpha_var,
        nu=artifacts.forecast.nu,
    )
    sim = StandardizedTSimulator(
        artifacts.forecast.mu_oos[mask_oos],
        artifacts.forecast.sigma_oos[mask_oos],
        artifacts.forecast.nu,
    )
    battery = run_full_battery(
        u_t=artifacts.u_t[mask_oos],
        oos_returns=artifacts.forecast.oos_returns[mask_oos],
        var_t=var_at_var,
        es_t=es_at_var,
        sigma_oos=artifacts.forecast.sigma_oos[mask_oos],
        h_t=artifacts.h_t[mask_oos],
        alpha_es=alpha_es,
        alpha_var=alpha_var,
        m=m,
        es_simulator=sim,
        n_sim=n_sim,
        n_bootstrap=n_bootstrap,
    )
    return {"artifacts": artifacts, "battery": battery, "n_oos": int(mask_oos.sum())}


def run_grid(
    assets: Iterable[AssetSpec],
    periods: Iterable[PeriodSpec],
    *,
    data_start: str,
    data_end: str,
    in_sample_years: int = 5,
    alpha_es: float = 0.10,
    alpha_var: float = 0.05,
    m: int = 5,
    n_sim: int = 10_000,
    n_bootstrap: int = 10_000,
) -> pd.DataFrame:
    """Run the battery on every (asset, period) combination.

    Returns a long-form DataFrame with columns
    ``(asset, period, test, statistic, p_value, alpha, decision)``.
    """
    log = get_logger("esback.multi_asset")
    rows: list[dict[str, Any]] = []
    for asset in assets:
        log.info("Loading %s (%s)", asset.name, asset.ticker)
        try:
            data = load_yfinance(asset.ticker, start=data_start, end=data_end)
        except Exception as exc:
            log.warning("Failed to load %s: %s", asset.ticker, exc)
            continue

        for period in periods:
            is_start, is_end = _is_window(period.oos_start, in_sample_years)
            log.info(
                "  %s x %s: IS [%s, %s], OOS [%s, %s]",
                asset.name,
                period.name,
                is_start,
                is_end,
                period.oos_start,
                period.oos_end,
            )
            try:
                outcome = run_one(
                    data,
                    is_start=is_start,
                    is_end=is_end,
                    oos_start=period.oos_start,
                    oos_end=period.oos_end,
                    alpha_es=alpha_es,
                    alpha_var=alpha_var,
                    m=m,
                    n_sim=n_sim,
                    n_bootstrap=n_bootstrap,
                )
            except Exception as exc:
                log.warning("    failed: %s", exc)
                continue

            for test_name, result in outcome["battery"].items():
                rows.append(
                    {
                        "asset": asset.name,
                        "period": period.name,
                        "test": test_name,
                        "statistic": result.statistic,
                        "p_value": result.p_value,
                        "alpha": result.alpha,
                        "decision": result.decision(),
                        "n_oos": outcome["n_oos"],
                    }
                )

    return pd.DataFrame(rows)


def pivot_p_values(grid: pd.DataFrame, period: str | None = None) -> pd.DataFrame:
    """Pivot the long-form grid into ``test x asset`` p-value matrix."""
    sub = grid if period is None else grid[grid["period"] == period]
    return sub.pivot_table(index="test", columns="asset", values="p_value", aggfunc="first")


def reject_rate_per_test(grid: pd.DataFrame, level: float = 0.05) -> pd.DataFrame:
    """For each test, compute the share of (asset, period) cells that reject at ``level``."""
    rejects = grid.assign(reject=(grid["p_value"] < level).astype(float))
    return rejects.groupby("test")["reject"].mean().to_frame("reject_rate")


def reject_rate_per_period(grid: pd.DataFrame, level: float = 0.05) -> pd.DataFrame:
    """For each (test, period), compute the share of assets that reject at ``level``."""
    rejects = grid.assign(reject=(grid["p_value"] < level).astype(float))
    return rejects.pivot_table(index="test", columns="period", values="reject", aggfunc="mean")
