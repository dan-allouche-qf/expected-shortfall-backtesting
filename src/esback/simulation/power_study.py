"""Monte Carlo size and power study.

For each ``(scenario, n)`` pair, draws ``n_replications`` paths from
the scenario DGP, fits the claimed model (AR(1)-GARCH(1,1)-t baseline)
on the first ``in_sample_fraction`` of the data, runs the unified
backtest battery on the remainder, and records the rejection
indicator at the chosen significance level. Aggregating over
replications gives the empirical size (under H0 scenarios) or power
(under H1 scenarios) of every test.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from esback._logging import get_logger
from esback.backtests.battery import run_full_battery
from esback.backtests.es_baselines import StandardizedTSimulator
from esback.pipeline import run_period
from esback.risk_measures import es_t as _compute_es_t
from esback.risk_measures import var_t as _compute_var_t
from esback.simulation.dgp import DGP


@dataclass(frozen=True)
class Scenario:
    """Named DGP plus a free-form description for the report."""

    name: str
    dgp: DGP
    description: str = ""


def _run_one_replication(
    returns: pd.Series,
    *,
    n_is: int,
    alpha_es: float,
    alpha_var: float,
    m: int,
    n_bootstrap: int,
    n_sim_power: int,
) -> dict[str, float] | None:
    """Fit the AR(1)-GARCH(1,1)-t baseline and run the battery on one path.

    Returns a ``{test_name: p_value}`` dict, or ``None`` if the fit or
    battery raised.
    """
    df = returns.to_frame(name="Return")
    is_start = str(df.index[0].date())
    is_end = str(df.index[n_is - 1].date())
    oos_end = str(df.index[-1].date())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            artifacts = run_period(
                df,
                is_start=is_start,
                is_end=is_end,
                oos_end=oos_end,
                label="",
                alpha_es=alpha_es,
                alpha_var=alpha_var,
                m=m,
            )
        except Exception:
            return None

        var_at_var = _compute_var_t(
            artifacts.forecast.mu_oos,
            artifacts.forecast.sigma_oos,
            alpha_var,
            artifacts.forecast.nu,
        )
        es_at_var = _compute_es_t(
            artifacts.forecast.mu_oos,
            artifacts.forecast.sigma_oos,
            alpha_var,
            artifacts.forecast.nu,
        )
        es_simulator = StandardizedTSimulator(
            mu_oos=artifacts.forecast.mu_oos,
            sigma_oos=artifacts.forecast.sigma_oos,
            nu=artifacts.forecast.nu,
        )

        try:
            battery = run_full_battery(
                u_t=artifacts.u_t,
                oos_returns=artifacts.forecast.oos_returns,
                var_t=var_at_var,
                es_t=es_at_var,
                sigma_oos=artifacts.forecast.sigma_oos,
                h_t=artifacts.h_t,
                alpha_es=alpha_es,
                alpha_var=alpha_var,
                m=m,
                n_bootstrap=n_bootstrap,
                es_simulator=es_simulator,
                n_sim=n_sim_power,
            )
        except Exception:
            return None

    return {name: result.p_value for name, result in battery.items()}


def run_power_study(
    scenarios: list[Scenario],
    sample_sizes: list[int],
    *,
    n_replications: int = 200,
    in_sample_fraction: float = 0.6,
    alpha_es: float = 0.10,
    alpha_var: float = 0.05,
    m: int = 5,
    n_bootstrap: int = 200,
    n_sim_power: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Drive the Monte Carlo grid and return the long-form rejection log.

    Returns a DataFrame with columns ``scenario, n, rep, test, p_value,
    rejected``. ``rejected`` is True if ``p_value < 0.05``; NaN p-values
    count as not-rejected.

    The Acerbi-Szekely Z1 / Z2 parametric MC p-values are computed with
    ``n_sim_power`` draws per replication. The default (2000) is lower
    than the 10000 used in single-period runs to keep the total grid
    affordable; raise it for sharper p-values at the cost of runtime.
    """
    log = get_logger("esback.power_study")
    master_rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    total = len(scenarios) * len(sample_sizes) * n_replications
    done = 0

    for scenario in scenarios:
        for n in sample_sizes:
            n_is = max(int(n * in_sample_fraction), 100)
            for rep in range(n_replications):
                rep_seed = int(master_rng.integers(0, 2**31 - 1))
                rep_rng = np.random.default_rng(rep_seed)
                returns = scenario.dgp.simulate(n, rep_rng)
                p_values = _run_one_replication(
                    returns,
                    n_is=n_is,
                    alpha_es=alpha_es,
                    alpha_var=alpha_var,
                    m=m,
                    n_bootstrap=n_bootstrap,
                    n_sim_power=n_sim_power,
                )
                done += 1
                if p_values is None:
                    continue
                for test_name, p in p_values.items():
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "n": n,
                            "rep": rep,
                            "test": test_name,
                            "p_value": p,
                            "rejected": (not np.isnan(p)) and (p < 0.05),
                        }
                    )
            log.info("  %s @ n=%d: %d/%d replications done", scenario.name, n, done, total)
    return pd.DataFrame(rows)


def summarize_power(power_df: pd.DataFrame, level: float = 0.05) -> pd.DataFrame:
    """Aggregate replications into ``(scenario, n, test) -> rejection_rate``."""
    if power_df.empty:
        return pd.DataFrame(columns=["scenario", "n", "test", "rejection_rate"])
    df = power_df.copy()
    df["rejected"] = (df["p_value"] < level) & (~df["p_value"].isna())
    return (
        df.groupby(["scenario", "n", "test"])["rejected"].mean().reset_index(name="rejection_rate")
    )
