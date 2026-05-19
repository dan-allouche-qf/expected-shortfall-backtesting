"""Unified backtest battery.

Wraps the per-paper modules (:mod:`esback.backtests.du_escanciano`,
:mod:`esback.backtests.var_baselines`, :mod:`esback.backtests.es_baselines`)
into a single ``run_full_battery`` entry point so the CLI and the
notebooks evaluate every test on the same OOS slice without having to
juggle individual function signatures.

The battery is opportunistic: it returns whatever it can given the
inputs you pass. Robust Du-Escanciano statistics need a fitted model
and the in-sample window indices; Acerbi-Szekely tests benefit from a
parametric simulator. When those optional inputs are missing the
corresponding tests are simply skipped.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from esback.backtests import du_escanciano
from esback.backtests.base import BacktestResult
from esback.backtests.es_baselines import (
    ESSimulator,
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    mcneil_frey,
)
from esback.backtests.var_baselines import (
    christoffersen_cc,
    christoffersen_cci,
    kratz_lok_mcneil,
    kupiec_pof,
)
from esback.models.base import ModelFit


def run_full_battery(
    *,
    u_t: np.ndarray,
    oos_returns: np.ndarray,
    var_t: np.ndarray,
    es_t: np.ndarray,
    sigma_oos: np.ndarray,
    h_t: np.ndarray,
    alpha_es: float,
    alpha_var: float,
    m: int = 5,
    klm_levels: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10),
    fit: ModelFit | None = None,
    returns_all: pd.Series | None = None,
    is_end_idx: int | None = None,
    oos_end_idx: int | None = None,
    es_simulator: ESSimulator | None = None,
    n_sim: int = 10_000,
    n_bootstrap: int = 10_000,
    seed: int | None = 42,
) -> dict[str, BacktestResult]:
    """Evaluate every available backtest on a single OOS slice.

    The Du-Escanciano basic tests, the Kupiec / Christoffersen / KLM
    VaR baselines, and the McNeil-Frey ES bootstrap always run. The
    Du-Escanciano robust tests (``MU_ES``, ``MC_ES``) require ``fit``,
    ``returns_all``, ``is_end_idx``, and ``oos_end_idx``. The
    Acerbi-Szekely tests return a Monte-Carlo p-value when
    ``es_simulator`` is supplied; otherwise only the statistic is
    reported.

    Note: the Acerbi-Szekely / McNeil-Frey statistics assume
    ``alpha_VaR == alpha_ES``. When the project's Du-Escanciano
    convention ``alpha_ES = 2 * alpha_VaR`` is in use, ``var_t`` and
    ``es_t`` need to be re-evaluated at ``alpha_var`` and supplied as
    such to this function. This keeps the unified battery
    interpretable.
    """
    results: dict[str, BacktestResult] = {}

    results.update(du_escanciano.run_battery(u_t, alpha_es=alpha_es, alpha_var=alpha_var, m=m))

    if (
        fit is not None
        and returns_all is not None
        and is_end_idx is not None
        and oos_end_idx is not None
    ):
        results.update(
            du_escanciano.run_robust_battery(
                u_t,
                returns_all,
                fit,
                is_end_idx,
                oos_end_idx,
                alpha_es=alpha_es,
                m=m,
            )
        )

    results["Kupiec_POF"] = kupiec_pof(h_t, alpha_var)
    results["Christoffersen_CCI"] = christoffersen_cci(h_t, alpha_var)
    results["Christoffersen_CC"] = christoffersen_cc(h_t, alpha_var)
    klm_result = kratz_lok_mcneil(u_t, klm_levels)
    results[klm_result.name] = klm_result

    results["AS_Z1"] = acerbi_szekely_z1(
        oos_returns, var_t, es_t, simulator=es_simulator, n_sim=n_sim, seed=seed
    )
    results["AS_Z2"] = acerbi_szekely_z2(
        oos_returns,
        var_t,
        es_t,
        alpha_var,
        simulator=es_simulator,
        n_sim=n_sim,
        seed=seed,
    )
    results["McNeil_Frey"] = mcneil_frey(
        oos_returns, var_t, es_t, sigma_oos, n_bootstrap=n_bootstrap, seed=seed
    )

    return results


def battery_dataframe(
    results: Mapping[str, BacktestResult],
    decision_level: float = 0.05,
) -> pd.DataFrame:
    """Convenience wrapper around :func:`esback.reporting.tables.battery_to_dataframe`.

    Re-exposed here so the unified battery has a one-call rendering
    path without forcing callers to import a second module.
    """
    from esback.reporting.tables import battery_to_dataframe

    return battery_to_dataframe(results, decision_level=decision_level)
