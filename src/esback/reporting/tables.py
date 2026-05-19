"""Result tables and summaries.

Pure functions that turn dictionaries of :class:`BacktestResult` into
pandas DataFrames suitable for printing, exporting to markdown or LaTeX,
or feeding figure helpers.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from esback.backtests.base import BacktestResult


def battery_to_dataframe(
    battery: Mapping[str, BacktestResult],
    decision_level: float = 0.05,
) -> pd.DataFrame:
    """Format a battery of backtest results as a tidy DataFrame.

    Returns a DataFrame indexed by test name with columns ``statistic``,
    ``p_value``, ``alpha``, ``decision``, ``reference``.
    """
    rows = [
        {
            "name": name,
            "statistic": result.statistic,
            "p_value": result.p_value,
            "alpha": result.alpha,
            "decision": result.decision(level=decision_level),
            "reference": result.reference,
        }
        for name, result in battery.items()
    ]
    return pd.DataFrame(rows).set_index("name")


def battery_summary(
    battery: Mapping[str, BacktestResult],
    decision_level: float = 0.05,
) -> pd.DataFrame:
    """Compact summary without the reference column."""
    return battery_to_dataframe(battery, decision_level)[
        ["statistic", "p_value", "alpha", "decision"]
    ]


def comparison_table(
    period_batteries: Mapping[str, Mapping[str, BacktestResult]],
    decision_level: float = 0.05,
) -> pd.DataFrame:
    """Side-by-side comparison of one battery across multiple periods.

    Returns a DataFrame indexed by test name; columns are a MultiIndex on
    ``(period, {statistic, p_value, alpha, decision})``.
    """
    pieces = []
    for period_name, battery in period_batteries.items():
        df = battery_summary(battery, decision_level).copy()
        df.columns = pd.MultiIndex.from_product([[period_name], df.columns])
        pieces.append(df)
    return pd.concat(pieces, axis=1)
