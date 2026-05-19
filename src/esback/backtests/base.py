"""Shared types for backtest outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a single backtest evaluation.

    Parameters
    ----------
    name
        Short identifier such as ``"U_ES"`` or ``"C_ES(5)"``.
    statistic
        Realised value of the test statistic.
    p_value
        Asymptotic p-value under the null of correct specification.
    alpha
        Risk level the test was evaluated at.
    reference
        Bibliographic reference for the test.
    extra
        Optional structured payload (autocorrelations, intermediate
        quantities) that downstream reporters may use.
    """

    name: str
    statistic: float
    p_value: float
    alpha: float
    reference: str
    extra: dict[str, Any] = field(default_factory=dict)

    def decision(self, level: float = 0.05) -> str:
        if math.isnan(self.p_value):
            return "N/A"
        return "REJECT" if self.p_value < level else "Fail to reject"
