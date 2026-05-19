"""Common types for conditional risk models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelFit:
    """Outcome of estimating a conditional risk model.

    Attributes
    ----------
    params
        Estimated parameters keyed by their model-specific name (e.g.
        ``"omega"``, ``"alpha[1]"``, ``"nu"``).
    param_names
        Order in which the parameters appear in ``param_cov``.
    param_cov
        Asymptotic covariance matrix of the parameter estimator,
        rows/columns in ``param_names`` order. Used by the robust
        backtests of Du-Escanciano (2017).
    in_sample_n
        Number of in-sample observations the fit was based on.
    log_likelihood
        Maximised log-likelihood, when the underlying optimiser exposes it.
    """

    params: dict[str, float]
    param_names: list[str]
    param_cov: np.ndarray
    in_sample_n: int
    log_likelihood: float | None = None


@runtime_checkable
class RiskModel(Protocol):
    """Common interface for conditional risk models.

    Implementations live in :mod:`esback.models.ar1_garch_t` and
    :mod:`esback.models.garch_variants`.
    """

    def fit(self, returns: pd.Series) -> ModelFit: ...
