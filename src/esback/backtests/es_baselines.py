"""ES baselines: Acerbi-Szekely Z1 / Z2 and McNeil-Frey bootstrap.

Acerbi-Szekely (2014) introduced two simulation-based tests that are
recommended by regulators for ES validation. ``Z1`` looks at the average
standardised exceedance over violations; ``Z2`` looks at the average
weighted by the violation indicator over the whole window. Both are
test statistics whose null distribution must be approximated by Monte
Carlo simulation under a user-supplied generative model -- here the
:class:`StandardizedTSimulator` from the AR(1)-GARCH(1,1)-t baseline.

McNeil & Frey (2000) test that the standardised exceedances
``(X_t - ES_t) / sigma_t`` averaged over violations have zero mean,
with a stationary bootstrap p-value computed by resampling the
centred exceedances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from esback.backtests.base import BacktestResult


class ESSimulator(Protocol):
    """Protocol for parametric simulators used by the Acerbi-Szekely tests.

    Any object with a ``simulate_returns(n_sim, rng) -> ndarray`` method
    returning a matrix of shape ``(n_sim, n_oos)`` is a valid simulator.
    """

    def simulate_returns(self, n_sim: int, rng: np.random.Generator) -> np.ndarray: ...


@dataclass(frozen=True)
class StandardizedTSimulator:
    """Parametric simulator for AR(1)-GARCH(1,1) with standardised-t innovations.

    Used by Acerbi-Szekely tests to draw realisations of the OOS return
    process under H0. Each draw uses the same conditional moments
    (``mu_oos``, ``sigma_oos``) as the actual data and replaces the
    realised innovations with fresh i.i.d. ``standardised t(nu)`` draws.
    """

    mu_oos: np.ndarray
    sigma_oos: np.ndarray
    nu: float

    def simulate_returns(self, n_sim: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``shape (n_sim, n_oos)`` matrix of simulated returns under H0."""
        n_oos = len(self.mu_oos)
        scale = float(np.sqrt((self.nu - 2) / self.nu))
        eps = rng.standard_t(df=self.nu, size=(n_sim, n_oos)) * scale
        return self.mu_oos[None, :] + self.sigma_oos[None, :] * eps


@dataclass(frozen=True)
class MDNSimulator:
    """Parametric simulator for an MDN that outputs a Student-$t$ per day.

    Each OOS day ``t`` carries its own ``(loc_t, scale_t, df_t)``; a
    single simulation path draws an independent Student-$t$ sample at
    every day under those conditional parameters. The simulator
    satisfies the :class:`ESSimulator` protocol and is therefore a
    drop-in replacement for :class:`StandardizedTSimulator` in the
    Acerbi-Szekely tests when the conditional distribution is supplied
    by an MDN rather than by a single GARCH fit.
    """

    loc_oos: np.ndarray
    scale_oos: np.ndarray
    df_oos: np.ndarray

    def simulate_returns(self, n_sim: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``shape (n_sim, n_oos)`` matrix of simulated returns under H0."""
        n_oos = len(self.loc_oos)
        eps = rng.standard_t(df=self.df_oos[None, :], size=(n_sim, n_oos))
        return self.loc_oos[None, :] + self.scale_oos[None, :] * eps


def _z1_value(losses: np.ndarray, es_t: np.ndarray, violations: np.ndarray) -> float:
    n_viol = int(np.sum(violations))
    if n_viol == 0:
        return float("nan")
    return float(np.sum(losses[violations] / es_t[violations]) / n_viol - 1)


def acerbi_szekely_z1(
    oos_returns: np.ndarray,
    var_t: np.ndarray,
    es_t: np.ndarray,
    *,
    simulator: ESSimulator | None = None,
    n_sim: int = 10_000,
    seed: int | None = 42,
) -> BacktestResult:
    """Acerbi-Szekely (2014) ``Z1`` test, the average-exceedance ES check.

    Z1 = (1 / N_viol) * sum_{t: violation} (-Y_t / ES_t) - 1.
    Under correct ES specification, ``E[Z1] = 0``. Z1 > 0 indicates the
    model underestimates ES.

    A parametric Monte-Carlo p-value is returned when ``simulator`` is
    provided; otherwise the p-value is NaN and only the statistic is
    reported. The p-value is one-sided in the upper tail (``P(Z1_sim >=
    Z1_obs)``) following the regulator-relevant convention of testing
    ES underestimation.
    """
    losses = -oos_returns
    violations = losses > var_t
    n_viol = int(np.sum(violations))

    if n_viol == 0:
        return BacktestResult(
            name="AS_Z1",
            statistic=float("nan"),
            p_value=float("nan"),
            alpha=float("nan"),
            reference="Acerbi, Szekely (2014), Risk Magazine",
            extra={"n_violations": 0, "reason": "no violations in OOS window"},
        )

    z1 = _z1_value(losses, es_t, violations)

    p_value = float("nan")
    if simulator is not None:
        rng = np.random.default_rng(seed)
        sim_returns = simulator.simulate_returns(n_sim, rng)
        sim_losses = -sim_returns
        sim_violations = sim_losses > var_t[None, :]
        ratios = sim_losses / es_t[None, :]
        masked_sums = np.where(sim_violations, ratios, 0.0).sum(axis=1)
        counts = sim_violations.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z1_sims = np.where(counts > 0, masked_sums / counts - 1, np.nan)
        valid = ~np.isnan(z1_sims)
        if valid.any():
            p_value = float(np.mean(z1_sims[valid] >= z1))

    return BacktestResult(
        name="AS_Z1",
        statistic=z1,
        p_value=p_value,
        alpha=float("nan"),
        reference="Acerbi, Szekely (2014), Risk Magazine",
        extra={"n_violations": n_viol, "n_total": len(oos_returns)},
    )


def acerbi_szekely_z2(
    oos_returns: np.ndarray,
    var_t: np.ndarray,
    es_t: np.ndarray,
    alpha: float,
    *,
    simulator: ESSimulator | None = None,
    n_sim: int = 10_000,
    seed: int | None = 42,
) -> BacktestResult:
    """Acerbi-Szekely (2014) ``Z2`` test, the quantile-respect ES check.

    Z2 = (1 / (N * alpha)) * sum_t (X_t / ES_t) * I_t - 1, where X_t is
    the loss, I_t is the violation indicator, and the sum runs over the
    full OOS window. Under correct specification ``E[Z2] = 0`` because
    ``E[X_t * I_t] = alpha * ES_t``. ``Z2 > 0`` indicates ES under-
    estimation. Unlike ``Z1``, ``Z2`` does not vanish when the OOS
    window happens to contain no violations.

    The function expects ``var_t`` and ``es_t`` evaluated at the same
    ``alpha``. A parametric Monte-Carlo p-value is returned when
    ``simulator`` is provided; one-sided upper-tail by convention
    (``P(Z2_sim >= Z2_obs)``).
    """
    n = len(oos_returns)
    losses = -oos_returns
    violations = losses > var_t

    z2 = float(np.sum((losses / es_t) * violations) / (n * alpha) - 1)

    p_value = float("nan")
    if simulator is not None:
        rng = np.random.default_rng(seed)
        sim_returns = simulator.simulate_returns(n_sim, rng)
        sim_losses = -sim_returns
        sim_violations = sim_losses > var_t[None, :]
        sim_ratios = (sim_losses / es_t[None, :]) * sim_violations
        z2_sims = sim_ratios.sum(axis=1) / (n * alpha) - 1
        p_value = float(np.mean(z2_sims >= z2))

    return BacktestResult(
        name="AS_Z2",
        statistic=z2,
        p_value=p_value,
        alpha=alpha,
        reference="Acerbi, Szekely (2014), Risk Magazine",
        extra={"n_total": n, "n_violations": int(violations.sum())},
    )


def mcneil_frey(
    oos_returns: np.ndarray,
    var_t: np.ndarray,
    es_t: np.ndarray,
    sigma_oos: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    seed: int | None = 42,
) -> BacktestResult:
    """McNeil & Frey (2000) bootstrap test on standardised exceedances.

    Forms ``r_t = (X_t - ES_t) / sigma_t`` over the violation days. Under
    H0, and **provided ES_t is computed at the same alpha as VaR_t**
    (the standard convention), ``E[r_t | violation] = 0``. The test
    statistic is the t-ratio ``mean(r) / (std(r) / sqrt(N_viol))``; its
    p-value is the two-sided bootstrap probability ``P(|t_boot| >=
    |t_obs|)``, with bootstrap samples drawn from the centred ``r``.

    Note: this test is invalid when ``alpha_ES != alpha_VaR``, because
    ``E[r_t]`` then has a deterministic non-zero offset and any size
    interpretation would be wrong.
    """
    losses = -oos_returns
    violations = losses > var_t
    n_viol = int(np.sum(violations))

    if n_viol < 2:
        return BacktestResult(
            name="McNeil_Frey",
            statistic=float("nan"),
            p_value=float("nan"),
            alpha=float("nan"),
            reference="McNeil, Frey (2000), J. Empirical Finance 7",
            extra={"n_violations": n_viol, "reason": "fewer than 2 violations"},
        )

    r = (losses[violations] - es_t[violations]) / sigma_oos[violations]
    se = float(np.std(r, ddof=1) / np.sqrt(n_viol))
    statistic = float(np.mean(r) / se) if se > 0 else float("nan")

    rng = np.random.default_rng(seed)
    r_centered = r - np.mean(r)
    indices = rng.integers(0, n_viol, size=(n_bootstrap, n_viol))
    boot_means = r_centered[indices].mean(axis=1)
    boot_stat = boot_means / se if se > 0 else np.full(n_bootstrap, np.nan)

    if np.isnan(statistic):
        p_value = float("nan")
    else:
        p_value = float(np.mean(np.abs(boot_stat) >= abs(statistic)))

    return BacktestResult(
        name="McNeil_Frey",
        statistic=statistic,
        p_value=p_value,
        alpha=float("nan"),
        reference="McNeil, Frey (2000), J. Empirical Finance 7",
        extra={"n_violations": n_viol, "exceedance_mean": float(np.mean(r))},
    )
