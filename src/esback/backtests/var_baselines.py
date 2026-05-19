"""VaR baselines: Kupiec proportion-of-failures, Christoffersen tests,
Kratz-Lok-McNeil multinomial.

These complement the Du-Escanciano U_VaR / C_VaR statistics with the
classical likelihood-ratio versions and the Basel-FRTB-compatible
multinomial check on multiple VaR levels.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from esback.backtests.base import BacktestResult


def kupiec_pof(h_t: np.ndarray, alpha: float) -> BacktestResult:
    """Kupiec (1995) Proportion-of-Failures likelihood-ratio test.

    Tests H0: ``P(violation) = alpha`` against H1 by comparing the
    binomial likelihood at the model-implied alpha to the unrestricted MLE
    ``pi_hat = n1 / n``.

    Asymptotic null: chi-squared with 1 degree of freedom.
    Note: returns NaN when there are zero or n violations (degenerate
    likelihood at the boundary).
    """
    n = len(h_t)
    n1 = int(np.sum(h_t))
    n0 = n - n1

    if n1 == 0 or n0 == 0:
        return BacktestResult(
            name="Kupiec_POF",
            statistic=float("nan"),
            p_value=float("nan"),
            alpha=alpha,
            reference="Kupiec (1995), J. Derivatives 3(2)",
            extra={
                "n_violations": n1,
                "n_total": n,
                "reason": "degenerate (0 or n violations)",
            },
        )

    pi_hat = n1 / n
    log_l_restricted = n1 * np.log(alpha) + n0 * np.log(1 - alpha)
    log_l_unrestricted = n1 * np.log(pi_hat) + n0 * np.log(1 - pi_hat)
    statistic = float(-2 * (log_l_restricted - log_l_unrestricted))
    p_value = float(1 - stats.chi2.cdf(statistic, df=1))

    return BacktestResult(
        name="Kupiec_POF",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        reference="Kupiec (1995), J. Derivatives 3(2)",
        extra={"n_violations": n1, "n_total": n, "pi_hat": float(pi_hat)},
    )


def christoffersen_cci(h_t: np.ndarray, alpha: float) -> BacktestResult:
    """Christoffersen (1998) Conditional Coverage Independence test.

    Models the violation series as a two-state Markov chain and tests
    H0: ``pi_01 = pi_11`` (i.e. transitioning to a violation does not
    depend on whether yesterday was a violation) against H1.

    Asymptotic null: chi-squared with 1 degree of freedom.
    Note: returns NaN when the chain is degenerate (no transitions of
    one of the four kinds).
    """
    n_total = len(h_t)
    h_int = h_t.astype(int)
    prev = h_int[:-1]
    curr = h_int[1:]

    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    n_pairs = n00 + n01 + n10 + n11
    n_violations_in_pairs = n01 + n11

    if n_violations_in_pairs == 0 or n_pairs == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return BacktestResult(
            name="Christoffersen_CCI",
            statistic=float("nan"),
            p_value=float("nan"),
            alpha=alpha,
            reference="Christoffersen (1998), International Economic Review 39(4)",
            extra={
                "n00": n00,
                "n01": n01,
                "n10": n10,
                "n11": n11,
                "reason": "degenerate Markov chain",
            },
        )

    pi_01 = n01 / (n00 + n01)
    pi_11 = n11 / (n10 + n11)
    pi_uncond = n_violations_in_pairs / n_pairs

    def _binom_loglik(n_zero: int, n_one: int, p: float) -> float:
        if not 0.0 < p < 1.0:
            return 0.0
        return float(n_zero * np.log(1 - p) + n_one * np.log(p))

    log_l_restricted = _binom_loglik(n00 + n10, n_violations_in_pairs, pi_uncond)
    log_l_unrestricted = _binom_loglik(n00, n01, pi_01) + _binom_loglik(n10, n11, pi_11)

    statistic = float(-2 * (log_l_restricted - log_l_unrestricted))
    p_value = float(1 - stats.chi2.cdf(statistic, df=1))

    return BacktestResult(
        name="Christoffersen_CCI",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        reference="Christoffersen (1998), International Economic Review 39(4)",
        extra={
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "pi_01": float(pi_01),
            "pi_11": float(pi_11),
            "n_total": n_total,
        },
    )


def christoffersen_cc(h_t: np.ndarray, alpha: float) -> BacktestResult:
    """Combined conditional coverage test ``LR_CC = LR_UC + LR_IND``.

    Asymptotic null: chi-squared with 2 degrees of freedom.
    """
    uc = kupiec_pof(h_t, alpha)
    ind = christoffersen_cci(h_t, alpha)

    if np.isnan(uc.statistic) or np.isnan(ind.statistic):
        return BacktestResult(
            name="Christoffersen_CC",
            statistic=float("nan"),
            p_value=float("nan"),
            alpha=alpha,
            reference="Christoffersen (1998), International Economic Review 39(4)",
            extra={"LR_UC": uc.statistic, "LR_IND": ind.statistic},
        )

    statistic = float(uc.statistic + ind.statistic)
    p_value = float(1 - stats.chi2.cdf(statistic, df=2))
    return BacktestResult(
        name="Christoffersen_CC",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        reference="Christoffersen (1998), International Economic Review 39(4)",
        extra={"LR_UC": uc.statistic, "LR_IND": ind.statistic},
    )


def kratz_lok_mcneil(
    u_t: np.ndarray,
    levels: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10),
) -> BacktestResult:
    """Kratz, Lok, McNeil (2018) multinomial backtest on multiple VaR levels.

    Bins the PIT residuals into the cells ``(0, l_1], (l_1, l_2], ...,
    (l_K, 1)`` and compares the observed counts to the expected
    proportions via a Pearson chi-squared statistic. Approves the model
    only if the *whole* multi-level violation profile is consistent,
    which is the FRTB-recommended way of backtesting ES indirectly.

    Asymptotic null: chi-squared with ``K`` degrees of freedom (one less
    than the number of bins).
    """
    levels_sorted = tuple(sorted(levels))
    edges = (0.0, *levels_sorted, 1.0)
    n = len(u_t)

    observed = np.histogram(u_t, bins=edges)[0]
    expected = np.diff(edges) * n
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    df = len(levels_sorted)
    p_value = float(1 - stats.chi2.cdf(statistic, df=df))

    return BacktestResult(
        name=f"KLM_multinomial({df})",
        statistic=statistic,
        p_value=p_value,
        alpha=levels_sorted[-1],
        reference="Kratz, Lok, McNeil (2018), J. Banking & Finance 88",
        extra={
            "levels": list(levels_sorted),
            "observed": observed.tolist(),
            "expected": expected.tolist(),
            "df": df,
            "n": n,
        },
    )
