"""Du-Escanciano (2017) ES and VaR backtests.

Implements the four basic statistics from the paper -- ``U_ES`` and
``C_ES(m)`` for Expected Shortfall, ``U_VaR`` and ``C_VaR(m)`` for
Value-at-Risk -- plus the robust ``MU_ES`` and ``MC_ES(m)`` variants
that correct for parameter estimation uncertainty by numerical
differentiation of the test functionals against the fitted parameters.

Notation follows the paper: ``h_t = 1{u_t <= alpha}`` is the binary
VaR violation indicator and ``H_t = (1/alpha) * (alpha - u_t) * h_t`` is
the continuous ES violation severity. Under correct conditional
specification, ``E[H_t] = alpha / 2`` and ``Var(H_t) = alpha * (1/3 -
alpha/4)``, which underpin all four asymptotic null distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from esback.backtests.base import BacktestResult
from esback.models.base import ModelFit
from esback.pit import pit_from_theta

REFERENCE = "Du, Escanciano (2017), Management Science 63(2)"

# Central-difference step heuristic for the numerical Jacobian used in the
# robust statistics MU_ES and MC_ES: ``step = max(|theta_k| * STEP_REL, STEP_MIN)``.
# These defaults trade rounding error (favoured by larger steps) against
# truncation error (favoured by smaller steps) for typical GARCH parameter
# scales; expose as kwargs of the robust functions so callers can tune.
DEFAULT_STEP_REL: float = 1e-5
DEFAULT_STEP_MIN: float = 1e-7


def var_violations(u_t: np.ndarray, alpha: float) -> np.ndarray:
    """Binary VaR violation indicator ``h_t = 1{u_t <= alpha}``."""
    return (u_t <= alpha).astype(float)


def es_severity(u_t: np.ndarray, alpha: float) -> np.ndarray:
    """Cumulative ES violation severity ``H_t = (1/alpha) * (alpha - u_t)+``."""
    return (1.0 / alpha) * (alpha - u_t) * (u_t <= alpha)


def _autocorrs(centered: np.ndarray, m: int) -> np.ndarray:
    """Sample autocorrelations ``rho_1, ..., rho_m`` of a centered series."""
    n = len(centered)
    gamma_0 = float(np.mean(centered**2))
    rho = np.zeros(m)
    for j in range(1, m + 1):
        gamma_j = float(np.sum(centered[j:] * centered[:-j]) / n)
        rho[j - 1] = gamma_j / gamma_0 if gamma_0 > 0 else 0.0
    return rho


def unconditional_es(u_t: np.ndarray, alpha: float) -> BacktestResult:
    """``U_ES``: unconditional ES test, eq. (5) of Du-Escanciano (2017).

    Asymptotic null distribution: standard normal.
    """
    n = len(u_t)
    H_t = es_severity(u_t, alpha)
    H_bar = float(np.mean(H_t))
    var_H = alpha * (1.0 / 3 - alpha / 4)
    statistic = float(np.sqrt(n) * (H_bar - alpha / 2) / np.sqrt(var_H))
    p_value = float(2 * (1 - stats.norm.cdf(abs(statistic))))
    return BacktestResult(
        name="U_ES",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        reference=REFERENCE,
        extra={"H_bar": H_bar, "n": n, "H_t": H_t},
    )


def conditional_es(u_t: np.ndarray, alpha: float, m: int = 5) -> BacktestResult:
    """``C_ES(m)``: conditional ES Box-Pierce test on cumulative violations.

    Asymptotic null distribution: chi-squared with ``m`` degrees of freedom.
    """
    n = len(u_t)
    H_t = es_severity(u_t, alpha)
    rho = _autocorrs(H_t - alpha / 2, m)
    statistic = float(n * np.sum(rho**2))
    p_value = float(1 - stats.chi2.cdf(statistic, df=m))
    return BacktestResult(
        name=f"C_ES({m})",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        reference=REFERENCE,
        extra={"rho": rho, "n": n, "m": m, "H_t": H_t},
    )


def unconditional_var(u_t: np.ndarray, alpha: float) -> BacktestResult:
    """``U_VaR``: unconditional VaR test, Kupiec-type proportion-of-failures.

    Asymptotic null distribution: standard normal.
    """
    n = len(u_t)
    h_t = var_violations(u_t, alpha)
    h_bar = float(np.mean(h_t))
    statistic = float(np.sqrt(n) * (h_bar - alpha) / np.sqrt(alpha * (1 - alpha)))
    p_value = float(2 * (1 - stats.norm.cdf(abs(statistic))))
    return BacktestResult(
        name="U_VaR",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        reference=REFERENCE,
        extra={"h_bar": h_bar, "n": n, "h_t": h_t},
    )


def conditional_var(u_t: np.ndarray, alpha: float, m: int = 5) -> BacktestResult:
    """``C_VaR(m)``: conditional VaR Box-Pierce test on binary violations.

    Asymptotic null distribution: chi-squared with ``m`` degrees of freedom.
    """
    n = len(u_t)
    h_t = var_violations(u_t, alpha)
    rho = _autocorrs(h_t - alpha, m)
    statistic = float(n * np.sum(rho**2))
    p_value = float(1 - stats.chi2.cdf(statistic, df=m))
    return BacktestResult(
        name=f"C_VaR({m})",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        reference=REFERENCE,
        extra={"rho": rho, "n": n, "m": m, "h_t": h_t},
    )


def _theta_vector(fit: ModelFit) -> np.ndarray:
    return np.array([fit.params[name] for name in fit.param_names])


def _numerical_step(theta_k: float, rel: float, floor: float) -> float:
    return max(abs(theta_k) * rel, floor)


def robust_unconditional_es(
    u_t: np.ndarray,
    returns_all: pd.Series,
    fit: ModelFit,
    is_end_idx: int,
    oos_end_idx: int,
    alpha: float,
    *,
    step_rel: float = DEFAULT_STEP_REL,
    step_min: float = DEFAULT_STEP_MIN,
) -> BacktestResult:
    """``MU_ES``: robust unconditional ES test (eq. 8 of Du-Escanciano).

    Corrects ``U_ES`` for parameter estimation uncertainty by numerically
    differentiating ``H_bar`` with respect to the fitted parameters and
    inflating the variance with ``(n/T) * R' W R``, where ``W = T *
    Cov(theta_hat)``.

    Asymptotic null distribution: standard normal.
    """
    n = len(u_t)
    in_sample_T = fit.in_sample_n
    theta_hat = _theta_vector(fit)
    n_params = len(theta_hat)

    H_bar = float(np.mean(es_severity(u_t, alpha)))
    var_H = alpha * (1.0 / 3 - alpha / 4)

    R_ES = np.zeros(n_params)
    for k in range(n_params):
        step = _numerical_step(theta_hat[k], step_rel, step_min)
        theta_plus = theta_hat.copy()
        theta_minus = theta_hat.copy()
        theta_plus[k] += step
        theta_minus[k] -= step
        u_plus = pit_from_theta(theta_plus, fit.param_names, returns_all, is_end_idx, oos_end_idx)
        u_minus = pit_from_theta(theta_minus, fit.param_names, returns_all, is_end_idx, oos_end_idx)
        H_bar_plus = float(np.mean(es_severity(u_plus, alpha)))
        H_bar_minus = float(np.mean(es_severity(u_minus, alpha)))
        R_ES[k] = (H_bar_plus - H_bar_minus) / (2 * step)

    W_T = in_sample_T * fit.param_cov
    correction = (n / in_sample_T) * R_ES @ W_T @ R_ES
    denom = var_H + correction

    if denom <= 0:
        return BacktestResult(
            name="MU_ES",
            statistic=float("nan"),
            p_value=float("nan"),
            alpha=alpha,
            reference=REFERENCE,
            extra={"reason": "denominator non-positive", "R_ES": R_ES},
        )

    statistic = float(np.sqrt(n) * (H_bar - alpha / 2) / np.sqrt(denom))
    p_value = float(2 * (1 - stats.norm.cdf(abs(statistic))))
    return BacktestResult(
        name="MU_ES",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        reference=REFERENCE,
        extra={"R_ES": R_ES, "H_bar": H_bar, "n": n, "T": in_sample_T},
    )


def robust_conditional_es(
    u_t: np.ndarray,
    returns_all: pd.Series,
    fit: ModelFit,
    is_end_idx: int,
    oos_end_idx: int,
    alpha: float,
    m: int = 5,
    *,
    step_rel: float = DEFAULT_STEP_REL,
    step_min: float = DEFAULT_STEP_MIN,
) -> BacktestResult:
    """``MC_ES(m)``: robust conditional ES test (eq. 9 of Du-Escanciano).

    Corrects ``C_ES(m)`` for parameter estimation uncertainty by replacing
    the identity covariance of the autocorrelation vector with
    ``I + (n/T) * R W R'``, where ``R`` collects the gradients of the
    autocorrelations against the fitted parameters.

    Asymptotic null distribution: chi-squared with ``m`` degrees of freedom.
    """
    n = len(u_t)
    in_sample_T = fit.in_sample_n
    theta_hat = _theta_vector(fit)
    n_params = len(theta_hat)

    H_t = es_severity(u_t, alpha)
    rho_hat = _autocorrs(H_t - alpha / 2, m)

    R_j = np.zeros((m, n_params))
    for k in range(n_params):
        step = _numerical_step(theta_hat[k], step_rel, step_min)
        theta_plus = theta_hat.copy()
        theta_minus = theta_hat.copy()
        theta_plus[k] += step
        theta_minus[k] -= step
        u_plus = pit_from_theta(theta_plus, fit.param_names, returns_all, is_end_idx, oos_end_idx)
        u_minus = pit_from_theta(theta_minus, fit.param_names, returns_all, is_end_idx, oos_end_idx)
        rho_plus = _autocorrs(es_severity(u_plus, alpha) - alpha / 2, m)
        rho_minus = _autocorrs(es_severity(u_minus, alpha) - alpha / 2, m)
        R_j[:, k] = (rho_plus - rho_minus) / (2 * step)

    W_T = in_sample_T * fit.param_cov
    Sigma = np.eye(m) + (n / in_sample_T) * R_j @ W_T @ R_j.T
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return BacktestResult(
            name=f"MC_ES({m})",
            statistic=float("nan"),
            p_value=float("nan"),
            alpha=alpha,
            reference=REFERENCE,
            extra={"reason": "Sigma matrix singular", "R_j": R_j},
        )

    statistic = float(n * rho_hat @ Sigma_inv @ rho_hat)
    p_value = float(1 - stats.chi2.cdf(statistic, df=m))
    return BacktestResult(
        name=f"MC_ES({m})",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        reference=REFERENCE,
        extra={"R_j": R_j, "rho": rho_hat, "n": n, "T": in_sample_T, "m": m},
    )


def run_battery(
    u_t: np.ndarray,
    alpha_es: float = 0.10,
    alpha_var: float = 0.05,
    m: int = 5,
) -> dict[str, BacktestResult]:
    """Run the four basic Du-Escanciano backtests on a PIT series."""
    return {
        "U_ES": unconditional_es(u_t, alpha_es),
        f"C_ES({m})": conditional_es(u_t, alpha_es, m),
        "U_VaR": unconditional_var(u_t, alpha_var),
        f"C_VaR({m})": conditional_var(u_t, alpha_var, m),
    }


def run_robust_battery(
    u_t: np.ndarray,
    returns_all: pd.Series,
    fit: ModelFit,
    is_end_idx: int,
    oos_end_idx: int,
    alpha_es: float = 0.10,
    m: int = 5,
) -> dict[str, BacktestResult]:
    """Run the two robust ES backtests."""
    return {
        "MU_ES": robust_unconditional_es(u_t, returns_all, fit, is_end_idx, oos_end_idx, alpha_es),
        f"MC_ES({m})": robust_conditional_es(
            u_t, returns_all, fit, is_end_idx, oos_end_idx, alpha_es, m
        ),
    }
