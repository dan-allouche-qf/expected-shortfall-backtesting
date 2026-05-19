# Methodology

This document explains the statistical machinery behind `esback`: the
notation, the eleven backtests, the forecasting models, and the Monte
Carlo size-and-power study. It is written for a reader who knows the
definitions of Value-at-Risk and Expected Shortfall but has not read
the underlying papers.

- [1. Setup and notation](#1-setup-and-notation)
- [2. Backtest battery](#2-backtest-battery)
- [3. Forecasting models](#3-forecasting-models)
- [4. Monte Carlo size-and-power study](#4-monte-carlo-size-and-power-study)
- [5. References](#5-references)

---

## 1. Setup and notation

Let `Y_t` be the (log-)return at date `t`, and let `F_t` denote the
conditional distribution of `Y_t` given the information available
through `t-1`. Two risk measures sit on the left tail of `F_t`:

- **Value-at-Risk** at level `alpha_VaR`:
  `VaR_t(alpha) = -F_t^{-1}(alpha)`. With our sign convention a loss is
  `-Y_t > 0`, so `P(-Y_t > VaR_t(alpha)) = alpha`.
- **Expected Shortfall** at level `alpha_ES`:
  `ES_t(alpha) = E[-Y_t | -Y_t > VaR_t(alpha)]`. ES averages the loss
  conditional on a VaR exception.

### 1.1 PIT residuals

Given a one-step-ahead forecast `F_t`, the **Probability Integral
Transform** of the realised return is `u_t = F_t(Y_t)`. Under correct
conditional specification, `{u_t}` is i.i.d. Uniform(0,1).

PIT residuals are the substrate the Du-Escanciano battery operates on:
all four basic statistics are written in terms of `u_t` alone. The
project computes them in `src/esback/pit.py` (Student-t standardised
and Normal innovation flavours). The KS test against Uniform(0,1) is
exposed as `ks_test_uniform` and is the diagnostic surfaced in the
`plot_pit_diagnostic` figure.

### 1.2 Two violation indicators: `h_t` and `H_t`

Two derived series carry the test signal:

- **Binary VaR violation**: `h_t = 1{u_t <= alpha}`. Under correct
  specification `E[h_t] = alpha` and `Var(h_t) = alpha(1 - alpha)`.
- **Cumulative ES severity**:
  `H_t = (1/alpha) * (alpha - u_t) * 1{u_t <= alpha}`. Under correct
  specification `E[H_t] = alpha/2` and
  `Var(H_t) = alpha * (1/3 - alpha/4)`.

`H_t` is the continuous counterpart of `h_t`: it equals zero on a
non-violation day, and on a violation day it grows linearly with
how far in the tail the loss lies. These four moments underpin all
four Du-Escanciano asymptotic null distributions.

### 1.3 Two alpha conventions

The library deliberately keeps two conventions side by side, so the
reader of a results table can tell them apart:

- **Du-Escanciano**: `alpha_ES = 2 * alpha_VaR`. The ES is reported at
  twice the VaR level (e.g. `alpha_VaR = 0.05`, `alpha_ES = 0.10`),
  which is what the 2017 paper uses.
- **Acerbi-Szekely / McNeil-Frey**: `alpha_ES = alpha_VaR`. The ES is
  reported at the same level as the VaR (e.g. both at 0.05).

`run_full_battery` re-evaluates VaR and ES at `alpha_var` before
calling the Acerbi-Szekely / McNeil-Frey tests so the conventions never
get mixed up, while the Du-Escanciano statistics keep their own
`alpha_es`. The figure legends carry `alpha_var` and `alpha_es` as
required keyword arguments so a legend cannot silently misreport the
convention used by the caller.

---

## 2. Backtest battery

Eleven base tests are evaluated on the same out-of-sample slice (four
basic Du-Escanciano, four VaR baselines, three ES baselines). Two
robust Du-Escanciano variants (`MU_ES`, `MC_ES`) augment the basic
unconditional and conditional ES tests, so `run_full_battery` returns
up to thirteen `BacktestResult` entries when a fitted model is
supplied. The tests split into three groups: Du-Escanciano (basic +
robust), Value-at-Risk baselines, and Expected-Shortfall baselines.

`backtests.battery.run_full_battery` returns a `dict[str,
BacktestResult]` where each `BacktestResult` carries the statistic,
the p-value, the alpha used, the asymptotic / bootstrap reference, and
an extras dict for diagnostic quantities (number of violations, R
vector, denominator sign).

### 2.1 Du-Escanciano basic statistics

Implemented in `src/esback/backtests/du_escanciano.py`.

#### `U_ES` — unconditional ES (eq. 5)

`U_ES = sqrt(n) * (H_bar - alpha/2) / sqrt(alpha (1/3 - alpha/4))`,
where `H_bar = mean(H_t)`. Tests whether the average cumulative
severity matches its theoretical mean `alpha/2`.

Asymptotic null distribution: standard normal.

#### `C_ES(m)` — conditional ES (eq. 7)

Box-Pierce statistic on the first `m` autocorrelations of `H_t`,
centered at the theoretical mean `alpha/2`:

`C_ES(m) = n * sum_{j=1..m} rho_j^2`

where `rho_j` is the lag-`j` sample autocorrelation of
`H_t - alpha/2`. Detects clustering of ES exceedances.

Asymptotic null distribution: chi-squared with `m` degrees of freedom
(default `m = 5`).

#### `U_VaR`, `C_VaR(m)` — VaR analogues

Same construction with `h_t - alpha` in place of `H_t - alpha/2`. They
target VaR rather than ES coverage, and are reported alongside the ES
statistics in the same table for direct comparison.

### 2.2 Du-Escanciano robust statistics (eq. 8 and 9)

The robust statistics `MU_ES` and `MC_ES(m)` correct the basic
statistics for *parameter-estimation uncertainty*. The reason this
matters: `H_t` is computed from PIT residuals that depend on the
fitted parameters, so when the in-sample window `T` is comparable in
length to the out-of-sample window `n`, the variance of the test
statistic is no longer `alpha (1/3 - alpha/4)` and the test
size-distorts.

The fix is to add a `(n/T) * R' W R` term to the variance, where:

- `R = d H_bar / d theta` is the gradient of `H_bar` against the
  parameter vector `theta`, computed numerically by central differences
  with step `max(|theta_k| * DEFAULT_STEP_REL, DEFAULT_STEP_MIN)`. The
  default `DEFAULT_STEP_REL = 1e-5`, `DEFAULT_STEP_MIN = 1e-7` are
  the standard Nocedal-Wright trade-off between rounding and
  truncation error.
- `W = T * Cov(theta_hat)` is the asymptotic covariance of the
  parameter estimates, lifted directly from the `arch` fit.

`MU_ES` is the corrected unconditional statistic, `MC_ES(m)` the
corrected conditional one. The pedagogical figure
`plot_lambda_diagram` illustrates when this correction matters:
when `lambda = n/T` is small (long in-sample, short out-of-sample) the
basic statistics are fine; when `lambda ≈ 1` (comparable lengths) the
robust correction is essential.

### 2.3 Value-at-Risk baselines

Implemented in `src/esback/backtests/var_baselines.py`. These are the
classical pre-ES tests, kept for direct comparison.

#### `Kupiec_POF` (1995)

Likelihood-ratio "proportion-of-failures" test against the null that
the violation rate equals `alpha_VaR`:

`LR_POF = -2 [log(L0) - log(L1)] ~ chi2(1)`

where `L0` is the Bernoulli likelihood at the nominal rate and `L1`
at the empirical rate.

#### `Christoffersen_CCI` (1998) — independence

Markov-chain test of independence: are violations on consecutive
days independent? `LR_IND ~ chi2(1)`.

#### `Christoffersen_CC` (1998) — conditional coverage

Joint test of unconditional coverage *and* independence:
`LR_CC = LR_POF + LR_IND ~ chi2(2)`. The decomposition is exact, which
is why if `LR_CC` rejects but `LR_IND` does not, the rejection is
inflated coverage rather than clustering.

#### `KLM_multinomial(K)` — Kratz-Lok-McNeil (2018)

Tests VaR jointly at `K` levels using a multinomial likelihood. The
project uses `K = 4` (alpha levels `0.025, 0.05, 0.075, 0.10`,
matching the default `klm_levels` in `run_full_battery`); this is the
simplest implementation of "test ES by testing VaR at several levels
at once", an alternative to Acerbi-Szekely that avoids parametric
simulation. `chi2(K)` asymptotic null.

### 2.4 Expected-Shortfall baselines

Implemented in `src/esback/backtests/es_baselines.py`.

#### `AS_Z1` and `AS_Z2` — Acerbi-Szekely (2014)

Both statistics compare the realised tail losses against the model's
ES. They differ in the normaliser:

- `Z1`: average over the *empirical* violation set,
  `Z1 = (1/N_viol) sum_{t: viol} (-Y_t / ES_t) - 1`. Sensitive to
  whether losses *given a violation* are bigger than ES.
- `Z2`: average over the full window,
  `Z2 = sum_t (-Y_t * 1{viol}) / (N * alpha * ES_t) - 1`. Sensitive
  also to the *number* of violations.

The p-value is parametric Monte-Carlo: draw `n_sim` paths from the
fitted AR(1)-GARCH(1,1)-t at the same conditional moments and read off
the empirical quantile of the observed statistic against the
simulated distribution. The simulator is `StandardizedTSimulator` in
the same module.

#### `McNeil_Frey` (2000) — standardised exceedance bootstrap

For each VaR violation `t` define the *standardised exceedance*
`r_t = (-Y_t - VaR_t) / sigma_t`. Under correct ES specification the
`r_t` form a zero-mean i.i.d. sample. The test compares
`r_bar = mean(r_t)` against a stationary bootstrap distribution. The
`run_full_battery` default is `n_bootstrap = 10_000`; the power
study reduces this to `n_bootstrap = 200` to keep the Monte Carlo
grid affordable. Sensitive to ES being too small in the tail.

### 2.5 Summary table

| Test | Module | Targets | Asymptotic null | Typical use |
|------|--------|---------|-----------------|-------------|
| `U_ES`, `C_ES(m)` | `du_escanciano.py` | ES (uncond. + cond.) | N(0,1), chi2(m) | Headline ES backtest |
| `MU_ES`, `MC_ES(m)` | `du_escanciano.py` | ES with param-correction | N(0,1), chi2(m) | When n / T is not tiny |
| `U_VaR`, `C_VaR(m)` | `du_escanciano.py` | VaR (uncond. + cond.) | N(0,1), chi2(m) | VaR sanity check |
| `Kupiec_POF` | `var_baselines.py` | VaR coverage | chi2(1) | Classical, simple |
| `Christoffersen_CCI` | `var_baselines.py` | VaR independence | chi2(1) | Clustering-only |
| `Christoffersen_CC` | `var_baselines.py` | VaR coverage + indep. | chi2(2) | Joint VaR test |
| `KLM_multinomial(K)` | `var_baselines.py` | ES via multi-level VaR | chi2(K) | Non-parametric ES |
| `AS_Z1`, `AS_Z2` | `es_baselines.py` | ES via standardised tails | parametric MC | Acerbi-Szekely workhorses |
| `McNeil_Frey` | `es_baselines.py` | ES via exceedances | bootstrap | EVT-flavoured |

---

## 3. Forecasting models

### 3.1 AR(1)-GARCH(1,1)-t baseline

The baseline used throughout the project:

`Y_t = const + a0 * Y_{t-1} + eps_t`
`eps_t = sigma_t * z_t`
`sigma2_t = omega + alpha1 * eps_{t-1}^2 + beta1 * sigma2_{t-1}`
`z_t ~ standardised t(nu)`  (i.e. `Var(z_t) = 1`)

Maximum-likelihood estimation is delegated to the `arch` package;
the deterministic recursions (volatility path, OOS residual
extraction) are reimplemented in
`src/esback/models/ar1_garch_t.py` so they can be reused by the robust
statistics, which need to evaluate them at perturbed parameter
vectors.

### 3.2 GJR-GARCH and Normal variants

`src/esback/models/garch_variants.py` adds two orthogonal degrees of
freedom on top of the baseline:

- **Volatility model**: `GARCH` (symmetric, default) or `GJR` (asymmetric
  Glosten-Jagannathan-Runkle, adds the leverage term
  `gamma * 1{eps_{t-1} < 0} * eps_{t-1}^2`).
- **Innovation distribution**: `t` (default, fat-tailed) or `normal`.

This gives four (`vol`, `dist`) combinations: `(GARCH, t)`,
`(GARCH, normal)`, `(GJR, t)`, `(GJR, normal)`. They share a single
`_garch_recursion` helper in `ar1_garch_t.py`; `gamma = 0` recovers the
symmetric case. The helper clips `sigma2` to be non-negative before
taking the square root, so ill-conditioned MLE fits on very short OOS
windows do not produce `NaN` volatilities.

### 3.3 LSTM-Quantile baseline

For comparison, `src/esback/models/dl/lstm_quantile.py` ships a small
PyTorch sequence model: stacked LSTM hidden state mapped to two heads,
one for VaR and one for ES, parameterised so that `ES >= VaR` by
construction (the ES head adds `softplus(.)` to the VaR head). The
loss combines the pinball loss (for VaR) and the FZ0 loss of
Fissler-Ziegel (for ES jointly with VaR).

This baseline is not part of the headline replication; it lives in
notebook 04 as a contrast between a classical parametric model and a
modern neural one trained on the same data.

### 3.4 Mixture-density LSTM

`src/esback/models/dl/lstm_mdn.py` extends the LSTM-Quantile head into
a mixture-density network: the same stacked LSTM, but three linear
heads on the final hidden state output the Student-$t$ parameters
$(\mu_t, \sigma_t, \nu_t)$ of the conditional return distribution
$F_t = t_{\nu_t}(\mu_t, \sigma_t)$ rather than a single
$(\widehat{\VaR}_t, \widehat{\ES}_t)$ pair. The constraints
$\sigma_t > 0$ and $\nu_t > 2$ (finite variance) are enforced via
$\mathrm{softplus}$. The training loss is the negative log-likelihood
of the predicted distribution, which is a strictly proper scoring
rule.

Because the MDN supplies a closed-form CDF, every quantity the
unified battery needs falls out without estimation:

- the PIT residual $u_t = F_t(Y_t)$ for the Du-Escanciano
  unconditional and conditional statistics;
- $\widehat{\VaR}_t = -F_t^{-1}(\alpha)$ and $\widehat{\ES}_t$ via
  the closed-form Student-$t$ ES (no numerical integration);
- a parametric simulator `MDNSimulator` that draws fresh
  Student-$t$ paths under the per-day $(\mu_t, \sigma_t, \nu_t)$ for
  the Acerbi-Szekely Monte-Carlo p-values.

The two robust Du-Escanciano statistics ($MU_{ES}$, $MC_{ES}$) remain
out of reach because they need the parameter covariance of an MLE
fit; the SGD-trained network does not expose a tractable Hessian.
The MDN therefore covers eleven of the thirteen statistics. Notebook
06 reports the contrast against `GARCH-t` and `LSTM-Quantile` on the
COVID-19 OOS slice.

---

## 4. Monte Carlo size-and-power study

`src/esback/simulation/power_study.py` drives a Monte Carlo grid that
quantifies, for each test, how often it rejects under correct
specification (the *empirical size*) versus under three alternative
data-generating processes (the *power*).

### 4.1 Four scenarios

- `H0_garch_t`: AR(1)-GARCH(1,1)-t with `nu = 8.0`. The DGP matches
  the fitted model — every test should reject at ~5% as `n` grows.
- `H1_gjr_t`: GJR-GARCH(1,1,1)-t with `gamma = 0.15`. Moderate
  asymmetric leverage; the symmetric AR(1)-GARCH(1,1)-t baseline is
  *slightly* misspecified.
- `H1_gjr_strong`: GJR-GARCH(1,1,1)-t with `gamma = 0.25`. Strong
  asymmetric leverage; the misspecification is material.
- `H1_normal`: AR(1)-GARCH(1,1)-Normal. Same volatility dynamics, but
  Gaussian innovations instead of Student-t. The misspecification is
  on the tail thickness only.

### 4.2 Four sample sizes, one thousand replications

Sample sizes `n ∈ {250, 500, 1000, 2500}` cover the range that
practitioners typically face. Each `(scenario, n)` runs 1,000
replications; the in-sample fraction is 60 % and the OOS fraction is
40 %. The full grid contains 16,000 simulated paths and 176,000
test-replication rows.

### 4.3 The Acerbi-Szekely simulator is active in the power study

`run_power_study` instantiates a `StandardizedTSimulator` per
replication so the AS Z1 / Z2 p-values come from a parametric
Monte-Carlo draw on every path. The number of simulator draws per
replication is `n_sim_power = 2000` by default — lower than the
10000 used in single-period CLI runs to keep the total grid time
manageable. Raise it for sharper p-values at the cost of runtime.

### 4.4 What the curves should show

- Under H0 the empirical rejection rate should sit close to 5 %.
  Curves *above* 5 % indicate size distortion (over-rejection);
  curves *below* indicate conservative tests. The `plot_size_vs_nominal`
  helper renders a binomial-CI band around 5 % so a reader can tell
  the difference between sampling noise and a real distortion.
- Under H1_normal the *tail* is mis-specified — tests sensitive to
  ES depth (AS Z1, McNeil_Frey, MU_ES) should rise faster than tests
  sensitive only to coverage (Kupiec, U_VaR).
- Under H1_gjr_* the *leverage* is mis-specified — conditional tests
  (C_ES(m), C_VaR(m), MC_ES(m)) and AS Z2 should rise as the leverage
  strengthens, while tests with no time-series structure (Kupiec_POF,
  Christoffersen_CCI alone) are blind to it.

---

## 5. References

- Du, Z., & Escanciano, J. C. (2017). Backtesting Expected Shortfall:
  Accounting for Tail Risk. *Management Science*, 63(2), 474-495.
- Acerbi, C., & Szekely, B. (2014). Back-testing Expected Shortfall.
  *Risk Magazine*, 27(11), 76-81.
- Kratz, M., Lok, Y. H., & McNeil, A. J. (2018). Multinomial VaR
  Backtests: A Simple Implicit Approach to Backtesting Expected
  Shortfall. *Journal of Banking and Finance*, 88, 393-407.
- Kupiec, P. H. (1995). Techniques for Verifying the Accuracy of Risk
  Measurement Models. *Journal of Derivatives*, 3(2).
- Christoffersen, P. F. (1998). Evaluating Interval Forecasts.
  *International Economic Review*, 39(4), 841-862.
- McNeil, A. J., & Frey, R. (2000). Estimation of Tail-Related Risk
  Measures for Heteroscedastic Financial Time Series: An Extreme Value
  Approach. *Journal of Empirical Finance*, 7, 271-300.
- Fissler, T., & Ziegel, J. F. (2016). Higher Order Elicitability and
  Osband's Principle. *Annals of Statistics*, 44(4), 1680-1707.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the
  Relation between the Expected Value and the Volatility of the
  Nominal Excess Return on Stocks. *Journal of Finance*, 48(5),
  1779-1801.
