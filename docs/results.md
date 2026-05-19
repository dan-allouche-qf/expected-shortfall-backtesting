# Results

This document collects the headline numbers and figures produced by
`esback`. The commentary refers only to the numbers reported below,
which are the outputs of the executed notebooks under `notebooks/`.

- [1. Single-asset replication: S&P 500](#1-single-asset-replication-sp-500)
- [2. Multi-asset, multi-crisis grid](#2-multi-asset-multi-crisis-grid)
- [3. Monte Carlo size and power](#3-monte-carlo-size-and-power)
- [4. LSTM vs GARCH on the COVID-19 window](#4-lstm-vs-garch-on-the-covid-19-window)
- [5. Key takeaways](#5-key-takeaways)

---

## 1. Single-asset replication: S&P 500

Notebook: [`notebooks/01_replication.ipynb`](../notebooks/01_replication.ipynb).
Source data: S&P 500 daily returns, 1997-01-03 → 2024-12-30 (7044 obs).
Risk levels: `alpha_VaR = 0.05`, `alpha_ES = 0.10`, Box-Pierce lag
`m = 5`.

Two periods are reported, in line with Du-Escanciano (2017) plus a
COVID-19 extension:

- **Period 1 — 2008 Financial Crisis.** In-sample 1997-01-01 →
  2007-06-30 (`T ≈ 2640` days), out-of-sample 2007-07-01 →
  2009-06-30 (`n = 504` days, 44 VaR violations). Fitted
  `nu = 8.72`.
- **Period 2 — COVID-19 Crash.** In-sample 2010-01-01 → 2019-12-31
  (`T ≈ 2516` days), out-of-sample 2020-01-01 → 2021-12-31
  (`n = 505` days, 40 VaR violations). Fitted `nu = 4.83` — much
  fatter tails than the GFC-era estimate, consistent with the steep
  March-2020 shock dominating the in-sample window of the COVID OOS.

### 1.1 Full battery results

| Test                  | GFC stat | GFC p-value | GFC decision    | COVID stat | COVID p-value | COVID decision  |
|-----------------------|---------:|------------:|-----------------|-----------:|--------------:|-----------------|
| `U_ES`                |   4.197  |      0.0000 | REJECT          |     2.313  |        0.0207 | REJECT          |
| `C_ES(5)`             |  14.953  |      0.0106 | REJECT          |    13.844  |        0.0166 | REJECT          |
| `U_VaR`               |   3.842  |      0.0001 | REJECT          |     3.012  |        0.0026 | REJECT          |
| `C_VaR(5)`            |   9.539  |      0.0894 | Fail to reject  |    17.072  |        0.0044 | REJECT          |
| `MU_ES`               |   3.740  |      0.0002 | REJECT          |     2.205  |        0.0275 | REJECT          |
| `MC_ES(5)`            |  14.894  |      0.0108 | REJECT          |    12.682  |        0.0265 | REJECT          |
| `Kupiec_POF`          |  12.194  |      0.0005 | REJECT          |     7.762  |        0.0053 | REJECT          |
| `Christoffersen_CCI`  |   3.417  |      0.0645 | Fail to reject  |     1.084  |        0.2979 | Fail to reject  |
| `Christoffersen_CC`   |  15.611  |      0.0004 | REJECT          |     8.846  |        0.0120 | REJECT          |
| `KLM_multinomial(4)`  |  23.310  |      0.0001 | REJECT          |    13.504  |        0.0091 | REJECT          |
| `AS_Z1`               |   0.038  |      0.2168 | Fail to reject  |     0.099  |        0.1021 | Fail to reject  |
| `AS_Z2`               |   0.812  |      0.0002 | REJECT          |     0.741  |        0.0004 | REJECT          |
| `McNeil_Frey`         |   1.080  |      0.2759 | Fail to reject  |     1.361  |        0.1721 | Fail to reject  |

### 1.2 Discussion

On the 2008 Financial Crisis window, **9 of the 13 tests reject at
5 %**; the four that fail to reject are `C_VaR(5)` (p = 0.089),
`Christoffersen_CCI` (p = 0.065), `AS_Z1` (p = 0.22) and `McNeil_Frey`
(p = 0.28). On the COVID-19 window, **10 of the 13 tests reject**;
the same three tests `Christoffersen_CCI`, `AS_Z1` and `McNeil_Frey`
keep p > 0.10 (0.30, 0.10 and 0.17 respectively).

The marginal-coverage block (`Kupiec_POF`, `U_VaR`, `U_ES`,
`KLM_multinomial(4)`) and the conditional-coverage block
(`C_ES(5)`, `MU_ES`, `MC_ES(5)`, plus `C_VaR(5)` on COVID) fire on
both crises, while the day-to-day independence test
(`Christoffersen_CCI`) and the standardised-exceedance tests
(`AS_Z1`, `McNeil_Frey`) do not detect a model violation at the
level of consecutive-day pairs.

The asymmetry — the aggregate-violation component rejects even when
the clustering component does not — illustrates the contrast between
unconditional and conditional coverage, and is the operational case
for not backtesting ES through a single statistic.

### 1.3 Figures

Three figures are exported to `assets/figures/` for the README
gallery:

- `01_returns_var_es_2008.png` — returns vs VaR and ES envelopes
  over the 2007-07 → 2009-06 OOS window, with an annotated zoom on
  the worst single-day drawdown.
- `01_pit_diagnostic_covid.png` — QQ-plot and histogram of the
  COVID PIT residuals against Uniform(0,1), with the
  Kolmogorov-Smirnov p-value reported inline.
- `01_violations_2008.png` — stacked timeseries of binary
  violations `h_t` and cumulative severities `H_t` over the GFC
  window.

---

## 2. Multi-asset, multi-crisis grid

Notebook: [`notebooks/02_multi_asset_battery.ipynb`](../notebooks/02_multi_asset_battery.ipynb).
Five equity indices × five crisis windows × eleven tests = **275
p-values**. The GARCH recursion clips `sigma2` to be non-negative
before taking the square root, which keeps the very short SVB OOS
window (seven months) numerically tractable on every asset.

Each `(asset, period)` cell fits AR(1)-GARCH(1,1)-t on the 5-year
in-sample window ending the day before the OOS slice, runs the
unified battery on the OOS slice, and records the statistic, p-value
and decision per test.

### 2.1 Assets and windows

| Asset | Ticker | Dotcom | GFC | COVID | Inflation 2022 | SVB |
|-------|--------|--------|-----|-------|----------------|-----|
| SPX | `^GSPC` | 2000-04 → 2002-12 | 2007-07 → 2009-06 | 2020-01 → 2021-12 | 2022-01 → 2022-12 | 2023-03 → 2023-09 |
| NDX | `^NDX` | same | same | same | same | same |
| FTSE | `^FTSE` | same | same | same | same | same |
| DAX | `^GDAXI` | same | same | same | same | same |
| Nikkei | `^N225` | same | same | same | same | same |

### 2.2 Rejection rates per `(test, period)`

Share of assets (out of 5) that reject at the 5 % level, per test and
period:

| Test                  | COVID | Dotcom | GFC | Inflation_2022 | SVB |
|-----------------------|------:|-------:|----:|---------------:|----:|
| `U_ES`                |  1.0  |   1.0  | 1.0 |           1.0  | 0.0 |
| `C_ES(5)`             |  0.4  |   0.4  | 0.6 |           0.4  | 0.0 |
| `U_VaR`               |  0.8  |   1.0  | 1.0 |           1.0  | 0.0 |
| `C_VaR(5)`            |  0.2  |   0.4  | 0.4 |           0.4  | 0.0 |
| `AS_Z1`               |  0.4  |   0.0  | 0.6 |           0.0  | 0.0 |
| `AS_Z2`               |  1.0  |   0.8  | 1.0 |           1.0  | 0.2 |
| `McNeil_Frey`         |  0.4  |   0.2  | 0.6 |           0.2  | 0.2 |
| `Kupiec_POF`          |  0.8  |   1.0  | 1.0 |           1.0  | 0.0 |
| `Christoffersen_CCI`  |  0.0  |   0.0  | 0.2 |           0.2  | 0.0 |
| `Christoffersen_CC`   |  0.6  |   1.0  | 1.0 |           1.0  | 0.0 |
| `KLM_multinomial(4)`  |  1.0  |   1.0  | 1.0 |           1.0  | 0.0 |

### 2.3 Discussion

**Unconditional coverage vs independence.** Across the four
non-SVB windows (20 cells per test), `Christoffersen_CCI` rejects
2/20, while `Christoffersen_CC` rejects 18/20 and `Kupiec_POF` rejects
19/20. Since `LR_CC = LR_UC + LR_IND` (chi-squared with 2 dof =
chi-squared with 1 dof + chi-squared with 1 dof), and `LR_UC` is
precisely the Kupiec component, the gap between `Christoffersen_CC`
and `Christoffersen_CCI` is carried almost entirely by `LR_UC`: the
marginal inflation in the number of violations, not the clustering
pattern.

**ES test hierarchy outside SVB.** `KLM_multinomial(4)` and `U_ES`
reject 20/20. `AS_Z2` reaches 19/20, `Kupiec_POF` 19/20,
`Christoffersen_CC` 18/20. The conditional Box-Pierce statistics
`C_ES(5)` (9/20) and `C_VaR(5)` (7/20) sit in the middle. `AS_Z1`
(5/20) and `McNeil_Frey` (7/20) lag behind; both depend heavily on
the number of effective violations in the OOS window, and several
cells have too few violations to give those statistics teeth.

**Most stressed regimes.** GFC and COVID-19 are the windows where
unconditional coverage is massively violated: `Kupiec_POF` reaches
100 % (GFC) and 80 % (COVID), `KLM_multinomial(4)` reaches 100 % on
both, `AS_Z2` reaches 100 % on both. The independence-only component
stays low (20 % on GFC, 0 % on COVID): on these windows the excess
violations are *numerous* without being *consecutively clustered* in
the sense of the two-state Markov chain tested by
`Christoffersen_CCI`.

**SVB is the quiet outlier.** Across all 11 tests, SVB triggers only
2 rejections (1 from `AS_Z2`, 1 from `McNeil_Frey`) out of 55 cells.
The seven-month 2023-03 → 2023-09 window saw a sharp regional-bank
shock but limited index-level volatility, so the
AR(1)-GARCH(1,1)-t baseline scales its envelope correctly. SVB is a
useful negative control: a "stress" window where the model does not
in fact mis-specify ES on equity indices, illustrating that crisis
labels and ES-mis-specification do not always coincide.

### 2.4 Figures

- `02_battery_heatmap.png` — p-value heatmap across (test) ×
  (asset, period) cells.
- `02_reject_rate_per_test.png` — bar chart of rejection rate
  per test across all 25 cells.

---

## 3. Monte Carlo size and power

Notebook: [`notebooks/03_power_study.ipynb`](../notebooks/03_power_study.ipynb).
4 scenarios × 4 sample sizes × 1,000 replications = 16,000 simulated
paths. On each path AR(1)-GARCH(1,1)-t is fitted on the first 60%
and the unified battery runs on the remaining 40%. The Acerbi-Szekely
Z1 / Z2 p-values are parametric Monte-Carlo readings produced by the
`StandardizedTSimulator` (`n_sim_power = 2000` draws per replication).
The full log totals 176,000 rows.

### 3.1 Scenarios

- `H0_garch_t`: AR(1)-GARCH(1,1)-t with `nu = 8.0` — correctly
  specified.
- `H1_gjr_t`: GJR-GARCH(1,1,1)-t with `gamma = 0.15` — moderate
  asymmetric leverage.
- `H1_gjr_strong`: GJR-GARCH(1,1,1)-t with `gamma = 0.25` — strong
  asymmetric leverage.
- `H1_normal`: AR(1)-GARCH(1,1) with Normal innovations — same
  volatility, tail mis-specification.

The robust statistics `MU_ES` and `MC_ES(5)` are not in this table:
the power-study harness runs `run_full_battery` without passing the
`fit` and `returns_all` arguments, so the basic statistics are
returned and the robust ones are skipped.

### 3.2 Empirical size and power at `n = 1000`

Rejection rates at the 5% level:

| Test                  | H0_garch_t | H1_gjr_t | H1_gjr_strong | H1_normal |
|-----------------------|-----------:|---------:|--------------:|----------:|
| `U_ES`                |       0.16 |     0.17 |          0.31 |      0.17 |
| `C_ES(5)`             |       0.08 |     0.16 |          0.30 |      0.09 |
| `U_VaR`               |       0.14 |     0.13 |          0.28 |      0.15 |
| `C_VaR(5)`            |       0.08 |     0.14 |          0.24 |      0.07 |
| `AS_Z1`               |       0.12 |     0.10 |          0.12 |      0.07 |
| `AS_Z2`               |       0.14 |     0.14 |          0.23 |      0.12 |
| `McNeil_Frey`         |       0.16 |     0.16 |          0.14 |      0.14 |
| `Kupiec_POF`          |       0.13 |     0.14 |          0.26 |      0.15 |
| `Christoffersen_CCI`  |       0.04 |     0.05 |          0.10 |      0.04 |
| `Christoffersen_CC`   |       0.12 |     0.13 |          0.26 |      0.11 |
| `KLM_multinomial(4)`  |       0.13 |     0.15 |          0.25 |      0.13 |

### 3.3 Discussion

**Under H0_garch_t (correct specification).** `Christoffersen_CCI`
(0.04), `C_ES(5)` and `C_VaR(5)` (0.08) are within four percentage
points of the nominal 5% size. The other tests overshoot: `U_ES` and
`McNeil_Frey` at 0.16, `U_VaR` and `AS_Z2` at 0.14, `Kupiec_POF` and
`KLM_multinomial(4)` at 0.13, `AS_Z1` and `Christoffersen_CC` at 0.12.
The asymptotic N(0,1) and chi-squared distributions predict
convergence to 5% as `n` grows; at `n = 1000` that convergence has
not yet been reached for the marginal-coverage tests.

**Under H1_gjr_t (γ=0.15, moderate leverage).** All rejection rates
stay below 20%: `U_ES` 0.17, `C_ES(5)` 0.16, `McNeil_Frey` 0.16,
`KLM_multinomial(4)` 0.15, `AS_Z2` 0.14, `C_VaR(5)` 0.14,
`Kupiec_POF` 0.14, `Christoffersen_CC` 0.13, `U_VaR` 0.13, `AS_Z1`
0.10, `Christoffersen_CCI` 0.05. Several tests (`AS_Z1`,
`Christoffersen_CCI`) sit at or below their H0 size at this leverage
intensity.

**Under H1_gjr_strong (γ=0.25, strong leverage).** The conditional
Box-Pierce statistics on cumulative violations lead the power
ranking: `U_ES` reaches 0.31, `C_ES(5)` 0.30, `U_VaR` 0.28,
`Kupiec_POF` and `Christoffersen_CC` 0.26, `KLM_multinomial(4)` 0.25,
`C_VaR(5)` 0.24, `AS_Z2` 0.23. Net power (rejection rate at H1
minus rejection rate at H0): +22 pp for `C_ES(5)`, +16 pp for
`C_VaR(5)`, +15 pp for `U_ES`, +14 pp for `U_VaR` and
`Christoffersen_CC`, +13 pp for `Kupiec_POF`, +12 pp for
`KLM_multinomial(4)`, +9 pp for `AS_Z2`, +6 pp for
`Christoffersen_CCI`, 0 pp for `AS_Z1`, −2 pp for `McNeil_Frey`.

`AS_Z2` is the most active *ES-specific* test under leverage: it
tracks both the *frequency* and the *magnitude* of violations and
gains +9 pp net of H0 size. `AS_Z1` (0.12 at H1, 0.12 at H0) gains
0 pp — it normalises by the empirical violation set, so it is blind
to whether violations are *too numerous*, only to whether the
*severity given a violation* exceeds ES.

**Under H1_normal (Gaussian innovations).** The rejection rates hug
the H0 size (`U_ES` 0.17, `Christoffersen_CCI` 0.04, `Kupiec_POF`
0.15, `McNeil_Frey` 0.14, `AS_Z1` 0.07, `AS_Z2` 0.12,
`KLM_multinomial(4)` 0.13, `Christoffersen_CC` 0.11). At `n = 1000`,
the mismatch between the Gaussian tail and the Student-t tail predicted
by the fitted model is not pronounced enough in the PIT for any of
the `u_t`-based tests to detect at typical equity-return scales.

**Headline pattern.** The conditional Box-Pierce statistics on
cumulative violations (`C_ES(5)`, `C_VaR(5)`) gain the most net
power against leverage misspecification (+22 pp and +16 pp
respectively); the marginal-coverage block (`U_ES`, `U_VaR`,
`Kupiec_POF`, `KLM_multinomial(4)`) follows at +12-15 pp;
`AS_Z2` sits at +9 pp; the tests that condition on the empirical
violation set (`AS_Z1`, `McNeil_Frey`) or only test independence
(`Christoffersen_CCI`) gain little or nothing.

### 3.4 Figures

- `03_size_under_h0.png` — size curves vs sample size under the
  correctly-specified DGP.
- `03_power_curves.png` — power curves vs sample size against the
  three H1 alternatives.

---

## 4. LSTM vs GARCH on the COVID-19 window

Notebook: [`notebooks/04_neural_baseline.ipynb`](../notebooks/04_neural_baseline.ipynb).

The LSTM-Quantile head (30 epochs, hidden size 32, sequence length
20, Adam at 1e-3, pinball + FZ0 loss) is trained on 2010-2019 and
evaluated on the 2020-2021 COVID OOS slice, alongside the
AR(1)-GARCH(1,1)-t baseline. Both models output one-step-ahead
`(VaR, ES)` for every OOS day. Risk level: `alpha = 0.05`.

### 4.1 Training and forecast summaries

- Final training losses: pinball = 0.7906, fz0 = 0.8943.
- **LSTM** mean `VaR = 0.695%`, mean `ES = 2.839%` over the OOS slice.
- **GARCH-t** mean `VaR = 1.831%`, mean `ES = 2.682%` over the OOS slice.
- **Underestimation factor**: LSTM mean VaR is **2.6 × smaller** than
  GARCH-t's. The mean ES values are closer (GARCH ES = 2.682%, LSTM
  ES = 2.839%) but the LSTM's VaR / ES *spread* is much wider —
  symptomatic of a trained head that has lost the link between the
  two quantiles for tail days.

### 4.2 Battery comparison

| Test                  | GARCH stat | GARCH p-value | GARCH decision   | LSTM stat | LSTM p-value | LSTM decision   |
|-----------------------|-----------:|--------------:|------------------|----------:|-------------:|-----------------|
| `Kupiec_POF`          |    7.762   |       0.0053  | REJECT           |   150.636 |        0.000 | REJECT          |
| `Christoffersen_CCI`  |    1.084   |       0.2979  | Fail to reject   |     0.468 |        0.494 | Fail to reject  |
| `Christoffersen_CC`   |    8.846   |       0.0120  | REJECT           |   151.103 |        0.000 | REJECT          |
| `AS_Z1`               |    0.099   |       N/A     | N/A              |    -0.348 |          N/A | N/A             |
| `AS_Z2`               |    0.741   |       N/A     | N/A              |     1.685 |          N/A | N/A             |

AS Z1 / Z2 p-values are NaN here because the lightweight LSTM helper
does not ship a `StandardizedTSimulator` of its own; only the
statistic is reported.

### 4.3 Discussion

On the COVID-19 OOS slice, the AR(1)-GARCH(1,1)-t baseline scales
its VaR / ES envelope with the volatility regime (mean VaR =
1.83%), while the LSTM-Quantile head trained on the calmer 2010-2019
decade fails to scale during the March 2020 spike (mean VaR =
0.70%). Both models reject Kupiec POF, but the LSTM's rejection
(stat ≈ 151) is **about 20× stronger** than GARCH-t's (stat ≈ 7.8) —
the LSTM's violation rate is far above 5 %.

`Christoffersen_CCI` does not reject for either model: the excess
violations, when they happen, are spread out enough that the
day-to-day independence component is satisfied. `Christoffersen_CC`
rejects in proportion to the unconditional violation gap.
Acerbi-Szekely Z1 / Z2 are reported for both models as statistics
only, since their p-values would require a parametric simulator
that the LSTM does not provide; a mixture-density network would
yield the PIT needed to run the full battery.

The honest takeaway: a 30-epoch LSTM-Quantile trained on a
low-volatility decade does not generalise to a once-in-a-decade tail
event the way an explicitly volatility-clustered model does.
Sensible next steps: a longer training window covering the 2008
crisis, a mixture-density-network head, or an ensemble of GARCH and
LSTM forecasts.

### 4.4 Figure

- `04_lstm_vs_garch_covid.png` — returns and the two negative
  VaR / ES envelopes superimposed.

---

## 5. Forecasting-model sensitivity: four GARCH variants on COVID-19

Notebook: [`notebooks/05_multi_model.ipynb`](../notebooks/05_multi_model.ipynb).
Same OOS window as Section 1.1 (S&P 500, 2020-01-01 → 2021-12-31).
Four model variants share the AR(1) mean and the GARCH(1,1) recursion;
they differ only in the leverage term (`GARCH` vs `GJR`) and the
innovation distribution (Student-$t$ vs Normal). Eleven base tests
run per variant; the robust Du-Escanciano statistics are not included
in this grid (they would require the parameter covariance passed
through).

### 5.1 In-sample fit (log-likelihood, 2010-2019)

| Variant | Log-likelihood | $\Delta$ vs best |
|---|---:|---:|
| `GJR-t` | $-2860.23$ | $0$ |
| `GARCH-t` | $-2915.35$ | $-55.11$ |
| `GJR-normal` | $-2936.60$ | $-76.37$ |
| `GARCH-normal` | $-3002.10$ | $-141.86$ |

Student-$t$ innovations dominate Normal ones; the GJR leverage term
adds a modest within-distribution premium.

### 5.2 COVID-19 OOS rejection counts

| Variant | Rejects at 5% / tests with p-value |
|---|---:|
| `GJR-t` | 5 / 11 |
| `GARCH-t` | 8 / 11 |
| `GJR-normal` | 3 / 9 |
| `GARCH-normal` | 7 / 9 |

The two Normal-innovation variants are missing the Acerbi-Szekely
p-values because the parametric simulator wired into the battery is
Student-$t$-only.

### 5.3 Key inversions between `GARCH-t` and `GJR-t`

The four basic Du-Escanciano statistics and the multinomial test
shift materially when the leverage term is added on top of GARCH-$t$:

| Test | `GARCH-t` p-value | `GJR-t` p-value |
|---|---:|---:|
| `C_ES(5)` | 0.0164 (REJECT) | 0.2366 (Fail) |
| `C_VaR(5)` | 0.0045 (REJECT) | 0.9899 (Fail) |
| `Christoffersen_CC` | 0.0117 (REJECT) | 0.0763 (Fail) |
| `KLM_multinomial(4)` | 0.0088 (REJECT) | 0.0838 (Fail) |
| `AS_Z1` | 0.1070 (Fail) | 0.0140 (REJECT) |

The asymmetric variance recursion routes more variance toward days
following negative shocks, which is precisely the clustering pattern
that $C_{ES}(m)$ and $C_{VaR}(m)$ detect; with `gamma > 0` those
signals collapse. The $AS_{Z_1}$ inversion is genuine: with leverage
on, `GJR-t` predicts a *smaller* ES on the violation days, and the
realised severities then exceed it on average.

### 5.4 Figure

- `05_multi_model_heatmap.png` — p-value heatmap, 11 tests × 4
  variants, ordered by in-sample log-likelihood.

---

## 6. Mixture-density LSTM on COVID-19

Notebook: [`notebooks/06_mdn_baseline.ipynb`](../notebooks/06_mdn_baseline.ipynb).
The MDN-LSTM replaces the two-output LSTM-Quantile head with three
output heads that produce the Student-$t$ parameters
$(\mu_t, \sigma_t, \nu_t)$ of the conditional return distribution
$F_t$. The training loss is the negative log-likelihood of the
predicted distribution. From the trained network the PIT residual,
VaR, ES and a parametric simulator follow in closed form, so eleven
of the thirteen statistics in the battery apply (the two robust
Du-Escanciano variants still require the parameter covariance of an
MLE fit).

### 6.1 Forecast statistics on the COVID-19 OOS window

| Model | Mean VaR (%) | Mean ES (%) | Training-loss final |
|---|---:|---:|---:|
| `AR(1)-GARCH(1,1)-t` | $1.831$ | $2.682$ | (closed-form MLE) |
| `LSTM-Quantile` | $0.695$ | $2.839$ | pinball $= 0.7906$ |
| `MDN-LSTM` | $1.445$ | $2.136$ | NLL $= 1.1207$ |

The MDN-LSTM scales the VaR envelope to $1.445\%$ — almost twice the
LSTM-Quantile's $0.695\%$ — but still under-calls compared to
GARCH-$t$'s $1.831\%$. Average conditional parameters on the OOS
window: $\bar{\mu} = 0.11$, $\bar{\sigma} = 0.77$, $\bar{\nu} = 4.67$
(low df, fat-tailed).

### 6.2 Battery comparison

| Test | GARCH-$t$ stat | GARCH-$t$ p-value | MDN stat | MDN p-value |
|---|---:|---:|---:|---:|
| `U_ES` | 3.302 | 0.0010 | 5.535 | 0.0000 |
| `C_ES(5)` | 7.532 | 0.1840 | 37.767 | 0.0000 |
| `U_VaR` | 3.012 | 0.0026 | 5.258 | 0.0000 |
| `C_VaR(5)` | 17.072 | 0.0044 | 24.669 | 0.0002 |
| `AS_Z1` | 0.099 | 0.1021 | 0.158 | 0.0426 |
| `AS_Z2` | 0.741 | 0.0004 | 1.340 | 0.0000 |
| `McNeil_Frey` | 1.361 | 0.1721 | 1.502 | 0.1192 |
| `Kupiec_POF` | 7.762 | 0.0053 | 21.614 | 0.0000 |
| `Christoffersen_CCI` | 1.084 | 0.2979 | 0.006 | 0.9370 |
| `Christoffersen_CC` | 8.846 | 0.0120 | 21.620 | 0.0000 |
| `KLM_multinomial(4)` | 13.504 | 0.0091 | 29.583 | 0.0000 |

GARCH-$t$ rejects 8 of the 13 statistics that apply. The MDN-LSTM
rejects 9 of the 11 statistics that apply. The MDN's Kupiec POF
statistic is 21.6 — a factor of $2.8$ over GARCH-$t$'s 7.8 — versus
the LSTM-Quantile's 150.6 from §4, a factor of $19.4$. The MDN
under-calls in the same direction as the LSTM-Quantile but with
much less severity, because the parametric output lets it learn
a heavier-tailed conditional law on the training window.

### 6.3 Operational takeaway

Adding the MDN head fixes the LSTM-Quantile's architectural limit
(no analytic CDF $\Rightarrow$ no PIT $\Rightarrow$ five tests at most)
without re-architecting the LSTM backbone. The eleven backtests now
all apply, and the comparison with GARCH-$t$ moves to the same
language used for the GARCH variants. Generalisation across regime
shifts is a separate problem from the architecture choice; the MDN,
like the LSTM-Quantile, fails to extrapolate from a calm in-sample
window to a once-in-a-decade tail event.

### 6.4 Figure

- `06_mdn_vs_lstm_vs_garch.png` — three negative-VaR envelopes
  overlaid on S&P 500 COVID-19 OOS returns.

---

## 7. Key takeaways

1. **Eleven tests give eleven different answers.** Even on the same
   OOS slice, the basic Du-Escanciano, the robust variants, the
   Kupiec / Christoffersen baselines, the Kratz-Lok-McNeil multinomial,
   the Acerbi-Szekely Z1 / Z2 and McNeil-Frey land in different
   regions of the p-value space. Backtesting ES through a single
   statistic is operationally fragile.

2. **Marginal vs conditional coverage are not redundant.** The
   `LR_CC = LR_UC + LR_IND` decomposition shows that the joint test
   can reject for two distinct reasons; on real crises (GFC, COVID)
   the rejection is almost always driven by the marginal-coverage
   component, not by clustering.

3. **AS_Z2 is the most active ES test on real data.** Across the
   five non-SVB crises × five assets, `AS_Z2` reaches 19/20
   rejections, on par with `KLM_multinomial(4)` (20/20) and
   `U_ES` (20/20), and clearly above `AS_Z1` (5/20) or
   `McNeil_Frey` (7/20).

4. **SVB is a useful negative control.** The 2023 regional-bank
   shock did not mis-specify ES on equity indices; the model scales
   its envelope correctly. Crisis labels and ES mis-specification
   do not always coincide.

5. **The Monte Carlo size study warns about over-rejection at `n =
   1000`.** Under correctly-specified DGP, the marginal-coverage
   tests overshoot the nominal 5 % level (`U_ES` and `McNeil_Frey`
   0.16, `U_VaR` and `AS_Z2` 0.14, `Kupiec_POF` and
   `KLM_multinomial(4)` 0.13, …); only `Christoffersen_CCI`,
   `C_ES(5)` and `C_VaR(5)` sit within four points of the nominal
   size. The asymptotic distributions converge only at substantially
   larger sample sizes.

6. **AS_Z2 is materially more powerful than AS_Z1 under leverage.**
   Under `H1_gjr_strong` (γ=0.25), `AS_Z2` reaches 0.23 versus
   `AS_Z1` at 0.12; the gap (+11 pp absolute, +9 pp net of H0 size)
   reflects that `Z2` aggregates over the full window while `Z1`
   only normalises over the empirical violation set.

7. **LSTM-Quantile does not generalise to a once-in-a-decade tail
   event.** Trained on calm 2010-2019, it scales its VaR to 0.70 %
   on a COVID OOS where GARCH-t scales to 1.83 % — a 2.6 × under-call
   that the Kupiec POF rejects with stat ≈ 151 versus GARCH-t's
   stat ≈ 7.8. Classical volatility-clustered models dominate at
   tails of this magnitude.

8. **GARCH variant choice is not innocuous.** On the same S&P 500
   COVID-19 window, switching from `GARCH-t` to `GJR-t` cuts the
   rejection count from 8/11 to 5/11. The asymmetric leverage term
   relaxes `C_ES(5)`, `C_VaR(5)`, `Christoffersen_CC` and
   `KLM_multinomial(4)`. The Acerbi-Szekely $Z_1$ test inverts:
   `GARCH-t` does not reject ($p=0.107$); `GJR-t` does ($p=0.014$).

9. **A mixture-density LSTM unlocks the full battery for the
   neural baseline.** Outputting Student-$t$ parameters
   $(\mu, \sigma, \nu)$ rather than just $(\widehat{\VaR},
   \widehat{\ES})$ supplies the PIT residual and the parametric
   simulator that the LSTM-Quantile cannot. On COVID-19 the MDN
   under-calls VaR (mean $1.45\%$ versus GARCH-$t$ at $1.83\%$),
   but its Kupiec POF rejection is a factor of $2.8$ vs the
   LSTM-Quantile's $19.4$.
