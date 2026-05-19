# Reproducibility

This document describes how to run `esback` end-to-end from a clean
checkout, what is pinned to guarantee bit-stable reference outputs,
and the few places where determinism is bounded by external factors.

- [1. Setup](#1-setup)
- [2. Frozen reference data](#2-frozen-reference-data)
- [3. CLI commands](#3-cli-commands)
- [4. Notebook order and timing](#4-notebook-order-and-timing)
- [5. Regression tests](#5-regression-tests)
- [6. Determinism caveats](#6-determinism-caveats)
- [7. Re-creating the full report](#7-re-creating-the-full-report)

---

## 1. Setup

`esback` targets Python 3.11+. The recommended setup:

```bash
git clone <your-clone-url>
cd expected-shortfall-backtesting
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional but recommended:

```bash
pre-commit install
```

This wires `ruff format`, `ruff check`, and the trailing-whitespace
hook into git. The configuration lives in `.pre-commit-config.yaml`.

`pyproject.toml` pins all run-time and dev dependencies under the
relevant `[project]` and `[project.optional-dependencies]` blocks.
Notable choices:

- `numpy>=2.1`, `pandas>=2.2`, `scipy>=1.14`
- `arch>=7` for GARCH maximum-likelihood estimation
- `yfinance>=0.2.40` for data ingestion (gated by a parquet cache)
- `torch>=2.3` for the LSTM-Quantile baseline (MPS-friendly on Apple
  Silicon)
- `typer>=0.12` for the CLI, `pyyaml>=6` for configs

---

## 2. Frozen reference data

The S&P 500 returns series used by the regression tests lives in
`tests/data/sp500_returns_19970101_20241231.parquet`. This file is
*committed* and treated as immutable, so a silent `yfinance` back-fill
cannot change any reference value.

The `data/cache/` directory contains the same parquet under the
filename schema `<ticker>_<start>_<end>.parquet`. It is generated on
first use by `load_sp500` (which wraps `yfinance` via
`esback.data.cache.parquet_cache`). The cache is excluded from the
working tree under `.gitignore`; reproducing the project from scratch
re-downloads the data and re-creates the cache.

Reference outputs for the regression tests are pinned under
`tests/data/`:

- `reference_p1_days.parquet`, `reference_p2_days.parquet`: VaR/ES
  paths for the two replication periods (2008 GFC and COVID-19).
- `reference_summary.json`: p-values for every test in the basic and
  robust batteries, period by period.

These files are produced by the helper at
`tests/integration/test_regression.py::generate_references` and are
re-generated only on a deliberate refresh.

---

## 3. CLI commands

`esback` exposes a Typer CLI with two sub-commands. The entry point
is declared in `pyproject.toml` under `[project.scripts]`.

### `esback replicate`

Run the Du-Escanciano (2017) replication on the periods declared in a
YAML config.

```bash
esback replicate -c configs/replication_du_escanciano.yaml
```

The shipped config drives the S&P 500 over two windows: the 2008
financial crisis (in-sample 1997-01-01 → 2007-06-30, OOS
2007-07-01 → 2009-06-30) and the COVID-19 crash (in-sample
2010-01-01 → 2019-12-31, OOS 2020-01-01 → 2021-12-31). The command
writes:

- Five PDF figures per period under `results/figures/`:
  loss distribution, violations timeline, returns-vs-risk-envelope, ACF
  comparison, PIT diagnostic.
- Three pedagogical PDFs (`numerical_example.pdf`,
  `lambda_diagram.pdf`, `results_table.pdf`).
- Three CSV tables per period (`_basic.csv`, `_robust.csv`,
  `_full_battery.csv`) under `results/tables/`, plus a
  cross-period `comparison.csv`.

End-to-end runtime: ~2 seconds on Apple Silicon.

### `esback multi-asset`

Run the unified battery on a grid of `(asset, crisis-period)`
combinations.

```bash
esback multi-asset -c configs/multi_asset_crisis.yaml
```

The shipped config drives five equity indices (SPX, NDX, FTSE, DAX,
Nikkei) over five crisis windows (Dotcom, GFC, COVID-19,
Inflation_2022, SVB), each with a 5-year in-sample window ending the
day before the crisis OOS slice. The command writes:

- `results/multi_asset/grid.parquet` and `.csv` with the full table
  of statistics, p-values and decisions (275 rows).
- `results/multi_asset/reject_rate_per_test.csv`.
- One heatmap PDF per period under `results/multi_asset/figures/`.

End-to-end runtime: a few minutes (dominated by the AR(1)-GARCH(1,1)-t
fits and the Acerbi-Szekely Monte-Carlo p-values).

---

## 4. Notebook order and timing

Four notebooks live under `notebooks/`. Each is self-contained and
caches its heaviest outputs, so re-running them is cheap.

| Notebook | Topic | First-run wall clock | Cache file |
|----------|-------|----------------------|------------|
| `01_replication.ipynb` | Du-Escanciano on S&P 500 (GFC + COVID) | ~30 s | `data/cache/gspc_*.parquet` |
| `02_multi_asset_battery.ipynb` | 5×5 grid, heatmaps, rejection rates | ~3 min | `results/multi_asset/grid.parquet` |
| `03_power_study.ipynb` | Monte Carlo size and power | ~15 min | `results/power_study/results.parquet` |
| `04_neural_baseline.ipynb` | LSTM-Quantile vs GARCH on COVID | ~5 min | (none) |

Re-execute one notebook from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=esback \
  notebooks/01_replication.ipynb
```

(Install the `esback` kernel once with `python -m ipykernel install
--user --name=esback`.)

To force a full re-execution from scratch, wipe the caches first:

```bash
rm -rf data/cache/* results/power_study results/multi_asset results/neural
```

---

## 5. Regression tests

Twelve integration tests under `tests/integration/test_regression.py`
assert bit-stable behaviour on the frozen reference data:

- The S&P 500 series loaded from the parquet cache matches the
  committed reference (to bit).
- The AR(1)-GARCH(1,1)-t fit produces the same parameter vector and
  asymptotic covariance on both periods (1e-10 / 1e-12 tolerances).
- The PIT residuals match the reference path day-by-day.
- The VaR / ES forecasts match the reference path day-by-day.
- Every backtest p-value matches the reference, basic and robust,
  for both periods.

Any change that moves these values surfaces in the test output. The
intent is that *any* drift in the model, the recursion, or the test
implementation is caught before it propagates into the figures.

Run them:

```bash
pytest -q tests/integration/test_regression.py
```

The unit-test suite under `tests/unit/` covers the rest of the
package: PIT transforms, risk-measure analytics, individual backtest
mechanics, viz smoke tests, CLI smoke tests, etc.

---

## 6. Determinism caveats

A few sources of non-determinism are bounded but not eliminated:

- **MPS non-determinism (notebook 04).** PyTorch's MPS backend on
  Apple Silicon has known non-deterministic kernels. Even with all
  seeds pinned (`seed=42` in the LSTM trainer and DataLoader), the
  final pinball / FZ0 loss values drift by ~`1e-3` between runs.
  Discussed in the notebook's last cell.
- **`yfinance` back-fills.** `yfinance` occasionally back-fills
  historical values after corporate actions or vendor corrections.
  The parquet cache and the frozen reference under `tests/data/`
  insulate the project from this, but if you delete the cache *and*
  the reference parquet, you will get whatever `yfinance` returns
  *today*.
- **`arch` package updates.** The `arch` MLE uses
  `scipy.optimize.minimize` under the hood; minor updates to the
  scipy optimiser have historically shifted parameter estimates by
  `1e-7`. The regression-test tolerance (1e-10 strict, 1e-12 for
  p-values) is calibrated to catch that. Tighten or loosen by
  editing the constants in `tests/integration/test_regression.py`.

---

## 7. Re-creating the full report

The full set of artefacts that ships with the project — the eight PNGs
in `assets/figures/`, the executed notebooks, and the per-period
result tables — is produced by running this sequence from a clean
checkout:

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Validate
pytest -q tests/
ruff check src/ tests/
mypy src/esback

# 3. Wipe caches and re-run the CLI
rm -rf data/cache/* results/*
esback replicate -c configs/replication_du_escanciano.yaml
esback multi-asset -c configs/multi_asset_crisis.yaml

# 4. Re-execute the four notebooks (sequential, in order)
for nb in 01_replication 02_multi_asset_battery 03_power_study 04_neural_baseline; do
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=esback \
    notebooks/${nb}.ipynb
done
```

The four notebooks save their headline figures back into
`assets/figures/` so the README gallery refreshes automatically.

Total wall-clock for the full report: 20-30 minutes on Apple Silicon
(dominated by the Monte Carlo grid in notebook 03).
