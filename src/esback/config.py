"""Project-wide constants and default parameters.

Centralises seed values, default risk levels, and filesystem paths so the
rest of the package never hard-codes them.
"""

from __future__ import annotations

from pathlib import Path

SEED: int = 42
"""Default RNG seed used everywhere ``numpy.random.default_rng`` is created."""

DEFAULT_ALPHA_ES: float = 0.10
"""Risk level for the ES backtest. Du and Escanciano (2017) recommend
``alpha_ES ~= 2 * alpha_VaR`` so the expected number of informative days
is comparable across the two backtests."""

DEFAULT_ALPHA_VAR: float = 0.05
"""Risk level for the VaR backtest."""

DEFAULT_M: int = 5
"""Number of autocorrelation lags used in the conditional Box-Pierce
statistic ``C_ES(m)`` and ``C_VaR(m)``."""

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
"""Repository root, computed relative to this file."""

DATA_CACHE_DIR: Path = REPO_ROOT / "data" / "cache"
"""Where parquet caches of downloaded market data live."""

RESULTS_DIR: Path = REPO_ROOT / "results"
"""Where CLI-generated tables and figures land."""

TESTS_DATA_DIR: Path = REPO_ROOT / "tests" / "data"
"""Where reference outputs for regression tests live."""
