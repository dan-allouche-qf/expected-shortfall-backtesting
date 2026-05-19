"""Market data loaders.

Generic yfinance loader plus per-asset wrappers. Every loader is wrapped
in :func:`esback.data.cache.parquet_cache` so repeated calls are
bit-stable and protected from yfinance's silent back-fills of split- and
dividend-adjusted history.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yfinance as yf

from esback.data.cache import parquet_cache


def _ticker_slug(ticker: str) -> str:
    """File-system-safe slug for a yfinance ticker."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").lower()


def _yf_cache_key(ticker: str, start: str = "1997-01-01", end: str = "2024-12-31") -> str:
    return f"{_ticker_slug(ticker)}_{start.replace('-', '')}_{end.replace('-', '')}"


@parquet_cache(_yf_cache_key)
def load_yfinance(ticker: str, start: str = "1997-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Download daily closes from yfinance and compute 100x log-returns.

    Returns a DataFrame with columns ``Close`` (split- and dividend-
    adjusted) and ``Return`` (100 times daily log-returns), indexed by
    trading date. The first row is dropped because its return is
    undefined.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df["Return"] = 100.0 * np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()


def load_sp500(start: str = "1997-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Convenience wrapper for the S&P 500 (``^GSPC``)."""
    return load_yfinance("^GSPC", start=start, end=end)
