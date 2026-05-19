"""Parquet-based cache for slow data-fetch functions.

The :func:`parquet_cache` decorator wraps a DataFrame-returning function
so its result is persisted in a parquet file the first time it is called
and read back from disk on subsequent calls. The cache key is computed by
a user-supplied ``key_fn`` and serves as the file stem.

Why we need this: yfinance can return slightly different histories for
the same query depending on when it is hit (split/dividend back-fills),
which would silently change baselines and break regression tests. Caching
freezes the first response and gives bit-stable rerunnable pipelines.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pandas as pd

from esback import config

DataFrameFn = TypeVar("DataFrameFn", bound=Callable[..., pd.DataFrame])


def parquet_cache(
    key_fn: Callable[..., str],
    cache_dir: Path | str | None = None,
) -> Callable[[DataFrameFn], DataFrameFn]:
    """Decorate a DataFrame-returning function with a parquet cache.

    Parameters
    ----------
    key_fn
        Callable that takes the same arguments as the wrapped function and
        returns a string used as the cache filename stem.
    cache_dir
        Directory holding the parquet files. Defaults to
        :data:`esback.config.DATA_CACHE_DIR`.
    """

    def deco(fn: DataFrameFn) -> DataFrameFn:
        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> pd.DataFrame:
            target = Path(cache_dir) if cache_dir is not None else config.DATA_CACHE_DIR
            cache_name = key_fn(*args, **kwargs)
            cache_path = target / f"{cache_name}.parquet"
            if cache_path.exists():
                return pd.read_parquet(cache_path)
            df = fn(*args, **kwargs)
            target.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            return df

        return wrapper  # type: ignore[return-value]

    return deco
