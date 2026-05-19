"""Unit tests for esback.data.cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from esback.data.cache import parquet_cache


def _frame(x: int) -> pd.DataFrame:
    return pd.DataFrame({"x": [x, x + 1, x + 2]})


def test_cache_miss_creates_parquet(tmp_path: Path) -> None:
    @parquet_cache(lambda x: f"k_{x}", cache_dir=tmp_path)
    def fetch(x: int) -> pd.DataFrame:
        return _frame(x)

    df = fetch(7)
    assert (tmp_path / "k_7.parquet").exists()
    pd.testing.assert_frame_equal(df, _frame(7))


def test_cache_hit_skips_function(tmp_path: Path) -> None:
    calls = {"n": 0}

    @parquet_cache(lambda x: f"k_{x}", cache_dir=tmp_path)
    def fetch(x: int) -> pd.DataFrame:
        calls["n"] += 1
        return _frame(x)

    fetch(3)
    assert calls["n"] == 1
    fetch(3)
    assert calls["n"] == 1


def test_cache_returns_equal_dataframe(tmp_path: Path) -> None:
    @parquet_cache(lambda x: f"k_{x}", cache_dir=tmp_path)
    def fetch(x: int) -> pd.DataFrame:
        return _frame(x)

    first = fetch(11)
    second = fetch(11)
    pd.testing.assert_frame_equal(first, second)


def test_distinct_keys_create_distinct_files(tmp_path: Path) -> None:
    @parquet_cache(lambda x: f"k_{x}", cache_dir=tmp_path)
    def fetch(x: int) -> pd.DataFrame:
        return _frame(x)

    fetch(1)
    fetch(2)
    files = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert files == ["k_1.parquet", "k_2.parquet"]


def test_cache_dir_is_created_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "cache"

    @parquet_cache(lambda x: f"k_{x}", cache_dir=target)
    def fetch(x: int) -> pd.DataFrame:
        return _frame(x)

    fetch(0)
    assert target.is_dir()
    assert (target / "k_0.parquet").exists()


@pytest.fixture
def isolated_cache(tmp_path: Path) -> Path:
    return tmp_path
