"""Unit tests for esback.config constants."""

from __future__ import annotations

from esback import config


def test_seed_is_42() -> None:
    assert config.SEED == 42


def test_default_alphas() -> None:
    assert config.DEFAULT_ALPHA_ES == 0.10
    assert config.DEFAULT_ALPHA_VAR == 0.05


def test_default_m_is_positive() -> None:
    assert config.DEFAULT_M >= 1


def test_paths_are_absolute_under_repo_root() -> None:
    assert config.REPO_ROOT.is_absolute()
    assert config.DATA_CACHE_DIR.parent.parent == config.REPO_ROOT
    assert config.RESULTS_DIR.parent == config.REPO_ROOT
    assert config.TESTS_DATA_DIR.parent.parent == config.REPO_ROOT
