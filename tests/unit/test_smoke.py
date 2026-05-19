"""Smoke tests — package imports and version metadata."""

from __future__ import annotations

import esback


def test_package_imports() -> None:
    assert hasattr(esback, "__version__")


def test_version_is_non_empty_string() -> None:
    assert isinstance(esback.__version__, str)
    assert len(esback.__version__) > 0
