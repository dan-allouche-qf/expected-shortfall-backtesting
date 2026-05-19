"""Unit tests for esback._logging."""

from __future__ import annotations

import logging

from esback._logging import get_logger


def test_get_logger_returns_logger() -> None:
    logger = get_logger("esback.test")
    assert isinstance(logger, logging.Logger)


def test_get_logger_does_not_duplicate_handlers() -> None:
    name = "esback.test_no_dup"
    a = get_logger(name)
    n_initial = len(a.handlers)
    b = get_logger(name)
    assert a is b
    assert len(b.handlers) == n_initial


def test_get_logger_does_not_propagate() -> None:
    logger = get_logger("esback.test_propagate")
    assert logger.propagate is False
