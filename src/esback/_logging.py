"""Structured logging setup.

Returns a logger configured with ``rich`` formatting if the optional
dependency is available, falling back to a plain stderr handler otherwise.
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s - %(message)s"


def get_logger(name: str = "esback", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler: logging.Handler
    formatter: logging.Formatter
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, markup=False, show_path=False)
        formatter = logging.Formatter("%(message)s")
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(_DEFAULT_FORMAT)

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
