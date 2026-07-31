"""Profiling helper placeholders."""

from __future__ import annotations

from time import perf_counter


def now() -> float:
    """Return a high-resolution timestamp."""

    return perf_counter()
