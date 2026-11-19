"""Input checks shared by the runtime entry points."""

from __future__ import annotations

import numpy as np

from ._pool import Workers
from ._sinks import sink_dtype, wants_inverse_norms


def check_workers(workers: object) -> int | None:
    """``workers`` of a run: None (the usable CPUs), a positive int or the ``Workers`` of a call."""
    if isinstance(workers, Workers):
        return workers.workers
    return None if workers is None else _positive(workers, "workers")


def _non_negative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, not {value!r}")
    return int(value)


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive int, not {value!r}")
    return int(value)


def check_sink_rows(sink: object, dim: int):
    """``sink`` has ``target`` and ``commit`` and takes rows of ``dim`` values of a storage
    dtype, with or without inverse norms."""
    if not callable(getattr(sink, "target", None)) or not callable(getattr(sink, "commit", None)):
        raise TypeError(f"the sink must have target and commit methods, not {type(sink).__name__}")
    if getattr(sink, "dim", None) != dim:
        raise ValueError(f"the sink must take rows of {dim} values")
    sink_dtype(sink)
    wants_inverse_norms(sink)
