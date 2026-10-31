#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The backends of the runtime behind one contract (docs/architecture/backends.md, "Backends").

A fixed set chosen by name, without a registry: ``cpu`` (the native kernel) and
``numpy`` (the reference).
"""

from fse import _log

from ._base import (
    Backend,
    Capabilities,
    PoolInput,
    Prepared,
    UnsupportedByBackendError,
)
from ._cpu import CpuBackend
from ._numpy import NumpyBackend

__all__ = [
    "BACKEND_NAMES",
    "Backend",
    "Capabilities",
    "CpuBackend",
    "NumpyBackend",
    "PoolInput",
    "Prepared",
    "UnsupportedByBackendError",
    "get_backend",
]

BACKEND_NAMES = ("cpu", "numpy")


_SELECTED: set[tuple[str, str | None]] = set()  # logged once per process each


def get_backend(name: str, *, isa: str | None = None) -> Backend:
    """The backend called ``name``; ``isa`` selects the kernel of the cpu backend."""
    backend = _make(name, isa)
    kernel = getattr(backend, "isa", None)
    if (backend.name, kernel) not in _SELECTED:
        _SELECTED.add((backend.name, kernel))
        _log.emit(
            "backend.selected",
            "fse %s backend: kernel %s",
            backend.name,
            kernel,
            backend=backend.name,
            isa=kernel,
        )
    return backend


def _make(name: str, isa: str | None) -> Backend:
    if name == "cpu":
        return CpuBackend(isa)
    if name == "numpy" and isa is not None:
        raise ValueError("isa is an option of the cpu backend only")
    if name == "numpy":
        return NumpyBackend()
    raise ValueError(f"backend must be one of {', '.join(BACKEND_NAMES)}, not {name!r}")
