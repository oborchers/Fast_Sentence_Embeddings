#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The backends of the runtime behind one contract (ARCHITECTURE.md, "Backends").

A fixed set chosen by name, without a registry: ``cpu`` (the native kernel), ``numpy``
(the reference) and the backends of the coming PRs.
"""

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


def get_backend(name: str, *, isa: str | None = None) -> Backend:
    """The backend called ``name``; ``isa`` selects the kernel of the cpu backend."""
    if name == "cpu":
        return CpuBackend(isa)
    if name == "numpy":
        if isa is not None:
            raise ValueError("isa is an option of the cpu backend only")
        return NumpyBackend()
    raise ValueError(f"backend must be one of {', '.join(BACKEND_NAMES)}, not {name!r}")
