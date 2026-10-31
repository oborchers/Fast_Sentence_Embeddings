#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The backends of the runtime behind one contract (docs/architecture/backends.md, "Backends").

A fixed set chosen by name, without a registry: ``cpu`` (the native kernel), ``numpy``
(the reference) and ``mlx`` (the GPU of Apple silicon, ``pip install "fse[mlx]"``; the
cpu backend is its host), later ``cuda``. The mlx package, which imports MLX, is imported
only when ``get_backend("mlx")`` makes that backend, so ``import fse`` and the other
backends never load MLX.
"""

import importlib

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

BACKEND_NAMES = ("cpu", "numpy", "mlx")


_SELECTED: set[tuple[str, str | None]] = set()  # logged once per process each


def get_backend(name: str, *, isa: str | None = None) -> Backend:
    """The backend called ``name``; ``isa`` selects the kernel of the cpu backend."""
    backend = _make(name, isa)
    kernel = getattr(getattr(backend, "host", backend), "isa", None)
    if (backend.name, kernel) not in _SELECTED:
        _SELECTED.add((backend.name, kernel))
        concurrent = backend.capabilities().concurrent
        _log.emit(
            "backend.selected",
            "fse %s backend: kernel %s",
            backend.name,
            kernel,
            backend=backend.name,
            isa=kernel,
            concurrent=concurrent,
        )
    return backend


def _make(name: str, isa: str | None) -> Backend:
    if name == "cpu":
        return CpuBackend(isa)
    if name in ("numpy", "mlx") and isa is not None:
        raise ValueError("isa is an option of the cpu backend only")
    if name == "numpy":
        return NumpyBackend()
    if name == "mlx":
        try:
            mlx = importlib.import_module("fse._backends._mlx._backend")
        except ImportError as err:
            raise UnsupportedByBackendError(
                'the mlx backend needs pip install "fse[mlx]" on macOS 14 or newer on Apple '
                f"silicon: mlx cannot be imported ({err})"
            ) from err
        return mlx.MlxBackend(CpuBackend())
    raise ValueError(f"backend must be one of {', '.join(BACKEND_NAMES)}, not {name!r}")
