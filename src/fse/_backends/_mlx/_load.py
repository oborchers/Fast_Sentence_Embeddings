#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The only module of fse that imports MLX (tools/check_guardrails.py checks it).

MLX is optional (``pip install "fse[mlx]"``) and imported on first use, never by
``import fse`` (ruff allows the import inside a function in the loaders of the device
backends only): ``core()`` gives ``mlx.core`` once this machine can run the backend
(macOS 14 or newer on Apple silicon, a supported MLX, a Metal device) and otherwise
raises an UnsupportedByBackendError that says why. There is no fallback to the CPU.
"""

from __future__ import annotations

import functools
import platform
import re
import sys
from types import ModuleType

from .._base import UnsupportedByBackendError

MIN_VERSION = (0, 32, 4)
_INSTALL = 'pip install "fse[mlx]" on macOS 14 or newer on Apple silicon'


def core() -> ModuleType:
    """``mlx.core``, or an UnsupportedByBackendError naming what is missing."""
    mx, reason = _probe()
    if mx is None:
        raise UnsupportedByBackendError(f"the mlx backend needs {_INSTALL}: {reason}")
    return mx


@functools.cache
def _probe() -> tuple[ModuleType | None, str | None]:
    """mlx.core and None, or None and the reason it cannot run here (checked once)."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return None, f"this is {sys.platform} on {platform.machine()}"
    release = _version(platform.mac_ver()[0])
    if release < (14,):
        return None, f"this is macOS {platform.mac_ver()[0]}"
    try:
        import mlx.core as mx
    except ImportError as err:
        return None, f"mlx cannot be imported ({err})"
    installed = getattr(mx, "__version__", "")
    if _version(installed) < MIN_VERSION:
        need = ".".join(map(str, MIN_VERSION))
        return None, f"mlx {installed} is installed, {need} or newer is needed"
    if not mx.metal.is_available():
        return None, "Metal is not available (a virtual machine or a headless session?)"
    return mx, None


def _version(text: str) -> tuple[int, ...]:
    """The leading numbers of a version string: "0.32.4.dev1" is (0, 32, 4)."""
    match = re.match(r"\d+(?:\.\d+)*", text)
    return tuple(int(part) for part in match.group().split(".")) if match else ()
