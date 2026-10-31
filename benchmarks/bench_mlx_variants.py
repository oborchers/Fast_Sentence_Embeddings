#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The two ablations of the MLX device thread, as patches of ``Device`` in the benchmark
process (no change in ``src``; DR-178 PR 4, ARCHITECTURE.md, "Device thread").

- ``copy``: the caller's copy (``use``) runs in the device thread, as part of the job's
  ``finish``, with nothing lent: the MLX arrays are dropped right after the copy, and the
  worker only waits (the opposite of lending them until the worker has copied).
- ``single``: one group at a time, launch, complete, settle, instead of launching the
  next group before the one before is complete (at most one group in flight).
- ``both``: the two together. ``base`` is the code as it is; ``base2`` is the same again,
  the A/A control that gives the noise between two equal sides.

``patched(name)`` is a context manager; the patches are on the class, so it applies to
every worker and to the device thread that starts inside it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from fse._backends._mlx._device import Device, Job, _collect

# base2 is base again (A/A); the others name the patches they apply.
VARIANTS: dict[str, tuple[str, ...]] = {
    "base": (),
    "base2": (),
    "copy": ("copy",),
    "single": ("single",),
    "both": ("copy", "single"),
}


def _copy_in_device_thread(run: Callable) -> Callable:
    """``Device.run`` with ``use`` after ``finish`` in the device thread and no lease."""

    def patched(self, launch, finish, tokens=0, use=None):
        if use is not None:
            first = finish

            def finish(arrays, first=first, use=use):
                return use(first(arrays))

        return run(self, launch, finish, tokens, None)

    return patched


def _serial_serve(self: Device, previous: threading.Thread | None = None) -> None:
    """``Device._serve`` with one group at a time: launch it, complete it, take the next."""
    group: list[Job] = []
    launched: list[tuple[Job, tuple]] = []
    lent: list[Job] = []
    try:
        if previous is not None:
            previous.join()
        mx = self.core()
        if self._stream is None:
            self._stream = mx.new_thread_unsafe_stream(mx.gpu)
        group = self._take()
        while group:
            launched, failed = self._launch(mx, group)
            done = [*failed, *self._complete(mx, launched)]
            launched = []
            group = self._take()
            lent = self._settle(done, lent, last=not group)
    except BaseException as err:  # as _serve: a failure must not leave a waiter hanging
        self._abandon([*group, *(job for job, _ in launched)], err)
        _collect(lent, wait=True)


@contextmanager
def patched(name: str) -> Iterator[None]:
    """The device thread as variant ``name`` of ``VARIANTS`` runs it."""
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; one of {', '.join(VARIANTS)}")
    saved: dict[str, Any] = {"run": Device.run, "_serve": Device._serve}
    try:
        if "copy" in VARIANTS[name]:
            Device.run = _copy_in_device_thread(saved["run"])  # type: ignore[method-assign]
        if "single" in VARIANTS[name]:
            Device._serve = _serial_serve  # type: ignore[method-assign]
        yield
    finally:
        Device.run = saved["run"]  # type: ignore[method-assign]
        Device._serve = saved["_serve"]  # type: ignore[method-assign]
