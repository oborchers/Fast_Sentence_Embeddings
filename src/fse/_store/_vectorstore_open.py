#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Opening a ``fse.VectorStore`` from one path or from several (a view).

Split from ``_vectorstore`` so that module keeps room for new methods; it holds the
construction only and behaves as before: the same checks in the same order, the same
``store.opened`` event, the same errors."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fse import _log
from fse._backends import BACKEND_NAMES, get_backend
from fse._encoder import Encoder
from fse._validation import validate_arguments

from ._layout import read_store
from ._recovery import clear_exception_frames, snapshot
from ._vectors_format import VECTORS
from ._view import open_view


@validate_arguments
def _open(cls: Any, path: Any, *, encoder: Any = None, backend: str = "cpu") -> Any:
    try:
        return _opened(cls, path, encoder=encoder, backend=backend)
    except BaseException as error:
        clear_exception_frames(error)
        raise


def _opened(cls: Any, path: Any, *, encoder: Any, backend: str) -> Any:
    if backend not in BACKEND_NAMES:
        raise ValueError(f"backend must be one of {list(BACKEND_NAMES)}, not {backend!r}")
    path = Path(path)
    info = read_store(path, VECTORS.name)
    if encoder is not None:
        if not isinstance(encoder, Encoder):
            raise TypeError(f"encoder must be an Encoder, not {type(encoder).__name__}")
        if encoder.fingerprint != info.encoder:
            raise ValueError(
                f"the store {path} holds the rows of encoder {info.encoder}, not of {encoder.fingerprint}"
            )
    out = cls.__new__(cls)
    out.path, out._info, out._view = path, info, None
    checked = snapshot(path, info, verify=True)
    out._shards = checked.shards
    out._backend = get_backend(backend)
    out._lock = threading.Lock()
    out._refreshing = threading.Lock()
    out._active = 0
    out._closed = False
    _log.emit(
        "store.opened",
        "fse store: opened %d rows in %d shards: %d dimensions, %s, %s backend",
        len(out),
        len(out._shards),
        info.dim,
        info.dtype,
        backend,
        rows=len(out),
        shards=len(out._shards),
        dim=info.dim,
        dtype=info.dtype,
        backend=backend,
        unsynced_shards=checked.unsynced_shards,
        verified_bytes=checked.verified_bytes,
        verify_seconds=checked.verify_seconds,
        sign_bits_shards=sum(shard.sign_bits is not None for shard, _ in out._shards),
        sign_bits_bytes=sum(
            shard.sign_bits.nbytes for shard, _ in out._shards if shard.sign_bits is not None
        ),
    )
    return out


@validate_arguments
def _opened_view(cls: Any, paths: Any, *, encoder: Any = None, backend: str = "cpu") -> Any:
    def member(path: Path) -> Any:
        return cls.open(path, encoder=encoder, backend=backend)

    return open_view(cls, paths, member, {"_active": 0})


# The decorated functions keep their names: pydantic titles a validation error with them.
open_store = _open
open_store_view = _opened_view
