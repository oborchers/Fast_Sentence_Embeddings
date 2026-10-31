#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.VectorStore``: a store of sentence vectors opened for search and score.

The shards published when the store is opened are mapped read-only (a snapshot: later
appends need a new open), and every search runs on the stored float32, float16 or int8
rows directly, through their inverse norms, without a normalized float32 copy. ``mode``
picks the arithmetic: ``"exact"`` gives the cosines, scores and the order of the results
of ``fse._reference``, bit for bit; ``"blas"`` decodes each block to float32 and
multiplies with BLAS, whose sums differ from the contract in the last bits; ``"auto"``
(the default) takes the faster one, the contract for at most 8 queries or weight vectors
and BLAS above, so its bits depend on how many are asked at once (maintainer decision).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy import ndarray

from fse import _codes, _log
from fse._backends import BACKEND_NAMES, get_backend
from fse._container import CorruptArtifactError, NpyContainer
from fse._cpus import usable_cpus
from fse._encoder import Encoder
from fse._validation import NonNegativeInt, PositiveInt, validate_arguments

from . import _search
from ._layout import Shard, StoreInfo, read_store
from ._recovery import check_values, snapshot

F32 = np.float32


class SearchResult(NamedTuple):
    """The rows of the best matches (int64) and their cosines (float32), best first."""

    ids: ndarray
    scores: ndarray


class VectorStore:
    """The rows of a store at the time it was opened, searched by cosine (``search``) or
    scored with linear weights (``score``). Close it (or use it as a context manager) to
    release the mappings; a closed store raises."""

    path: Path
    _info: StoreInfo
    _shards: list[tuple[Shard, dict[str, ndarray]]]
    _backend: Any
    _lock: threading.Lock
    _active: int
    _closed: bool

    @classmethod
    def open(cls, path: Any, *, encoder: Any = None, backend: str = "cpu") -> VectorStore:
        """Open the store ``path``; with an ``encoder``, the store must hold its rows."""
        return _open(cls, path, encoder=encoder, backend=backend)

    def __init__(self, *_args: Any, **_kwargs: Any):
        raise TypeError("open a store with VectorStore.open(path)")

    # --- what it holds --------------------------------------------------------------

    def __len__(self) -> int:
        return sum(shard.rows for shard, _ in self._shards)

    @property
    def dim(self) -> int:
        return self._info.dim

    @property
    def dtype(self) -> str:
        return self._info.dtype

    @property
    def encoder_fingerprint(self) -> str:
        """The fingerprint of the encoder of the rows (``open(encoder=)`` takes the encoder)."""
        return self._info.encoder

    # --- search and score -----------------------------------------------------------

    @validate_arguments
    def search(
        self,
        queries: Any,
        k: PositiveInt = 10,
        *,
        workers: PositiveInt | None = None,
        mode: Literal["auto", "exact", "blas"] = "auto",
    ) -> SearchResult:
        """The ``k`` rows with the largest cosine to every query (a vector, or a matrix of one
        per row): ids (positions in the store) and scores of shape (k,), or (queries, k),
        best first, ties to the smaller id.

        ``mode="exact"`` computes every cosine by the numerical contract of fse, so every
        query gives the same bits however many are asked at once; ``"blas"`` decodes
        blocks to float32 and uses BLAS, faster for many queries, its bits depending on the
        batch; ``"auto"`` (the default) takes exact for up to 8 queries and BLAS above.
        Use ``"exact"`` for results that must be reproducible.
        """
        self._check_open()
        matrix, single = self._queries(queries, "queries")
        if k > len(self):
            raise ValueError(f"k is {k}, but the store holds {len(self)} rows")
        with self._using():
            ids, scores = _search.search(
                self._backend,
                self._shards,
                matrix,
                k,
                workers=workers or usable_cpus(),
                exact=_search.is_exact(mode, len(matrix)),
            )
        return SearchResult(ids[0], scores[0]) if single else SearchResult(ids, scores)

    @validate_arguments
    def score(
        self,
        weights: Any,
        bias: Any = 0.0,
        *,
        workers: PositiveInt | None = None,
        mode: Literal["auto", "exact", "blas"] = "auto",
    ) -> ndarray:
        """The linear score ``dot16(row, w) * scale + bias`` of every row with every weight
        vector (``weights`` a vector or a matrix of one per row; ``bias`` one number or one
        per vector): shape (rows,) or (rows, vectors). ``mode`` as for ``search``: exact
        for up to 8 weight vectors by default, BLAS above.
        """
        self._check_open()
        matrix, single = self._queries(weights, "weights")
        with self._using():
            out = _search.score(
                self._backend,
                self._shards,
                matrix,
                bias,
                workers=workers or usable_cpus(),
                exact=_search.is_exact(mode, len(matrix)),
            )
        return out[:, 0] if single else out

    @validate_arguments
    def get_vectors(
        self, *, start: NonNegativeInt = 0, stop: NonNegativeInt | None = None
    ) -> ndarray:
        """A float32 copy of the rows ``start`` to ``stop - 1`` (all by default), decoded
        from the stored float16 or int8 values."""
        self._check_open()
        stop = len(self) if stop is None else stop
        if not start <= stop <= len(self):
            raise ValueError(f"rows {start} to {stop} are not within the {len(self)} rows")
        parts = []
        with self._using():
            for shard, tensors in self._shards:
                lo, hi = max(start, shard.first_row), min(stop, shard.first_row + shard.rows)
                if lo < hi:
                    a, b = lo - shard.first_row, hi - shard.first_row
                    scales = tensors["scales"][a:b] if "scales" in tensors else None
                    parts.append(_codes.dequantize(tensors["vectors"][a:b], scales))
        return np.concatenate(parts) if parts else np.empty((0, self.dim), F32)

    def verify(self) -> None:
        """Hash every shard on disk against its checksums and check its values: finite
        scales and inverse norms of at least 0, finite float rows, int8 codes within
        +-127."""
        start = time.perf_counter()
        with self._using():
            for shard, tensors in self._shards:
                NpyContainer().verify(self.path / shard.name)
                check_values(shard, tensors)
        _log.emit(
            "store.verified",
            "fse store: verified %d rows in %d shards",
            len(self),
            len(self._shards),
            rows=len(self),
            shards=len(self._shards),
            seconds=time.perf_counter() - start,
        )

    # --- life cycle ---------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("the store is being searched; close it when the search is done")
            self._shards = []
            self._closed = True

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __reduce__(self):
        raise TypeError("a VectorStore cannot be pickled; open it again from its path")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("the VectorStore is closed")

    def _using(self):
        return _Use(self)

    def _queries(self, value: Any, name: str) -> tuple[ndarray, bool]:
        array = np.asarray(value)
        if array.dtype.kind not in "fiu":
            raise TypeError(f"{name} must be numbers (float or int), not strings, objects or bools")
        single = array.ndim == 1
        # Converted as the contract does, whatever rounding the calling thread set.
        matrix = self._backend.float32(array[None, :] if single else array)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim or not len(matrix):
            raise ValueError(
                f"{name} must be a vector of {self.dim} values or a matrix of them, not shape {array.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be finite in float32")
        return matrix, single


class _Use:
    """Counts a search or score in flight, so ``close`` cannot unmap rows it reads."""

    def __init__(self, store: VectorStore):
        self.store = store

    def __enter__(self):
        with self.store._lock:
            if self.store._closed:
                raise RuntimeError("the VectorStore is closed")
            self.store._active += 1

    def __exit__(self, *_exc: object) -> None:
        with self.store._lock:
            self.store._active -= 1


@validate_arguments
def _open(cls: Any, path: Any, *, encoder: Any = None, backend: str = "cpu") -> Any:
    if backend not in BACKEND_NAMES:
        raise ValueError(f"backend must be one of {list(BACKEND_NAMES)}, not {backend!r}")
    path = Path(path)
    info = read_store(path)
    if encoder is not None:
        if not isinstance(encoder, Encoder):
            raise TypeError(f"encoder must be an Encoder, not {type(encoder).__name__}")
        if encoder.fingerprint != info.encoder:
            raise ValueError(
                f"the store {path} holds the rows of encoder {info.encoder}, not of {encoder.fingerprint}"
            )
    out = cls.__new__(cls)
    out.path, out._info = path, info
    checked = snapshot(path, info, verify=True)
    out._shards = checked.shards
    out._backend = get_backend(backend)
    out._lock = threading.Lock()
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
    )
    return out
