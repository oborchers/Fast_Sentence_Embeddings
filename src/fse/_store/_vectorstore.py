#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.VectorStore``: a store of sentence vectors opened for search and score.

The shards published when the store is opened are mapped read-only (a snapshot: ``refresh``
adds the shards published since), and every search runs on the stored float32, float16 or int8
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
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy import ndarray

from fse import _codes, _log
from fse._container import NpyContainer
from fse._cpus import usable_cpus
from fse._validation import NonNegativeInt, PositiveInt, validate_arguments

from . import _search, _take, _vector_batches
from ._id_stats import check_removed
from ._kind import Shard
from ._kinds import kind_of
from ._layout import read_store
from ._recovery import check_chain, rows_in, stop_row
from ._refresh import refreshed
from ._result import SearchResult
from ._vector_batches import VectorBatch
from ._vectors_format import VECTORS, StoreInfo
from ._vectorstore_open import open_store, open_store_view
from ._vectorstore_search import rescore_store, search_store
from ._view import Joined, MemberReaders, verify_members

F32 = np.float32
DEFAULT_BLOCK_ROWS = 65_536


class VectorStore(MemberReaders):
    """The rows of a store at the time it was opened (or last refreshed), searched by cosine
    (``search``) or scored with linear weights (``score``). Every call reads the shards it
    finds when it starts, so a ``refresh`` meanwhile does not reach it. Close it (or use it
    as a context manager) to release the mappings; a closed store raises, and ``close``
    raises while a search, score, read or refresh runs."""

    path: Path | None  # None for a view
    _info: StoreInfo
    _view: Joined | None
    _shards: list[tuple[Shard, dict[str, ndarray]]]
    _backend: Any
    _lock: threading.Lock
    _refreshing: threading.Lock
    _active: int
    _closed: bool

    @classmethod
    def open(cls, path: Any, *, encoder: Any = None, backend: str = "cpu") -> VectorStore:
        """Open the store ``path``; with an ``encoder``, the store must hold its rows."""
        return open_store(cls, path, encoder=encoder, backend=backend)

    @classmethod
    def open_view(cls, paths: Any, *, encoder: Any = None, backend: str = "cpu") -> VectorStore:
        """Open the stores ``paths`` (a list of paths) as one store whose rows follow each
        other, without merging them on disk (``paths``, ``members``, ``source`` and
        ``locate`` describe them). Every search, score and read gives what one store holding
        the same rows gives, ties included (with ``mode="exact"``; BLAS blocks follow the
        shards). The stores must hold the same encoder, dtype, dimension and document-ID
        mode; stores written by ``Encoder.encode(shard=)`` must be parts of one source and are
        read in the order of the parts (all finished: positions are source rows). A view has no ``refresh`` and no ``path``,
        and ``close`` closes its stores."""
        return open_store_view(cls, paths, encoder=encoder, backend=backend)

    def __init__(self, *_args: Any, **_kwargs: Any):
        raise TypeError("open a store with VectorStore.open(path)")

    # --- what it holds --------------------------------------------------------------

    def __len__(self) -> int:
        return rows_in(self._shards)

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

    @property
    def has_doc_ids(self) -> bool:
        """True if every row of the store has a document ID (``ShardSink(doc_ids=True)``)."""
        return self._info.has_doc_ids

    # --- search and score -----------------------------------------------------------

    @validate_arguments
    def search(
        self,
        queries: Any,
        k: PositiveInt = 10,
        *,
        workers: PositiveInt | None = None,
        mode: Literal["auto", "exact", "blas"] = "auto",
        doc_ids: Any = None,
        rows: Any = None,
        max_per_doc: PositiveInt | None = None,
    ) -> SearchResult:
        """The ``k`` rows with the largest cosine to every query (a vector, or a matrix of one
        per row): ids (positions in the store) and scores of shape (k,), or (queries, k),
        best first, ties to the smaller id.

        ``mode="exact"`` computes every cosine by the numerical contract of fse, so every
        query gives the same bits however many are asked at once; ``"blas"`` decodes
        blocks to float32 and uses BLAS, faster for many queries, its bits depending on the
        batch; ``"auto"`` (the default) takes exact for up to 8 queries and BLAS above.
        Use ``"exact"`` for results that must be reproducible.

        At most ``k`` hits are returned: fewer if the selection holds fewer rows (or, with
        ``max_per_doc``, fewer documents), without an error. The result then has that
        many entries, none if nothing is selected.

        ``max_per_doc=1`` (a store with doc_ids) returns the ``k`` best documents
        instead: of every document ID only its best row (the largest score, ties to the
        smaller id), each hit of a different document. Only 1 is allowed in this version;
        larger values may be added later. It combines with ``doc_ids`` and ``rows``: the
        rows are selected first, then grouped.
        """
        return search_store(
            self,
            queries,
            k,
            workers=workers,
            mode=mode,
            doc_ids=doc_ids,
            rows=rows,
            max_per_doc=max_per_doc,
        )

    @validate_arguments
    def rescore(
        self,
        queries: Any,
        candidates: Any,
        /,
        k: PositiveInt = 10,
        *,
        workers: PositiveInt | None = None,
        max_per_doc: PositiveInt | None = None,
    ) -> SearchResult:
        """The ``k`` best rows of every query among its own ``candidates``, scored exactly:
        the result of ``search(query, k, mode="exact", rows=sorted(set(candidates) - {-1}))``,
        bit for bit (also with ``max_per_doc=1``), for the hits of an external index.

        ``candidates`` are int64 positions of the store, an array of shape (m,) for one
        query vector or (queries, m) for a matrix, or a flat list; ``-1`` marks a gap.
        Lists may be unsorted, hold duplicates, be empty or shorter than ``k``. A hit list
        holds ``min(k, candidates)`` rows (``max_per_doc=1``: ``min(k, documents)``). A
        matrix result is as wide as the longest list, filled up on the right with ids
        ``-1``, scores ``-inf`` and document IDs ``-1``; a single query is not filled up.
        Keep ``candidates`` unchanged until the call returns."""
        return rescore_store(self, queries, candidates, k, workers=workers, max_per_doc=max_per_doc)

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
        for up to 8 weight vectors by default, BLAS above. Every row is scored;
        ``get_doc_ids`` returns the labels of the rows.
        """
        self._check_open()
        matrix, single = _search.queries(self._backend, self.dim, weights, "weights")
        with self._using() as shards:
            out = _search.score(
                self._backend,
                shards,
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

        def decode(tensors: Any, a: int, b: int) -> ndarray:
            scales = tensors["scales"][a:b] if "scales" in tensors else None
            return _codes.dequantize(tensors["vectors"][a:b], scales)

        parts = self._row_parts(start, stop, decode)
        return np.concatenate(parts) if parts else np.empty((0, self.dim), F32)

    @validate_arguments
    def get_doc_ids(
        self, *, start: NonNegativeInt = 0, stop: NonNegativeInt | None = None
    ) -> ndarray:
        """The document IDs (int64) of the rows ``start`` to ``stop - 1`` (all by default),
        a copy."""
        self._check_open()
        if not self._info.has_doc_ids:
            raise ValueError("this store has no doc_ids")
        parts = self._row_parts(start, stop, lambda tensors, a, b: tensors["doc_ids"][a:b])
        return (
            np.concatenate(parts).astype(np.int64, copy=False) if parts else np.empty(0, np.int64)
        )

    @validate_arguments
    def take_vectors(self, rows: Any, /) -> ndarray:
        """The float32 rows at the store positions ``rows`` (a list or an int64 array of one or
        two dimensions, so ``result.ids`` of a search works), in that order, decoded as
        ``get_vectors``: shape ``rows.shape + (dim,)``, a new array. Order and duplicates
        are kept, unlike ``search(rows=)``, which is a set; negative positions raise.
        Keep ``rows`` unchanged until the call returns."""
        self._check_open()
        wanted = _take.positions(rows)
        with self._using() as shards:
            return _take.gather(
                shards,
                wanted,
                rows_in(shards),
                _take.read_vectors,
                tail=(self.dim,),
                dtype=F32,
                row_bytes=_take.vector_row_bytes(self.dim, self.dtype),
            )

    @validate_arguments
    def take_doc_ids(self, rows: Any, /) -> ndarray:
        """The document IDs (int64) at the store positions ``rows``, as ``take_vectors`` takes
        them: shape ``rows.shape``, a new array, in the order asked, duplicates kept."""
        self._check_open()
        if not self._info.has_doc_ids:
            raise ValueError("this store has no doc_ids")
        wanted = _take.positions(rows)
        with self._using() as shards:
            return _take.gather(
                shards,
                wanted,
                rows_in(shards),
                _take.read_doc_ids,
                tail=(),
                dtype=np.int64,
                row_bytes=_take.ID_ROW_BYTES,
            )

    @validate_arguments
    def iter_batches(
        self,
        *,
        block_rows: PositiveInt = DEFAULT_BLOCK_ROWS,
        start: NonNegativeInt = 0,
        stop: NonNegativeInt | None = None,
    ) -> Iterator[VectorBatch]:
        """The stored rows ``start`` to ``stop - 1`` (all by default) in blocks of at most
        ``block_rows`` rows, never over the end of a shard: ``VectorBatch`` objects whose
        ``vectors`` (and ``doc_ids`` and ``inv_norms``) are read-only views of the mapped shards, in the stored
        dtype (int8 as ``QuantizedRows``). No row is copied or decoded.

        The views and a suspended generator keep the mapped files open, which on Windows
        locks them: drop the views, and call ``close()`` on the generator (or use
        ``contextlib.closing``) when you stop early. After ``VectorStore.close`` the next
        block raises ``RuntimeError``, also before the first; views given before stay
        readable."""
        self._check_open()
        with self._using() as shards:  # the shards and the rows of this call, also for a generator
            stop = stop_row(shards, start, stop)
        return self._batches(shards, block_rows, start, stop)

    def _batches(
        self, shards: list, block_rows: int, start: int, stop: int
    ) -> Iterator[VectorBatch]:
        self._check_open()  # also for no rows, and before the first block
        for shard, tensors, first, end in _vector_batches.blocks(shards, start, stop, block_rows):
            self._check_open()  # a closed store gives nothing more
            yield _vector_batches.make(tensors, shard.first_row, first, end)

    def _row_parts(self, start: int, stop: int | None, read: Any) -> list[ndarray]:
        """``read(tensors, a, b)`` for each shard part within the rows ``start`` to ``stop - 1``;
        ``np.concatenate`` of the parts copies them out of the mappings."""
        parts = []
        with self._using() as shards:
            stop = stop_row(shards, start, stop)
            for shard, tensors in shards:
                lo, hi = max(start, shard.first_row), min(stop, shard.first_row + shard.rows)
                if lo < hi:
                    parts.append(read(tensors, lo - shard.first_row, hi - shard.first_row))
        return parts

    def verify(self) -> None:
        """Hash every shard on disk against its checksums and check its values: finite
        scales and inverse norms of at least 0, finite float rows, int8 codes within
        +-127. A view verifies each of its stores and the boundaries between them."""
        self._check_open()
        if self._view is not None:
            with self._using():
                return verify_members(self._view)
        start = time.perf_counter()
        read_store(self._root, VECTORS.name)
        kind = kind_of(self._info)
        with self._using() as shards:
            for shard, tensors in shards:
                file = self._root / shard.name / "manifest.json"
                check_removed(NpyContainer().metadata(self._root / shard.name), file)
                NpyContainer().verify(self._root / shard.name)
                kind.check_values(shard, tensors)
            check_chain(self._root, kind, shards)
        _log.emit(
            "store.verified",
            "fse store: verified %d rows in %d shards",
            rows_in(shards),
            len(shards),
            rows=rows_in(shards),
            shards=len(shards),
            seconds=time.perf_counter() - start,
        )

    def refresh(self) -> int:
        """Add the shards published since the store was opened (or last refreshed) and return
        the number of rows they hold (0 if there are none). The store changes in place: a
        search, score or read that runs, or a generator already made, keeps the rows it
        started with. Only the new shards are checked and, if tagged ``durability="none"``,
        hashed. A gap, an orphan or a break of the chain raises ``CorruptArtifactError``; a
        store that is no longer the one that was opened raises ``RuntimeError``; the rows
        stay as they were. Nothing is promised against ``repair_store`` running meanwhile.
        ``close`` raises while a refresh runs. A view raises ``ValueError``: open it again."""
        self._check_open()
        if self._view is not None:
            raise ValueError("a view does not refresh; open it again")
        with self._refreshing, self._using() as shards:
            checked = refreshed(self._root, self._info, shards)
            added = rows_in(checked.shards)
            total = rows_in(shards) + added
            with self._lock:
                self._shards = [*shards, *checked.shards]
        _log.emit(
            "store.refreshed",
            "fse store: refreshed to %d rows: %d rows in %d shards added",
            total,
            added,
            len(checked.shards),
            rows=total,
            added_rows=added,
            added_shards=len(checked.shards),
            unsynced_shards=checked.unsynced_shards,
            verified_bytes=checked.verified_bytes,
            verify_seconds=checked.verify_seconds,
        )
        return added

    # --- life cycle ---------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError(
                    "the store is in use (a search, score or read is running); "
                    "close it when it is done"
                )
            self._shards = []
            self._closed = True
        if self._view is not None:
            self._view.close()

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


class _Use:
    """Counts a search, score, read or refresh in flight, so ``close`` cannot unmap rows it
    reads, and gives its shards."""

    def __init__(self, store: VectorStore):
        self.store = store

    def __enter__(self) -> list:
        """The shards of the store at this moment: a list that is replaced, never changed."""
        with self.store._lock:
            if self.store._closed:
                raise RuntimeError("the VectorStore is closed")
            self.store._active += 1
            return self.store._shards

    def __exit__(self, *_exc: object) -> None:
        with self.store._lock:
            self.store._active -= 1
