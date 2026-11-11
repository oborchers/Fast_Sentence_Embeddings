#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.ScoreStore``: a score store opened for reading (docs/architecture/scores.md, "Reading").
The shards published when the store is opened are mapped read-only (a snapshot: ``refresh``
adds the shards published since). ``iter_batches`` gives the stored values as they are, read-only
views of the mapping; ``predict_batches`` gives labels or values as ``fse.Heads`` or a single head does, derived
from what is stored, with the thresholds of the multilabel heads replaced if asked. No
encoder, tokenizer or weight table is loaded."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, overload

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from fse import _log
from fse._backends import BACKEND_NAMES, get_backend
from fse._container import NpyContainer
from fse._head._heads_run import OutputInput, OutputRule, resolve_outputs
from fse._head._thresholds import updated
from fse._store._id_stats import check_removed
from fse._store._kind import Shard
from fse._store._kinds import kind_of
from fse._store._layout import read_store
from fse._store._recovery import (
    check_chain,
    clear_exception_frames,
    rows_in,
    snapshot,
    stop_row,
)
from fse._store._refresh import refreshed
from fse._store._scores_info import ScoreHead, ScoreInfo
from fse._store._vector_batches import blocks
from fse._store._view import Joined, MemberReaders, open_view, verify_members
from fse._validation import (
    NonNegativeInt,
    PathLike,
    PathSequence,
    PositiveInt,
    UncheckedPath,
    file_path,
    validate_arguments,
)

from . import _predict
from ._batches import ScoreBatch
from ._schema import group_of

DEFAULT_BLOCK_ROWS = 65_536
KIND = "prediction_scores"


class ScoreStore(MemberReaders):
    """The rows of a score store at the time it was opened (or last refreshed). Every call
    reads the shards it finds when it starts, so a ``refresh`` meanwhile does not reach it,
    not even a generator already made. Close it (or use it as a context
    manager) to release the mappings; views given before stay readable, and a closed store
    gives nothing more: the next block of ``iter_batches`` or ``predict_batches`` raises
    ``RuntimeError``, also before the first."""

    path: Path | None  # None for a view
    _info: ScoreInfo
    _view: Joined | None
    _shards: list[tuple[Shard, dict[str, ndarray]]]
    _backend: str
    _lock: threading.Lock
    _refreshing: threading.Lock
    _closed: bool

    @classmethod
    def open(
        cls, path: PathLike, *, heads: Any = None, name: Any = None, backend: str = "cpu"
    ) -> ScoreStore:
        """Open the store ``path``; with ``heads`` (a ``fse.Heads``), the store must hold
        exactly those heads, in that order, with those models. ``heads`` may also be one
        ``Classifier`` or ``Regressor``: then ``name`` is required and the store must hold
        exactly that head under that name, as that model, of that encoder."""
        return _open(cls, path, heads=heads, name=name, backend=backend)

    @classmethod
    def open_view(
        cls, paths: PathSequence, *, heads: Any = None, name: Any = None, backend: str = "cpu"
    ) -> ScoreStore:
        """Open the stores ``paths`` (a list of paths) as one store whose rows follow each
        other, without merging them on disk; ``heads`` and ``name`` are checked against every
        store as ``open`` does. Every read gives what one store holding the same rows gives (blocks end at the
        shards of the stores).
        The stores must hold the same heads, columns and paths, the same encoder and
        document-ID mode and, across their boundaries, the same thresholds; stores written by
        ``Encoder.encode(shard=)`` must be parts of one source and are read in the order
        of the parts (all finished: positions are source rows). A view has no ``refresh`` and no ``path`` (``paths`` names them), and
        ``close`` closes its stores."""
        return _opened_view(cls, paths, heads=heads, name=name, backend=backend)

    def __init__(self, *_args: Any, **_kwargs: Any):
        raise TypeError("open a store with ScoreStore.open(path)")

    # --- what it holds --------------------------------------------------------------

    def __len__(self) -> int:
        return rows_in(self._shards)

    @property
    def heads(self) -> Mapping[str, ScoreHead]:
        """The heads of the store by name, in column order: what each stores, the path it
        ran on, its labels and its columns."""
        return MappingProxyType({head.name: head for head in self._info.heads})

    @property
    def encoder_fingerprint(self) -> str:
        """The fingerprint of the encoder of the heads."""
        return self._info.encoder

    @property
    def has_doc_ids(self) -> bool:
        """True if every row of the store has a document ID (``ScoreSink(doc_ids=True)``)."""
        return self._info.has_doc_ids

    # --- reading --------------------------------------------------------------------

    @validate_arguments
    def iter_batches(
        self,
        *,
        block_rows: PositiveInt = DEFAULT_BLOCK_ROWS,
        start: NonNegativeInt = 0,
        stop: NonNegativeInt | None = None,
    ) -> Generator[ScoreBatch, None, None]:
        """The stored values of the rows ``start`` to ``stop - 1`` (all by default) in
        blocks of at most ``block_rows`` rows, never over the end of a shard: for each head
        a read-only view of the mapped shard (the stored decision, probabilities or values;
        ``(n,)`` for single-label decisions or vector regression; multilabel stays ``(n, k)``).
        A head of several columns is not contiguous in the view; copy it if you need that.
        Raises ``RuntimeError`` for the next block after
        ``close``, also before the first."""
        shards = self._snapshot()
        stop = stop_row(shards, start, stop)
        return self._blocks(shards, block_rows, start, stop, _predict_raw, None)

    @overload
    def predict_batches(
        self,
        *,
        output: None = None,
        thresholds: Any = None,
        block_rows: PositiveInt = DEFAULT_BLOCK_ROWS,
        backend: str | None = None,
    ) -> Generator[ScoreBatch, None, None]: ...

    @overload
    def predict_batches(
        self,
        *,
        output: OutputInput,
        thresholds: Any = None,
        block_rows: PositiveInt = DEFAULT_BLOCK_ROWS,
        backend: str | None = None,
    ) -> Generator[ScoreBatch, None, None]: ...

    @validate_arguments
    def predict_batches(
        self,
        *,
        output: Any = None,
        thresholds: Any = None,
        block_rows: PositiveInt = DEFAULT_BLOCK_ROWS,
        backend: str | None = None,
    ) -> Generator[ScoreBatch, None, None]:
        """Labels or values for every row, in blocks, as ``Heads.predict_batches`` gives
        them: by default the labels of a classifier and the values of a regressor,
        ``output`` as for ``Heads`` (one name or a mapping of head names to names). A
        multilabel output keeps shape ``(n, k)``, including k = 1. What a
        head can give follows from what it stored (see ``fse._scores._predict``); anything
        else raises before a row is read. ``thresholds`` maps multilabel head names to
        new cutoffs (a partial mapping keeps the stored cutoffs of the labels it does not
        name; None resets to 0.5); no model, encoder or tokenizer is loaded. Raises
        ``RuntimeError`` for the next block after ``close``, also before the first."""
        shards = self._snapshot()
        name = self._backend if backend is None else backend
        if name not in BACKEND_NAMES:
            raise ValueError(f"backend must be one of {list(BACKEND_NAMES)}, not {name!r}")
        plans = self._plans(shards, output, thresholds)
        return self._blocks(
            shards,
            block_rows,
            0,
            rows_in(shards),
            _predict_derived,
            (get_backend(name), plans, {}),
        )

    @validate_arguments
    def get_scores(
        self,
        head: str,
        stored: str,
        *,
        start: NonNegativeInt = 0,
        stop: NonNegativeInt | None = None,
    ) -> NDArray[np.float32]:
        """A copy of the values of ``head`` for the rows ``start`` to ``stop - 1`` (all by
        default): ``stored`` must be what the head stored (``decision``, ``proba`` or
        ``values``), so that logits and probabilities are never taken for each other."""
        shards = self._snapshot()
        found = self.heads.get(head)
        if found is None:
            raise ValueError(f"the store holds the heads {list(self.heads)}, not {head!r}")
        if stored != found.stored:
            raise ValueError(
                f"head {head!r} stored {found.stored}, not {stored!r}; the values are not "
                "converted here (predict_batches derives other outputs)"
            )
        stop = stop_row(shards, start, stop)
        parts = [
            np.array(_columns(found, tensors["scores"][first:end]))
            for _, tensors, first, end in blocks(shards, start, stop, 1 << 62)
        ]
        shape = (0,) if _predict._store_flat(found, found.stored) else (0, found.width)
        return np.concatenate(parts) if parts else np.empty(shape, np.float32)

    @validate_arguments
    def get_doc_ids(
        self, *, start: NonNegativeInt = 0, stop: NonNegativeInt | None = None
    ) -> NDArray[np.int64]:
        """The document IDs (int64) of the rows ``start`` to ``stop - 1`` (all by default),
        a copy."""
        shards = self._snapshot()
        if not self._info.has_doc_ids:
            raise ValueError("this store has no doc_ids")
        stop = stop_row(shards, start, stop)
        parts = [
            np.array(tensors["doc_ids"][first:end])
            for _, tensors, first, end in blocks(shards, start, stop, 1 << 62)
        ]
        return (
            np.concatenate(parts).astype(np.int64, copy=False) if parts else np.empty(0, np.int64)
        )

    def verify(self) -> None:
        """Hash every shard on disk against its checksums and check its values: finite
        scores, probabilities within [0, 1], thresholds within [0, 1], the document-ID
        statistics, and that the thresholds of neighbours are equal. A view verifies each of its
        stores and the boundaries between them."""
        shards = self._snapshot()
        if self._view is not None:
            return verify_members(self._view)
        start = time.perf_counter()
        read_store(self._root, KIND)
        kind = kind_of(self._info)
        for shard, tensors in shards:
            file = self._root / shard.name / "manifest.json"
            check_removed(NpyContainer().metadata(self._root / shard.name), file)
            NpyContainer().verify(self._root / shard.name)
            kind.check_values(shard, tensors)
        check_chain(self._root, kind, shards)
        _log.emit(
            "scores.verified",
            "fse scores: verified %d rows in %d shards",
            rows_in(shards),
            len(shards),
            rows=rows_in(shards),
            shards=len(shards),
            seconds=time.perf_counter() - start,
        )

    def refresh(self) -> int:
        """Add the shards published since the store was opened (or last refreshed) and return
        the number of rows they hold (0 if there are none). The store changes in place: a
        read or a generator already made keeps the rows it started with. Only the new
        shards are checked and, if tagged ``durability="none"``, hashed. A gap, an orphan,
        a break of the chain or a shard whose thresholds differ raises
        ``CorruptArtifactError``; a store that is no longer the one that was opened raises
        ``RuntimeError``; the rows stay as they were. Nothing is promised against
        ``repair_store`` running meanwhile. A view raises ``ValueError``: open it again."""
        if self._view is not None:
            self._check_open()
            raise ValueError("a view does not refresh; open it again")
        with self._refreshing:
            shards = self._snapshot()
            checked = refreshed(self._root, self._info, shards)
            added = rows_in(checked.shards)
            total = rows_in(shards) + added
            with self._lock:
                closed = self._closed
                if not closed:
                    self._shards = [*shards, *checked.shards]
            if closed:
                del checked  # the mappings go with the error
                raise RuntimeError("the ScoreStore is closed")
        _log.emit(
            "scores.refreshed",
            "fse scores: refreshed to %d rows: %d rows in %d shards added",
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

    # --- life cycle -----------------------------------------------------------------

    def close(self) -> None:
        """Release the store's shard references. Later calls raise ``RuntimeError``.
        Previously returned views remain readable and keep their mappings until released."""
        with self._lock:
            self._shards = []
            self._closed = True
        if self._view is not None:
            self._view.close()

    def __enter__(self) -> ScoreStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __reduce__(self):
        raise TypeError("a ScoreStore cannot be pickled; open it again from its path")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("the ScoreStore is closed")

    # --- internals ------------------------------------------------------------------

    def _snapshot(self) -> list:
        """The shards of the store now (a list that is replaced, never changed); a closed
        store raises."""
        with self._lock:
            self._check_open()
            return self._shards

    def _blocks(
        self, shards: list, block_rows: int, start: int, stop: int, make: Any, state: Any
    ) -> Generator[ScoreBatch, None, None]:
        self._check_open()  # also when closed after the call and before the first block
        for shard, tensors, first, end in blocks(shards, start, stop, block_rows):
            self._check_open()  # a closed store gives nothing more
            scores = tensors["scores"][first:end]
            ids = tensors["doc_ids"][first:end] if self._info.has_doc_ids else None
            yield make(self._info, state, scores, shard.first_row + first, ids)

    def _plans(self, shards: list, output: Any, thresholds: Any) -> dict[str, _predict.HeadPlan]:
        """How each head gives its output and what its cutoffs are, checked now."""
        heads = self._info.heads
        rules = {h.name: OutputRule(h.task, None if h.link == "none" else h.link) for h in heads}
        wanted = dict(zip(rules, resolve_outputs(rules, output), strict=True))
        given = _threshold_names(self.heads, thresholds)
        plans = {}
        for head in heads:
            limits = self._limits(shards, head, given.get(head.name, _NOT_GIVEN))
            plans[head.name] = _predict.HeadPlan(head, wanted[head.name], limits)
        return plans

    def _limits(self, shards: list, head: ScoreHead, given: Any) -> ndarray | None:
        """The cutoffs of a multilabel head: the stored ones, with ``given`` on top."""
        if not head.multilabel:
            return None
        labels = head.labels
        assert labels is not None
        if shards:
            tensors = shards[0][1]
            stored = tensors[head._threshold_name]
        else:
            stored = np.full(len(labels), 0.5, np.float32)
        if given is _NOT_GIVEN:
            return np.array(stored, dtype=np.float32)
        current = dict(zip(labels, map(float, stored), strict=True))
        return updated(current, given, labels)


_NOT_GIVEN: Any = object()


def _threshold_names(heads: Mapping[str, ScoreHead], thresholds: Any) -> dict[str, Any]:
    """The new cutoffs by head name, for multilabel heads only."""
    if thresholds is None:
        return {}
    if not isinstance(thresholds, Mapping):
        raise TypeError(
            f"thresholds must map head names to cutoffs, not {type(thresholds).__name__}"
        )
    for name in thresholds:
        if not isinstance(name, str) or name not in heads:
            raise ValueError(f"thresholds name the head {name!r}, but the heads are {list(heads)}")
        if not heads[name].multilabel:
            raise ValueError(f"head {name!r} has no thresholds: only multilabel heads do")
    return dict(thresholds)


def _columns(head: ScoreHead, scores: ndarray) -> ndarray:
    """The columns of ``head`` in the rows ``scores``, (n,) for a flat head."""
    view = scores[:, head.columns[0] : head.columns[1]]
    return view[:, 0] if _predict._store_flat(head, head.stored) else view


def _predict_raw(info: ScoreInfo, state: Any, scores: ndarray, store_row: int, ids: Any):
    del state
    outputs = {}
    for head in info.heads:
        view = _columns(head, scores).view()
        view.flags.writeable = False
        outputs[head.name] = view
    return ScoreBatch(store_row=store_row, n_rows=len(scores), outputs=outputs, doc_ids=ids)


def _predict_derived(info: ScoreInfo, state: Any, scores: ndarray, store_row: int, ids: Any):
    host, plans, work = state
    columns = {h.name: scores[:, h.columns[0] : h.columns[1]] for h in info.heads}
    return _predict.predict(host, plans, columns, store_row, work, ids)


def _check_heads(path: Path, info: ScoreInfo, heads: Any, name: Any = None) -> None:
    """The store must hold exactly the heads of ``heads``, in order, as the same models."""
    heads = group_of(heads, name)
    names = [head.name for head in info.heads]
    if names != list(heads):
        raise ValueError(f"the store {path} holds the heads {names}, not {list(heads)}")
    for head, (name, model) in zip(info.heads, heads.items(), strict=True):
        if head.model != model.fingerprint:
            raise ValueError(
                f"head {name!r} of the store {path} is another model than the one given"
            )
    if info.encoder != heads.encoder_fingerprint:
        raise ValueError(
            f"the store {path} holds the scores of another encoder than the heads read"
        )


@validate_arguments
def _open(
    cls: Any, path: UncheckedPath, *, heads: Any = None, name: Any = None, backend: str = "cpu"
) -> Any:
    try:
        return _opened(cls, path, heads=heads, name=name, backend=backend)
    except BaseException as error:
        clear_exception_frames(error)
        raise


def _opened(cls: Any, path: UncheckedPath, *, heads: Any, name: Any, backend: str) -> Any:
    if backend not in BACKEND_NAMES:
        raise ValueError(f"backend must be one of {list(BACKEND_NAMES)}, not {backend!r}")
    path = Path(file_path(path, "path"))
    info = read_store(path, KIND)
    if heads is not None:
        _check_heads(path, info, heads, name)
    elif name is not None:
        raise ValueError("name goes with a single head given as heads")
    out = cls.__new__(cls)
    out.path, out._info, out._view = path, info, None
    checked = snapshot(path, info, verify=True)
    out._shards = checked.shards
    out._backend = backend
    out._lock = threading.Lock()
    out._refreshing = threading.Lock()
    out._closed = False
    _log.emit(
        "scores.opened",
        "fse scores: opened %d rows in %d shards: %d heads, %d columns",
        len(out),
        len(out._shards),
        len(info.heads),
        info.dim,
        rows=len(out),
        shards=len(out._shards),
        heads=len(info.heads),
        columns=info.dim,
        unsynced_shards=checked.unsynced_shards,
        verified_bytes=checked.verified_bytes,
        verify_seconds=checked.verify_seconds,
    )
    return out


@validate_arguments
def _opened_view(
    cls: Any, paths: Any, *, heads: Any = None, name: Any = None, backend: str = "cpu"
) -> Any:
    def member(path: Path) -> Any:
        return cls.open(path, heads=heads, name=name, backend=backend)

    return open_view(cls, paths, member, scores=True)
