#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.ScoreStore``: a score store opened for reading (docs/architecture/scores.md,
"Reading").

The shards published when the store is opened are mapped read-only (a snapshot: later
appends need a new open). ``iter_batches`` gives the stored values as they are, read-only
views of the mapping; ``predict_batches`` gives labels or values as ``fse.Heads`` does, derived
from what is stored, with the thresholds of the multilabel heads replaced if asked. No
encoder, tokenizer or weight table is loaded.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _log
from fse._backends import BACKEND_NAMES, get_backend
from fse._container import NpyContainer
from fse._head import Heads
from fse._head._heads_run import OutputRule, resolve_outputs
from fse._head._thresholds import updated
from fse._store._id_stats import check_removed
from fse._store._kind import Shard
from fse._store._kinds import kind_of
from fse._store._layout import read_store
from fse._store._recovery import check_chain, clear_exception_frames, snapshot
from fse._store._scores_info import ScoreHead, ScoreInfo
from fse._validation import NonNegativeInt, PositiveInt, validate_arguments

from . import _predict
from ._batches import ScoreBatch, blocks

DEFAULT_BLOCK_ROWS = 65_536
KIND = "prediction_scores"


class ScoreStore:
    """The rows of a score store at the time it was opened. Close it (or use it as a context
    manager) to release the mappings; views given before stay readable, and a closed store
    gives nothing more: the next block of ``iter_batches`` or ``predict_batches`` raises
    ``RuntimeError``, also before the first."""

    path: Path
    _info: ScoreInfo
    _shards: list[tuple[Shard, dict[str, ndarray]]]
    _backend: str
    _lock: threading.Lock
    _closed: bool

    @classmethod
    def open(cls, path: Any, *, heads: Any = None, backend: str = "cpu") -> ScoreStore:
        """Open the store ``path``; with ``heads`` (a ``fse.Heads``), the store must hold
        exactly those heads, in that order, with those models."""
        return _open(cls, path, heads=heads, backend=backend)

    def __init__(self, *_args: Any, **_kwargs: Any):
        raise TypeError("open a store with ScoreStore.open(path)")

    # --- what it holds --------------------------------------------------------------

    def __len__(self) -> int:
        return sum(shard.rows for shard, _ in self._shards)

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
    ) -> Iterator[ScoreBatch]:
        """The stored values of the rows ``start`` to ``stop - 1`` (all by default) in
        blocks of at most ``block_rows`` rows, never over the end of a shard: for each head
        a read-only view of the mapped shard (the stored decision, probabilities or values;
        ``(n,)`` for a head that is flat). A head of several columns is not contiguous in
        the view; copy it if you need that. Raises ``RuntimeError`` for the next block after
        ``close``, also before the first."""
        self._check_open()
        stop = self._rows(start, stop)
        return self._blocks(block_rows, start, stop, _predict_raw, None)

    @validate_arguments
    def predict_batches(
        self,
        *,
        output: Any = None,
        thresholds: Any = None,
        block_rows: PositiveInt = DEFAULT_BLOCK_ROWS,
        backend: str | None = None,
    ) -> Iterator[ScoreBatch]:
        """Labels or values for every row, in blocks, as ``Heads.predict_batches`` gives
        them: by default the labels of a classifier and the values of a regressor,
        ``output`` as for ``Heads`` (one name or a mapping of head names to names). What a
        head can give follows from what it stored (see ``fse._scores._predict``); anything
        else raises before a row is read. ``thresholds`` maps multilabel head names to
        new cutoffs (a partial mapping keeps the stored cutoffs of the labels it does not
        name; None resets to 0.5); no model, encoder or tokenizer is loaded. Raises
        ``RuntimeError`` for the next block after ``close``, also before the first."""
        self._check_open()
        name = self._backend if backend is None else backend
        if name not in BACKEND_NAMES:
            raise ValueError(f"backend must be one of {list(BACKEND_NAMES)}, not {name!r}")
        plans = self._plans(output, thresholds)
        return self._blocks(
            block_rows, 0, len(self), _predict_derived, (get_backend(name), plans, {})
        )

    @validate_arguments
    def get_scores(
        self,
        head: str,
        stored: str,
        *,
        start: NonNegativeInt = 0,
        stop: NonNegativeInt | None = None,
    ) -> ndarray:
        """A copy of the values of ``head`` for the rows ``start`` to ``stop - 1`` (all by
        default): ``stored`` must be what the head stored (``decision``, ``proba`` or
        ``values``), so that logits and probabilities are never taken for each other."""
        self._check_open()
        found = self.heads.get(head)
        if found is None:
            raise ValueError(f"the store holds the heads {list(self.heads)}, not {head!r}")
        if stored != found.stored:
            raise ValueError(
                f"head {head!r} stored {found.stored}, not {stored!r}; the values are not "
                "converted here (predict_batches derives other outputs)"
            )
        stop = self._rows(start, stop)
        parts = [
            np.array(_columns(found, tensors["scores"][first:end]))
            for _, tensors, first, end in blocks(self._shards, start, stop, 1 << 62)
        ]
        shape = (0,) if _flat(found) else (0, found.width)
        return np.concatenate(parts) if parts else np.empty(shape, np.float32)

    @validate_arguments
    def get_doc_ids(
        self, *, start: NonNegativeInt = 0, stop: NonNegativeInt | None = None
    ) -> ndarray:
        """The document IDs (int64) of the rows ``start`` to ``stop - 1`` (all by default),
        a copy."""
        self._check_open()
        if not self._info.has_doc_ids:
            raise ValueError("this store has no doc_ids")
        stop = self._rows(start, stop)
        parts = [
            np.array(tensors["doc_ids"][first:end])
            for _, tensors, first, end in blocks(self._shards, start, stop, 1 << 62)
        ]
        return (
            np.concatenate(parts).astype(np.int64, copy=False) if parts else np.empty(0, np.int64)
        )

    def verify(self) -> None:
        """Hash every shard on disk against its checksums and check its values: finite
        scores, probabilities within [0, 1], thresholds within [0, 1], the document-ID
        statistics, and that the thresholds of neighbours are equal."""
        self._check_open()
        start = time.perf_counter()
        read_store(self.path, KIND)
        kind = kind_of(self._info)
        for shard, tensors in self._shards:
            file = self.path / shard.name / "manifest.json"
            check_removed(NpyContainer().metadata(self.path / shard.name), file)
            NpyContainer().verify(self.path / shard.name)
            kind.check_values(shard, tensors)
        check_chain(self.path, kind, self._shards)
        _log.emit(
            "scores.verified",
            "fse scores: verified %d rows in %d shards",
            len(self),
            len(self._shards),
            rows=len(self),
            shards=len(self._shards),
            seconds=time.perf_counter() - start,
        )

    # --- life cycle -----------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._shards = []
            self._closed = True

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

    def _rows(self, start: int, stop: int | None) -> int:
        stop = len(self) if stop is None else stop
        if not start <= stop <= len(self):
            raise ValueError(f"rows {start} to {stop} are not within the {len(self)} rows")
        return stop

    def _blocks(self, block_rows: int, start: int, stop: int, make: Any, state: Any):
        self._check_open()  # also when closed after the call and before the first block
        shards = list(self._shards)
        for shard, tensors, first, end in blocks(shards, start, stop, block_rows):
            self._check_open()  # a closed store gives nothing more
            scores = tensors["scores"][first:end]
            ids = tensors["doc_ids"][first:end] if self._info.has_doc_ids else None
            yield make(self._info, state, scores, shard.first_row + first, ids)

    def _plans(self, output: Any, thresholds: Any) -> dict[str, _predict.HeadPlan]:
        """How each head gives its output and what its cutoffs are, checked now."""
        heads = self._info.heads
        rules = {h.name: OutputRule(h.task, None if h.link == "none" else h.link) for h in heads}
        wanted = dict(zip(rules, resolve_outputs(rules, output), strict=True))
        given = _threshold_names(self.heads, thresholds)
        plans = {}
        for head in heads:
            limits = self._limits(head, given.get(head.name, _NOT_GIVEN))
            plans[head.name] = _predict.HeadPlan(head, wanted[head.name], limits)
        return plans

    def _limits(self, head: ScoreHead, given: Any) -> ndarray | None:
        """The cutoffs of a multilabel head: the stored ones, with ``given`` on top."""
        if not head.multilabel:
            return None
        labels = head.labels
        assert labels is not None
        if self._shards:
            tensors = self._shards[0][1]
            stored = tensors[head.threshold_name]
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


def _flat(head: ScoreHead) -> bool:
    """Whether one column of the head is given as (n,): as ``Heads`` gives a classifier's
    decision of one logit and the values of a regressor that was a vector."""
    if head.width != 1:
        return False
    return (head.task == "classification" and head.stored == "decision") or head.flat is True


def _columns(head: ScoreHead, scores: ndarray) -> ndarray:
    """The columns of ``head`` in the rows ``scores``, (n,) for a flat head."""
    view = scores[:, head.columns[0] : head.columns[1]]
    return view[:, 0] if _flat(head) else view


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


def _check_heads(path: Path, info: ScoreInfo, heads: Any) -> None:
    """The store must hold exactly the heads of ``heads``, in order, as the same models."""
    if not isinstance(heads, Heads):
        raise TypeError(f"heads must be a fse.Heads, not {type(heads).__name__}")
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
def _open(cls: Any, path: Any, *, heads: Any = None, backend: str = "cpu") -> Any:
    try:
        return _opened(cls, path, heads=heads, backend=backend)
    except BaseException as error:
        clear_exception_frames(error)
        raise


def _opened(cls: Any, path: Any, *, heads: Any, backend: str) -> Any:
    if backend not in BACKEND_NAMES:
        raise ValueError(f"backend must be one of {list(BACKEND_NAMES)}, not {backend!r}")
    path = Path(path)
    info = read_store(path, KIND)
    if heads is not None:
        _check_heads(path, info, heads)
    out = cls.__new__(cls)
    out.path, out._info = path, info
    checked = snapshot(path, info, verify=True)
    out._shards = checked.shards
    out._backend = backend
    out._lock = threading.Lock()
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
