#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Several stores read as one (docs/architecture/store.md, "Views").

``VectorStore.open_view(paths)`` and ``ScoreStore.open_view(paths)`` open each store as
``open`` does and chain their shards: the shards are immutable, so a view is the list of the
member shards with ``first_row`` moved by the rows of the members before them. Nothing is
merged on disk, and every read of the stores (search, score, take, batches) runs on that
list unchanged, so a view gives the bits and the order of ties of one store that holds the
same rows.

What the members must share is what ``info.check`` says of any two stores (encoder, dtype,
dimension, document IDs; for scores the heads, their columns and the paths they ran on) and,
for scores, the thresholds across the boundary of two members (the same ``follows`` as the
chain inside a store). Members of the mode ``source_rows@1`` are the parts of one source:
the view takes them in the order of their part and reads the source rows as positions. A
view has no ``refresh``.

This module also holds the readers that ``VectorStore`` and ``ScoreStore`` share for the
members they are made of (``paths``, ``members``, ``source``, ``locate``); a store that is no
view is its own single member.
"""

from __future__ import annotations

import itertools
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fse import _log
from fse._validation import is_file_path

from ._kind import Kind
from ._kinds import kind_of
from ._position_format import MODE as SOURCE_POSITION
from ._position_format import SourcePosition, anchor_of, skip_of
from ._recovery import NO_SHARD, chain_breach, clear_exception_frames
from ._resume_format import MODE as SOURCE_ROWS
from ._resume_format import SourceRows, execution_fingerprint
from ._take import positions

LIST_ORDER = "list"  # the members in the order given; positions are their concatenation


@dataclass(frozen=True)
class Joined:
    """What a view keeps of its members: the opened stores in view order, the view row each
    one starts at and its rows, the mode that ordered them and, if they are all the parts of
    one source, the identity of the whole source."""

    stores: tuple[Any, ...]
    offsets: tuple[int, ...]
    rows: tuple[int, ...]
    mode: str
    whole: SourceRows | None

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(store.path for store in self.stores)

    def close(self) -> None:
        for store in self.stores:
            store.close()


def source_mapping(resume: SourceRows | SourcePosition | None) -> Mapping[str, Any] | None:
    """The source a store was written from, as a read-only mapping (a new one on every call),
    or None for a store without one, also for a store of the mode ``source_position@1``: a
    stream is no source with rows of its own."""
    if not isinstance(resume, SourceRows):
        return None
    return MappingProxyType(
        {
            "mode": SOURCE_ROWS,
            "fingerprint": resume.fingerprint,
            "form": resume.form,
            "first_row": resume.first_row,
            "rows": resume.rows,
            "shard": list(resume.shard),
        }
    )


class MemberReaders:
    """``paths``, ``members``, ``source`` and ``locate`` of a store: those of its members if
    it is a view, else of the store itself."""

    path: Path | None
    _info: Any
    _view: Joined | None
    _shards: list

    def __len__(self) -> int:
        raise NotImplementedError

    def _check_open(self) -> None:
        raise NotImplementedError

    @property
    def _root(self) -> Path:
        """The path of a store that is no view."""
        assert self.path is not None
        return self.path

    @property
    def paths(self) -> tuple[Path, ...]:
        """The paths of the stores, in the order of the positions (one path if this store is
        no view)."""
        return (self._root,) if self._view is None else self._view.paths

    @property
    def source(self) -> Mapping[str, Any] | None:
        """The source the rows were written from, as a mapping with ``mode``,
        ``fingerprint``, ``form``, ``first_row``, ``rows`` and ``shard`` (the part ``[i, n]``
        of the source the store holds), or None. A view that holds all parts of a source
        reads it as one part ``[0, 1]`` and the positions are the rows of the source; any
        other view has none."""
        return source_mapping(self._info.resume if self._view is None else self._view.whole)

    @property
    def members(self) -> tuple[Mapping[str, Any], ...]:
        """One read-only mapping for each store of the positions, in order: its ``path``, the
        position ``first_row`` it starts at, its ``rows`` and its ``source`` (as ``source``);
        a member of the mode ``source_position@1`` also its ``position``, the pair of
        ``source_position``."""
        if self._view is None:
            return (_member(self._root, 0, len(self), self._info.resume, self.source_position),)
        view = self._view
        return tuple(
            _member(store.path, first, rows, store._info.resume, store.source_position)
            for store, first, rows in zip(view.stores, view.offsets, view.rows, strict=True)
        )

    @property
    def source_position(self) -> tuple[str, int] | None:
        """``(P, skip)`` of a store of the mode ``source_position@1``: the position of the
        stream to seek to and the records to drop after it so that the next record is the
        next row of the store (the anchor of its last shard, else of its root; ``skip`` is the
        rows from that anchor to the end). None for any other store and for a view, whose
        members hold theirs in ``members``."""
        self._check_open()
        if self._view is not None:
            return None
        shards = self._shards
        anchor = anchor_of(self._info, shards)
        rows = sum(shard.rows for shard, _ in shards)
        return None if anchor is None else skip_of(anchor, rows)

    def locate(self, ids: Any, /) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """For store positions ``ids`` (a list or an int64 array of one or two dimensions, as
        ``take_vectors`` takes them), the index into ``paths`` of the store that holds each
        position and the row in that store: two int64 arrays of the shape of ``ids``. A
        negative position or one beyond the rows raises ``ValueError``."""
        self._check_open()
        wanted = positions(ids)
        total = len(self)
        if wanted.size and (int(wanted.min()) < 0 or int(wanted.max()) >= total):
            raise ValueError(f"positions must lie within the {total} rows of the store")
        view = self._view
        if view is None:
            return np.zeros(wanted.shape, np.int64), wanted.copy()
        offsets = np.array(view.offsets, dtype=np.int64)
        full = np.flatnonzero(np.array(view.rows, dtype=np.int64))  # an empty store holds none
        member = full[np.searchsorted(offsets[full], wanted, side="right") - 1]
        return member.astype(np.int64, copy=False), wanted - offsets[member]


def _member(
    path: Path, first: int, rows: int, resume: Any, position: tuple[str, int] | None
) -> Mapping[str, Any]:
    entry = {"path": path, "first_row": first, "rows": rows, "source": source_mapping(resume)}
    if position is not None:
        entry["position"] = position
    return MappingProxyType(entry)


# --- opening ---------------------------------------------------------------------------


def open_view(
    cls: Any,
    paths: Any,
    opener: Callable[[Path], Any],
    attributes: dict[str, Any] | None = None,
    *,
    scores: bool = False,
) -> Any:
    """The view of the stores ``paths``, each opened by ``opener``, as a new object of
    ``cls`` with the ``attributes`` of its kind set as they are, and logged (``scores``: as a
    view of score stores). A member that fails to open or does not fit closes all
    that were opened and releases their mappings."""
    named = _paths(paths)
    stores: list = []
    try:
        for path in named:
            stores.append(opener(path))
        ordered, mode, whole = _ordered(stores)
        _check_alike(ordered)
        check_boundaries(kind_of(ordered[0]._info), ordered)
        joined = _joined(ordered, mode, whole)
        out = _view_of(cls, joined, attributes or {})
    except BaseException as error:
        for store in stores:
            store.close()
        clear_exception_frames(error)
        raise
    _announce(out, joined, scores=scores)
    return out


def _announce(out: Any, joined: Joined, *, scores: bool) -> None:
    fields = {
        "stores": len(joined.stores),
        "rows": len(out),
        "shards": len(out._shards),
        "mode": joined.mode,
    }
    message = "fse store: opened a view of %d stores: %d rows in %d shards, %s"
    args = (*fields.values(),)
    if scores:
        _log.emit("scores.view_opened", message, *args, **fields)
    else:
        _log.emit("store.view_opened", message, *args, **fields, dtype=out._info.dtype)


def _paths(paths: Any) -> list[Path]:
    """The paths of a view: a non-empty list or tuple of paths, none twice."""
    if not isinstance(paths, (list, tuple)):
        raise TypeError(f"paths must be a list or tuple of store paths, not {type(paths).__name__}")
    if not paths:
        raise ValueError("paths must name at least one store")
    named = []
    for path in paths:
        if not is_file_path(path):
            raise TypeError(f"paths must hold str or path objects, not {type(path).__name__}")
        named.append(Path(path))
    seen: set[tuple[int, int] | str] = set()
    for path in named:
        key = _identity(path)
        if key in seen:
            raise ValueError(f"the store {path} is listed twice")
        seen.add(key)
    return named


def _identity(path: Path) -> tuple[int, int] | str:
    """What names the directory ``path`` whatever its spelling: device and inode (so a
    case-insensitive file system and links agree), else the resolved path of one that does
    not exist, which opening it refuses."""
    try:
        found = path.stat()
    except OSError:
        return os.path.realpath(path)
    return found.st_dev, found.st_ino


def _ordered(stores: list) -> tuple[list, str, SourceRows | None]:
    """The stores in the order of the positions, the mode that gave it and the identity of the
    whole source if the stores are all its parts, finished. Stores of the mode
    ``source_rows@1`` must be parts of one source (maintainer decision of 2026-11-06), each
    part at most once; the order is that of the parts. Parts may be missing or unfinished (a
    producer still runs): the view is then a snapshot, its positions the concatenation, and
    it has no source. Stores without a source keep the order given; a mix is refused."""
    resumes = [store._info.resume for store in stores]
    if all(resume is None for resume in resumes):
        return stores, LIST_ORDER, None
    if all(isinstance(resume, SourcePosition) for resume in resumes):
        return stores, LIST_ORDER, None  # streams: the order given, no claim on a source
    for store, resume in zip(stores, resumes, strict=True):
        if resume is None:
            raise ValueError(
                f"the store {store.path} has no source, but other stores of the view were "
                f"written from one ({SOURCE_ROWS}); a view reads either the parts of one source "
                "or stores without a source"
            )
        if not isinstance(resume, SourceRows):
            raise ValueError(
                f"the store {store.path} was written from a stream ({SOURCE_POSITION}), the "
                f"others from a source ({SOURCE_ROWS}); a view reads either the parts of one "
                "source or stores of streams"
            )
    first = resumes[0]
    for store, resume in zip(stores, resumes, strict=True):
        _same_source(first, resume, store.path)
    order = sorted(range(len(stores)), key=lambda k: resumes[k].shard[0])
    _check_parts(first, [(resumes[k].shard[0], stores[k].path) for k in order])
    ordered = [stores[k] for k in order]
    complete = len(order) == first.shard[1] and all(
        len(store) == store._info.resume.span for store in ordered
    )
    whole = SourceRows(first.fingerprint, first.form, first.rows, (0, 1), 0) if complete else None
    return ordered, SOURCE_ROWS, whole


def _same_source(first: SourceRows, other: SourceRows, path: Path) -> None:
    for name, what in (
        ("fingerprint", "another source"),
        ("form", "another form of the source"),
        ("rows", "another number of rows of the source"),
    ):
        if getattr(first, name) != getattr(other, name):
            raise ValueError(f"the store {path} was written from {what} than the first store")
    if first.shard[1] != other.shard[1]:
        raise ValueError(
            f"the store {path} is a part of {other.shard[1]} parts of the source, the first "
            f"store of {first.shard[1]}"
        )


def _check_parts(first: SourceRows, parts: list[tuple[int, Path]]) -> None:
    """No part twice (``parts`` sorted by part)."""
    count = first.shard[1]
    for (index, path), (after, _) in itertools.pairwise(parts):
        if index == after:
            raise ValueError(f"part {index} of {count} is given twice (the store {path} is one)")


def _check_alike(stores: list) -> None:
    """The stores hold the same kind of rows (``info.check``) and, if resumable, come from the
    same execution."""
    first = stores[0]
    for store in stores[1:]:
        store._info.check(first._info, store.path)
        if (
            store._info.resume is not None
            and first._info.resume is not None
            and execution_fingerprint(store._info.metadata())
            != execution_fingerprint(first._info.metadata())
        ):
            raise ValueError(f"the store {store.path} was written by another execution")


def check_boundaries(kind: Kind, stores: list) -> None:
    """The chain across the boundaries of the stores: the first shard of each store must
    follow the last shard of the store before it (an empty store is passed over). For
    vectors nothing differs between shards; scores must keep their thresholds."""
    previous: object = NO_SHARD
    before: Path | None = None
    for store in stores:
        shards = store._shards
        if not shards:
            continue
        reason = chain_breach(kind, previous, kind.without_row(kind.summary(*shards[0])))
        if reason is not None:
            raise ValueError(
                f"the store {store.path} cannot follow the store {before} in a view: {reason}"
            )
        previous, before = kind.without_row(kind.summary(*shards[-1])), store.path


def _joined(stores: list, mode: str, whole: SourceRows | None) -> Joined:
    offsets, rows, first = [], [], 0
    for store in stores:
        offsets.append(first)
        rows.append(len(store))
        first += len(store)
    return Joined(tuple(stores), tuple(offsets), tuple(rows), mode, whole)


def _view_of(cls: Any, joined: Joined, attributes: dict[str, Any]) -> Any:
    """The object of ``cls`` that reads the shards of ``joined`` as one list."""
    first = joined.stores[0]
    out = cls.__new__(cls)
    out.path = None
    out._info = replace(first._info, resume=None)
    out._backend = first._backend
    out._lock = threading.Lock()
    out._refreshing = threading.Lock()
    out._closed = False
    out._view = joined
    out._shards = [
        (replace(shard, first_row=offset + shard.first_row), tensors)
        for store, offset in zip(joined.stores, joined.offsets, strict=True)
        for shard, tensors in store._shards
    ]
    for name, value in attributes.items():
        setattr(out, name, value)
    return out


def verify_members(view: Joined) -> None:
    """``verify`` of a view: each member over its own path, then the chain across their
    boundaries."""
    for store in view.stores:
        store.verify()
    check_boundaries(kind_of(view.stores[0]._info), list(view.stores))
