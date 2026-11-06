#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The resume mode ``source_position@1`` of a shard store (docs/architecture/document-ids.md,
"Resume mode"; store.md, "Stream positions").

A store written from a stream that can be read again from a position keeps, in ``store.json``
and in every shard, an anchor ``(P, R)``: ``P`` is an opaque text of the caller (a Kafka
offset, a byte offset, a cursor) from which the source delivers, in the same order, the
record that became store row ``R``. fse never reads ``P``: it stores it, compares it for
equality in the root of two stores and hands it back. This module holds the keys, the
check of a position, the parser of an anchor and the dispatch on the two resume modes. It
knows nothing of what a row holds, so the stores of vectors and of scores use it alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fse._container import INTEGER, CorruptArtifactError, is_fingerprint

from ._resume_format import EXECUTION_KEY, SOURCE_KEYS, SourceRows, execution_fingerprint
from ._resume_format import MODE as SOURCE_ROWS
from ._resume_format import parse as parse_rows

MODE = "source_position@1"
POSITION_KEY = "fse.source_position"
ROW_KEY = "fse.source_position_row"
POSITION_KEYS = (EXECUTION_KEY, POSITION_KEY, ROW_KEY)
MAX_POSITION_BYTES = 1 << 16
MODES = ("none", SOURCE_ROWS, MODE)


@dataclass(frozen=True)
class SourcePosition:
    """An anchor of a store of the mode ``source_position@1``: the position ``position`` from
    which the source delivers the record of store row ``row``. The root of a store holds the
    anchor of its start, row 0; a shard holds its own."""

    position: str
    row: int

    mode = MODE

    def metadata(self, shared: dict[str, str]) -> dict[str, str]:
        """The three keys of a root; ``shared`` are the other keys of the root (the execution
        fingerprint is made of them)."""
        return {EXECUTION_KEY: execution_fingerprint(shared), **self.anchor_metadata()}

    def anchor_metadata(self) -> dict[str, str]:
        """The two keys of the anchor, which a shard sets to its own."""
        return {POSITION_KEY: self.position, ROW_KEY: str(self.row)}


def check_position(position: Any) -> str:
    """``position`` if it is a text of 1 to 65,536 UTF-8 bytes. fse reads nothing else of it."""
    if not isinstance(position, str):
        raise TypeError(f"a position is a str, not {type(position).__name__}")
    try:
        size = len(position.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError("a position must be text that encodes as UTF-8") from error
    if not 1 <= size <= MAX_POSITION_BYTES:
        raise ValueError(
            f"a position holds 1 to {MAX_POSITION_BYTES} bytes of UTF-8, not {size} bytes"
        )
    return position


def parse(
    metadata: dict[str, Any], shared: dict[str, str], file: Path, *, root: bool
) -> SourcePosition:
    """The anchor in ``metadata`` (a root or a shard), which holds the three keys. Its
    execution fingerprint must be that of ``shared``; the anchor of a root is at row 0."""
    execution, position, row = (metadata[k] for k in POSITION_KEYS)
    if not (isinstance(execution, str) and is_fingerprint(execution)):
        raise CorruptArtifactError(f"{file}: the execution fingerprint is invalid")
    if not (isinstance(row, str) and INTEGER.fullmatch(row)):
        raise CorruptArtifactError(f"{file}: {ROW_KEY} must be a canonical decimal number")
    try:
        check_position(position)
    except (TypeError, ValueError) as error:
        raise CorruptArtifactError(f"{file}: {POSITION_KEY} is invalid ({error})") from error
    if execution_fingerprint(shared) != execution:
        raise CorruptArtifactError(f"{file}: the execution fingerprint does not match the store")
    if root and row != "0":
        raise CorruptArtifactError(f"{file}: the anchor of a root is at row 0")
    return SourcePosition(position, int(row))


# --- the two resume modes ------------------------------------------------------------


def resume_keys(mode: Any) -> tuple[str, ...]:
    """The keys a root or a shard holds for the resume mode ``mode`` (the value of
    ``fse.resume``)."""
    if mode == SOURCE_ROWS:
        return SOURCE_KEYS
    return POSITION_KEYS if mode == MODE else ()


def parse_resume(
    mode: Any, metadata: dict[str, Any], shared: dict[str, str], file: Path, *, root: bool = False
) -> SourceRows | SourcePosition | None:
    """The identity or anchor in ``metadata`` for the resume mode ``mode``, or None for a
    store without one."""
    if mode == SOURCE_ROWS:
        return parse_rows(metadata, shared, file)
    if mode == MODE:
        return parse(metadata, shared, file, root=root)
    return None


def same_resume(
    store: SourceRows | SourcePosition | None, shard: SourceRows | SourcePosition | None
) -> bool:
    """Whether a shard describes the resume mode of its store: the same identity for
    ``source_rows@1``, any anchor for ``source_position@1`` (each shard has its own)."""
    if isinstance(store, SourcePosition):
        return isinstance(shard, SourcePosition)
    return store == shard


def anchor_of(info: Any, shards: list) -> SourcePosition | None:
    """The anchor of a store: that of its last shard, else that of its root; None if the
    store is not of the mode ``source_position@1``. ``shards`` are the ``(shard, tensors)``
    of its snapshot."""
    if not isinstance(info.resume, SourcePosition):
        return None
    return shards[-1][0].anchor if shards else info.resume


def skip_of(anchor: SourcePosition, rows: int) -> tuple[str, int]:
    """``(P, skip)`` of a store of ``rows`` rows with the anchor ``anchor``: the position to
    seek to and the records to drop after it."""
    return anchor.position, rows - anchor.row
