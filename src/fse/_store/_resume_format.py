#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The resume mode ``source_rows@1`` of a shard store (docs/architecture/document-ids.md,
"Resume mode").

A store written from a source that can be read again carries the identity of that source,
of the execution and of the partition it holds, in ``store.json`` and in every shard, under
the same names. This module holds the keys, the parser of the six values and the comparison of two identities. It knows nothing of what a
row holds, so the stores of vectors and of scores use it alike.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fse._container import (
    INTEGER,
    CorruptArtifactError,
    PublishedNotDurableError,
    canonical_json,
    fingerprint,
    is_fingerprint,
)
from fse._container._common import _sync_directory
from fse._partition import partition

MODE = "source_rows@1"
FORMS = ("lines", "strings", "tokens")
EXECUTION_KEY = "fse.execution_fingerprint"
SOURCE_PREFIX = "fse.source_"
SOURCE_KEYS = (
    "fse.source_fingerprint",
    "fse.source_form",
    EXECUTION_KEY,
    "fse.source_first_row",
    "fse.source_rows",
    "fse.source_shard",
)


def execution_fingerprint(metadata: dict[str, str]) -> str:
    """The fingerprint of the root metadata without the keys of the source and of the
    execution fingerprint itself, and without tensors: what must be equal for two runs to
    compute the same rows. Scheduling (workers, batches, shard size) is not part of it."""
    content = {
        key: value
        for key, value in metadata.items()
        if not key.startswith(SOURCE_PREFIX) and key != EXECUTION_KEY
    }
    return fingerprint({}, content)


@dataclass(frozen=True)
class SourceRows:
    """The identity a store of the mode ``source_rows@1`` keeps: the source (its
    fingerprint, form and number of rows), the part of it the store holds (``shard``, the
    rows from ``first_row``) and the fingerprint of the execution."""

    fingerprint: str
    form: str
    rows: int
    shard: tuple[int, int]
    first_row: int

    @property
    def span(self) -> int:
        """How many rows the part holds when it is complete."""
        return partition(self.rows, *self.shard)[1] - self.first_row

    def metadata(self, shared: dict[str, str]) -> dict[str, str]:
        """The six keys; ``shared`` are the other keys of the root (the execution
        fingerprint is made of them)."""
        return {
            "fse.source_fingerprint": self.fingerprint,
            "fse.source_form": self.form,
            EXECUTION_KEY: execution_fingerprint(shared),
            "fse.source_first_row": str(self.first_row),
            "fse.source_rows": str(self.rows),
            "fse.source_shard": canonical_json(list(self.shard)),
        }

    def same_source(self, other: SourceRows) -> str | None:
        """What differs between this identity and ``other`` in the source or the part, or
        None."""
        for name, what in (
            ("fingerprint", "the source"),
            ("form", "the form of the source"),
            ("rows", "the number of rows of the source"),
            ("shard", "the partition"),
        ):
            if getattr(self, name) != getattr(other, name):
                return what
        return None


def parse(metadata: dict[str, Any], shared: dict[str, str], file: Path) -> SourceRows:
    """The identity in ``metadata`` (a root or a shard), which holds the six keys. Its
    execution fingerprint must be that of ``shared``, the keys the shard shares with the
    root; a root or a shard that was edited fails here."""
    fingerprint_, form, execution = (metadata[k] for k in SOURCE_KEYS[:3])
    first, rows, shard = (metadata[k] for k in SOURCE_KEYS[3:])
    if not (is_fingerprint(fingerprint_) and is_fingerprint(execution)) or form not in FORMS:
        raise CorruptArtifactError(f"{file}: the source fingerprint, form or execution is invalid")
    if not all(isinstance(v, str) and INTEGER.fullmatch(v) for v in (first, rows)):
        raise CorruptArtifactError(f"{file}: fse.source_first_row and fse.source_rows are invalid")
    parts = _shard(shard, file)
    identity = SourceRows(fingerprint_, form, int(rows), parts, int(first))
    if execution_fingerprint(shared) != execution:
        raise CorruptArtifactError(f"{file}: the execution fingerprint does not match the store")
    if partition(identity.rows, *parts)[0] != identity.first_row:
        raise CorruptArtifactError(f"{file}: the first source row is not that of the partition")
    return identity


def _shard(text: Any, file: Path) -> tuple[int, int]:
    try:
        found = json.loads(text) if isinstance(text, str) else None
    except ValueError:
        found = None
    if (
        not isinstance(found, list)
        or len(found) != 2
        or any(type(v) is not int for v in found)
        or canonical_json(found) != text
        or not 0 <= found[0] < found[1]
    ):
        raise CorruptArtifactError(f"{file}: fse.source_shard must be the JSON list [i,n]")
    return found[0], found[1]


# --- what a resuming sink checks of a store (vector and score sinks alike) ------------


def check_mode(existing: Any, resume: bool, path: Path) -> None:
    """A store of the mode source_rows@1 is opened with ``resume=True`` only, and
    ``resume=True`` opens such a store only. ``existing`` is the info of the store."""
    if existing.resume is not None and not resume:
        raise ValueError(f"the store {path} is resumable (mode {MODE}); open it with resume=True")
    if existing.resume is None and resume:
        raise ValueError(
            f"the store {path} is not a resumable store; write a new store with resume=True"
        )


def check_source(existing: Any, wanted: SourceRows, path: Path) -> None:
    """The source and part of this run must be those the store was written from."""
    assert existing.resume is not None
    what = existing.resume.same_source(wanted)
    if what is not None:
        raise ValueError(
            f"the store {path} was written from another source or partition ({what} differs); "
            "a store is resumed with the same source and shard, or written anew"
        )


def sync_store(path: Path) -> None:
    """Make the entries of the store directory durable: a shard published by an earlier run
    may have been renamed in without its sync (docs/architecture/store.md, "Resume stores")."""
    try:
        _sync_directory(path)
        _sync_directory(path.parent)  # the entry of a store an earlier run made
    except OSError as error:
        raise PublishedNotDurableError(
            f"the shards of {path} are published, but syncing the directory failed ({error}); "
            "they are not confirmed durable"
        ) from error
