#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``refresh`` of a store: the shards published since the snapshot was taken
(docs/architecture/store.md, "Refresh").

A snapshot is a list of immutable shards, so a refresh only appends: it follows the chain
from the end of the snapshot with the summary of the last shard as the shard before, hashes
the new tagged shards only, and checks gaps and orphans over the whole directory. Before
that it checks that the store is still the one the snapshot came from: the root names the
same rows, and the last shard of the snapshot has the manifest fingerprint it had when it
was mapped. Nothing is promised against a repair that runs meanwhile (``repair_store`` is
offline); a repair that changed the last shard is recognized, one that changed an earlier
shard is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fse._container import FINGERPRINT_KEY, CorruptArtifactError, NpyContainer

from ._layout import read_store
from ._recovery import Snapshot, clear_exception_frames, extend


def refreshed(path: Path, info: Any, shards: list) -> Snapshot:
    """The shards published after ``shards`` (a snapshot of the store ``path``), mapped and
    checked; the snapshot itself is not changed. A store that is no longer the one of the
    snapshot, also one of another kind, is a ``RuntimeError``; a gap, an orphan or a break of
    the chain is a ``CorruptArtifactError``."""
    try:
        if read_store(path) != info:  # any kind of store: another kind is a changed store
            raise _changed()
        if shards:
            _check_last(path, shards[-1][0])
        return extend(path, info, shards, verify=True)
    except BaseException as error:
        clear_exception_frames(error)
        raise


def _check_last(path: Path, last: Any) -> None:
    try:
        found = NpyContainer().metadata(path / last.name).get(FINGERPRINT_KEY)
    except (OSError, CorruptArtifactError) as error:
        raise _changed() from error
    if found != last.fingerprint:
        raise _changed()


def _changed() -> RuntimeError:
    return RuntimeError("the store changed below its snapshot; open it again")
