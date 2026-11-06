#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The kind ``prediction_scores`` (docs/architecture/scores.md, "prediction_scores 1").

``store.json`` names the encoder, the document-ID mode and the heads with their columns
(``_scores_info``); the rows are shards, each an NpyContainer artifact with one tensor
``scores`` of all heads side by side, with document IDs ``doc_ids`` and for each multilabel
head the constant tensor ``thresholds.<head>``. This module holds the metadata of a shard,
the checks of one shard, the chain (the thresholds of two neighbours are equal) and the kind
the store layer opens it by.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray

from fse._container import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    FINGERPRINT_KEY,
    INTEGER,
    CorruptArtifactError,
    NpyContainer,
)
from fse._contract import UnsupportedArtifactError

from ._id_stats import ID_KEYS, check_removed, parsed_stats, verify_stats
from ._kind import Shard, shard_name
from ._resume_format import SOURCE_KEYS
from ._scores_info import (
    FORMAT_VERSION,
    NAME,
    SHARED,
    STORE_VERSION,
    ScoreInfo,
    parse_info,
)

_VALUE_BYTES = 1 << 20


@dataclass(frozen=True)
class ScoreShard(Shard):
    """A shard of a score store: besides its rows, the column ranges of the stored
    probabilities and the names of its threshold tensors, both from the root of the store
    (no state of the kind: two stores of different layout can be open at once)."""

    proba_columns: tuple[tuple[int, int], ...] = ()
    thresholds: tuple[str, ...] = ()


def shard_metadata(
    info: ScoreInfo, first_row: int, rows: int, created_by: str, *, unsynced: bool = False
) -> dict[str, str]:
    metadata = info.metadata()
    metadata[CREATED_BY_KEY] = created_by
    metadata["fse.first_row"] = str(first_row)
    metadata["fse.rows"] = str(rows)
    if unsynced:
        metadata["fse.durability"] = "none"
    return metadata


def _shard_info(metadata: dict, info: ScoreInfo, file: Path) -> bool:
    """Check the metadata of a shard against the root; returns whether it is tagged
    ``fse.durability=none``."""
    content = {
        key: value
        for key, value in metadata.items()
        if key != FINGERPRINT_KEY and not key.startswith(CHECKSUM_PREFIX)
    }
    unsynced = "fse.durability" in content
    if unsynced and content["fse.durability"] != "none":
        raise UnsupportedArtifactError(
            f"{file}: fse.durability is {content['fse.durability']!r}; "
            "this fse supports only 'none' or an absent durability tag"
        )
    required = {CREATED_BY_KEY, "fse.first_row", "fse.rows"}
    if info.has_doc_ids:
        required.update(ID_KEYS)
    if required - set(content):
        raise CorruptArtifactError(f"{file}: the shard metadata do not hold the keys of its kind")
    shared = {k: v for k, v in content.items() if k not in required | {"fse.durability"}}
    if parse_info(shared, file) != info:
        raise CorruptArtifactError(f"{file}: the shard and the store describe different rows")
    return unsynced


def _check_tensors(tensors: dict[str, ndarray], info: ScoreInfo, n: int, file: Path) -> None:
    want = {"scores", *info.threshold_names} | ({"doc_ids"} if info.has_doc_ids else set())
    if set(tensors) != want:
        raise CorruptArtifactError(f"{file}: a shard holds exactly {sorted(want)}")
    if tensors["scores"].shape != (n, info.dim) or tensors["scores"].dtype != np.float32:
        raise CorruptArtifactError(f"{file}: scores must be float32 with {n} rows of {info.dim}")
    if info.has_doc_ids and (
        tensors["doc_ids"].shape != (n,) or tensors["doc_ids"].dtype != np.dtype("<i8")
    ):
        raise CorruptArtifactError(f"{file}: doc_ids must be int64, one per row")
    for head in info.heads:
        if head.multilabel:
            limits = tensors[head.threshold_name]
            if limits.shape != (len(head.labels or ()),) or limits.dtype != np.float32:
                raise CorruptArtifactError(
                    f"{file}: {head.threshold_name} must be float32, one value per label"
                )


def open_shard(
    path: Path, info: ScoreInfo, first_row: int
) -> tuple[ScoreShard, dict[str, ndarray]]:
    """The shard of the store ``path`` that starts at ``first_row``, mapped and checked
    against the store: its metadata, tensors and shapes."""
    name = shard_name(first_row)
    directory = path / name
    if not directory.is_dir() or directory.is_symlink():
        raise CorruptArtifactError(f"{directory} is not a regular shard directory")
    file = directory / "manifest.json"
    check_removed(NpyContainer().metadata(directory), file)
    tensors, metadata = NpyContainer().open(directory, mmap=True)
    unsynced = _shard_info(metadata, info, file)
    rows = metadata.get("fse.rows", "")
    if (
        metadata.get("fse.first_row") != str(first_row)
        or not INTEGER.fullmatch(rows)
        or rows == "0"
    ):
        raise CorruptArtifactError(f"{file}: the shard must hold rows from {first_row}")
    n = int(rows)
    if info.resume is not None and first_row + n > info.resume.span:
        raise CorruptArtifactError(f"{file}: the shard holds rows beyond the source")
    _check_tensors(tensors, info, n, file)
    expected = {key: metadata[CHECKSUM_PREFIX + key] for key in tensors}
    stats = parsed_stats(metadata, file) if info.has_doc_ids else None
    return (
        ScoreShard(
            name, first_row, n, unsynced, expected, stats, info.proba_columns, info.threshold_names
        ),
        tensors,
    )


def _finite_unit(shard: Shard, what: str, values: ndarray) -> None:
    if not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
        raise CorruptArtifactError(f"{shard.name}: {what} holds values outside [0, 1]")


def _scores(shard: ScoreShard, scores: ndarray) -> None:
    width = scores.shape[1]
    cols = max(1, min(width, _VALUE_BYTES // 4))  # a row wider than a block is cut as well
    step = max(1, _VALUE_BYTES // (4 * cols))
    for begin in range(0, len(scores), step):
        for left in range(0, width, cols):
            part = scores[begin : begin + step, left : left + cols]
            if not np.isfinite(part).all():
                raise CorruptArtifactError(f"{shard.name}: scores holds values that are not finite")
            for start, stop in shard.proba_columns:
                low, high = max(start, left), min(stop, left + cols)
                if low < high:
                    _finite_unit(
                        shard, "the probabilities of scores", part[:, low - left : high - left]
                    )


class ScoreKind:
    """``prediction_scores`` as a kind of the shard store."""

    name = NAME

    def parse_root(self, version: int, metadata: Any, file: Path) -> ScoreInfo:
        if version != STORE_VERSION:
            raise CorruptArtifactError(
                f"{file}: prediction_scores needs store version {STORE_VERSION}, not {version}"
            )
        return parse_info(metadata, file)

    def open_shard(
        self, path: Path, info: ScoreInfo, first_row: int
    ) -> tuple[ScoreShard, dict[str, ndarray]]:
        return open_shard(path, info, first_row)

    def check_values(self, shard: Shard, tensors: dict[str, ndarray]) -> None:
        """Check values in bounded blocks, without shard-sized temporary arrays."""
        if not isinstance(shard, ScoreShard):
            raise TypeError("a score shard is opened by the kind prediction_scores")
        _scores(shard, tensors["scores"])
        for name in shard.thresholds:
            _finite_unit(shard, name, tensors[name])
        if shard.id_stats is not None:
            verify_stats(tensors["doc_ids"], shard.id_stats, shard.name)

    def check_identifiers(self, metadata: dict, file: Path, info: Any) -> None:
        del info
        with contextlib.suppress(CorruptArtifactError):  # open_shard diagnoses it
            parse_info({k: v for k, v in metadata.items() if k in _ROOT_KEYS}, file)
        durability = metadata.get("fse.durability")
        if isinstance(durability, str) and durability != "none":
            raise UnsupportedArtifactError(f"{file}: unsupported fse.durability {durability!r}")

    def summary(self, shard: Shard, tensors: dict[str, ndarray]) -> object:
        """The checksums of the threshold tensors, from the manifest (no hashing)."""
        del tensors
        if not isinstance(shard, ScoreShard):
            raise TypeError("a score shard is opened by the kind prediction_scores")
        return tuple(shard.checksums[name] for name in shard.thresholds)

    def follows(self, previous: object, current: object) -> str | None:
        if previous != current:
            return "its thresholds differ from those of the shard before it"
        return None


_ROOT_KEYS = frozenset((*SHARED, *SOURCE_KEYS))
SCORES = ScoreKind()
__all__ = ["FORMAT_VERSION", "SCORES", "ScoreKind", "ScoreShard", "open_shard", "shard_metadata"]
