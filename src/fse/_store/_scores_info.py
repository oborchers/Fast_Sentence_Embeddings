#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""What every row of a ``prediction_scores`` store shares (docs/architecture/scores.md,
"prediction_scores 1"): the encoder of the heads, the document-ID mode and the list of
heads with what each one stores in which columns.

``ScoreHead`` is one entry of ``fse.score_heads``; ``ScoreInfo`` is the metadata of
``store.json`` and of every shard. The table of the combinations a head may have is data
(``_LINK_RULES`` and the checks below it), so a new link or rule adds a row, not a branch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fse._container import (
    FINGERPRINT_FORM,
    CorruptArtifactError,
    canonical_json,
    is_fingerprint,
)
from fse._container._common import _strict_json
from fse._contract import LINK_NONE, UnsupportedArtifactError, parse_id

from ._id_stats import MODE_KEYS, check_mode
from ._position_format import MODES, SourcePosition, parse_resume, resume_keys
from ._resume_format import SourceRows

NAME = "prediction_scores"
FORMAT_VERSION = "1"
STORE_VERSION = 2
THRESHOLD_PREFIX = "thresholds."

TASKS = ("classification", "regression")
STORED = ("decision", "proba", "values")
PATHS = ("folded", "normalized", "unfolded", "standalone")
MULTILABEL = "multilabel@1"
LINEAR1, LINEAR2 = "linear@1", "linear@2"
# The label rules each link can take (a head without a link takes linear@1 only).
_LINK_RULES = {
    LINK_NONE: (LINEAR1,),
    "sigmoid@1": (LINEAR1, LINEAR2),
    "softmax@1": (LINEAR1, LINEAR2),
    "ovr@1": (LINEAR1, LINEAR2),
    MULTILABEL: (LINEAR2,),
}
_ENTRY_KEYS = (
    "name",
    "model",
    "task",
    "rule",
    "link",
    "stored",
    "path",
    "labels",
    "width",
    "columns",
    "flat",
)
_NAME_FORM = re.compile(r"[a-z][a-z0-9_]*", re.ASCII)
_WORD = re.compile(r"[a-z][a-z_]*", re.ASCII)
# The metadata of the root and of every shard.
SHARED = (
    "fse.kind",
    "fse.format_version",
    "fse.transform_version",
    "fse.encoder",
    *MODE_KEYS,
    "fse.score_heads",
)


@dataclass(frozen=True)
class ScoreHead:
    """One head of a score store and the half-open ``columns`` range it occupies.

    ``name`` is the head's name and ``model`` its model fingerprint. ``task`` names
    classification or regression, ``rule`` the label rule, ``link`` the probability
    link and ``stored`` the stored output. ``path`` records the inference path
    (folded, normalized, unfolded or standalone). ``labels`` names the outputs,
    or is None for a regressor. ``width`` is the number of stored columns and
    ``flat`` records whether one output is returned as a vector.
    """

    name: str
    model: str
    task: str
    rule: str
    link: str
    stored: str
    path: str
    labels: tuple[str, ...] | tuple[int, ...] | None
    width: int
    columns: tuple[int, int]
    flat: bool | None

    def _entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "task": self.task,
            "rule": self.rule,
            "link": self.link,
            "stored": self.stored,
            "path": self.path,
            "labels": None if self.labels is None else list(self.labels),
            "width": self.width,
            "columns": list(self.columns),
            "flat": self.flat,
        }

    @property
    def multilabel(self) -> bool:
        return self.link == MULTILABEL

    @property
    def _threshold_name(self) -> str:
        return THRESHOLD_PREFIX + self.name


@dataclass(frozen=True)
class ScoreInfo:
    """What every row of a score store shares."""

    encoder: str
    has_doc_ids: bool
    heads: tuple[ScoreHead, ...]
    version: int = STORE_VERSION
    # The identity of a store of the mode source_rows@1, or the anchor of its start in the mode
    # source_position@1.
    resume: SourceRows | SourcePosition | None = None

    @property
    def kind(self) -> str:
        return NAME

    @property
    def dim(self) -> int:
        return sum(head.width for head in self.heads)

    @property
    def threshold_names(self) -> tuple[str, ...]:
        return tuple(head._threshold_name for head in self.heads if head.multilabel)

    @property
    def proba_columns(self) -> tuple[tuple[int, int], ...]:
        return tuple(head.columns for head in self.heads if head.stored == "proba")

    def metadata(self) -> dict[str, str]:
        metadata = {
            "fse.kind": NAME,
            "fse.format_version": FORMAT_VERSION,
            "fse.transform_version": "2",
            "fse.encoder": self.encoder,
            "fse.doc_ids": "int64" if self.has_doc_ids else "none",
            "fse.resume": "none" if self.resume is None else self.resume.mode,
            "fse.score_heads": canonical_json([head._entry() for head in self.heads]),
        }
        if self.resume is not None:
            metadata.update(self.resume.metadata(metadata))
        return metadata

    def check(self, other: ScoreInfo, path: Path, *, paths: bool = True) -> None:
        """Raise unless ``other`` describes the same rows as this store; with ``paths``
        False, without comparing the paths of the heads (a sink knows them only when a run
        attaches)."""
        for name in ("encoder", "has_doc_ids"):
            if getattr(self, name) != getattr(other, name):
                raise ValueError(
                    f"the store {path} holds rows of {name} {getattr(self, name)!r}, not "
                    f"{getattr(other, name)!r}"
                )
        mine = [head.name for head in self.heads]
        theirs = [head.name for head in other.heads]
        if mine != theirs:
            raise ValueError(f"the store {path} holds the heads {mine}, not {theirs}")
        for have, want in zip(self.heads, other.heads, strict=True):
            _same_head(have, want, path, paths)


def _same_head(have: ScoreHead, want: ScoreHead, path: Path, paths: bool) -> None:
    for name in ("model", "task", "rule", "link", "stored", "labels", "columns", "flat"):
        if getattr(have, name) != getattr(want, name):
            raise ValueError(
                f"the store {path} holds head {have.name!r} with {name} "
                f"{getattr(have, name)!r}, not {getattr(want, name)!r}"
            )
    if paths and have.path != want.path:
        raise ValueError(
            f"the store {path} holds head {have.name!r} on the path {have.path!r}, but this run "
            f"takes the path {want.path!r}; a head has the same bits only on the same path"
        )


# --- reading the root ---------------------------------------------------------------


def parse_info(metadata: Any, file: Path) -> ScoreInfo:
    """The ScoreInfo of the metadata of a root or a shard (the extra keys of a shard
    aside, which the caller removed), checked strictly."""
    if not isinstance(metadata, dict):
        raise CorruptArtifactError(f"{file}: metadata must be an object")
    check_mode(metadata, file)
    version = metadata.get("fse.format_version")
    if isinstance(version, str) and version != FORMAT_VERSION:
        raise UnsupportedArtifactError(f"{file}: unsupported prediction_scores format {version!r}")
    heads = metadata.get("fse.score_heads")
    if isinstance(heads, str):
        _identifiers(heads, file)
    mode = metadata.get("fse.resume")
    keys = (*SHARED, *resume_keys(mode))
    if set(metadata) != set(keys):
        raise CorruptArtifactError(f"{file}: the metadata must hold exactly {', '.join(keys)}")
    if not all(isinstance(value, str) for value in metadata.values()):
        raise CorruptArtifactError(f"{file}: every value of the metadata must be text")
    if metadata["fse.kind"] != NAME or metadata["fse.transform_version"] != "2":
        raise UnsupportedArtifactError(
            f"{file} holds {metadata['fse.kind']!r} transform {metadata['fse.transform_version']!r}; "
            f"this fse reads prediction_scores format {FORMAT_VERSION}, transform 2"
        )
    if not is_fingerprint(metadata["fse.encoder"]):
        raise CorruptArtifactError(
            f"{file}: fse.encoder must be an encoder fingerprint ({FINGERPRINT_FORM})"
        )
    if metadata["fse.doc_ids"] not in ("none", "int64") or metadata["fse.resume"] not in MODES:
        raise CorruptArtifactError(f"{file}: invalid document-ID or resume mode")
    parsed = _heads(metadata["fse.score_heads"], file)
    shared = {key: metadata[key] for key in SHARED}
    return ScoreInfo(
        metadata["fse.encoder"],
        metadata["fse.doc_ids"] == "int64",
        parsed,
        resume=parse_resume(mode, metadata, shared, file),
    )


def _entries(text: str, file: Path) -> list[Any]:
    try:
        found = _strict_json(text.encode("utf-8"), file)
    except UnicodeEncodeError as error:
        raise CorruptArtifactError(f"{file}: fse.score_heads is not UTF-8") from error
    if not isinstance(found, list) or not found or canonical_json(found) != text:
        raise CorruptArtifactError(f"{file}: fse.score_heads must be a canonical JSON list")
    return found


def _identifiers(text: str, file: Path) -> None:
    """Recognize readable future identifiers in ``fse.score_heads``, before today's
    structure rules apply: UnsupportedArtifactError."""
    try:
        found = _entries(text, file)
    except CorruptArtifactError:
        return  # the structure rules diagnose it
    for entry in found:
        if isinstance(entry, dict):
            _future(entry, file)


def _future(entry: dict, file: Path) -> None:
    for key, known in (("rule", (LINEAR1, LINEAR2)), ("link", (*_LINK_RULES,))):
        _unknown(entry.get(key), known, key, file, versioned=True)
    for key, known in (("path", PATHS), ("stored", STORED), ("task", TASKS)):
        _unknown(entry.get(key), known, key, file, versioned=False)


def _unknown(value: Any, known: tuple, key: str, file: Path, *, versioned: bool) -> None:
    if not isinstance(value, str) or value in known:
        return
    if versioned:
        try:
            parse_id(value)
        except ValueError:
            return
    elif not _WORD.fullmatch(value):
        return
    raise UnsupportedArtifactError(f"{file}: unsupported {key} {value!r}")


def _heads(text: str, file: Path) -> tuple[ScoreHead, ...]:
    heads, column = [], 0
    for entry in _entries(text, file):
        head = _head(entry, file)
        if head.columns != (column, column + head.width):
            raise CorruptArtifactError(
                f"{file}: the columns of head {head.name!r} must run from {column} to "
                f"{column + head.width} without a gap"
            )
        column = head.columns[1]
        heads.append(head)
    if len({head.name for head in heads}) != len(heads):
        raise CorruptArtifactError(f"{file}: the names of the heads must be unique")
    return tuple(heads)


def _head(entry: Any, file: Path) -> ScoreHead:
    if not isinstance(entry, dict) or set(entry) != set(_ENTRY_KEYS):
        raise CorruptArtifactError(f"{file}: a head must hold exactly {', '.join(_ENTRY_KEYS)}")
    _future(entry, file)
    texts = ("name", "model", "task", "rule", "link", "stored", "path")
    if not all(isinstance(entry[key], str) for key in texts):
        raise CorruptArtifactError(f"{file}: a head names its model, rule and link as text")
    columns, width, labels = entry["columns"], entry["width"], entry["labels"]
    if (
        not isinstance(columns, list)
        or len(columns) != 2
        or any(type(c) is not int for c in columns)
        or type(width) is not int
        or not isinstance(entry["flat"], bool | None)
        or not (labels is None or isinstance(labels, list))
    ):
        raise CorruptArtifactError(f"{file}: head {entry['name']!r} has malformed values")
    head = ScoreHead(
        name=entry["name"],
        model=entry["model"],
        task=entry["task"],
        rule=entry["rule"],
        link=entry["link"],
        stored=entry["stored"],
        path=entry["path"],
        labels=None if labels is None else tuple(labels),
        width=width,
        columns=(columns[0], columns[1]),
        flat=entry["flat"],
    )
    _check_combination(head, file)
    return head


def _check_combination(head: ScoreHead, file: Path) -> None:
    reason = combination_error(head)
    if reason is not None:
        raise CorruptArtifactError(f"{file}: head {head.name!r}: {reason}")


# --- the table of combinations --------------------------------------------------------


def combination_error(head: ScoreHead) -> str | None:
    """Why ``head`` is no combination this format can hold, or None (docs/architecture/
    scores.md, "Compatibility table")."""
    reason = _form_error(head)
    if reason is not None:
        return reason
    if head.task == "regression":
        return _regression_error(head)
    return _classification_error(head)


def _form_error(head: ScoreHead) -> str | None:
    if not _NAME_FORM.fullmatch(head.name):
        return "the name must match [a-z][a-z0-9_]*"
    if not is_fingerprint(head.model):
        return "model must be a fingerprint"
    if head.width < 1 or head.columns[1] - head.columns[0] != head.width:
        return "width must be at least 1 and span its columns"
    if head.path not in PATHS or head.task not in TASKS or head.stored not in STORED:
        return "path, task or stored is not one of the known values"
    if head.link not in _LINK_RULES or head.rule not in (LINEAR1, LINEAR2):
        return "rule or link is not a known identifier"
    return None


def _regression_error(head: ScoreHead) -> str | None:
    if (head.rule, head.link, head.stored) != (LINEAR1, LINK_NONE, "values") or head.labels:
        return "a regression head is linear@1, has no link and labels, and stores values"
    if (head.flat is None) != (head.width > 1):
        return "flat is given exactly for a head of one value"
    return None


def _classification_error(head: ScoreHead) -> str | None:
    labels = head.labels
    if head.stored == "values" or head.flat is not None:
        return "a classification head stores decision or proba and has no flat"
    if head.rule not in _LINK_RULES[head.link]:
        return f"the rule {head.rule} does not go with the link {head.link}"
    if (
        labels is None
        or not (all(type(v) is int for v in labels) or all(type(v) is str for v in labels))
        or len(set(labels)) != len(labels)
    ):
        return "labels must be unique, all int or all str"
    if head.stored == "proba" and head.link == LINK_NONE:
        return "a head without a link has no probabilities"
    if head.multilabel:
        return None if head.width == len(labels) else "a multilabel head stores one value per label"
    return _single_label_error(head, len(labels))


def _single_label_error(head: ScoreHead, labels: int) -> str | None:
    if labels < 2:
        return "a single-label head has at least two labels"
    if head.link == "sigmoid@1" and labels != 2:
        return "sigmoid has two labels"
    if head.stored == "proba":
        return None if head.width == labels else "probabilities are stored one per label"
    return _logit_error(head, labels)


def _logit_error(head: ScoreHead, labels: int) -> str | None:
    # The logits: one per label, or one for two labels.
    if not (head.width == labels or (head.width == 1 and labels == 2)):
        return "the logits are one per label, or one for two labels"
    if head.link == "sigmoid@1" and head.width != 1:
        return "sigmoid has one logit"
    if head.link == "ovr@1" and head.width < 2:
        return "ovr has at least two logits"
    return None
