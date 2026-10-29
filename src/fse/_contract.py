#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Contracts of fse 2.0: versions, aggregations, the normalize step and batches.

ARCHITECTURE.md, section "2.0", states the rules these types encode. Aggregations and
postprocessing steps are identified as ``"<name>@<version>"``; saved artifacts carry
these strings instead of Python class paths, so the strings are the persistence API.
A released identifier never changes its meaning, and an unknown one is rejected when an
artifact is read, never skipped.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy import ndarray

# The version of the numerical rules shared by every aggregation and step (ARCHITECTURE.md,
# "Numerical contract"). 1 is the arithmetic of fse 1.x, which only exported data carries.
TRANSFORM_VERSION = 2
SUPPORTED_TRANSFORM_VERSIONS = frozenset({TRANSFORM_VERSION})

# Token id of an unknown FastText word in a Batch; its ngram buckets follow in the batch.
OOV_TOKEN = 0xFFFFFFFE

_ID = re.compile(r"([a-z][a-z0-9_]*)@([1-9][0-9]*)")
_INT64_MAX = 2**63 - 1


class UnsupportedArtifactError(ValueError):
    """An artifact needs an aggregation, step or version this fse does not know."""


def parse_id(text: object) -> tuple[str, int]:
    """Split ``"<name>@<version>"`` into its name and version, or raise a ValueError."""
    if not isinstance(text, str):
        raise ValueError(f"an identifier must be a str, not {type(text).__name__}")
    match = _ID.fullmatch(text)
    if match is None:
        raise ValueError(
            f"{text!r} is not an identifier of the form name@version, such as 'weighted_mean@1'"
        )
    return match.group(1), int(match.group(2))


class _Versioned(Enum):
    """An enum whose members are ``(name, version)`` pairs, written as ``name@version``."""

    @property
    def id(self) -> str:
        name, version = self.value
        return f"{name}@{version}"

    def __str__(self) -> str:
        return self.id

    @classmethod
    def parse(cls, text: object):
        """The member written as ``text``; raises UnsupportedArtifactError for unknown ones."""
        key = parse_id(text)
        for member in cls:
            if member.value == key:
                return member
        supported = ", ".join(member.id for member in cls)
        raise UnsupportedArtifactError(
            f"{_KINDS[cls.__name__]} {text} is not supported by this fse; supported: {supported}"
        )


class Aggregation(_Versioned):
    """How the weighted token rows of a sentence combine into one row (stage 3)."""

    WEIGHTED_MEAN = ("weighted_mean", 1)

    def output_dim(self, input_dim: int) -> int:
        """The dimension of a row aggregated from rows of ``input_dim`` values."""
        return check_dim(input_dim, "input_dim")


class Op(_Versioned):
    """A postprocessing step applied to every aggregated row (stage 4)."""

    NORMALIZE = ("normalize", 1)


class Tokenizer(_Versioned):
    """A native tokenizer that splits the text of a sentence into tokens (stage 1)."""

    FSE_WORDS = ("fse_words", 1)
    # BERT's WordPiece as Hugging Face tokenizers 0.23.3 runs it; its parameters are data
    # of the model (fse._tokenizer_spec).
    WORDPIECE = ("wordpiece", 1)


# The tokens were given by the caller (or split at whitespace), who owns reproducing them.
EXTERNAL_TOKENIZER = "external"


def check_tokenizer(text: object) -> str:
    """Return ``text`` if it names a supported tokenizer, or raise UnsupportedArtifactError."""
    if isinstance(text, str) and text == EXTERNAL_TOKENIZER:
        return EXTERNAL_TOKENIZER
    supported = ", ".join([EXTERNAL_TOKENIZER, *(member.id for member in Tokenizer)])
    try:
        return Tokenizer.parse(text).id
    except ValueError:
        raise UnsupportedArtifactError(
            f"tokenizer {text!r} is not supported by this fse; supported: {supported}"
        ) from None


_KINDS = {"Aggregation": "aggregation", "Op": "postprocessing step", "Tokenizer": "tokenizer"}


def check_dim(dim: object, name: str = "dim") -> int:
    """Return ``dim`` if it is a positive int (not a bool), or raise."""
    if isinstance(dim, bool) or not isinstance(dim, int | np.integer):
        raise TypeError(f"{name} must be an int, not {type(dim).__name__}")
    if dim < 1:
        raise ValueError(f"{name} must be at least 1, not {dim}")
    return int(dim)


def check_transform_version(version: object) -> int:
    """Return ``version`` if this fse can apply it, or raise UnsupportedArtifactError."""
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"transform_version must be an int, not {type(version).__name__}")
    if version not in SUPPORTED_TRANSFORM_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_TRANSFORM_VERSIONS))
        raise UnsupportedArtifactError(
            f"transform_version {version} is not supported by this fse; supported: {supported}"
        )
    return version


@dataclass(frozen=True)
class Normalize:
    """Scale a row to unit length; an all-zero row stays zero."""

    op = Op.NORMALIZE


Step = Normalize


def check_steps(steps: Sequence[object], dim: int) -> tuple[Step, ...]:
    """Return ``steps`` as a tuple if they form an allowed list for rows of ``dim`` values:
    none, or ``normalize`` once. Linear maps belong in the rows of a finished table."""
    check_dim(dim)
    checked: list[Step] = []
    for step in steps:
        if not isinstance(step, Normalize):
            raise TypeError(f"{step!r} is not a postprocessing step")
        checked.append(step)
    if len(checked) > 1:
        raise ValueError(
            f"the postprocessing steps are none or normalize once, not {[s.op.id for s in checked]}"
        )
    return tuple(checked)


def prerelease_error(kind: str, found: Sequence[str]) -> UnsupportedArtifactError:
    """The error for an artifact of a pre-release of 2.0, which held ``found`` (keys or
    tensors this fse removed, such as counts, recipes or learned steps)."""
    return UnsupportedArtifactError(
        f"this {kind} was written by a pre-release of 2.0 (it holds {', '.join(sorted(found))}, "
        "which 2.0 no longer has); rebuild the model with this fse"
    )


def _int_array(value: object, name: str, dtype: type) -> ndarray:
    if not isinstance(value, ndarray):
        raise TypeError(f"{name} must be a numpy array, not {type(value).__name__}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}, not {value.dtype}")
    if value.ndim != 1:
        raise ValueError(f"{name} must have one dimension, not {value.ndim}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _readonly(array: ndarray) -> ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


def _check_offsets(offsets: ndarray, total: int, name: str):
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError(f"{name} must start with 0")
    # Compare neighbors instead of np.diff, whose int64 differences can wrap around.
    if (offsets[1:] < offsets[:-1]).any():
        raise ValueError(f"{name} must not decrease")
    if offsets[-1] != total:
        raise ValueError(f"{name} must end with {total}, the number of entries, not {offsets[-1]}")


@dataclass(frozen=True, eq=False)
class Batch:
    """The token ids of consecutive sentences, the input of an aggregation.

    Sentence s has the tokens ``tokens[offsets[s]:offsets[s + 1]]`` and is row
    ``first_row + s`` of the output. A token is a vocabulary id or ``OOV_TOKEN`` for an
    unknown FastText word, whose ngram buckets are ``ngrams[ngram_offsets[k]:ngram_offsets[k
    + 1]]`` for the k-th unknown word of the batch. Unknown words without FastText are not
    tokens of a batch. Weights are no part of a batch; the encoder supplies them.
    """

    first_row: int
    offsets: ndarray
    tokens: ndarray
    ngram_offsets: ndarray | None = None
    ngrams: ndarray | None = None

    def __post_init__(self):
        if isinstance(self.first_row, bool) or not isinstance(self.first_row, int | np.integer):
            raise TypeError(f"first_row must be an int, not {type(self.first_row).__name__}")
        if not 0 <= self.first_row <= _INT64_MAX:
            raise ValueError(f"first_row must be between 0 and 2**63 - 1, not {self.first_row}")
        tokens = _int_array(self.tokens, "tokens", np.uint32)
        offsets = _int_array(self.offsets, "offsets", np.int64)
        _check_offsets(offsets, len(tokens), "offsets")
        # Read-only views, so a checked batch cannot be changed through its fields.
        object.__setattr__(self, "tokens", _readonly(tokens))
        object.__setattr__(self, "offsets", _readonly(offsets))
        if int(self.first_row) + len(offsets) - 2 > _INT64_MAX:
            raise ValueError(f"the rows of a batch from {self.first_row} exceed the int64 range")
        unknown = int(np.count_nonzero(tokens == OOV_TOKEN))
        if (self.ngram_offsets is None) != (self.ngrams is None):
            raise ValueError("ngram_offsets and ngrams must be given together")
        if self.ngram_offsets is None:
            if unknown:
                raise ValueError(f"{unknown} unknown words in a batch without ngrams")
            return
        ngrams = _int_array(self.ngrams, "ngrams", np.uint32)
        ngram_offsets = _int_array(self.ngram_offsets, "ngram_offsets", np.int64)
        if len(ngram_offsets) != unknown + 1:
            raise ValueError(
                f"ngram_offsets must have {unknown + 1} entries for {unknown} unknown words, "
                f"not {len(ngram_offsets)}"
            )
        _check_offsets(ngram_offsets, len(ngrams), "ngram_offsets")
        object.__setattr__(self, "ngrams", _readonly(ngrams))
        object.__setattr__(self, "ngram_offsets", _readonly(ngram_offsets))

    @property
    def n_sentences(self) -> int:
        return len(self.offsets) - 1
