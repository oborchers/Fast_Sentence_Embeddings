#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Contracts of fse 2.0: versions, aggregations, postprocessing steps and batches.

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
# How far the rows of components and bases may be from orthonormal: float32 eigenvectors
# of a 300 x 300 Gram matrix are orthonormal to about 1e-6.
ORTHONORMAL_TOLERANCE = 1e-4


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

    REMOVE_COMPONENTS = ("remove_components", 1)
    PROJECT = ("project", 1)
    NORMALIZE = ("normalize", 1)


_KINDS = {"Aggregation": "aggregation", "Op": "postprocessing step"}


def check_dim(dim: object, name: str = "dim") -> int:
    """Return ``dim`` if it is a positive int (not a bool), or raise."""
    if isinstance(dim, bool) or not isinstance(dim, int | np.integer):
        raise TypeError(f"{name} must be an int, not {type(dim).__name__}")
    if dim < 1:
        raise ValueError(f"{name} must be at least 1, not {dim}")
    return int(dim)


def _frozen(array: ndarray) -> ndarray:
    """A read-only copy, so a validated step cannot change through the caller's array."""
    copy = array.copy()
    copy.setflags(write=False)
    return copy


def _check_orthonormal(rows: ndarray, name: str):
    gram = rows.astype(np.float64) @ rows.astype(np.float64).T
    error = float(np.abs(gram - np.eye(len(rows))).max())
    if error > ORTHONORMAL_TOLERANCE:
        raise ValueError(
            f"the rows of {name} must be orthonormal; their Gram matrix differs from the "
            f"identity by {error:.3g}, more than {ORTHONORMAL_TOLERANCE}"
        )


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


def _float32_matrix(value: object, name: str, ndim: int) -> ndarray:
    if not isinstance(value, ndarray):
        raise TypeError(f"{name} must be a numpy array, not {type(value).__name__}")
    if value.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32, not {value.dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, not {value.ndim}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return value


@dataclass(frozen=True, eq=False)
class RemoveComponents:
    """Subtract the projection of a row onto ``components``, optionally weighted.

    ``components`` holds k unit rows of the row dimension d (1 <= k <= d); ``weights``,
    if given, scales the coefficient of each component (uSIF).
    """

    components: ndarray
    weights: ndarray | None = None
    op = Op.REMOVE_COMPONENTS

    def __post_init__(self):
        components = _frozen(_float32_matrix(self.components, "components", 2))
        object.__setattr__(self, "components", components)
        k, d = components.shape
        if not 1 <= k <= d:
            raise ValueError(f"components must hold between 1 and {d} rows, not {k}")
        _check_orthonormal(components, "components")
        if self.weights is not None:
            weights = _frozen(_float32_matrix(self.weights, "weights", 1))
            object.__setattr__(self, "weights", weights)
            if weights.shape != (k,):
                raise ValueError(f"weights must have shape ({k},), not {weights.shape}")

    @property
    def dim(self) -> int:
        return int(self.components.shape[1])


@dataclass(frozen=True, eq=False)
class Project:
    """Replace a row of d values by its k coordinates along the rows of ``basis`` (k <= d).

    The basis holds k orthonormal rows, such as the leading principal axes from ``fit``;
    the step reduces the dimension of every row from d to k.
    """

    basis: ndarray
    op = Op.PROJECT

    def __post_init__(self):
        basis = _frozen(_float32_matrix(self.basis, "basis", 2))
        object.__setattr__(self, "basis", basis)
        k, d = basis.shape
        if not 1 <= k <= d:
            raise ValueError(f"basis must hold between 1 and {d} rows, not {k}")
        _check_orthonormal(basis, "basis")

    @property
    def dim(self) -> int:
        return int(self.basis.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.basis.shape[0])


@dataclass(frozen=True)
class Normalize:
    """Scale a row to unit length; an all-zero row stays zero."""

    op = Op.NORMALIZE


Step = RemoveComponents | Project | Normalize

# Steps that are linear in the row: an encoder may fold them into its word vectors.
LINEAR_OPS = frozenset({Op.REMOVE_COMPONENTS, Op.PROJECT})

# The order of steps: each at most once, removal before projection before normalization.
_STEP_ORDER = (Op.REMOVE_COMPONENTS, Op.PROJECT, Op.NORMALIZE)


def check_steps(steps: Sequence[object], dim: int) -> tuple[Step, ...]:
    """Return ``steps`` as a tuple if they form an allowed list for rows of ``dim`` values."""
    dim = check_dim(dim)
    checked: list[Step] = []
    for step in steps:
        if not isinstance(step, RemoveComponents | Project | Normalize):
            raise TypeError(f"{step!r} is not a postprocessing step")
        checked.append(step)
    ops = [step.op for step in checked]
    if ops != [op for op in _STEP_ORDER if op in ops]:
        raise ValueError(
            "postprocessing steps must be a subset of remove_components, project and "
            f"normalize in this order, each at most once, not {[op.id for op in ops]}"
        )
    for step in checked:
        if isinstance(step, RemoveComponents | Project):
            if step.dim != dim:
                raise ValueError(
                    f"the {step.op.id} step takes rows of {step.dim} values, not {dim}"
                )
            if isinstance(step, Project):
                dim = step.output_dim
    return tuple(checked)


def output_dim(steps: Sequence[Step], dim: int) -> int:
    """The dimension of rows of ``dim`` values after ``steps``."""
    for step in check_steps(steps, dim):
        if isinstance(step, Project):
            dim = step.output_dim
    return dim


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
