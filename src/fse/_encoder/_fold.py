#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Folding learned linear steps into word vectors (ARCHITECTURE.md, "2.0 design").

The weighted mean is linear, and so are ``remove_components`` and ``project``: applied to
every word and ngram row before averaging they give the rows of applying them to every
averaged row, within float32 rounding. ``fold`` makes the folded word vectors in blocks:
every stored row is read as float32 (int8 as code * scale, rounded once), multiplied by
the map of the steps in float64 (``A x``, one BLAS product per block) and rounded once to
float32; float16 and int8 vectors are stored again by ``quantize_rows@1``. This is fit
arithmetic, deterministic per machine and BLAS, and frozen into the encoder.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import ndarray

from fse._backends import get_backend
from fse._container import CorruptArtifactError, Hasher, checksum, fingerprint, tensor_code
from fse._contract import Project, RemoveComponents
from fse._wordvectors import WordVectors
from fse._wordvectors_format import FORMAT_VERSION

BLOCK_ROWS = 16_384
# The keys of word vectors that folding keeps; the others are made anew.
_KEPT_KEYS = ("fse.fasttext", "fse.ngram_hash")


def linear_map(steps: Sequence[RemoveComponents | Project], dim: int) -> ndarray:
    """The matrix A of the linear steps in float64: a folded row is A x."""
    a = np.eye(dim)
    for step in steps:
        if isinstance(step, RemoveComponents):
            u = step.components.astype(np.float64)
            w = np.ones(len(u)) if step.weights is None else step.weights.astype(np.float64)
            a = (np.eye(u.shape[1]) - u.T @ (w[:, None] * u)) @ a
        else:
            a = step.basis.astype(np.float64) @ a
    return a


def fold(word_vectors: WordVectors, steps: Sequence[RemoveComponents | Project]) -> WordVectors:
    """New word vectors of the dtype of ``word_vectors`` with ``steps`` folded into every
    word and ngram row; without counts. Data loaded from a file are hashed as they are
    read and must still match their checksums, so damaged or changed data never become
    folded vectors."""
    tensors, metadata, expected = word_vectors._parts()
    a = linear_map(steps, word_vectors.dim)
    dtype = word_vectors.dtype
    out: dict[str, ndarray] = {}
    float32: dict[str, tuple[str, tuple[int, ...], str]] = {}
    for name in ("vocab.bytes", "vocab.offsets"):
        # Hashed once: for the check against the loaded checksum and for quantized_from.
        out[name], digest = _checked_copy(tensors[name], name, expected, dtype != "float32")
        if digest is not None:
            float32[name] = (tensor_code(out[name]), out[name].shape, digest)
    for rows, scales in (("vectors", "scales"), ("ngrams", "ngram_scales")):
        if rows not in tensors:
            continue
        stored, stored_scales, digest = _fold_rows(tensors, rows, scales, a, dtype, expected)
        out[rows] = stored
        if stored_scales is not None:
            out[scales] = stored_scales
        if digest is not None:
            float32[rows] = ("F32", (len(stored), a.shape[0]), digest)
    new = {
        "fse.kind": "word_vectors",
        "fse.format_version": FORMAT_VERSION,
        "fse.dim": str(a.shape[0]),
        "fse.dtype": "float32",
        **{k: metadata[k] for k in _KEPT_KEYS if k in metadata},
    }
    if dtype != "float32":
        new_float32 = dict(new)
        new["fse.dtype"] = dtype
        new["fse.quantization"] = "quantize_rows@1"
        # The fingerprint of the folded float32 vectors, which are never held whole.
        new["fse.quantized_from"] = fingerprint(float32, new_float32)
    return word_vectors._derived(out, new)


def _checked_copy(
    array: ndarray, name: str, expected: dict[str, str] | None, hashed: bool = False
) -> tuple[ndarray, str | None]:
    """A frozen copy of ``array``, which must still match its loaded checksum, and the
    checksum of the copy if one was computed (for ``expected`` or if ``hashed``)."""
    copy = np.array(array, copy=True)
    digest = checksum(copy) if expected is not None or hashed else None
    if expected is not None and digest != expected[name]:
        raise CorruptArtifactError(f"tensor {name} changed since it was loaded")
    copy.setflags(write=False)
    return copy, digest


def _fold_rows(
    tensors: dict[str, ndarray],
    rows: str,
    scales: str,
    a: ndarray,
    dtype: str,
    expected: dict[str, str] | None,
) -> tuple[ndarray, ndarray | None, str | None]:
    """The folded rows of tensor ``rows``, stored as ``dtype``, and for float16 and int8
    the checksum of the folded float32 rows."""
    source = tensors[rows]
    source_scales = (
        None if scales not in tensors else _checked_copy(tensors[scales], scales, expected)[0]
    )
    n = len(source)
    stored = np.empty((n, a.shape[0]), dtype=source.dtype)
    stored_scales = np.empty(n, dtype=np.float32) if source_scales is not None else None
    # Hashed only where needed: data loaded with checksums, and the folded float32 rows
    # of vectors stored again (their fingerprint is quantized_from).
    read = Hasher() if expected is not None else None
    folded = Hasher() if dtype != "float32" else None
    store = get_backend("cpu")
    for begin in range(0, n, BLOCK_ROWS):
        block = np.array(source[begin : begin + BLOCK_ROWS], copy=True)
        _update(read, block)
        y = _folded_block(block, source_scales, begin, a)
        _update(folded, y)
        end = begin + len(block)
        if dtype == "float32":
            stored[begin:end] = y
        else:
            target = None if stored_scales is None else stored_scales[begin:end]
            store.store(y, stored[begin:end], target, begin)
    if expected is not None and read is not None and read.checksum() != expected[rows]:
        raise CorruptArtifactError(f"tensor {rows} changed since it was loaded")
    stored.setflags(write=False)
    if stored_scales is not None:
        stored_scales.setflags(write=False)
    return stored, stored_scales, None if folded is None else folded.checksum()


def _update(hasher: Hasher | None, array: ndarray):
    """Feed ``array`` to ``hasher``, if a checksum is needed."""
    if hasher is not None:
        hasher.update(array)


def _folded_block(block: ndarray, scales: ndarray | None, begin: int, a: ndarray) -> ndarray:
    """A x of every stored row of ``block``, read as float32 (int8 as code * scale)."""
    x = block.astype(np.float32)
    if scales is not None:
        x *= scales[begin : begin + len(block), None]  # code * scale, rounded once
    return np.ascontiguousarray((x.astype(np.float64) @ a.T).astype(np.float32))
