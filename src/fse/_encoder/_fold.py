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

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy import ndarray

from fse._backends import get_backend
from fse._container import CorruptArtifactError, array_sha256, fingerprint, tensor_code
from fse._contract import Project, RemoveComponents
from fse._wordvectors import WordVectors

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
        out[name] = _checked_copy(tensors[name], name, expected)
        float32[name] = (tensor_code(out[name]), out[name].shape, array_sha256(out[name]))
    for rows, scales in (("vectors", "scales"), ("ngrams", "ngram_scales")):
        if rows not in tensors:
            continue
        stored, stored_scales, digest = _fold_rows(tensors, rows, scales, a, dtype, expected)
        out[rows] = stored
        if stored_scales is not None:
            out[scales] = stored_scales
        float32[rows] = ("F32", (len(stored), a.shape[0]), digest)
    new = {
        "fse.kind": "word_vectors",
        "fse.format_version": "1",
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


def _checked_copy(array: ndarray, name: str, expected: dict[str, str] | None) -> ndarray:
    copy = np.array(array, copy=True)
    if expected is not None and array_sha256(copy) != expected[name]:
        raise CorruptArtifactError(f"tensor {name} changed since it was loaded")
    copy.setflags(write=False)
    return copy


def _fold_rows(
    tensors: dict[str, ndarray],
    rows: str,
    scales: str,
    a: ndarray,
    dtype: str,
    expected: dict[str, str] | None,
) -> tuple[ndarray, ndarray | None, str]:
    """The folded rows of tensor ``rows``, stored as ``dtype``, and the sha256 of the
    folded float32 rows."""
    source = tensors[rows]
    source_scales = (
        None if scales not in tensors else _checked_copy(tensors[scales], scales, expected)
    )
    n, k = len(source), a.shape[0]
    stored = np.empty((n, k), dtype=source.dtype)
    stored_scales = np.empty(n, dtype=np.float32) if source_scales is not None else None
    store = get_backend("cpu")
    read, folded = hashlib.sha256(), hashlib.sha256()
    for begin in range(0, n, BLOCK_ROWS):
        block = np.array(source[begin : begin + BLOCK_ROWS], copy=True)
        read.update(memoryview(block.reshape(-1).view(np.uint8)))
        x = block.astype(np.float32)
        if source_scales is not None:
            x *= source_scales[begin : begin + len(block), None]  # code * scale, rounded once
        y = np.ascontiguousarray((x.astype(np.float64) @ a.T).astype(np.float32))
        folded.update(memoryview(y.reshape(-1).view(np.uint8)))
        end = begin + len(block)
        if dtype == "float32":
            stored[begin:end] = y
        else:
            store.store(
                y,
                stored[begin:end],
                None if stored_scales is None else stored_scales[begin:end],
                begin,
            )
    if expected is not None and read.hexdigest() != expected[rows]:
        raise CorruptArtifactError(f"tensor {rows} changed since it was loaded")
    stored.setflags(write=False)
    if stored_scales is not None:
        stored_scales.setflags(write=False)
    return stored, stored_scales, folded.hexdigest()
