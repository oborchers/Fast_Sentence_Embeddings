#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""What the readers of models of other libraries share.

docs/architecture/hub.md, "Model2Vec models and from_pretrained" and "sentence-transformers
models": the files of a model (``Files``), each read once into bytes that are checked and
then parsed; strict JSON configs; the float32 table of one tensor, whose header is checked
before its data are read; the parts of the encoder a reader gives (``Converted``); and the
module types of sentence-transformers as each version writes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy import ndarray

from fse._container import CorruptArtifactError, foreign_header, foreign_tensors, strict_json
from fse._contract import UnsupportedArtifactError
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors

MAX_CONFIG_BYTES = 2**20
# The rule of a model itself, which an argument of from_pretrained (max_length, truncate_dim,
# normalize) keeps.
MODEL_RULE = "model"
_BLOCK_ROWS = 1 << 16
# The tensors of vocabulary quantization (model2vec), which no reader of fse reads.
QUANTIZED = frozenset({"weights", "mapping"})
_DTYPE_OF_CODE = {"F32": "float32", "F16": "float16", "I8": "int8", "U8": "uint8"}
_DTYPE_OF_CODE |= {"U32": "uint32", "I64": "int64", "F64": "float64"}

# The type of a module in a modules.json of sentence-transformers, as its versions write
# it: up to 5.3 the module of the class, from 5.4 the module and the class (Normalize
# moved again in 6.0).
STATIC_EMBEDDING = frozenset(
    {
        "sentence_transformers.models.StaticEmbedding",
        "sentence_transformers.sentence_transformer.modules.static_embedding.StaticEmbedding",
    }
)
NORMALIZE = frozenset(
    {
        "sentence_transformers.models.Normalize",
        "sentence_transformers.sentence_transformer.modules.normalize.Normalize",
        "sentence_transformers.base.modules.normalize.Normalize",
    }
)
DENSE = frozenset(
    {"sentence_transformers.models.Dense", "sentence_transformers.base.modules.dense.Dense"}
)


class Files(Protocol):
    """The files of a model, each read once into bytes: ``read`` checks them (a regular
    file within ``limit`` bytes; on the Hub against its commit, ``lfs`` for files that
    must be LFS files there; ``inspect(handle, size)`` may check a prefix before the rest
    is read), ``has`` says whether the model has a file, ``where`` names it in errors, and
    ``source`` and ``revision`` say where the model came from."""

    source: str | None
    revision: str | None

    def has(self, name: str) -> bool: ...

    def where(self, name: str) -> Path: ...

    def read(
        self, name: str, *, lfs: bool = False, limit: int | None = None, inspect: Any = None
    ) -> bytes: ...


@dataclass(frozen=True, eq=False)
class Converted:
    """The parts of the encoder of a converted model; ``dims`` is ``(width of the
    model's table, width kept)`` when its table was cut to its first dimensions."""

    word_vectors: WordVectors
    tokenizer: TokenizerSpec
    normalize: bool
    dims: tuple[int, int] | None = None


def kept_dims(truncate_dim: Any, stored: int | None, width: int, where: Any) -> int | None:
    """The dimensions kept of a table ``width`` columns wide, or None for all of them:
    ``truncate_dim``, or with the model's rule the ``stored`` one (None: all)."""
    kept = stored if truncate_dim == MODEL_RULE else truncate_dim
    if kept is None or kept == width:
        return None
    if kept > width:
        source = f"{where} states" if truncate_dim == MODEL_RULE else "the argument is"
        raise ValueError(f"truncate_dim {kept}: {source} more than the {width} dimensions")
    return kept


def cut_table(table: ndarray, kept: int | None) -> ndarray:
    """The first ``kept`` columns of the table as a read-only copy; the table itself for
    None. The columns that are dropped must be finite too, as the whole table is when it is
    not cut, so a corrupt table is never accepted by a cut."""
    if kept is None:
        return table
    for begin in range(0, len(table), _BLOCK_ROWS):
        dropped = table[begin : begin + _BLOCK_ROWS, kept:]
        bad = np.flatnonzero(~np.isfinite(dropped).all(axis=1))
        if len(bad):
            raise ValueError(
                f"row {begin + int(bad[0])} of vectors contains NaN or infinite values "
                f"(in the columns that truncate_dim {kept} drops)"
            )
    table = np.ascontiguousarray(table[:, :kept])
    table.setflags(write=False)
    return table


def parse_config(data: bytes, path: Path) -> Any:
    """The parsed JSON config (strict JSON of at most MAX_CONFIG_BYTES)."""
    if len(data) > MAX_CONFIG_BYTES:
        raise CorruptArtifactError(f"{path} has more than {MAX_CONFIG_BYTES} bytes")
    return strict_json(data, path)


def read_config(files: Files, name: str) -> Any:
    """The parsed JSON config ``name`` of ``files``."""
    return parse_config(files.read(name, limit=MAX_CONFIG_BYTES), files.where(name))


def read_table(
    files: Files, name: str, names: frozenset[str], n_tokens: int, check: Any = None
) -> ndarray:
    """The one tensor of the safetensors file ``name``: a float32 matrix named one of
    ``names`` with one row per token. Its header is checked before the data are read
    (``check(path, dtype, shape)`` may check more first), so a file that holds other
    tensors costs no memory."""
    path = files.where(name)

    def inspect(handle, size: int) -> None:
        _check_header(path, foreign_header(handle, path, size)[1], names, n_tokens, check)

    data = files.read(name, lfs=True, inspect=inspect)
    return next(iter(foreign_tensors(data, path).values()))


def _check_header(path: Path, entries: dict, names: frozenset[str], n_tokens: int, check: Any):
    if "weights" in entries and "mapping" not in entries:
        raise UnsupportedArtifactError(
            f"{path} holds separate token weights (the tensor weights without a mapping), which "
            "this fse does not read; a table with the weights multiplied in, as Model2Vec's "
            "to_pipeline and to_static_model write it, is read"
        )
    quantized = sorted(QUANTIZED & set(entries))
    if quantized:
        raise UnsupportedArtifactError(
            f"{path} holds {quantized}, the tensors of vocabulary quantization, which this "
            "fse does not read"
        )
    if len(entries) != 1 or not set(entries) <= names:
        what = " or ".join(sorted(names))
        raise CorruptArtifactError(
            f"{path} must hold exactly one tensor, {what}, not {sorted(entries)}"
            if len(names) > 1
            else f"{path} must hold exactly the tensor {what}, not {sorted(entries)}"
        )
    code, shape, _, _ = next(iter(entries.values()))
    check_table(path, _DTYPE_OF_CODE[code], shape, n_tokens, check)


def check_table(path: Path, dtype: str, shape: tuple[int, ...], n_tokens: int, check: Any) -> None:
    """The table of ``dtype`` and ``shape`` is a float32 matrix with one row per token
    (``check(path, dtype, shape)`` may check more first); anything else raises."""
    if len(shape) != 2:
        raise CorruptArtifactError(f"{path}: the table must be a matrix, not of shape {shape}")
    if check is not None:
        check(path, dtype, shape)
    if dtype != "float32":
        raise UnsupportedArtifactError(
            f"{path} has a {dtype} table; this fse reads float32 tables (the original "
            "library computes others in their own dtype, so their vectors would differ)"
        )
    if shape[0] != n_tokens:
        raise CorruptArtifactError(
            f"{path} has {shape[0]} rows, but the tokenizer has {n_tokens} tokens"
        )
