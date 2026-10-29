#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Model2Vec models as the parts of an encoder (ARCHITECTURE.md, "Model2Vec models").

A Model2Vec model is ``config.json``, ``model.safetensors`` with the tensor ``embeddings``
and a BERT ``tokenizer.json``. Its encoder pools the rows of the token ids with
``weighted_mean@1``, the tokenizer is ``wordpiece@1`` with the unknown id dropped, and the
length rule is model2vec 0.10.0's (``model.py``): ``max_length`` tokens (512 unless the
config says otherwise, null for none) of the first ``max_length`` times the median token
length characters. ``normalize`` comes from the config (default false). Only float32
tables: model2vec computes the mean of a float16 table in float16, so the bound of the
comparison would not hold. Every key of the config is known: the ones used are checked
against the table, provenance is accepted and ignored, vocabulary quantization and heads
are refused, and an unknown key raises, so a new key that changes encoding cannot be
ignored silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numpy import ndarray

from fse._container import CorruptArtifactError, read_foreign, strict_json
from fse._contract import UnsupportedArtifactError
from fse._tokenizer_json import read_tokenizer_json
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors

FILES = ("config.json", "model.safetensors", "tokenizer.json")
DEFAULT_MAX_LENGTH = 512
MAX_CONFIG_BYTES = 2**20
# The length rule of the model itself, which max_length= of from_pretrained keeps.
MODEL_RULE = "model"
# Written by model2vec and its distillation; they do not change how a text is encoded.
_PROVENANCE = {"model_type", "architectures", "tokenizer_name", "apply_pca", "apply_zipf"}
_PROVENANCE |= {"sif_coefficient", "seq_length", "pooling"}
_REFUSED = {
    "vocabulary_quantization": "vocabulary quantization (a map of tokens to rows)",
    "head_config": "a classification head",
}
_DTYPES = {"float32": "float32", "float16": "float16", "int8": "int8", "float64": "float64"}


@dataclass(frozen=True, eq=False)
class Model2Vec:
    """The parts of the encoder of a Model2Vec model."""

    word_vectors: WordVectors
    tokenizer: TokenizerSpec
    normalize: bool


def read_model2vec(
    files: dict[str, Path], *, truncate: bool = True, max_length: Any = MODEL_RULE
) -> Model2Vec:
    """The encoder parts of the files of a Model2Vec model (``FILES``, each a regular
    file). With ``truncate`` the length rule is the model's, or ``max_length`` tokens
    (None: no cut); without, texts are read whole."""
    config = _config(files["config.json"])
    tokens, parameters = read_tokenizer_json(files["tokenizer.json"])
    table = _table(files["model.safetensors"], config, len(tokens))
    words = WordVectors.from_arrays(table, words=tokens)
    length = _length(config, truncate, max_length)
    max_chars = None if length is None else length * median_length(tokens)
    spec = TokenizerSpec("wordpiece@1", parameters, "drop", length, max_chars)
    return Model2Vec(words, spec, config.get("normalize", False))


def _config(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise CorruptArtifactError(f"{path} has more than {MAX_CONFIG_BYTES} bytes")
    config = strict_json(path.read_bytes(), path)
    if not isinstance(config, dict):
        raise CorruptArtifactError(f"{path} must be a JSON object")
    for key, what in _REFUSED.items():
        if key in config:
            raise UnsupportedArtifactError(
                f"{path}: the model has {what} ({key}), which this fse does not read"
            )
    unknown = (
        set(config) - _PROVENANCE - {"normalize", "max_length", "hidden_dim", "embedding_dtype"}
    )
    if unknown:
        raise UnsupportedArtifactError(
            f"{path} holds keys this fse does not know: {sorted(unknown)}; they could change "
            "how the model encodes"
        )
    if not isinstance(config.get("normalize", False), bool):
        raise CorruptArtifactError(f"{path}: normalize must be true or false")
    length = config.get("max_length", DEFAULT_MAX_LENGTH)
    if length is not None and (
        isinstance(length, bool) or not isinstance(length, int) or length < 1
    ):
        raise CorruptArtifactError(
            f"{path}: max_length must be a positive int or null, not {length!r}"
        )
    return config


def _table(path: Path, config: dict[str, Any], n_tokens: int) -> ndarray:
    """The tensor embeddings: float32 rows, one per token."""
    tensors = read_foreign(path)
    quantized = sorted({"weights", "mapping"} & set(tensors))
    if quantized:
        raise UnsupportedArtifactError(
            f"{path} holds {quantized}, the tensors of vocabulary quantization, which this "
            "fse does not read"
        )
    if set(tensors) != {"embeddings"}:
        raise CorruptArtifactError(
            f"{path} must hold exactly the tensor embeddings, not {sorted(tensors)}"
        )
    table = tensors["embeddings"]
    if table.ndim != 2:
        raise CorruptArtifactError(
            f"{path}: embeddings must be a matrix, not of shape {table.shape}"
        )
    declared = config.get("embedding_dtype")
    if declared is not None and _DTYPES.get(declared) != table.dtype.name:
        raise CorruptArtifactError(
            f"{path}: embedding_dtype is {declared!r}, the table {table.dtype}"
        )
    if table.dtype.name != "float32":
        raise UnsupportedArtifactError(
            f"{path} has a {table.dtype} table; this fse reads float32 Model2Vec models "
            "(model2vec computes others in their own dtype, so their vectors would differ)"
        )
    if config.get("hidden_dim", table.shape[1]) != table.shape[1]:
        raise CorruptArtifactError(
            f"{path}: hidden_dim is {config['hidden_dim']!r}, the table has {table.shape[1]} columns"
        )
    if table.shape[0] != n_tokens:
        raise CorruptArtifactError(
            f"{path} has {table.shape[0]} rows, but the tokenizer has {n_tokens} tokens"
        )
    return table


def _length(config: dict[str, Any], truncate: bool, max_length: Any) -> int | None:
    """The token cut: the model's (512 by default), ``max_length``, or none."""
    if not truncate:
        if max_length != MODEL_RULE:
            raise ValueError("max_length is a length rule; it needs truncate=True")
        return None
    if max_length == MODEL_RULE:
        return config.get("max_length", DEFAULT_MAX_LENGTH)
    return max_length


def median_length(tokens: list[str]) -> int:
    """``int(np.median(len(t) for t in tokens))`` as model2vec computes it, in integers."""
    lengths = sorted(len(t) for t in tokens)
    mid = len(lengths) // 2
    median = lengths[mid] if len(lengths) % 2 else (lengths[mid - 1] + lengths[mid]) // 2
    if median < 1:
        raise CorruptArtifactError("the median length of the tokens is 0")
    return median
