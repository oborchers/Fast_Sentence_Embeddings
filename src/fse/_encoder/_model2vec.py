#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Model2Vec models as the parts of an encoder (docs/architecture/hub.md, "Model2Vec models and from_pretrained").

A Model2Vec model is ``config.json``, ``model.safetensors`` with the tensor ``embeddings``
and a ``tokenizer.json``. Its encoder pools the rows of the token ids with
``weighted_mean@1``, the tokenizer is ``wordpiece@1`` (a WordPiece tokenizer.json) or
``unigram@1`` (a Unigram one, such as that of potion-multilingual-128M) with the unknown id
dropped, and the length rule is model2vec 0.10.0's (``model.py``): ``max_length`` tokens (512 unless the
config says otherwise, null for none) of the first ``max_length`` times the median token
length characters. ``normalize`` comes from the config (default false). Only float32
tables: model2vec computes the mean of a float16 table in float16, so the bound of the
comparison would not hold. Every key of the config is known: the ones used are checked
against the table, provenance is accepted and ignored, vocabulary quantization and heads
are refused, and an unknown key raises, so a new key that changes encoding cannot be
ignored silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fse import _log
from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._tokenizer_json import MAX_BYTES as MAX_TOKENIZER_BYTES
from fse._tokenizer_json import tokenizer_of
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors

from ._files import Converted, Files, read_table

FILES = ("config.json", "model.safetensors", "tokenizer.json")
DEFAULT_MAX_LENGTH = 512
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


def read_model2vec(
    files: Files, config: Any, *, truncate: bool = True, max_length: Any = MODEL_RULE
) -> Converted:
    """The encoder parts of the files of a Model2Vec model (``FILES``) with its parsed
    ``config``. With ``truncate`` the length rule is the model's, or ``max_length`` tokens
    (None: no cut); without, texts are read whole."""
    config = check_config(config, files.where("config.json"))
    where = files.where("tokenizer.json")
    tokens, parameters = tokenizer_of(
        files.read("tokenizer.json", limit=MAX_TOKENIZER_BYTES), where
    )
    table = model_table(files, config, len(tokens))
    # The table is a read-only view of the bytes that were read once and checked, which
    # nothing else holds: handed over without a copy, with every check of from_arrays.
    words = WordVectors._from_arrays(table, tokens, None, None, None, {}, copy=False)
    length = _length(config, truncate, max_length)
    max_chars = None if length is None else length * median_length(tokens)
    spec = TokenizerSpec(parameters.identifier, parameters, "drop", length, max_chars)
    return Converted(words, spec, config.get("normalize", False))


def check_config(config: Any, path: Path) -> dict[str, Any]:
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
    ignored = sorted(_PROVENANCE & set(config))  # fse's own names of these keys
    if ignored:
        _log.emit(
            "artifact.ignored",
            "fse artifact: ignored the provenance keys %s of a Model2Vec config",
            ignored,
            source="model2vec",
            keys=ignored,
        )
    length = config.get("max_length", DEFAULT_MAX_LENGTH)
    if length is not None and (
        isinstance(length, bool) or not isinstance(length, int) or length < 1
    ):
        raise CorruptArtifactError(
            f"{path}: max_length must be a positive int or null, not {length!r}"
        )
    return config


def model_table(files: Files, config: dict[str, Any], n_tokens: int):
    """The tensor embeddings: float32 rows, one per token, as the config states them."""

    def check(path, dtype: str, shape: tuple[int, ...]) -> None:
        declared = config.get("embedding_dtype")
        if declared is not None and _DTYPES.get(declared) != dtype:
            raise CorruptArtifactError(
                f"{path}: embedding_dtype is {declared!r}, the table {dtype}"
            )
        if config.get("hidden_dim", shape[1]) != shape[1]:
            raise CorruptArtifactError(
                f"{path}: hidden_dim is {config['hidden_dim']!r}, the table has {shape[1]} columns"
            )

    return read_table(files, "model.safetensors", frozenset({"embeddings"}), n_tokens, check)


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
