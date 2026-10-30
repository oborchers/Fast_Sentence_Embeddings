#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The tokenizer part of encoder 1 (ARCHITECTURE.md, "encoder 1"): the keys of a tokenizer
with parameters, written from and parsed into a TokenizerSpec, and the table it needs.
Moved from fse._encoder._format, which uses it.
"""

from __future__ import annotations

import json

from numpy import ndarray

from fse._container import INTEGER, canonical_json
from fse._contract import UnsupportedArtifactError, check_tokenizer
from fse._tokenizer_spec import WORDPIECE, TokenizerSpec, WordPieceConfig

# The keys of a tokenizer with parameters (wordpiece@1).
TOKENIZER_KEYS = ("fse.tokenizer_config", "fse.unknown", "fse.max_tokens", "fse.max_chars")


def tokenizer_metadata(spec: TokenizerSpec) -> dict[str, str]:
    if spec.config is None:
        return {}
    out = {"fse.tokenizer_config": canonical_json(spec.config.to_json())}
    out["fse.unknown"] = str(spec.unknown)
    if spec.max_tokens is not None:
        out["fse.max_tokens"] = str(spec.max_tokens)
    if spec.max_chars is not None:
        out["fse.max_chars"] = str(spec.max_chars)
    return out


def integer(metadata: dict[str, str], key: str) -> int:
    value = metadata.get(key)
    if value is None or not INTEGER.fullmatch(value):
        raise ValueError(f"{key} must be a non-negative decimal integer, not {value!r}")
    return int(value)


def check_tokenizer_table(
    tokenizer: TokenizerSpec,
    dtype: str | None,
    fasttext: bool,
    vocab_bytes: ndarray | None,
    vocab_offsets: ndarray | None,
) -> None:
    """The table a tokenizer with parameters needs: float32 rows without FastText, whose
    vocabulary holds every id of its parameters. Model2vec computes the mean of a float16
    table in float16, so only float32 tables keep the bound of a comparison for now."""
    if tokenizer.config is None:
        return
    if fasttext:
        raise ValueError(f"an encoder of {tokenizer.name} has no FastText table")
    if dtype != "float32":
        raise UnsupportedArtifactError(
            f"encoders of {tokenizer.name} with a {dtype} table are not supported by this "
            "fse; their tables are float32"
        )
    if vocab_bytes is None or vocab_offsets is None:
        raise ValueError("an encoder needs the tensors vocab.bytes and vocab.offsets")
    tokenizer.check_vocabulary(vocab_bytes, vocab_offsets)


def tokenizer_of(metadata: dict[str, str]) -> TokenizerSpec:
    """The tokenizer of ``fse.tokenizer`` with the keys of its parameters, which exist
    exactly for a tokenizer that has them."""
    name = check_tokenizer(metadata["fse.tokenizer"])
    present = [k for k in TOKENIZER_KEYS if k in metadata]
    if name != WORDPIECE:
        if present:
            raise ValueError(f"{present} are keys of wordpiece@1, not of the tokenizer {name}")
        return TokenizerSpec(name)
    for key in TOKENIZER_KEYS[:2]:
        if key not in metadata:
            raise ValueError(f"an encoder of {name} needs the metadata key {key}")
    config = WordPieceConfig.from_json(json_object(metadata, "fse.tokenizer_config"))
    return TokenizerSpec(
        name,
        config,
        metadata["fse.unknown"],
        integer(metadata, "fse.max_tokens") if "fse.max_tokens" in metadata else None,
        integer(metadata, "fse.max_chars") if "fse.max_chars" in metadata else None,
    )


def json_object(metadata: dict[str, str], key: str) -> dict:
    """The canonical JSON object of a key (no duplicate keys, NaN or whitespace)."""
    text = metadata[key]
    try:
        value = json.loads(text)
        canonical = canonical_json(value)
    except ValueError as error:  # malformed JSON, or NaN, which canonical JSON refuses
        raise ValueError(f"{key} is not canonical JSON: {error}") from None
    if not isinstance(value, dict) or canonical != text:
        raise ValueError(f"{key} must be a canonical JSON object")
    return value
