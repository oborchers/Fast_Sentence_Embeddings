#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The tokenizer part of encoder 1 (docs/architecture/formats.md, "encoder 1"): the keys of a tokenizer
with parameters, written from and parsed into a TokenizerSpec, and the table it needs.
Moved from fse._encoder._format, which uses it.
"""

from __future__ import annotations

import json

import numpy as np
from numpy import ndarray

from fse._container import INTEGER, canonical_json
from fse._contract import check_tokenizer
from fse._tokenizer_spec import UNIGRAM, WORDPIECE, TokenizerSpec, WordPieceConfig
from fse._tokenizer_unigram import UnigramConfig

# The keys of a tokenizer with parameters (wordpiece@1, unigram@1).
TOKENIZER_KEYS = ("fse.tokenizer_config", "fse.unknown", "fse.max_tokens", "fse.max_chars")
# The tensors of a tokenizer with data beyond its vocabulary (unigram@1): the float64
# scores of the pieces, and the charsmap of a precompiled step, verbatim.
SCORES, CHARSMAP = "tokenizer.scores", "tokenizer.charsmap"
TOKENIZER_TENSORS = frozenset({SCORES, CHARSMAP})


def tokenizer_tensors(spec: TokenizerSpec) -> dict[str, ndarray]:
    """The tensors of the tokenizer (none but for unigram@1)."""
    config = spec.config
    if not isinstance(config, UnigramConfig):
        return {}
    out = {SCORES: config.scores_array()}
    if config.charsmap is not None:
        out[CHARSMAP] = np.frombuffer(config.charsmap, dtype=np.uint8)
    return out


def _unigram_data(tensors: dict[str, ndarray]) -> tuple[bytes, bytes | None]:
    """The scores (F64, one per piece) and the charsmap (U8) of the tensors."""
    scores, charsmap = tensors.get(SCORES), tensors.get(CHARSMAP)
    if scores is None or scores.dtype != np.dtype("<f8") or scores.ndim != 1:
        raise ValueError(f"an encoder of {UNIGRAM} needs the F64 vector {SCORES}")
    if charsmap is not None and (charsmap.dtype != np.uint8 or charsmap.ndim != 1):
        raise ValueError(f"{CHARSMAP} must be a U8 vector")
    return scores.tobytes(), None if charsmap is None else charsmap.tobytes()


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
    fasttext: bool,
    vocab_bytes: ndarray | None,
    vocab_offsets: ndarray | None,
) -> None:
    """The table a tokenizer with parameters needs: rows of any dtype without FastText,
    whose vocabulary holds every id of its parameters. A float16 or int8 table is a
    quantized one (word_vectors 1 requires ``fse.quantization`` of it)."""
    if tokenizer.config is None:
        return
    if fasttext:
        raise ValueError(f"an encoder of {tokenizer.name} has no FastText table")
    if vocab_bytes is None or vocab_offsets is None:
        raise ValueError("an encoder needs the tensors vocab.bytes and vocab.offsets")
    tokenizer.check_vocabulary(vocab_bytes, vocab_offsets)


def tokenizer_of(metadata: dict[str, str], tensors: dict[str, ndarray]) -> TokenizerSpec:
    """The tokenizer of ``fse.tokenizer`` with the keys of its parameters, which exist
    exactly for a tokenizer that has them, and its tensors (exactly those it defines)."""
    name = check_tokenizer(metadata["fse.tokenizer"])
    present = [k for k in TOKENIZER_KEYS if k in metadata]
    if name not in (WORDPIECE, UNIGRAM):
        if present:
            raise ValueError(f"{present} are keys of a tokenizer with parameters, not of {name}")
        held = sorted(TOKENIZER_TENSORS & set(tensors))
        if held:
            raise ValueError(f"{held} are tensors of unigram@1, not of the tokenizer {name}")
        return TokenizerSpec(name)
    for key in TOKENIZER_KEYS[:2]:
        if key not in metadata:
            raise ValueError(f"an encoder of {name} needs the metadata key {key}")
    value = json_object(metadata, "fse.tokenizer_config")
    config: WordPieceConfig | UnigramConfig
    if name == UNIGRAM:
        config = UnigramConfig.from_json(value, *_unigram_data(tensors))
    else:
        config = WordPieceConfig.from_json(value)
    held = sorted(TOKENIZER_TENSORS & set(tensors))
    spec_tensors = sorted(tokenizer_tensors(TokenizerSpec(name, config, metadata["fse.unknown"])))
    if held != spec_tensors:
        raise ValueError(f"an encoder of {name} holds the tensors {spec_tensors}, not {held}")
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
