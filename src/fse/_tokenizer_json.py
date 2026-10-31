#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""A BERT ``tokenizer.json`` of Hugging Face tokenizers as the parameters of wordpiece@1.

docs/architecture/hub.md, "Model2Vec models and from_pretrained". Only what wordpiece@1 reproduces is accepted, and
everything else is refused by name: a WordPiece model with a vocabulary of dense ids, a
BertNormalizer, a BertPreTokenizer, added tokens matched on the raw text (``normalized``,
``lstrip``, ``rstrip`` and ``single_word`` false) that are tokens of the vocabulary with
the same id, and a post-processor that is null or a TemplateProcessing whose single template holds
the sequence A once (tokenizers 0.23.3 runs it also without special tokens; it then drops
the special tokens but repeats the sequences, so only such a template changes no id).
``padding`` is ignored, as model2vec and sentence-transformers disable it, and ``decoder``,
as nothing decodes. ``truncation`` is ignored by ``tokenizer_of`` (model2vec overrides
it); ``tokenizer_and_truncation`` gives it as a token limit (sentence-transformers keeps
it), only in its plain form: the first ``max_length`` ids (direction Right, stride 0,
strategy LongestFirst), any other form raises. The file is strict JSON of at most 64 MiB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fse._container import CorruptArtifactError, read_regular, strict_json
from fse._contract import UnsupportedArtifactError
from fse._tokenizer_json_common import (
    FILE_KEYS,
    MAX_BYTES,
    added_tokens,
    json_object,
    post_processor,
    truncation,
)
from fse._tokenizer_json_unigram import parse_unigram
from fse._tokenizer_spec import WordPieceConfig
from fse._tokenizer_unigram import UnigramConfig

_MODEL_KEYS = {"type", "unk_token", "continuing_subword_prefix", "max_input_chars_per_word"}
_MODEL_KEYS |= {"vocab"}
_NORMALIZER_KEYS = {"type", "clean_text", "handle_chinese_chars", "strip_accents", "lowercase"}


def read_tokenizer_json(path: Path) -> tuple[list[str], WordPieceConfig | UnigramConfig]:
    """The tokens in id order and the parameters of wordpiece@1 or unigram@1 of a
    tokenizer.json."""
    return tokenizer_of(read_regular(Path(path), limit=MAX_BYTES), Path(path))


def tokenizer_of(data: bytes, path: Path) -> tuple[list[str], WordPieceConfig | UnigramConfig]:
    """The same of the bytes of a tokenizer.json (``path`` names it in errors): a Unigram
    model gives the parameters of unigram@1 (fse._tokenizer_json_unigram), any other those
    of wordpiece@1, or the reason it is refused."""
    if len(data) > MAX_BYTES:
        raise CorruptArtifactError(f"{path} has more than {MAX_BYTES} bytes")
    value = strict_json(data, path)
    model = value.get("model") if isinstance(value, dict) else None
    if isinstance(model, dict) and model.get("type") == "Unigram":
        return parse_unigram(value, path)
    return parse_tokenizer(value, path)


def tokenizer_and_truncation(
    data: bytes, path: Path
) -> tuple[list[str], WordPieceConfig, int | None]:
    """The tokens, the parameters and the token limit of the plain truncation (None
    without one) of the bytes of a tokenizer.json."""
    if len(data) > MAX_BYTES:
        raise CorruptArtifactError(f"{path} has more than {MAX_BYTES} bytes")
    value = strict_json(data, path)
    tokens, config = parse_tokenizer(value, path)
    return tokens, config, truncation(value["truncation"], path)


def parse_tokenizer(value: Any, path: Path) -> tuple[list[str], WordPieceConfig]:
    """The tokens and parameters of the parsed JSON of a tokenizer.json at ``path``."""
    json_object(value, path, FILE_KEYS, "the file")
    model = json_object(value["model"], path, _MODEL_KEYS, "model", "type")
    if model["type"] != "WordPiece":
        raise UnsupportedArtifactError(
            f"{path} has a {model['type']} model; this fse reads WordPiece (wordpiece@1)"
        )
    json_object(model, path, _MODEL_KEYS, "model")
    tokens = _tokens(model["vocab"], path)
    unk = model["unk_token"]
    if unk not in model["vocab"]:
        raise CorruptArtifactError(f"{path}: the unknown token {unk!r} is not in the vocabulary")
    flags = _normalizer(value["normalizer"], path)
    _pre_tokenizer(value["pre_tokenizer"], path)
    post_processor(value["post_processor"], path)
    added = added_tokens(value["added_tokens"], model["vocab"], path)
    try:
        config = WordPieceConfig(
            added,
            model["vocab"][unk],
            model["continuing_subword_prefix"],
            model["max_input_chars_per_word"],
            **flags,
        )
    except UnsupportedArtifactError as error:
        raise UnsupportedArtifactError(f"{path}: {error}") from None
    except ValueError as error:
        raise CorruptArtifactError(f"{path}: {error}") from None
    return tokens, config


def _tokens(vocab: Any, path: Path) -> list[str]:
    """The tokens in id order; the ids run densely from 0."""
    if not isinstance(vocab, dict) or not vocab:
        raise CorruptArtifactError(f"{path}: the vocabulary must be a non-empty JSON object")
    tokens: list[str | None] = [None] * len(vocab)
    for token, i in vocab.items():
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(vocab):
            raise CorruptArtifactError(
                f"{path}: the id of {token!r} is {i!r}, not an int from 0 to {len(vocab) - 1}"
            )
        if tokens[i] is not None:
            raise CorruptArtifactError(f"{path}: the id {i} belongs to two tokens")
        tokens[i] = token
    return [t for t in tokens if t is not None]


def _normalizer(value: Any, path: Path) -> dict[str, bool]:
    if not isinstance(value, dict) or value.get("type") != "BertNormalizer":
        kind = value.get("type") if isinstance(value, dict) else value
        raise UnsupportedArtifactError(
            f"{path} has the normalizer {kind!r}; wordpiece@1 is a BertNormalizer"
        )
    json_object(value, path, _NORMALIZER_KEYS, "normalizer")
    strip = value["strip_accents"]
    # tokenizers strips accents when strip_accents is null and lowercase is on.
    flags = {
        "clean_text": value["clean_text"],
        "handle_chinese_chars": value["handle_chinese_chars"],
        "strip_accents": value["lowercase"] if strip is None else strip,
        "lowercase": value["lowercase"],
    }
    for name, flag in flags.items():
        if not isinstance(flag, bool):
            raise CorruptArtifactError(f"{path}: the normalizer's {name} must be a bool")
    return flags


def _pre_tokenizer(value: Any, path: Path) -> None:
    if value != {"type": "BertPreTokenizer"}:
        raise UnsupportedArtifactError(
            f"{path} has the pre-tokenizer {value!r}; wordpiece@1 is a BertPreTokenizer"
        )
