#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""A BERT ``tokenizer.json`` of Hugging Face tokenizers as the parameters of wordpiece@1.

ARCHITECTURE.md, "Model2Vec models". Only what wordpiece@1 reproduces is accepted, and
everything else is refused by name: a WordPiece model with a vocabulary of dense ids, a
BertNormalizer, a BertPreTokenizer, added tokens matched on the raw text (``normalized``,
``lstrip``, ``rstrip`` and ``single_word`` false) that are tokens of the vocabulary with
the same id, and a post-processor that is null or a TemplateProcessing whose single template holds
the sequence A once (tokenizers 0.23.3 runs it also without special tokens; it then drops
the special tokens but repeats the sequences, so only such a template changes no id).
``truncation`` and ``padding`` are ignored, as model2vec overrides both, and ``decoder``,
as nothing decodes. The file is strict JSON of at most 64 MiB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fse._container import CorruptArtifactError, read_regular, strict_json
from fse._contract import UnsupportedArtifactError
from fse._tokenizer_spec import WordPieceConfig

MAX_BYTES = 64 * 2**20
_KEYS = {"version", "truncation", "padding", "added_tokens", "normalizer"}
_KEYS |= {"pre_tokenizer", "post_processor", "decoder", "model"}
_MODEL_KEYS = {"type", "unk_token", "continuing_subword_prefix", "max_input_chars_per_word"}
_MODEL_KEYS |= {"vocab"}
_NORMALIZER_KEYS = {"type", "clean_text", "handle_chinese_chars", "strip_accents", "lowercase"}
_ADDED_KEYS = {"id", "content", "single_word", "lstrip", "rstrip", "normalized", "special"}


def read_tokenizer_json(path: Path) -> tuple[list[str], WordPieceConfig]:
    """The tokens in id order and the parameters of wordpiece@1 of a tokenizer.json."""
    return tokenizer_of(read_regular(Path(path), limit=MAX_BYTES), Path(path))


def tokenizer_of(data: bytes, path: Path) -> tuple[list[str], WordPieceConfig]:
    """The same of the bytes of a tokenizer.json (``path`` names it in errors)."""
    if len(data) > MAX_BYTES:
        raise CorruptArtifactError(f"{path} has more than {MAX_BYTES} bytes")
    return parse_tokenizer(strict_json(data, path), path)


def parse_tokenizer(value: Any, path: Path) -> tuple[list[str], WordPieceConfig]:
    """The tokens and parameters of the parsed JSON of a tokenizer.json at ``path``."""
    _object(value, path, _KEYS, "the file")
    model = _object(value["model"], path, _MODEL_KEYS, "model", "type")
    if model["type"] != "WordPiece":
        raise UnsupportedArtifactError(
            f"{path} has a {model['type']} model; this fse reads WordPiece (wordpiece@1)"
        )
    _object(model, path, _MODEL_KEYS, "model")
    tokens = _tokens(model["vocab"], path)
    unk = model["unk_token"]
    if unk not in model["vocab"]:
        raise CorruptArtifactError(f"{path}: the unknown token {unk!r} is not in the vocabulary")
    flags = _normalizer(value["normalizer"], path)
    _pre_tokenizer(value["pre_tokenizer"], path)
    _post_processor(value["post_processor"], path)
    added = _added_tokens(value["added_tokens"], model["vocab"], path)
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


def _object(value: Any, path: Path, keys: set[str], what: str, *required: str) -> dict:
    """A JSON object with exactly ``keys`` (or, with ``required``, at least those)."""
    if not isinstance(value, dict):
        raise CorruptArtifactError(f"{path}: {what} must be a JSON object")
    need = set(required) or keys
    if not need <= set(value) or (not required and set(value) - keys):
        missing, extra = sorted(need - set(value)), sorted(set(value) - keys)
        raise UnsupportedArtifactError(
            f"{path}: {what} must hold exactly {sorted(keys)}; it lacks {missing} and holds {extra}"
        )
    return value


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
    _object(value, path, _NORMALIZER_KEYS, "normalizer")
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


def _post_processor(value: Any, path: Path) -> None:
    """Null, or a TemplateProcessing whose template of a single text holds the sequence A
    once: tokenizers 0.23.3 runs it also without special tokens, drops the special tokens
    then and keeps the sequences, so only then are the ids unchanged ($A $A doubles them)."""
    if value is None:
        return
    if not isinstance(value, dict) or value.get("type") != "TemplateProcessing":
        kind = value.get("type") if isinstance(value, dict) else value
        raise UnsupportedArtifactError(
            f"{path} has the post-processor {kind!r}; this fse reads null or "
            "TemplateProcessing, which tokenizers runs without special tokens as no change"
        )
    single = value.get("single")
    pieces = single if isinstance(single, list) else []
    sequences = [p.get("Sequence") for p in pieces if isinstance(p, dict) and "Sequence" in p]
    others = [
        p
        for p in pieces
        if not (isinstance(p, dict) and set(p) <= {"Sequence", "SpecialToken"} and len(p) == 1)
    ]
    if (
        others
        or len(sequences) != 1
        or not isinstance(sequences[0], dict)
        or sequences[0].get("id") != "A"
    ):
        raise UnsupportedArtifactError(
            f"{path} has a TemplateProcessing whose single template is {single!r}; without "
            "special tokens only a template with the sequence A once leaves the ids unchanged"
        )


def _added_tokens(value: Any, vocab: dict, path: Path) -> tuple[tuple[int, str], ...]:
    """The added tokens, matched on the raw text, as tokens of the vocabulary."""
    if not isinstance(value, list):
        raise CorruptArtifactError(f"{path}: added_tokens must be a list")
    out = []
    for token in value:
        _object(token, path, _ADDED_KEYS, "an added token")
        content = token["content"]
        flags = [token[k] for k in ("single_word", "lstrip", "rstrip", "normalized")]
        if any(flag is not False for flag in flags):
            raise UnsupportedArtifactError(
                f"{path}: the added token {content!r} is normalized, stripped or single-word; "
                "wordpiece@1 matches added tokens on the raw text as they are"
            )
        if not isinstance(content, str) or vocab.get(content) != token["id"]:
            raise CorruptArtifactError(
                f"{path}: the added token {content!r} is not token {token['id']!r} of the vocabulary"
            )
        out.append((token["id"], content))
    return tuple(sorted(out))
