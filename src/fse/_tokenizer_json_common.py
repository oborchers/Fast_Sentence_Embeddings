#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""What the readers of a Hugging Face ``tokenizer.json`` share (wordpiece@1, unigram@1).

The keys of the file and its size limit, JSON objects with exactly their keys, the added
tokens matched on the raw text, the post-processors that change no id without special
tokens, and the plain truncation (docs/architecture/hub.md, "Model2Vec models and from_pretrained").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError

MAX_BYTES = 64 * 2**20
FILE_KEYS = {"version", "truncation", "padding", "added_tokens", "normalizer"}
FILE_KEYS |= {"pre_tokenizer", "post_processor", "decoder", "model"}
ADDED_KEYS = {"id", "content", "single_word", "lstrip", "rstrip", "normalized", "special"}


_PLAIN = {"direction": "Right", "stride": 0, "strategy": "LongestFirst"}


def truncation(value: Any, path: Path) -> int | None:
    """The token limit of a truncation of tokenizers: null is none; the plain form keeps
    the first ``max_length`` ids of a text, as step 6 of wordpiece@1 does."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"max_length", *_PLAIN}:
        raise CorruptArtifactError(
            f"{path}: truncation must be null or hold max_length, {', '.join(_PLAIN)}"
        )
    length = value["max_length"]
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise CorruptArtifactError(
            f"{path}: the max_length of truncation must be a positive int, not {length!r}"
        )
    stride = value["stride"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 0:
        raise CorruptArtifactError(
            f"{path}: the stride of truncation must be an int of at least 0, not {stride!r}"
        )
    other = {k: value[k] for k, v in _PLAIN.items() if value[k] != v}
    if other:
        raise UnsupportedArtifactError(
            f"{path} has a truncation with {other}; this fse reproduces only the plain one "
            f"({_PLAIN}), which keeps the first max_length ids"
        )
    return length


def json_object(value: Any, path: Path, keys: set[str], what: str, *required: str) -> dict:
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


def post_processor(value: Any, path: Path) -> None:
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


def added_tokens(value: Any, vocab: dict, path: Path) -> tuple[tuple[int, str], ...]:
    """The added tokens, matched on the raw text, as tokens of the vocabulary."""
    if not isinstance(value, list):
        raise CorruptArtifactError(f"{path}: added_tokens must be a list")
    out = []
    for token in value:
        json_object(token, path, ADDED_KEYS, "an added token")
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
