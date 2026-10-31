#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""A Unigram ``tokenizer.json`` of Hugging Face tokenizers as the parameters of unigram@1.

docs/architecture/tokenizers.md, "unigram@1". Only the forms of potion-multilingual-128M are accepted, and
everything else is refused by name: a Unigram model with unique pieces, float scores, an
unknown id and no byte fallback; a normalizer of Precompiled (at most one), Replace with a
String pattern or with exactly the regex `` {2,}`` or ``\\s+``, and Strip, in Sequences
nested in any way (a Sequence applies its normalizers in order, so nesting changes
nothing); Metaspace with the replacement U+2581, prepend scheme ``always`` and no split;
added tokens and the post-processor as for WordPiece. ``padding``, ``truncation`` and
``decoder`` are ignored, as for WordPiece in ``tokenizer_of``. The file is strict JSON of
at most 64 MiB.
"""

from __future__ import annotations

import base64
import binascii
import math
from pathlib import Path
from typing import Any

import numpy as np

from fse._container import CorruptArtifactError, strict_json
from fse._contract import UnsupportedArtifactError
from fse._tokenizer_json_common import (
    FILE_KEYS,
    MAX_BYTES,
    added_tokens,
    json_object,
    post_processor,
)
from fse._tokenizer_unigram import METASPACE, UnigramConfig, step_of

_MODEL_KEYS = {"type", "unk_id", "vocab", "byte_fallback"}
# The two regexes of Replace that unigram@1 reproduces, as runs of a class.
_RUNS = {
    " {2,}": {"type": "runs", "class": "space", "min": 2},
    "\\s+": {"type": "runs", "class": "white_space", "min": 1},
}
_METASPACE_KEYS = {"type", "replacement", "prepend_scheme", "split", "add_prefix_space"}
_METASPACE_KEYS |= {"str_rep"}  # written by old tokenizers, read and ignored by 0.23.3


def unigram_of(data: bytes, path: Path) -> tuple[list[str], UnigramConfig]:
    """The pieces in id order and the parameters of unigram@1 of the bytes of a
    tokenizer.json (``path`` names it in errors)."""
    if len(data) > MAX_BYTES:
        raise CorruptArtifactError(f"{path} has more than {MAX_BYTES} bytes")
    return parse_unigram(strict_json(data, path), path)


def parse_unigram(value: Any, path: Path) -> tuple[list[str], UnigramConfig]:
    """The pieces and parameters of the parsed JSON of a tokenizer.json at ``path``."""
    json_object(value, path, FILE_KEYS, "the file")
    model = json_object(value["model"], path, _MODEL_KEYS, "model", "type", "unk_id", "vocab")
    if model["type"] != "Unigram":
        raise UnsupportedArtifactError(
            f"{path} has a {model['type']} model; unigram@1 reads a Unigram model"
        )
    _known(model, _MODEL_KEYS, "model", path)
    pieces, scores = _vocab(model["vocab"], path)
    unk_id = model["unk_id"]
    if unk_id is None:
        raise UnsupportedArtifactError(f"{path}: unigram@1 needs an unknown id, not null")
    if isinstance(unk_id, bool) or not isinstance(unk_id, int) or not 0 <= unk_id < len(pieces):
        raise CorruptArtifactError(
            f"{path}: unk_id {unk_id!r} is no id of the {len(pieces)} pieces"
        )
    fallback = model.get("byte_fallback", False)
    added = added_tokens(value["added_tokens"], {p: i for i, p in enumerate(pieces)}, path)
    steps, charsmap = _normalizer(value["normalizer"], path)
    _metaspace(value["pre_tokenizer"], path)
    post_processor(value["post_processor"], path)
    try:
        config = UnigramConfig(added, unk_id, tuple(steps), scores, charsmap, fallback, METASPACE)
    except UnsupportedArtifactError as error:
        raise UnsupportedArtifactError(f"{path}: {error}") from None
    except ValueError as error:
        raise CorruptArtifactError(f"{path}: {error}") from None
    return pieces, config


def _known(value: dict, keys: set[str], what: str, path: Path) -> None:
    """``value`` holds no key beyond ``keys``."""
    extra = sorted(set(value) - keys)
    if extra:
        raise UnsupportedArtifactError(f"{path}: {what} holds keys this fse does not know: {extra}")


def _vocab(vocab: Any, path: Path) -> tuple[list[str], bytes]:
    """The pieces and their scores as float64 bytes; pieces unique and non-empty, scores
    numbers (an int is its float, as serde reads it), finite."""
    if not isinstance(vocab, list) or not vocab:
        raise CorruptArtifactError(f"{path}: the vocabulary must be a non-empty list")
    pieces, scores = [], []
    for entry in vocab:
        if not isinstance(entry, list) or len(entry) != 2:
            raise CorruptArtifactError(f"{path}: a piece is a [str, score] pair, not {entry!r}")
        piece, score = entry
        if not isinstance(piece, str) or not piece:
            raise CorruptArtifactError(f"{path}: a piece must be a non-empty str, not {piece!r}")
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            raise CorruptArtifactError(f"{path}: the score of {piece!r} is {score!r}, no number")
        pieces.append(piece)
        scores.append(float(score))
    if len(set(pieces)) != len(pieces):
        raise CorruptArtifactError(f"{path}: a piece is in the vocabulary twice")
    return pieces, np.array(scores, dtype="<f8").tobytes()


def _flatten(value: Any, path: Path) -> list[dict]:
    """The normalizers of a normalizer in the order they apply. null is allowed only for
    the normalizer of the file (``_normalizer``), not inside a Sequence, as in tokenizers."""
    if not isinstance(value, dict) or "type" not in value:
        raise CorruptArtifactError(f"{path}: a normalizer must be a JSON object with a type")
    if value["type"] != "Sequence":
        return [value]
    json_object(value, path, {"type", "normalizers"}, "a Sequence normalizer")
    if not isinstance(value["normalizers"], list):
        raise CorruptArtifactError(f"{path}: the normalizers of a Sequence must be a list")
    return [step for inner in value["normalizers"] for step in _flatten(inner, path)]


def _normalizer(value: Any, path: Path) -> tuple[list[tuple], bytes | None]:
    """The steps of unigram@1 and the charsmap of its precompiled step."""
    steps, charsmap = [], None
    for normalizer in [] if value is None else _flatten(value, path):
        kind = normalizer["type"]
        if kind == "Precompiled":
            if charsmap is not None:
                raise UnsupportedArtifactError(f"{path}: unigram@1 takes one Precompiled")
            charsmap = _charsmap(normalizer, path)
            steps.append(step_of({"type": "precompiled"}))
        elif kind == "Replace":
            steps.append(step_of(_replace(normalizer, path)))
        elif kind == "Strip":
            json_object(normalizer, path, {"type", "strip_left", "strip_right"}, "Strip")
            left, right = normalizer["strip_left"], normalizer["strip_right"]
            steps.append(step_of({"type": "strip", "left": left, "right": right}))
        else:
            raise UnsupportedArtifactError(
                f"{path} has the normalizer {kind!r}; unigram@1 reads Precompiled, Replace "
                "and Strip"
            )
    return steps, charsmap


def _charsmap(normalizer: dict, path: Path) -> bytes:
    json_object(normalizer, path, {"type", "precompiled_charsmap"}, "Precompiled")
    text = normalizer["precompiled_charsmap"]
    if not isinstance(text, str):
        raise CorruptArtifactError(f"{path}: precompiled_charsmap must be a base64 str")
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise CorruptArtifactError(f"{path}: precompiled_charsmap is no base64: {error}") from None


def _replace(normalizer: dict, path: Path) -> dict:
    json_object(normalizer, path, {"type", "pattern", "content"}, "Replace")
    pattern, content = normalizer["pattern"], normalizer["content"]
    if (
        not isinstance(pattern, dict)
        or len(pattern) != 1
        or not set(pattern) <= {"String", "Regex"}
    ):
        raise CorruptArtifactError(
            f"{path}: the pattern of Replace is String or Regex, not {pattern!r}"
        )
    if "String" in pattern:
        if pattern["String"] == "":
            raise UnsupportedArtifactError(f"{path}: a Replace of the empty string")
        return {"type": "replace", "pattern": pattern["String"], "content": content}
    regex = pattern["Regex"]
    if regex not in _RUNS:
        raise UnsupportedArtifactError(
            f"{path} has a Replace of the regex {regex!r}; unigram@1 reads only "
            f"{' and '.join(map(repr, _RUNS))}"
        )
    return {**_RUNS[regex], "content": content}


# The fields of Metaspace that tokenizers 0.23.3 reads as Option<bool> or Option<String>
# (null is the default), and the prepend schemes it knows.
_OPTIONAL_TYPES = {"split": bool, "add_prefix_space": bool, "str_rep": str}
_SCHEMES = ("first", "never", "always")


def _types(value: dict, path: Path) -> None:
    """The types tokenizers requires of the fields of Metaspace; it refuses others."""
    for key, kind in _OPTIONAL_TYPES.items():
        field = value.get(key)
        if field is not None and not isinstance(field, kind):
            raise CorruptArtifactError(
                f"{path}: {key} of Metaspace must be a {kind.__name__} or null, not {field!r}"
            )
    if "prepend_scheme" in value and value["prepend_scheme"] not in _SCHEMES:
        raise CorruptArtifactError(
            f"{path}: the prepend scheme of Metaspace is one of {_SCHEMES}, not "
            f"{value['prepend_scheme']!r}"
        )


def _metaspace(value: Any, path: Path) -> None:
    """Metaspace with the values of METASPACE; missing keys take the defaults of
    tokenizers 0.23.3 (prepend scheme always, split true)."""
    if not isinstance(value, dict) or value.get("type") != "Metaspace":
        kind = value.get("type") if isinstance(value, dict) else value
        raise UnsupportedArtifactError(
            f"{path} has the pre-tokenizer {kind!r}; unigram@1 is a Metaspace"
        )
    json_object(value, path, _METASPACE_KEYS, "Metaspace", "type", "replacement")
    _known(value, _METASPACE_KEYS, "Metaspace", path)
    replacement = value["replacement"]
    if not isinstance(replacement, str) or len(replacement) != 1:
        raise CorruptArtifactError(f"{path}: the replacement of Metaspace is one character")
    _types(value, path)
    scheme = value.get("prepend_scheme", "always")
    split = True if value.get("split") is None else value["split"]
    if value.get("add_prefix_space") is False and scheme != "never":
        raise CorruptArtifactError(
            f"{path}: Metaspace has add_prefix_space false but the prepend scheme {scheme!r}"
        )
    found = {"prepend_scheme": scheme, "replacement": replacement, "split": split}
    if found != dict(METASPACE):
        raise UnsupportedArtifactError(
            f"{path} has a Metaspace of {found}; unigram@1 supports {dict(METASPACE)}"
        )
