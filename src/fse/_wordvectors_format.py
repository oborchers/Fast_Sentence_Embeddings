#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The format rules of word_vectors 1 (ARCHITECTURE.md, "word_vectors 1").

Its keys and tensors, the checks of structure (on every open) and of values (on verify
and every write), the vocabulary encoding and the FastText parameters. fse._wordvectors
builds WordVectors on them.
"""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy import ndarray

from fse._container import (
    CREATED_BY_KEY,
    INTEGER,
    canonical_json,
    tensor_code,
)
from fse._contract import OOV_TOKEN, UnsupportedArtifactError

KIND = "word_vectors"
FORMAT_VERSION = "1"
QUANTIZATION = "quantize_rows@1"
NGRAM_HASH = "fasttext@1"

_KEYS = {
    "fse.kind",
    "fse.format_version",
    CREATED_BY_KEY,
    "fse.dim",
    "fse.dtype",
    "fse.quantization",
    "fse.quantized_from",
    "fse.fasttext",
    "fse.ngram_hash",
}
_TENSORS = {"vectors", "scales", "counts", "vocab.bytes", "vocab.offsets", "ngrams", "ngram_scales"}
_DTYPE_OF = {"F32": "float32", "F16": "float16", "I8": "int8"}
_BLOCK_ROWS = 16_384
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_WORDS = OOV_TOKEN  # word ids must stay below the reserved token ids


class FastTextParams(NamedTuple):
    """How unknown words are hashed into ngram buckets (the FastText scheme)."""

    min_n: int
    max_n: int
    bucket: int


def _encode_words(words: Sequence[str]) -> tuple[ndarray, ndarray]:
    """The UTF-8 bytes and offsets of ``words``, without a list of bytes objects."""
    for i, word in enumerate(words):
        if not isinstance(word, str):
            raise TypeError(f"word {i} must be a str, not {type(word).__name__}")
    try:
        lengths = np.fromiter(
            (len(w.encode("utf-8")) for w in words), dtype=np.int64, count=len(words)
        )
    except UnicodeEncodeError:
        bad = next(i for i, w in enumerate(words) if not _has_utf8(w))
        raise ValueError(f"word {bad} ({words[bad]!r}) has no UTF-8 form") from None
    offsets = np.zeros(len(words) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    # Every word encodes on its own, so their concatenation encodes to the joined bytes.
    data = np.frombuffer("".join(words).encode("utf-8"), dtype=np.uint8)
    return data, offsets


def _has_utf8(word: str) -> bool:
    try:
        word.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _integer(metadata: dict[str, str], key: str, error=UnsupportedArtifactError) -> int:
    value = metadata.get(key)
    if value is None or not INTEGER.fullmatch(value):
        raise error(f"{key} must be a non-negative decimal integer, not {value!r}")
    return int(value)


def _check_scales(tensors: dict[str, ndarray], name: str, rows: int, needed: bool):
    if needed != (name in tensors):
        raise ValueError(f"{name} are given exactly for int8 rows")
    if needed:
        _check_vector(tensors[name], name, "F32", rows)


def _check_vector(array: ndarray, name: str, code: str, length: int):
    if tensor_code(array) != code or array.shape != (length,):
        raise ValueError(
            f"{name} must be a vector of {length} {code} values, not {array.dtype.str} {array.shape}"
        )


def _decreases(offsets: ndarray) -> bool:
    """Whether ``offsets`` ever decrease, compared in blocks to bound the temporaries."""
    step = 2**20
    for begin in range(0, len(offsets) - 1, step):
        block = np.asarray(offsets[begin : begin + step + 1])
        if (block[1:] < block[:-1]).any():
            return True
    return False


def _check_utf8(data: ndarray, offsets: ndarray, chunk_bytes: int = 2**24):
    """Check that the vocabulary is UTF-8 and every word starts at a character boundary.

    An incremental decoder takes fixed blocks of bytes, whatever the words, so memory
    stays bounded also for a single huge word.
    """
    step = 2**20
    for begin in range(1, len(offsets) - 1, step):
        inner = np.asarray(offsets[begin : min(begin + step, len(offsets) - 1)])
        inner = inner[inner < len(data)]
        if len(inner) and ((np.asarray(data[inner]) & 0xC0) == 0x80).any():
            raise ValueError("a word of the vocabulary starts inside a UTF-8 character")
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    try:
        for begin in range(0, len(data), chunk_bytes):
            decoder.decode(data[begin : begin + chunk_bytes].tobytes())
        decoder.decode(b"", final=True)
    except UnicodeDecodeError as err:
        raise ValueError(f"the vocabulary is not valid UTF-8: {err}") from None


def _parse_fasttext(text: str) -> FastTextParams:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"fse.fasttext is not JSON: {text!r}") from None
    if not isinstance(value, dict) or set(value) != {"bucket", "max_n", "min_n"}:
        raise ValueError("fse.fasttext must hold exactly bucket, max_n and min_n")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value.values()):
        raise ValueError("fse.fasttext must hold integers")
    if canonical_json(value) != text:
        raise ValueError(f"fse.fasttext must be canonical JSON, not {text!r}")
    params = FastTextParams(value["min_n"], value["max_n"], value["bucket"])
    _check_fasttext(params)
    return params


def _check_fasttext(params: FastTextParams):
    if not 1 <= params.min_n <= params.max_n:
        raise ValueError(
            f"FastText needs 1 <= min_n <= max_n, not {params.min_n} and {params.max_n}"
        )
    if not 1 <= params.bucket <= 2**32 - 1:
        raise ValueError(f"FastText needs between 1 and 2**32 - 1 buckets, not {params.bucket}")


def _lookup(data: ndarray, offsets: ndarray) -> dict[bytes, int]:
    raw = data.tobytes()
    bounds = offsets.tolist()
    index: dict[bytes, int] = {}
    for i in range(len(bounds) - 1):
        word = raw[bounds[i] : bounds[i + 1]]
        if index.setdefault(word, i) != i:
            raise ValueError(f"word {i} ({word.decode('utf-8')!r}) is also word {index[word]}")
    return index
