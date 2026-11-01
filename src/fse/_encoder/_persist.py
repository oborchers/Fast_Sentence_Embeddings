#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""An encoder as an artifact and back (docs/architecture/formats.md, "encoder 1").

The tensors, metadata and expected checksums of an encoder, the parts of an encoder
artifact that was opened, the content of an encoder converted from another library, and
the fingerprint of an artifact. ``fse._encoder._encoder.Encoder`` holds the state; these
functions take its parts, so the encoder module stays the place of its API.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy import ndarray

from fse import _log
from fse._container import CHECKSUM_PREFIX, CorruptArtifactError, checksum, fingerprint, tensor_code
from fse._contract import Normalize, UnsupportedArtifactError
from fse._runtime import EncodeSpec
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors

from ._format import (
    TABLE_KEYS,
    TABLE_TENSORS,
    EncoderContent,
    Source,
    check_tokenizer_table,
    from_artifact,
    loaded_table,
    to_artifact,
)


def spec(content: EncoderContent, word_vectors: WordVectors) -> EncodeSpec:
    """What the pipelines of an encoder encode with: a finished table, so every word
    weighs 1."""
    return EncodeSpec(
        word_vectors,
        np.ones(len(word_vectors), dtype=np.float32),
        content.steps,
        content.tokenizer,
        oov_weight=None if content.oov_weight is None else content.oov_weight[()],
        max_ngrams=content.max_ngrams,
        aggregation=content.aggregation,
    )


def artifact(
    content: EncoderContent,
    word_vectors: WordVectors,
    checksums: dict[str, str] | None,
    oov_weight: float | None,
) -> tuple[dict[str, ndarray], dict[str, str], dict[str, str] | None]:
    """The tensors and metadata of an encoder, and the checksums its data must match
    (those it was loaded with), or None for data made in memory."""
    table, keys, loaded = word_vectors._parts()
    expected = dict(checksums or {})
    expected.update({k: v for k, v in (loaded or {}).items() if k in table})
    tensors, metadata = to_artifact(
        content, table, {k: v for k, v in keys.items() if k in TABLE_KEYS}
    )
    if oov_weight is not None:
        # The copy the pipelines encode with, so saving and verifying cannot pass a
        # value the encoder does not use.
        tensors["oov_weight"] = np.array(oov_weight, dtype=np.float32)
    return tensors, metadata, expected or None


def opened(
    tensors: dict[str, ndarray], metadata: dict[str, str], path: Path | None
) -> tuple[EncoderContent, WordVectors, dict[str, str]]:
    """The content, the table and the recorded checksums of an encoder artifact opened
    from ``path`` (also the encoder part of a head file)."""
    content, table, keys = from_artifact(tensors, metadata)
    checksums = {
        k[len(CHECKSUM_PREFIX) :]: v
        for k, v in metadata.items()
        if k.startswith(CHECKSUM_PREFIX) and k[len(CHECKSUM_PREFIX) :] in tensors
    }
    return content, loaded_table(table, keys, metadata, checksums, path), checksums


def converted(
    word_vectors: WordVectors,
    tokenizer: TokenizerSpec,
    normalize: bool,
    source: Source | None,
    license: str | tuple[str, ...] | None,
) -> EncoderContent:
    """The content of an encoder converted from another library (the readers of PR 12a2):
    its tokenizer with parameters, whether it normalizes, its provenance."""
    if not isinstance(word_vectors, WordVectors):
        raise TypeError(f"word_vectors must be WordVectors, not {type(word_vectors).__name__}")
    if not isinstance(tokenizer, TokenizerSpec):
        raise TypeError(f"tokenizer must be a TokenizerSpec, not {type(tokenizer).__name__}")
    if not isinstance(normalize, bool):
        raise TypeError(f"normalize must be a bool, not {type(normalize).__name__}")
    check_tokenizer_table(
        tokenizer,
        word_vectors.fasttext is not None,
        word_vectors.vocab_bytes,
        word_vectors.vocab_offsets,
    )
    _log.emit(
        "artifact.converted",
        "fse artifact: converted a %s model to an encoder: %d rows of %d dimensions",
        None if source is None else source.format,
        len(word_vectors),
        word_vectors.dim,
        kind="encoder",
        source=None if source is None else source.format,
        rows=len(word_vectors),
        dim=word_vectors.dim,
        tokenizer=tokenizer.name,
        normalize=normalize,
    )
    return EncoderContent(
        tokenizer=tokenizer,
        steps=(Normalize(),) if normalize else (),
        oov_weight=None,
        max_ngrams=None,
        source=source,
        license=license,
    )


def check_held(
    tensors: dict[str, ndarray], checksums: dict[str, str] | None, path: Path | None
) -> None:
    """Compare the tensors an encoder holds besides its table (the tokenizer's, the
    weight of unknown words) with the checksums it was loaded with."""
    known = checksums or {}
    for name, array in tensors.items():
        if name in known and name not in TABLE_TENSORS and checksum(array) != known[name]:
            raise CorruptArtifactError(
                f"tensor {name} of {path or 'this encoder'} does not match its checksum"
            )


def normalize_steps(steps: tuple[Any, ...], normalize: Any) -> tuple[Any, ...]:
    """The steps of an encoder loaded with ``normalize``: ``"model"`` keeps them, True
    ends them with ``normalize@1`` (the same tuple if they end with it), False drops a
    final ``normalize@1`` (the same tuple if there is none). A ``normalize@1`` before
    another step cannot be dropped without changing what that step receives, and raises."""
    if normalize == "model":
        return steps
    last = max((i for i, step in enumerate(steps) if isinstance(step, Normalize)), default=None)
    if normalize:
        return steps if last == len(steps) - 1 else (*steps, Normalize())
    if last is None:
        return steps
    if last != len(steps) - 1:
        raise UnsupportedArtifactError(
            "the model has other steps after its normalize@1, which normalize=False cannot "
            "drop without changing what they receive"
        )
    return steps[:-1]


def with_normalize(encoder: Any, normalize: Any) -> Any:
    """``encoder`` with its final ``normalize@1`` added or dropped (``"model"``: itself),
    sharing its table; its fingerprint is that of the result. The tensors besides the
    table keep the checksums they were loaded with."""
    steps = normalize_steps(encoder._content.steps, normalize)
    if steps is encoder._content.steps:
        return encoder
    return type(encoder)._new(
        replace(encoder._content, steps=steps),
        encoder._word_vectors,
        path=encoder._path,
        checksums=encoder._checksums,
        vocabulary=encoder._vocabulary,
    )


def quantized(
    content: EncoderContent,
    word_vectors: WordVectors,
    checksums: dict[str, str] | None,
    path: Path | None,
    oov_weight: float | None,
    dtype: Literal["float16", "int8"],
) -> tuple[EncoderContent, WordVectors]:
    """The content and the table of an encoder with a float16 or int8 copy of its float32
    table (``WordVectors.quantize``, which hashes the copy it quantizes). The tensors held
    besides the table are checked against the load checksums first; the weight of unknown
    words is the verified value the encoder uses, never the mapped array it was loaded as."""
    tensors, _, _ = artifact(content, word_vectors, checksums, oov_weight)
    check_held(tensors, checksums, path)
    table = word_vectors.quantize(dtype)
    check_tokenizer_table(
        content.tokenizer,
        table.fasttext is not None,
        table.vocab_bytes,
        table.vocab_offsets,
    )
    weight = None
    if oov_weight is not None:
        weight = np.array(oov_weight, dtype=np.float32)
        weight.setflags(write=False)
    return replace(content, oov_weight=weight), table


def fingerprint_of(
    tensors: dict[str, ndarray], metadata: dict[str, str], expected: dict[str, str] | None
) -> str:
    """The fingerprint of an artifact, with the checksums it is known to match."""
    known = expected or {}
    entries = {
        k: (tensor_code(v), v.shape, known.get(k) or checksum(v)) for k, v in tensors.items()
    }
    return fingerprint(entries, metadata)
