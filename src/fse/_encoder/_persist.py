#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""An encoder as an artifact and back (ARCHITECTURE.md, "encoder 1").

The tensors, metadata and expected checksums of an encoder, the parts of an encoder
artifact that was opened, the content of an encoder converted from another library, and
the fingerprint of an artifact. ``fse._encoder._encoder.Encoder`` holds the state; these
functions take its parts, so the encoder module stays the place of its API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy import ndarray

from fse import _log
from fse._container import CHECKSUM_PREFIX, checksum, fingerprint, tensor_code
from fse._contract import Normalize
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors

from ._format import (
    TABLE_KEYS,
    EncoderContent,
    Source,
    check_tokenizer_table,
    from_artifact,
    loaded_table,
    to_artifact,
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
        word_vectors.dtype,
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


def fingerprint_of(
    tensors: dict[str, ndarray], metadata: dict[str, str], expected: dict[str, str] | None
) -> str:
    """The fingerprint of an artifact, with the checksums it is known to match."""
    known = expected or {}
    entries = {
        k: (tensor_code(v), v.shape, known.get(k) or checksum(v)) for k, v in tensors.items()
    }
    return fingerprint(entries, metadata)
