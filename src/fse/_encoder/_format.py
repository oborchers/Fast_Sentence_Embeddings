#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The format rules of encoder 1 (ARCHITECTURE.md, "encoder 1").

An encoder is a finished model in one file: its table (the tensors and keys of
word_vectors 1), the tokenizer, the aggregation, whether every row is normalized and,
for FastText, ``max_ngrams`` and ``oov_weight``. ``to_artifact`` turns it into tensors
and metadata, ``from_artifact`` checks and parses them back. The keys and tensors are a
closed set: anything else raises, and what a pre-release of 2.0 wrote (recipes, word
weights, learned or folded steps, an encoder without its table) is refused by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from fse._container import CHECKSUM_PREFIX, CREATED_BY_KEY, FINGERPRINT_KEY, INTEGER, canonical_json
from fse._contract import (
    TRANSFORM_VERSION,
    Aggregation,
    Normalize,
    Op,
    Step,
    UnsupportedArtifactError,
    check_steps,
    check_tokenizer,
    check_transform_version,
    prerelease_error,
)

KIND = "encoder"
FORMAT_VERSION = "1"
# The keys and tensors of the table an encoder holds, as in word_vectors 1.
TABLE_KEYS = frozenset(
    {
        "fse.dim",
        "fse.dtype",
        "fse.quantization",
        "fse.quantized_from",
        "fse.fasttext",
        "fse.ngram_hash",
    }
)
TABLE_TENSORS = frozenset(
    {"vectors", "scales", "vocab.bytes", "vocab.offsets", "ngrams", "ngram_scales"}
)
_NEEDED = (
    "fse.transform_version",
    "fse.tokenizer",
    "fse.aggregation",
    "fse.steps",
    "fse.output_dim",
    "fse.dim",
)
_KEYS = frozenset(
    {"fse.kind", "fse.format_version", CREATED_BY_KEY, "fse.max_ngrams", *_NEEDED, *TABLE_KEYS}
)
# What encoders of a pre-release of 2.0 held and 2.0 removed: provenance of fit, the
# fingerprints of a table they referenced, word weights, and learned or folded steps.
_REMOVED_KEYS = frozenset(
    {
        "fse.recipe",
        "fse.folded",
        "fse.folded_from",
        "fse.input_dim",
        "fse.word_vectors",
        "fse.vocabulary",
    }
)
_REMOVED_PREFIXES = ("step.", "folded.")


@dataclass(frozen=True, eq=False)
class EncoderContent:
    """What an encoder is apart from its table: the tokenizer, the steps (none or
    normalize), and for FastText ``max_ngrams`` and ``oov_weight`` (a float32 scalar)."""

    tokenizer: str
    steps: tuple[Step, ...]
    oov_weight: ndarray | None
    max_ngrams: int | None
    aggregation: Aggregation = Aggregation.WEIGHTED_MEAN


def to_artifact(
    content: EncoderContent, table: dict[str, ndarray], keys: dict[str, str]
) -> tuple[dict[str, ndarray], dict[str, str]]:
    """The tensors and metadata of an encoder with the tensors and keys (``TABLE_KEYS``)
    of its table."""
    tensors = dict(table)
    if content.oov_weight is not None:
        tensors["oov_weight"] = content.oov_weight
    dim = int(keys["fse.dim"])
    metadata = {
        **keys,
        "fse.kind": KIND,
        "fse.format_version": FORMAT_VERSION,
        "fse.transform_version": str(TRANSFORM_VERSION),
        "fse.tokenizer": content.tokenizer,
        "fse.aggregation": content.aggregation.id,
        "fse.steps": canonical_json([s.op.id for s in content.steps]),
        "fse.output_dim": str(content.aggregation.output_dim(dim)),
    }
    if content.max_ngrams is not None:
        metadata["fse.max_ngrams"] = str(content.max_ngrams)
    return tensors, metadata


def from_artifact(
    tensors: dict[str, ndarray], metadata: dict[str, str]
) -> tuple[EncoderContent, dict[str, ndarray], dict[str, str]]:
    """Check and parse an encoder; returns its content and the tensors and keys of its
    table (whose own checks are those of word_vectors 1). Unsupported versions, kinds,
    keys and identifiers, and artifacts of a pre-release, raise UnsupportedArtifactError,
    anything else malformed ValueError."""
    _check_keys(tensors, metadata)
    check_transform_version(_integer(metadata, "fse.transform_version"))
    aggregation = Aggregation.parse(metadata["fse.aggregation"])
    dim = _integer(metadata, "fse.dim")
    output = aggregation.output_dim(dim)  # raises for fse.dim 0
    if str(output) != metadata["fse.output_dim"]:
        raise ValueError(
            f"fse.output_dim is {metadata['fse.output_dim']!r}, but rows of fse.dim {dim} "
            f"aggregate to {output}"
        )
    content = EncoderContent(
        tokenizer=check_tokenizer(metadata["fse.tokenizer"]),
        steps=_steps(metadata, output),
        oov_weight=_oov_weight(tensors, metadata),
        max_ngrams=_integer(metadata, "fse.max_ngrams") if "fse.max_ngrams" in metadata else None,
        aggregation=aggregation,
    )
    extra = set(tensors) - TABLE_TENSORS - {"oov_weight"}
    if extra:
        raise ValueError(f"tensors the encoder does not use: {sorted(extra)}")
    table = {k: v for k, v in tensors.items() if k in TABLE_TENSORS}
    return content, table, {k: v for k, v in metadata.items() if k in TABLE_KEYS}


def _check_keys(tensors: dict[str, ndarray], metadata: dict[str, str]):
    kind = metadata.get("fse.kind")
    if kind != KIND:
        raise UnsupportedArtifactError(f"the artifact is a {kind!r}, not {KIND!r}")
    if metadata.get("fse.format_version") != FORMAT_VERSION:
        raise UnsupportedArtifactError(
            f"encoders of format version {metadata.get('fse.format_version')!r} are not "
            f"supported by this fse; supported: {FORMAT_VERSION}"
        )
    removed = sorted(
        {k for k in metadata if k in _REMOVED_KEYS}
        | {k for k in tensors if k == "weights" or k.startswith(_REMOVED_PREFIXES)}
    )
    if removed or "vectors" not in tensors:
        raise prerelease_error("encoder", removed or ["no table of its own"])
    unknown = {
        k
        for k in metadata
        if k not in _KEYS and k != FINGERPRINT_KEY and not k.startswith(CHECKSUM_PREFIX)
    }
    if unknown:
        raise UnsupportedArtifactError(f"unknown metadata keys of an encoder: {sorted(unknown)}")
    for key in _NEEDED:
        if key not in metadata:
            raise ValueError(f"an encoder needs the metadata key {key}")


def _integer(metadata: dict[str, str], key: str) -> int:
    value = metadata.get(key)
    if value is None or not INTEGER.fullmatch(value):
        raise ValueError(f"{key} must be a non-negative decimal integer, not {value!r}")
    return int(value)


def _steps(metadata: dict[str, str], dim: int) -> tuple[Step, ...]:
    """The steps of ``fse.steps``: a canonical JSON list, [] or ["normalize@1"]."""
    text = metadata["fse.steps"]
    try:
        ids = json.loads(text)
    except json.JSONDecodeError:
        ids = None
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError("fse.steps must be a JSON list of step identifiers")
    if canonical_json(ids) != text:
        raise ValueError("fse.steps must be canonical JSON")
    ops = [Op.parse(i) for i in ids]  # an unknown or removed step raises here
    return check_steps([Normalize() for _ in ops], dim)  # normalize@1 is the only step


def _oov_weight(tensors: dict[str, ndarray], metadata: dict[str, str]) -> ndarray | None:
    """The weight of unknown FastText words: given exactly with fse.max_ngrams."""
    value = tensors.get("oov_weight")
    if (value is None) != ("fse.max_ngrams" not in metadata):
        raise ValueError("oov_weight and fse.max_ngrams are given exactly for FastText vectors")
    if value is None:
        return None
    if value.dtype != np.float32 or value.shape != () or not np.isfinite(value):
        raise ValueError("oov_weight must be one finite float32 value")
    return value
