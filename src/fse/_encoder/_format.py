#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The format rules of encoder 1 (ARCHITECTURE.md, "encoder 1").

An encoder is a finished model in one file: its table (the tensors and keys of
word_vectors 1), the tokenizer (with its parameters for ``wordpiece@1``), the aggregation,
whether every row is normalized, for FastText ``max_ngrams`` and ``oov_weight``, and the
provenance of a converted model. ``to_artifact`` turns it into tensors
and metadata, ``from_artifact`` checks and parses them back. The keys and tensors are a
closed set: anything else raises, and what a pre-release of 2.0 wrote (recipes, word
weights, learned or folded steps, an encoder without its table) is refused by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import ndarray

from fse._backends import get_backend
from fse._container import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    FINGERPRINT_KEY,
    INTEGER,
    canonical_json,
    fingerprint,
    tensor_code,
)
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
from fse._text import MAX_NGRAMS_LIMIT
from fse._tokenizer_spec import WORDPIECE, TokenizerSpec, WordPieceConfig
from fse._wordvectors import WordVectors
from fse._wordvectors_format import FORMAT_VERSION as WORD_VECTORS_VERSION

# The ngrams an unknown FastText word uses at most, unless an encoder is given another.
DEFAULT_MAX_NGRAMS = 40
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
# The keys of a tokenizer with parameters (wordpiece@1), and the provenance of a model.
TOKENIZER_KEYS = ("fse.tokenizer_config", "fse.unknown", "fse.max_tokens", "fse.max_chars")
PROVENANCE_KEYS = ("fse.source", "fse.license")
_KEYS = frozenset(
    {
        "fse.kind",
        "fse.format_version",
        CREATED_BY_KEY,
        "fse.max_ngrams",
        *_NEEDED,
        *TABLE_KEYS,
        *TOKENIZER_KEYS,
        *PROVENANCE_KEYS,
    }
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


@dataclass(frozen=True)
class Source:
    """The model a converted encoder came from (``fse.source``), never read to compute:
    its ``format`` (such as ``model2vec``), and its Hub ``repo`` and ``revision``."""

    format: str
    repo: str | None = None
    revision: str | None = None

    def __post_init__(self):
        if not isinstance(self.format, str) or not self.format:
            raise ValueError(f"the format of a source must be a non-empty str, not {self.format!r}")
        for name in ("repo", "revision"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"the {name} of a source must be None or a non-empty str")


@dataclass(frozen=True, eq=False)
class EncoderContent:
    """What an encoder is apart from its table: the tokenizer (a TokenizerSpec; a name
    stands for one without parameters), the steps (none or normalize), for FastText
    ``max_ngrams`` and ``oov_weight`` (a float32 scalar), and the provenance of a
    converted model (``source``, ``license``)."""

    tokenizer: TokenizerSpec
    steps: tuple[Step, ...]
    oov_weight: ndarray | None
    max_ngrams: int | None
    aggregation: Aggregation = Aggregation.WEIGHTED_MEAN
    source: Source | None = None
    license: str | tuple[str, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "tokenizer", TokenizerSpec.of(self.tokenizer))
        if self.source is not None and not isinstance(self.source, Source):
            raise TypeError(f"source must be a Source, not {type(self.source).__name__}")
        _check_license(self.license)


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
        "fse.tokenizer": content.tokenizer.name,
        "fse.aggregation": content.aggregation.id,
        "fse.steps": canonical_json([s.op.id for s in content.steps]),
        "fse.output_dim": str(content.aggregation.output_dim(dim)),
    }
    if content.max_ngrams is not None:
        metadata["fse.max_ngrams"] = str(content.max_ngrams)
    metadata.update(_tokenizer_metadata(content.tokenizer))
    if content.source is not None:
        source = content.source
        metadata["fse.source"] = canonical_json(
            {"format": source.format, "repo": source.repo, "revision": source.revision}
        )
    if content.license is not None:
        license_ = content.license
        metadata["fse.license"] = (
            license_ if isinstance(license_, str) else canonical_json(list(license_))
        )
    return tensors, metadata


def _tokenizer_metadata(spec: TokenizerSpec) -> dict[str, str]:
    if spec.config is None:
        return {}
    out = {"fse.tokenizer_config": canonical_json(spec.config.to_json())}
    out["fse.unknown"] = str(spec.unknown)
    if spec.max_tokens is not None:
        out["fse.max_tokens"] = str(spec.max_tokens)
    if spec.max_chars is not None:
        out["fse.max_chars"] = str(spec.max_chars)
    return out


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
        tokenizer=_tokenizer(metadata),
        steps=_steps(metadata, output),
        oov_weight=_oov_weight(tensors, metadata),
        max_ngrams=_integer(metadata, "fse.max_ngrams") if "fse.max_ngrams" in metadata else None,
        aggregation=aggregation,
        source=_source(metadata),
        license=_license(metadata),
    )
    check_tokenizer_table(
        content.tokenizer,
        metadata.get("fse.dtype"),
        "fse.fasttext" in metadata,
        tensors.get("vocab.bytes"),
        tensors.get("vocab.offsets"),
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


def _tokenizer(metadata: dict[str, str]) -> TokenizerSpec:
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
    config = WordPieceConfig.from_json(_json_object(metadata, "fse.tokenizer_config"))
    return TokenizerSpec(
        name,
        config,
        metadata["fse.unknown"],
        _integer(metadata, "fse.max_tokens") if "fse.max_tokens" in metadata else None,
        _integer(metadata, "fse.max_chars") if "fse.max_chars" in metadata else None,
    )


def _json_object(metadata: dict[str, str], key: str) -> dict:
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


def _source(metadata: dict[str, str]) -> Source | None:
    if "fse.source" not in metadata:
        return None
    value = _json_object(metadata, "fse.source")
    if set(value) != {"format", "repo", "revision"}:
        raise ValueError("fse.source holds exactly format, repo and revision")
    return Source(value["format"], value["repo"], value["revision"])


def loaded_table(
    tensors: dict[str, ndarray],
    keys: dict[str, str],
    metadata: dict[str, str],
    checksums: dict[str, str],
    path: Path | None,
) -> WordVectors:
    """The table an encoder holds, as word vectors loaded from its file, with the
    fingerprint of their recorded checksums."""
    content = {"fse.kind": "word_vectors", "fse.format_version": WORD_VECTORS_VERSION, **keys}
    entries = {k: (tensor_code(v), v.shape, checksums[k]) for k, v in tensors.items()}
    loaded = {
        **content,
        CREATED_BY_KEY: metadata.get(CREATED_BY_KEY, "fse"),
        FINGERPRINT_KEY: fingerprint(entries, content),
        **{CHECKSUM_PREFIX + k: checksums[k] for k in tensors},
    }
    return WordVectors._open(tensors, loaded, path)


def _check_license(value: object) -> None:
    """None, a license (a non-empty str) or the licenses a model card lists (a non-empty
    tuple of them)."""
    if value is None or (isinstance(value, str) and value and not value.startswith("[")):
        return
    if isinstance(value, tuple) and value and all(isinstance(v, str) and v for v in value):
        return
    raise ValueError(f"license must be None, a non-empty str or a tuple of them, not {value!r}")


def _license(metadata: dict[str, str]) -> str | tuple[str, ...] | None:
    """fse.license: a license, or the canonical JSON list of several."""
    value = metadata.get("fse.license")
    if value is None or not value.startswith("["):
        return value
    licenses = _json_list(metadata, "fse.license")
    if not licenses or not all(isinstance(v, str) and v for v in licenses):
        raise ValueError("fse.license must be a license or a JSON list of them")
    return tuple(licenses)


def _json_list(metadata: dict[str, str], key: str) -> list:
    text = metadata[key]
    try:
        value = json.loads(text)
        canonical = canonical_json(value)
    except ValueError as error:
        raise ValueError(f"{key} is not canonical JSON: {error}") from None
    if not isinstance(value, list) or canonical != text:
        raise ValueError(f"{key} must be a canonical JSON list")
    return value


def fasttext_parts(word_vectors: WordVectors, max_ngrams: int | None, oov_weight: float | None):
    """``max_ngrams`` and ``oov_weight`` of an encoder of ``word_vectors``, with their
    defaults for FastText vectors; other vectors take neither."""
    if word_vectors.fasttext is None:
        if max_ngrams is not None or oov_weight is not None:
            raise ValueError("max_ngrams and oov_weight apply only to FastText vectors")
        return {"max_ngrams": None, "oov_weight": None}
    if max_ngrams is not None and max_ngrams > MAX_NGRAMS_LIMIT:
        raise ValueError(f"max_ngrams must be at most {MAX_NGRAMS_LIMIT}, not {max_ngrams}")
    # Converted in the floating-point environment of the kernels (round to nearest, no
    # flush to zero), whatever the calling thread set, as the kernel converts it.
    given = 1.0 if oov_weight is None else oov_weight
    weight = get_backend("cpu").float32([given]).reshape(())
    if not np.isfinite(weight):
        raise ValueError(f"oov_weight must be finite in float32, not {oov_weight!r}")
    return {"max_ngrams": max_ngrams or DEFAULT_MAX_NGRAMS, "oov_weight": weight}
