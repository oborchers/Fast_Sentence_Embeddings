#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The format rules of encoder 2 (ARCHITECTURE.md, "encoder 2").

``to_artifact`` turns the content of an encoder into its tensors and metadata,
``from_artifact`` checks and parses them back. The keys and tensors of an encoder are a
closed set derived from its content: anything else raises. A folded encoder holds its
word vectors under their own names (word_vectors 2, without counts); the embedded word
vectors are built by the caller from what ``from_artifact`` returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy import ndarray

from fse._container import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    FINGERPRINT_FORM,
    FINGERPRINT_KEY,
    INTEGER,
    canonical_json,
    is_fingerprint,
)
from fse._contract import (
    TRANSFORM_VERSION,
    Aggregation,
    Normalize,
    Op,
    Project,
    RemoveComponents,
    Step,
    UnsupportedArtifactError,
    check_steps,
    check_tokenizer,
    check_transform_version,
    output_dim,
)

KIND = "encoder"
FORMAT_VERSION = "2"
# The keys of the word vectors a folded encoder holds, and their tensors (no counts: the
# weights of the encoder already took them in).
WORD_VECTOR_KEYS = frozenset(
    {
        "fse.dim",
        "fse.dtype",
        "fse.quantization",
        "fse.quantized_from",
        "fse.fasttext",
        "fse.ngram_hash",
    }
)
WORD_VECTOR_TENSORS = frozenset(
    {"vectors", "scales", "vocab.bytes", "vocab.offsets", "ngrams", "ngram_scales"}
)
_KEYS = frozenset(
    {
        "fse.kind",
        "fse.format_version",
        CREATED_BY_KEY,
        "fse.transform_version",
        "fse.tokenizer",
        "fse.aggregation",
        "fse.steps",
        "fse.input_dim",
        "fse.output_dim",
        "fse.max_ngrams",
        "fse.word_vectors",
        "fse.vocabulary",
        "fse.recipe",
        "fse.folded",
        "fse.folded_from",
        *WORD_VECTOR_KEYS,
    }
)
# The tensors of each step, by the name of their field; weights are optional.
_STEP_FIELDS = {Op.REMOVE_COMPONENTS: ("components", "weights"), Op.PROJECT: ("basis",)}


@dataclass(frozen=True, eq=False)
class EncoderContent:
    """What an encoder is, apart from its word vectors.

    ``weights`` is one float32 weight per word, or None for all ones; ``oov_weight`` a
    float32 scalar array exactly for FastText vectors, with ``max_ngrams``. ``input_dim``
    is the dimension of the word vectors the encoder pools. ``word_vectors`` and
    ``vocabulary`` are fingerprints. A folded encoder lists the steps folded into its
    word vectors in ``folded`` and names the original vectors in ``folded_from``.
    """

    tokenizer: str
    steps: tuple[Step, ...]
    input_dim: int
    weights: ndarray | None
    oov_weight: ndarray | None
    max_ngrams: int | None
    word_vectors: str
    vocabulary: str
    recipe: dict[str, Any]
    folded: tuple[Step, ...] = ()
    folded_from: str | None = None
    aggregation: Aggregation = Aggregation.WEIGHTED_MEAN

    @property
    def output_dim(self) -> int:
        return output_dim(self.steps, self.aggregation.output_dim(self.input_dim))


def to_artifact(
    content: EncoderContent, embedded: tuple[dict[str, ndarray], dict[str, str]] | None
) -> tuple[dict[str, ndarray], dict[str, str]]:
    """The tensors and metadata of an encoder; ``embedded`` holds the tensors and the
    metadata (``WORD_VECTOR_KEYS``) of the word vectors of a folded encoder."""
    tensors: dict[str, ndarray] = {}
    if content.weights is not None:
        tensors["weights"] = content.weights
    if content.oov_weight is not None:
        tensors["oov_weight"] = content.oov_weight
    tensors.update(_step_tensors("step", content.steps))
    metadata = {
        "fse.kind": KIND,
        "fse.format_version": FORMAT_VERSION,
        "fse.transform_version": str(TRANSFORM_VERSION),
        "fse.tokenizer": content.tokenizer,
        "fse.aggregation": content.aggregation.id,
        "fse.steps": canonical_json([s.op.id for s in content.steps]),
        "fse.input_dim": str(content.input_dim),
        "fse.output_dim": str(content.output_dim),
        "fse.word_vectors": content.word_vectors,
        "fse.vocabulary": content.vocabulary,
        "fse.recipe": canonical_json(content.recipe),
    }
    if content.max_ngrams is not None:
        metadata["fse.max_ngrams"] = str(content.max_ngrams)
    if content.folded:
        assert embedded is not None
        assert content.folded_from is not None
        tensors.update(_step_tensors("folded", content.folded))
        metadata["fse.folded"] = canonical_json([s.op.id for s in content.folded])
        metadata["fse.folded_from"] = content.folded_from
    if embedded is not None:
        tensors.update(embedded[0])
        metadata.update(embedded[1])
    return tensors, metadata


def _step_tensors(prefix: str, steps: tuple[Step, ...]) -> dict[str, ndarray]:
    tensors = {}
    for i, step in enumerate(steps):
        for field in _STEP_FIELDS.get(step.op, ()):
            value = getattr(step, field)
            if value is not None:
                tensors[f"{prefix}.{i}.{field}"] = value
    return tensors


def from_artifact(
    tensors: dict[str, ndarray], metadata: dict[str, str]
) -> tuple[EncoderContent, tuple[dict[str, ndarray], dict[str, str]] | None]:
    """Check and parse an encoder; returns its content and, for a folded encoder, the
    tensors and metadata of its word vectors. Unsupported versions, kinds, keys and
    identifiers raise UnsupportedArtifactError, anything else malformed ValueError."""
    _check_keys(metadata)
    check_transform_version(_integer(metadata, "fse.transform_version"))
    aggregation = Aggregation.parse(metadata["fse.aggregation"])
    input_dim = _integer(metadata, "fse.input_dim")
    used = {"weights", "oov_weight"}
    steps, names = _steps(tensors, _ids(metadata, "fse.steps"), "step")
    used |= names
    check_steps(steps, aggregation.output_dim(input_dim))
    folded_ids = _ids(metadata, "fse.folded") if "fse.folded" in metadata else None
    content = EncoderContent(
        tokenizer=check_tokenizer(metadata["fse.tokenizer"]),
        steps=steps,
        input_dim=input_dim,
        weights=_optional_vector(tensors, "weights"),
        oov_weight=_oov_weight(tensors, metadata),
        max_ngrams=_integer(metadata, "fse.max_ngrams") if "fse.max_ngrams" in metadata else None,
        word_vectors=_fingerprint(metadata, "fse.word_vectors"),
        vocabulary=_fingerprint(metadata, "fse.vocabulary"),
        recipe=_recipe(metadata),
        aggregation=aggregation,
    )
    if str(content.output_dim) != metadata.get("fse.output_dim"):
        raise ValueError(
            f"fse.output_dim is {metadata.get('fse.output_dim')!r}, but the steps give "
            f"{content.output_dim}"
        )
    embedded = None
    if folded_ids is not None:
        content, embedded, names = _folded(content, tensors, metadata, folded_ids)
        used |= names
    elif {"fse.folded_from", *WORD_VECTOR_KEYS} & set(metadata) or WORD_VECTOR_TENSORS & set(
        tensors
    ):
        raise ValueError("only a folded encoder holds word vectors")
    extra = set(tensors) - used
    if extra:
        raise ValueError(f"tensors the encoder does not use: {sorted(extra)}")
    return content, embedded


def _check_keys(metadata: dict[str, str]):
    kind = metadata.get("fse.kind")
    if kind != KIND:
        raise UnsupportedArtifactError(f"the artifact is a {kind!r}, not {KIND!r}")
    if metadata.get("fse.format_version") != FORMAT_VERSION:
        raise UnsupportedArtifactError(
            f"encoders of format version {metadata.get('fse.format_version')!r} are not "
            f"supported by this fse; supported: {FORMAT_VERSION}"
        )
    unknown = {
        k
        for k in metadata
        if k not in _KEYS and k != FINGERPRINT_KEY and not k.startswith(CHECKSUM_PREFIX)
    }
    if unknown:
        raise UnsupportedArtifactError(f"unknown metadata keys of an encoder: {sorted(unknown)}")
    for key in (
        "fse.transform_version",
        "fse.tokenizer",
        "fse.aggregation",
        "fse.steps",
        "fse.input_dim",
        "fse.output_dim",
        "fse.word_vectors",
        "fse.vocabulary",
        "fse.recipe",
    ):
        if key not in metadata:
            raise ValueError(f"an encoder needs the metadata key {key}")


def _integer(metadata: dict[str, str], key: str) -> int:
    value = metadata.get(key)
    if value is None or not INTEGER.fullmatch(value):
        raise ValueError(f"{key} must be a non-negative decimal integer, not {value!r}")
    return int(value)


def _fingerprint(metadata: dict[str, str], key: str) -> str:
    value = metadata[key]
    if not is_fingerprint(value):
        raise ValueError(f"{key} must be a fingerprint ({FINGERPRINT_FORM}), not {value!r}")
    return value


def _ids(metadata: dict[str, str], key: str) -> list[Op]:
    try:
        ids = json.loads(metadata[key])
    except json.JSONDecodeError:
        ids = None
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"{key} must be a JSON list of step identifiers")
    if canonical_json(ids) != metadata[key]:
        raise ValueError(f"{key} must be canonical JSON")
    return [Op.parse(i) for i in ids]


def _recipe(metadata: dict[str, str]) -> dict[str, Any]:
    try:
        recipe = json.loads(metadata["fse.recipe"])
    except json.JSONDecodeError:
        recipe = None
    if not isinstance(recipe, dict) or canonical_json(recipe) != metadata["fse.recipe"]:
        raise ValueError("fse.recipe must be a canonical JSON object")
    return recipe


def _steps(tensors: dict[str, ndarray], ops: list[Op], prefix: str) -> tuple[tuple[Step, ...], set]:
    """The steps ``ops`` from their tensors ``<prefix>.<i>.<field>``, and the names used."""
    steps: list[Step] = []
    names: set[str] = set()
    for i, op in enumerate(ops):
        fields = {}
        for field in _STEP_FIELDS.get(op, ()):
            name = f"{prefix}.{i}.{field}"
            if name in tensors:
                fields[field] = np.ascontiguousarray(tensors[name])
                names.add(name)
        steps.append(_step(op, fields, f"{prefix}.{i}"))
    return tuple(steps), names


def _step(op: Op, fields: dict[str, ndarray], name: str) -> Step:
    if op == Op.NORMALIZE:
        return Normalize()
    needed = _STEP_FIELDS[op][0]
    if needed not in fields:
        raise ValueError(f"the step {name} needs the tensor {name}.{needed}")
    if op == Op.REMOVE_COMPONENTS:
        return RemoveComponents(fields["components"], fields.get("weights"))
    return Project(fields["basis"])


def _optional_vector(tensors: dict[str, ndarray], name: str) -> ndarray | None:
    value = tensors.get(name)
    if value is None:
        return None
    if value.dtype != np.float32 or value.ndim != 1:
        raise ValueError(f"{name} must be a float32 vector")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return value


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


def _folded(
    content: EncoderContent, tensors: dict[str, ndarray], metadata: dict[str, str], ops: list[Op]
) -> tuple[EncoderContent, tuple[dict[str, ndarray], dict[str, str]], set[str]]:
    """The folded steps and the embedded word vectors of a folded encoder."""
    if not ops or any(op == Op.NORMALIZE for op in ops):
        raise ValueError("fse.folded must list linear steps (remove_components, project)")
    if any(s.op != Op.NORMALIZE for s in content.steps):
        raise ValueError("a folded encoder keeps only normalize as a step")
    if "fse.folded_from" not in metadata:
        raise ValueError("a folded encoder names its original word vectors in fse.folded_from")
    folded, names = _steps(tensors, ops, "folded")
    first = folded[0]
    assert isinstance(first, RemoveComponents | Project)
    original = first.dim
    if output_dim(check_steps(folded, original), original) != content.input_dim:
        raise ValueError(
            f"the folded steps give rows of {output_dim(folded, original)} values, not "
            f"fse.input_dim ({content.input_dim})"
        )
    words = {k: v for k, v in tensors.items() if k in WORD_VECTOR_TENSORS}
    keys = {k: v for k, v in metadata.items() if k in WORD_VECTOR_KEYS}
    if "vectors" not in words or keys.get("fse.dim") != str(content.input_dim):
        raise ValueError("a folded encoder holds its word vectors of fse.input_dim values")
    content = replace(content, folded=folded, folded_from=_fingerprint(metadata, "fse.folded_from"))
    return content, (words, keys), names | set(words)
