#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The format rules of linear_head 1 (docs/architecture/formats.md, "linear_head 1").

A head file is an encoder file (its tensors and keys, unchanged) with ``fse.kind`` and
``fse.format_version`` of the head, the ``fse.head.*`` keys, ``fse.encoder_fingerprint``
(the fingerprint of that encoder part) and the tensors ``head.weight`` and ``head.bias``.
A standalone file holds the folded table as its encoder part and no ``head.weight``. The
keys and tensors of the head are a closed set; reading checks them, takes them off and
hands the rest to the encoder's own parser as an encoder artifact, whose fingerprint must
equal ``fse.encoder_fingerprint`` (the adapter, so the encoder format stays unchanged).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy import ndarray

from fse._container import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    FINGERPRINT_KEY,
    CorruptArtifactError,
    canonical_json,
    fingerprint,
    is_fingerprint,
    tensor_code,
)
from fse._contract import LINK_NONE, Head, Link, UnsupportedArtifactError
from fse._encoder._format import FORMAT_VERSION as ENCODER_VERSION
from fse._encoder._format import KIND as ENCODER_KIND

from . import _content
from ._content import TASKS, HeadContent

KIND = "linear_head"
FORMAT_VERSION = "1"
_HEAD = "fse.head"
_KEYS = frozenset(
    {
        _HEAD,
        "fse.head.task",
        "fse.head.normalized",
        "fse.head.link",
        "fse.head.labels",
        "fse.head.standalone",
        "fse.head.flat",
        "fse.encoder_fingerprint",
    }
)
# Keys given only for some heads: the labels of a classifier, the shape of one target.
_OPTIONAL = frozenset({"fse.head.labels", "fse.head.flat"})
TENSORS = ("head.weight", "head.bias")
_BOOLS = {"true": True, "false": False}


def encoder_fingerprint(tensors: dict[str, ndarray], metadata: dict[str, str], checksums) -> str:
    """The fingerprint of an encoder artifact from the checksums of its tensors."""
    entries = {k: (tensor_code(v), v.shape, checksums[k]) for k, v in tensors.items()}
    return fingerprint(entries, metadata)


def to_artifact(
    content: HeadContent, tensors: dict[str, ndarray], metadata: dict[str, str], of_encoder: str
) -> tuple[dict[str, ndarray], dict[str, str]]:
    """The tensors and metadata of a head on the encoder artifact ``tensors`` and
    ``metadata``, whose fingerprint is ``of_encoder``."""
    out = dict(tensors)
    if content.weight is not None:
        out["head.weight"] = content.weight
    out["head.bias"] = content.bias
    keys = {
        k: v
        for k, v in metadata.items()
        if k not in (CREATED_BY_KEY, FINGERPRINT_KEY) and not k.startswith(CHECKSUM_PREFIX)
    }
    keys.update(
        {
            "fse.kind": KIND,
            "fse.format_version": FORMAT_VERSION,
            _HEAD: Head.LINEAR.id,
            "fse.head.task": content.task,
            "fse.head.normalized": str(content.normalized).lower(),
            "fse.head.link": content.link_id,
            "fse.head.standalone": str(content.standalone).lower(),
            "fse.encoder_fingerprint": of_encoder,
        }
    )
    if content.labels is not None:
        keys["fse.head.labels"] = canonical_json(list(content.labels))
    if content.task == "regression" and content.k == 1:
        keys["fse.head.flat"] = str(content.flat).lower()
    return out, keys


def from_artifact(
    tensors: dict[str, ndarray], metadata: dict[str, str]
) -> tuple[HeadContent, dict[str, ndarray], dict[str, str], str]:
    """Check and parse a head; returns its content, the encoder artifact (tensors and
    metadata, for ``fse._encoder._format.from_artifact``) and its fingerprint, checked
    against ``fse.encoder_fingerprint``."""
    _check_kind(metadata)
    task = _choice(metadata, "fse.head.task", TASKS)
    standalone = _bool(metadata, "fse.head.standalone")
    normalized = _bool(metadata, "fse.head.normalized")
    if standalone and normalized:
        raise ValueError("a standalone head reads raw means; fse.head.normalized must be false")
    weight, bias = _head_tensors(tensors, standalone)
    content = HeadContent(
        task,
        weight,
        bias,
        normalized,
        _link(metadata, task, len(bias)),
        _labels(metadata, task, len(bias)),
        standalone,
        _flat(metadata, task, len(bias)),
    )
    return (content, *_encoder_part(tensors, metadata, content))


def _head_tensors(tensors: dict[str, ndarray], standalone: bool) -> tuple[ndarray | None, ndarray]:
    bias = _tensor(tensors, "head.bias", 1)
    if standalone:
        if "head.weight" in tensors:
            raise ValueError("a standalone head holds no head.weight: its table is folded")
        return None, bias
    weight = _tensor(tensors, "head.weight", 2)
    if len(weight) != len(bias):
        raise ValueError(f"head.weight has {len(weight)} rows, but head.bias {len(bias)} values")
    return weight, bias


def _encoder_part(
    tensors: dict[str, ndarray], metadata: dict[str, str], content: HeadContent
) -> tuple[dict[str, ndarray], dict[str, str], str]:
    """The encoder artifact of a head and its fingerprint, checked against the head."""
    weight = content.weight
    if weight is not None and str(weight.shape[1]) != metadata.get("fse.output_dim"):
        raise ValueError(
            f"head.weight takes rows of {weight.shape[1]} values, but the encoder gives "
            f"{metadata.get('fse.output_dim')!r}"
        )
    if content.standalone:
        _check_folded(tensors, metadata, content.k)
    encoder_tensors = {k: v for k, v in tensors.items() if k not in TENSORS}
    dropped = {*_KEYS, FINGERPRINT_KEY, *(CHECKSUM_PREFIX + t for t in TENSORS)}
    encoder_metadata = {k: v for k, v in metadata.items() if k not in dropped}
    encoder_metadata.update({"fse.kind": ENCODER_KIND, "fse.format_version": ENCODER_VERSION})
    checksums = {
        k[len(CHECKSUM_PREFIX) :]: v for k, v in metadata.items() if k.startswith(CHECKSUM_PREFIX)
    }
    found = encoder_fingerprint(encoder_tensors, encoder_metadata, checksums)
    if found != metadata["fse.encoder_fingerprint"]:
        raise CorruptArtifactError(
            f"the encoder part of the head has the fingerprint {found}, not the "
            f"{metadata['fse.encoder_fingerprint']} it records"
        )
    return encoder_tensors, encoder_metadata, found


def _check_folded(tensors: dict[str, ndarray], metadata: dict[str, str], k: int) -> None:
    """The encoder part of a standalone head is its folded table: k float32 values per
    word and ngram, no scales, no steps."""
    if metadata.get("fse.steps") != "[]":
        raise ValueError("the folded table of a standalone head has no steps")
    if metadata.get("fse.output_dim") != str(k) or metadata.get("fse.dim") != str(k):
        raise ValueError(
            f"the folded table of a standalone head of {k} outputs has {k} values per row, not "
            f"{metadata.get('fse.output_dim')!r}"
        )
    if metadata.get("fse.dtype") != "float32" or "fse.quantization" in metadata:
        raise ValueError("the folded table of a standalone head is float32")
    extra = sorted({"scales", "ngram_scales"} & set(tensors))
    if extra:
        raise ValueError(f"the folded table of a standalone head has no scales, not {extra}")


def _check_kind(metadata: dict[str, str]) -> None:
    kind = metadata.get("fse.kind")
    if kind != KIND:
        raise UnsupportedArtifactError(f"the artifact is a {kind!r}, not {KIND!r}")
    if metadata.get("fse.format_version") != FORMAT_VERSION:
        raise UnsupportedArtifactError(
            f"linear heads of format version {metadata.get('fse.format_version')!r} are not "
            f"supported by this fse; supported: {FORMAT_VERSION}"
        )
    unknown = sorted(k for k in metadata if k.startswith(_HEAD + ".") and k not in _KEYS)
    if unknown:
        raise UnsupportedArtifactError(f"unknown metadata keys of a linear head: {unknown}")
    for key in sorted(_KEYS - _OPTIONAL):
        if key not in metadata:
            raise ValueError(f"a linear head needs the metadata key {key}")
    Head.parse(metadata[_HEAD])  # an unknown head rule raises
    if not is_fingerprint(metadata["fse.encoder_fingerprint"]):
        raise ValueError("fse.encoder_fingerprint must be a fingerprint")


def _choice(metadata: dict[str, str], key: str, allowed: tuple[str, ...]) -> str:
    value = metadata[key]
    if value not in allowed:
        raise ValueError(f"{key} must be one of {', '.join(allowed)}, not {value!r}")
    return value


def _bool(metadata: dict[str, str], key: str) -> bool:
    return _BOOLS[_choice(metadata, key, tuple(_BOOLS))]


def _tensor(tensors: dict[str, ndarray], name: str, ndim: int) -> ndarray:
    value = tensors.get(name)
    if value is None:
        raise ValueError(f"a linear head needs the tensor {name}")
    if value.dtype != np.float32 or value.ndim != ndim or not value.size:
        raise ValueError(f"{name} must be a non-empty F32 tensor of {ndim} dimensions")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} holds values that are not finite")
    return value


def _flat(metadata: dict[str, str], task: str, k: int) -> bool:
    """Whether one output gives (n,): ``fse.head.flat``, exactly for a regression head of
    one output (a classifier's decisions of one output are always (n,))."""
    if ("fse.head.flat" in metadata) != (task == "regression" and k == 1):
        raise ValueError("fse.head.flat is given exactly for regression heads of one output")
    return _bool(metadata, "fse.head.flat") if "fse.head.flat" in metadata else True


def _link(metadata: dict[str, str], task: str, k: int) -> Link | None:
    text = metadata["fse.head.link"]
    found = None if text == LINK_NONE else Link.parse(text)
    if task == "regression" and found is not None:
        raise ValueError(f"a regression head has no link, not {text}")
    if found is not None:
        _content.link(found.value[0], k)  # the link fits k
    return found


def _labels(metadata: dict[str, str], task: str, k: int) -> tuple | None:
    text = metadata.get("fse.head.labels")
    if (text is None) != (task == "regression"):
        raise ValueError("fse.head.labels is given exactly for classification")
    if text is None:
        return None
    try:
        values: Any = json.loads(text)
    except json.JSONDecodeError:
        values = None
    if not isinstance(values, list) or canonical_json(values) != text:
        raise ValueError("fse.head.labels must be a canonical JSON list")
    if any(isinstance(v, float) for v in values):
        raise ValueError("fse.head.labels must hold int or str labels")
    return _content.labels(values, k)
