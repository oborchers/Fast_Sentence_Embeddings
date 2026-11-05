#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Classifier heads from the files of a Model2Vec ``StaticModelPipeline`` export
(docs/architecture/hub.md, "Model2Vec classifiers").

``save_pretrained`` of a pipeline writes ``config.json`` (with ``head_config``),
``model.safetensors``, ``tokenizer.json`` and ``head.safetensors``; fse reads exactly that
layout, checks the header of ``head.safetensors`` before its data, and never opens
``pipeline.skops`` (a pickle). The encoder is converted as ``Encoder.from_pretrained``
converts a Model2Vec model, with the same checks of the Hub commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fse._container import (
    DTYPE_CODES,
    CorruptArtifactError,
    foreign_header,
    foreign_tensors,
)
from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder, _pretrained
from fse._encoder._files import Files
from fse._encoder._model2vec import FILES, read_model2vec

from . import _model2vec as spec_

HEAD = "head.safetensors"
LEGACY = "pipeline.skops"
_TENSORS = ("head.0.weight", "head.0.bias")


def claimed(files: Any) -> dict | None:
    """The parsed config.json if ``files`` hold a Model2Vec model or export that this
    reader judges (an fse folder, a local file or any other model is not its to judge:
    None)."""
    if isinstance(files, Path):
        return None
    config = _pretrained._config(files)
    if config is None or config.get("model_type") not in (None, "model2vec"):
        return None
    heads = "head_config" in config or files.has(HEAD) or files.has(LEGACY)
    return config if heads or all(files.has(name) for name in FILES) else None


def refuse_for_regressor(files: Any) -> None:
    """A Model2Vec model or export is no regressor."""
    if claimed(files) is not None:
        raise UnsupportedArtifactError(
            "the model is a Model2Vec model; fse imports its classifier head with "
            "fse.Classifier.from_pretrained (regression heads are not imported)"
        )


def read(files: Files, config: dict, mmap: bool) -> tuple[spec_.Imported, Encoder]:
    """The head and the encoder of the export in ``files`` with its parsed ``config``."""
    if mmap:
        raise ValueError(
            "mmap is for fse models: a Model2Vec export is converted in memory; save the "
            "classifier and load it with mmap"
        )
    where = files.where("config.json")
    has_head = files.has(HEAD)
    if not has_head and files.has(LEGACY):
        raise UnsupportedArtifactError(
            f"{where}: the model holds {LEGACY}, a legacy scikit-learn and skops pipeline; "
            "fse loads no pickle or skops (convert it with model2vec's "
            "convert_legacy_pipeline in an environment you trust, then save_pretrained)"
        )
    if "head_config" not in config and not has_head:
        raise UnsupportedArtifactError(
            f"{where}: the model has no head; read it with fse.Encoder.from_pretrained and "
            "make a head with fse.Classifier.from_sklearn or from_arrays"
        )
    if ("head_config" in config) != has_head:
        raise CorruptArtifactError(f"{where}: head_config and {HEAD} must come together")
    missing = [name for name in (*FILES, HEAD) if not files.has(name)]
    if missing:
        raise CorruptArtifactError(f"{where}: the export lacks {missing}")
    _pretrained._check_modules(files, config.get("model_type"))
    head = _head_config(config["head_config"], where)
    weight, bias = _tensors(files, head)
    own = {key: value for key, value in config.items() if key != "head_config"}
    found = _pretrained._found(files, "model2vec", read_model2vec(files, own))
    model = found.converted
    assert model is not None
    encoder = Encoder._converted(
        model.word_vectors,
        model.tokenizer,
        normalize=model.normalize,
        source=found.source,
        license=found.license,
    )
    if weight.shape[1] != encoder.output_dim:
        raise CorruptArtifactError(
            f"{files.where(HEAD)}: the head reads {weight.shape[1]} values, the table has "
            f"{encoder.output_dim} columns"
        )
    note = spec_.note(head, "StaticModelPipeline export", len(bias))
    return spec_.Imported(weight, bias, head.link, head.labels, note), encoder


def _head_config(value: Any, where: Any) -> spec_.Spec:
    if not isinstance(value, dict):
        raise CorruptArtifactError(f"{where}: head_config must be a JSON object")
    unknown = sorted(set(value) - spec_.KEYS)
    if unknown:
        raise UnsupportedArtifactError(
            f"{where}: head_config holds keys this fse does not know: {unknown}"
        )
    return spec_.spec(
        value.get("n_layers", 0),
        value.get("activation", "identity"),
        value.get("classes"),
        where,
        CorruptArtifactError,
    )


def _tensors(files: Files, head: spec_.Spec) -> tuple[np.ndarray, np.ndarray]:
    """A and b of ``head.safetensors``; the header is checked before the data are read."""
    path = files.where(HEAD)

    def inspect(handle: Any, size: int) -> None:
        entries = foreign_header(handle, path, size)[1]
        if set(entries) != set(_TENSORS):
            raise CorruptArtifactError(
                f"{path} must hold exactly {list(_TENSORS)}, not {sorted(entries)}"
            )
        described = [
            (np.dtype(DTYPE_CODES[entries[name][0]]).name, tuple(entries[name][1]))
            for name in _TENSORS
        ]
        spec_.check_layer(described[0], described[1], head, path, CorruptArtifactError)

    data = files.read(HEAD, lfs=True, inspect=inspect)
    tensors = foreign_tensors(data, path)
    return tensors[_TENSORS[0]], tensors[_TENSORS[1]]
