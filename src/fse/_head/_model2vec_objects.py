#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Classifier heads from Model2Vec objects (docs/architecture/hub.md, "Model2Vec
classifiers"): a ``StaticModelPipeline``, or a trained ``StaticModelForClassification``
whose ``to_pipeline`` makes one.

Nothing of model2vec or torch is imported: an object is
known by the name of its class and module, and read by its attributes, as the scikit-learn
importer does. The installed version of model2vec is read from its metadata and must be
0.10.* (the version whose layout this reader checks). Only ``to_pipeline`` of a trained
model runs model2vec code; it trains nothing.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from fse._encoder._format import Source
from fse._encoder._model2vec import read_object_table
from fse._validation import refuse_lost_input

from . import _model2vec as spec_

PIPELINE = ("model2vec.inference.model", "StaticModelPipeline")
TRAINED = ("model2vec.train.classifier", "StaticModelForClassification")
_WINDOW = re.compile(r"0\.10(\.|$)")
_WHERE = Path("Model2Vec pipeline")


def read(model: Any) -> tuple[spec_.Imported, Encoder]:
    """The head and the encoder of a Model2Vec pipeline or trained classifier."""
    kind = (type(model).__module__, type(model).__name__)
    if kind not in (PIPELINE, TRAINED) or not _is_class_of_module(model, kind):
        raise TypeError(
            f"model must be a model2vec StaticModelPipeline or a trained "
            f"StaticModelForClassification, not {kind[0]}.{kind[1]}; open files with "
            "Classifier.from_pretrained"
        )
    _check_version()
    what = "StaticModelPipeline"
    if kind == TRAINED:
        layers = getattr(model, "n_layers", None)
        if layers != 0:
            raise UnsupportedArtifactError(
                f"the trained model has n_layers={layers!r}; fse imports linear heads, "
                "trained with n_layers=0 (hidden layers with ReLU are not imported)"
            )
        model = model.to_pipeline()  # converts the trained tensors; fits nothing
        what = "StaticModelForClassification"
    return _pipeline(model, what)


def _is_class_of_module(model: Any, kind: tuple[str, str]) -> bool:
    """The class of ``model`` is the class of that name in the loaded module, not a class
    that merely carries the names (the module of an object is always loaded)."""
    module = sys.modules.get(kind[0])
    return getattr(module, kind[1], None) is type(model)


def _check_version() -> None:
    try:
        installed = version("model2vec")
    except PackageNotFoundError:  # an object of a package that is gone
        installed = ""
    if not _WINDOW.match(installed):
        raise UnsupportedArtifactError(
            f"model2vec {installed or '(unknown)'} is installed, but fse reads objects of "
            "model2vec 0.10 only; save the pipeline with save_pretrained and open the folder "
            "with Classifier.from_pretrained"
        )


def _classes(value: Any) -> list | None:
    """The classes as a list, from an integer or string array only: other containers and
    dtypes would be converted (a mix of str and int to str) before they could be checked."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        refuse_lost_input(value, "classes_")
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype.kind not in "iuU":
        raise ValueError("the classes must be a one-dimensional numpy array of int or str")
    return value.tolist()


def _array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"the head {name} must be a numpy array, not {type(value).__name__}")
    refuse_lost_input(value, name)
    return value


def _pipeline(pipeline: Any, what: str) -> tuple[spec_.Imported, Encoder]:
    head = pipeline.head
    layers = list(head.layers)
    classes = _classes(pipeline.classes_)
    refuse_lost_input(head.classes_, "head.classes_")
    same = (pipeline.classes_ is None) == (head.classes_ is None) and (
        classes is None or np.array_equal(pipeline.classes_, head.classes_)
    )
    if not same:
        raise ValueError("pipeline.classes_ differs from pipeline.head.classes_")
    given = getattr(head.activation, "value", head.activation)
    parsed = spec_.spec(len(layers), given, classes, _WHERE, ValueError)
    weight = _array(layers[0].weight, "weight")
    bias = _array(layers[0].bias, "bias")
    spec_.check_layer(
        (weight.dtype.name, weight.shape), (bias.dtype.name, bias.shape), parsed, _WHERE, ValueError
    )
    encoder = _encoder(pipeline.model, parsed, len(layers), weight.shape[1])
    weight, bias = np.array(weight, copy=True), np.array(bias, copy=True)
    note = spec_.note(parsed, what, len(bias))
    return spec_.Imported(weight, bias, parsed.link, parsed.labels, note), encoder


def _encoder(model: Any, head: spec_.Spec, n_layers: int, dim: int) -> Encoder:
    if model.token_mapping is not None:
        raise UnsupportedArtifactError(
            f"{_WHERE}: the model has vocabulary quantization (a token mapping), which this "
            "fse does not read"
        )
    if model.weights is not None:
        raise UnsupportedArtifactError(
            f"{_WHERE}: the model has separate token weights, which this fse does not read; "
            "to_pipeline of a trained model multiplies them into the table"
        )
    config = dict(model.config)
    expected = {"n_layers": n_layers, "activation": head.activation, "classes": head.labels}
    if "head_config" in config:
        stated = config.pop("head_config")
        exact = isinstance(stated, dict) and type(stated.get("n_layers")) is int
        if not exact or stated != expected:
            raise ValueError(f"{_WHERE}: the model.config head_config does not describe the head")
    for name in ("normalize", "max_length"):  # what model2vec applies, and what it states
        if getattr(model, name) != config.get(name) or (
            type(getattr(model, name)) is not type(config.get(name))
        ):
            raise ValueError(f"{_WHERE}: model.{name} differs from model.config[{name!r}]")
    tokenizer = model.tokenizer.to_str().encode("utf-8")
    converted = read_object_table(config, tokenizer, model.embedding, _WHERE)
    if converted.word_vectors.dim != dim:
        raise ValueError(
            f"{_WHERE}: the head reads {dim} values, the table has "
            f"{converted.word_vectors.dim} columns"
        )
    return Encoder._converted(
        converted.word_vectors,
        converted.tokenizer,
        normalize=converted.normalize,
        source=Source("model2vec"),
    )
