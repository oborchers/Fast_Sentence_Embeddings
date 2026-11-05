#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""What both importers of Model2Vec classifier heads share (docs/architecture/hub.md,
"Model2Vec classifiers").

A Model2Vec ``StaticModelPipeline`` is a static embedding model and an ``MLPHead``: layers
``x @ W.T + b`` (ReLU between them), an output activation and the classes. fse imports
exactly one linear layer (a head trained with ``n_layers=0``; the ``n_layers`` of an
export counts the linear layers, so it is 1) with the activation softmax or sigmoid, in
float32 with ``weight (k, d)`` and ``bias (k,)``. This module checks that description
before ``fse._head._content`` could round or reshape anything: the file reader gives the
header of ``head.safetensors``, the object reader the arrays of the pipeline, and both get
the same refusals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numpy import ndarray

from fse._contract import UnsupportedArtifactError

# The keys model2vec 0.10.0 writes into head_config; a new one could change the head.
KEYS = frozenset({"n_layers", "activation", "classes"})
# The activation of model2vec and the link of the head it becomes.
LINKS = {"softmax": "softmax", "sigmoid": "multilabel"}
_HIDDEN = (
    "the head has hidden layers with ReLU (a head trained with n_layers >= 1); fse imports "
    "linear heads, trained with n_layers=0 (the n_layers of an export counts its linear "
    "layers, so 1)"
)


@dataclass(frozen=True)
class Spec:
    """What ``head_config`` says: the link of the head and its classes."""

    link: str
    activation: str
    labels: list


@dataclass(frozen=True)
class Imported:
    """A head read from Model2Vec: A (k, d) and b (k,) in float32, the link name, the
    labels and a note for the model card."""

    weight: ndarray
    bias: ndarray
    link: str
    labels: list
    note: str


def spec(n_layers: Any, activation: Any, classes: Any, where: Any, fail: type) -> Spec:
    """The link and classes of a head of ``n_layers`` linear layers, the output
    ``activation`` and ``classes``; ``where`` names the source. A head that is no single
    linear layer with softmax or sigmoid raises ``UnsupportedArtifactError``, a malformed
    value ``fail``."""
    if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 0:
        raise fail(f"{where}: n_layers must be a non-negative int, not {n_layers!r}")
    if n_layers == 0:
        raise UnsupportedArtifactError(f"{where}: the head has no layer, so it is no linear head")
    if n_layers > 1:
        raise UnsupportedArtifactError(f"{where}: {_HIDDEN}")
    if not isinstance(activation, str):
        raise fail(f"{where}: the activation must be a str, not {activation!r}")
    if activation == "identity":
        raise UnsupportedArtifactError(
            f"{where}: the head is a regression or projection head (activation identity); "
            "fse imports classifiers with the activation softmax or sigmoid"
        )
    if activation not in LINKS:
        raise UnsupportedArtifactError(f"{where}: the activation {activation!r} is unknown")
    if not isinstance(classes, list) or not classes:
        raise fail(f"{where}: classes must be a non-empty list, not {classes!r}")
    return Spec(LINKS[activation], activation, classes)


def check_layer(
    weight: tuple[str, tuple], bias: tuple[str, tuple], head: Spec, where: Any, fail: type
) -> int:
    """k, the number of classes, of a layer whose tensors are given as ``(dtype name,
    shape)``. Only float32 is read: ``fse._head._content`` would round another dtype, and
    a bias of any other shape (such as a scalar) would be broadcast."""
    for name, (dtype, _) in (("weight", weight), ("bias", bias)):
        if dtype != "float32":
            raise UnsupportedArtifactError(
                f"{where}: the head {name} is {dtype}; fse reads float32 heads, as model2vec "
                "writes them"
            )
    if len(weight[1]) != 2 or len(bias[1]) != 1:
        raise fail(
            f"{where}: the head needs weight (k, d) and bias (k,), not shapes {weight[1]} "
            f"and {bias[1]}"
        )
    k = weight[1][0]
    if k < 1 or bias[1] != (k,) or k != len(head.labels):
        raise fail(
            f"{where}: weight {weight[1]}, bias {bias[1]} and {len(head.labels)} classes "
            "do not describe one head"
        )
    if head.link == "softmax" and k == 1:
        raise UnsupportedArtifactError(
            f"{where}: a softmax head of one class is constant; fse imports softmax heads of "
            "at least two classes"
        )
    return k


def note(head: Spec, what: str, k: int) -> str:
    """The sentence of the model card that names the import."""
    return f"head imported from a Model2Vec {what} ({head.activation}, {k} classes)"
