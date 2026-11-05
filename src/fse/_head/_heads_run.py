#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The inference of a group of heads: texts to the outputs of every head, in arrays or
batches (docs/architecture/runtime.md, "Runtime", Groups of heads).

Texts run through the one runtime of heads, in the workers: one tokenization per task for
all heads. A group has no sink and no store mode: its results are in memory. The ``output``
rule resolves what each head gives before any text is read.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any

from numpy import ndarray

from fse._encoder._batches import _named, batch_size, batch_source
from fse._encoder._sources import opened
from fse._runtime import check_workers
from fse._store import VectorStore

from ._outputs import HeadsBatch
from ._run import RUNTIME, check_backend, finish

# What a head can give: labels, probabilities or decisions of a classifier, values of a regressor.
OUTPUT_NAMES = ("labels", "proba", "decision", "values")


def resolve_outputs(models: Mapping[str, Any], output: Any) -> tuple[str, ...]:
    """The output of every head, in order, from ``output``: None (each head gives its
    ``predict``), one name for all heads, or a mapping of names to outputs (unnamed heads
    keep their default). Raises before anything is read."""
    given = _given(models, output)
    return tuple(_checked(name, model, given.get(name)) for name, model in models.items())


def _given(models: Mapping[str, Any], output: Any) -> dict[str, str]:
    if output is None:
        return {}
    if isinstance(output, str):
        return dict.fromkeys(models, _name(output))
    if not isinstance(output, Mapping):
        raise TypeError(
            f"output must be None, an output name or a mapping of head names to output names, "
            f"not {type(output).__name__}"
        )
    for key in output:
        if not isinstance(key, str):
            raise TypeError(f"the keys of output are head names (str), not {type(key).__name__}")
        if key not in models:
            raise ValueError(f"output names the head {key!r}, but the heads are {list(models)}")
    return {key: _name(value) for key, value in output.items()}


def _name(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"an output is one of {', '.join(OUTPUT_NAMES)}, not {value!r}")
    if value not in OUTPUT_NAMES:
        raise ValueError(f"output must be one of {', '.join(OUTPUT_NAMES)}, not {value!r}")
    return value


def _checked(name: str, model: Any, output: str | None) -> str:
    """``output`` for the head ``name`` (its default for None), if the head can give it."""
    regressor = model.TASK == "regression"
    if output is None:
        return "values" if regressor else "labels"
    if (output == "values") != regressor:
        what = (
            "a regressor gives values"
            if regressor
            else "a classifier gives labels, proba or decision"
        )
        raise ValueError(
            f"head {name!r} cannot give {output!r}: {what}; name the output of each head in a mapping"
        )
    if output == "proba" and model._content.link is None:
        raise ValueError(f"head {name!r} has no probabilities (link none); use decision")
    return output


def check_group_call(workers: Any, backend: Any, items: Any) -> None:
    check_workers(workers)
    check_backend(backend)
    if isinstance(items, VectorStore):
        raise ValueError(
            "a group of heads reads texts; take the outputs of a store from each head's predict"
        )


def infer(heads: Any, texts: Any, output: Any, workers: Any, backend: Any) -> Mapping[str, ndarray]:
    """The outputs of every head for every text of ``texts``, in head order."""
    check_group_call(workers, backend, texts)
    outputs = resolve_outputs(heads, output)
    pipeline = heads._pipeline(backend, with_vectors=False)
    with opened(texts, heads._tokenizer) as source:
        rows, _ = pipeline.infer(
            source, outputs=tuple(RUNTIME[o] for o in outputs), workers=workers
        )
    return _outputs(heads, rows, outputs)


def batches(
    heads: Any, items: Any, output: Any, return_vectors: bool, workers: Any, backend: Any
) -> Iterator[HeadsBatch]:
    """The checks of ``predict_batches`` now, then its batches as a generator."""
    check_group_call(workers, backend, items)
    outputs = resolve_outputs(heads, output)
    if return_vectors and heads._standalone:
        raise ValueError("a standalone group has no vectors: its table is folded")
    pipeline = heads._pipeline(backend, with_vectors=return_vectors)
    return _batches(heads, pipeline, iter(items), outputs, workers)


def _batches(heads, pipeline, items: Iterable[Any], outputs, workers) -> Iterator[HeadsBatch]:
    row = 0
    kinds = tuple(RUNTIME[o] for o in outputs)
    for k, item in enumerate(items):
        with (
            _named(k, row, batch_size(k, item)),
            batch_source(item, heads._tokenizer) as (source, n),
        ):
            rows, vectors = pipeline.infer(source, outputs=kinds, workers=workers, first_row=row)
            given = _outputs(heads, rows, outputs)
        yield HeadsBatch(first_row=row, n_rows=n, outputs=given, vectors=vectors)
        row += n


def _outputs(heads: Any, rows: list, outputs: tuple[str, ...]) -> Mapping[str, ndarray]:
    """The outputs as ``predict`` of each head gives them (labels from their indices, one
    column flattened where the head gives (n,)), read-only in head order."""
    return MappingProxyType(
        {
            name: finish(model, result, output)
            for (name, model), result, output in zip(heads.items(), rows, outputs, strict=True)
        }
    )
