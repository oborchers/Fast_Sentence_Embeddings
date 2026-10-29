#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The inference of a linear model: texts to its outputs, in arrays, batches or sinks.

Texts run through the head of the runtime (``fse._runtime._head``), in the workers.
Every check runs before any text is read.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from numpy import ndarray

from fse._backends import BACKEND_NAMES
from fse._encoder._batches import _named, check_sink, durable, flush
from fse._encoder._sources import opened
from fse._runtime import check_workers
from fse._store import ShardSink

from ._outputs import PredictedBatch

# The output a caller asks for, and what the runtime computes for it.
RUNTIME = {"labels": "labels", "proba": "proba", "decision": "logits", "values": "logits"}


def check_backend(backend: Any) -> str:
    if not isinstance(backend, str) or backend not in BACKEND_NAMES:
        raise ValueError(f"backend must be one of {list(BACKEND_NAMES)}, not {backend!r}")
    return backend


def check_output_sink(model: Any, sink: Any, output: str) -> None:
    """A sink of the outputs ``output``: float32 rows of their width, without inverse norms;
    one that names an ``encoder_fingerprint`` must name their output identity."""
    if output == "labels":
        raise ValueError("labels go to no sink in 2.0; take them from predict_batches")
    if isinstance(sink, ShardSink):
        raise ValueError(
            "a ShardSink stores the vectors of an encoder, not the outputs of a head; take the "
            "outputs from predict_batches, or store the vectors with return_vectors=True"
        )
    width = model._head.width(RUNTIME[output])
    dtype = check_sink(sink, None, model.output_identity(output), width)
    if dtype != "float32" or getattr(sink, "inverse_norms", False):
        raise ValueError("a sink of head outputs takes float32 rows without inverse norms")


def finish(model: Any, rows: ndarray, output: str) -> ndarray:
    """The outputs as a caller gets them: labels from their indices, one column flattened."""
    if output == "labels":
        return model.classes_[rows]
    if output != "proba" and rows.shape[1] == 1:
        return rows[:, 0].copy()
    return rows


def infer(model: Any, texts: Any, output: str, workers: Any, backend: Any, sink: Any) -> Any:
    """``output`` of every text of ``texts``: an array, or None when
    ``sink`` took them (its ``flush`` runs after the last row)."""
    check_workers(workers)
    check_backend(backend)
    if sink is not None:
        check_output_sink(model, sink, output)
    head = model._pipeline(backend, with_vectors=False)
    with opened(texts, model._tokenizer) as source:
        rows, _ = head.infer(source, output=RUNTIME[output], workers=workers, sink=sink)
    if sink is not None:
        flush(sink)
        return None
    return finish(model, rows, output)


def batches(
    model: Any,
    items: Any,
    output: str,
    return_vectors: bool,
    workers: Any,
    backend: Any,
    sink: Any,
) -> Iterator[PredictedBatch]:
    """The checks of ``predict_batches`` now, then its batches as a generator."""
    check_workers(workers)
    check_backend(backend)
    if return_vectors:
        if model.standalone:
            raise ValueError("a standalone model has no vectors: its table is folded")
        if sink is not None:  # the vectors go to the sink, the outputs stay in the batch
            check_sink(sink, None, model.encoder.fingerprint, model.encoder.output_dim)
    elif sink is not None:
        check_output_sink(model, sink, output)
    head = model._pipeline(backend, with_vectors=return_vectors)
    return _text_batches(model, head, iter(items), output, return_vectors, workers, sink)


def _text_batches(model, head, items: Iterable[Any], output, return_vectors, workers, sink):
    row, last = 0, None
    for k, item in enumerate(items):
        if not isinstance(item, list | tuple):
            raise TypeError(
                f"batch {k} is a {type(item).__name__}; every batch is a list (or tuple) of "
                "sentences"
            )
        n = len(item)
        with _named(k, row, n), opened(item, model._tokenizer) as source:
            sinks = {"vector_sink": sink} if return_vectors else {"sink": sink}
            rows, vectors = head.infer(
                source, output=RUNTIME[output], workers=workers, first_row=row, **sinks
            )
            if sink is not None:
                last = durable(sink, row + n, last)
            outputs = None if rows is None else finish(model, rows, output)
        yield PredictedBatch(row, n, outputs, vectors, last)
        row += n
    if sink is not None:
        flush(sink)
