#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The inference of a linear model: texts or a store to its outputs, in arrays, batches or
sinks.

Texts run through the head of the runtime (``fse._runtime._head``), in the workers;
a ``VectorStore`` is store mode: the logits come from the stored rows by ``score`` (the
contract) block by block, in row order, and every block becomes its outputs in the
worker that scored it, so ``predict_batches`` of a store holds about one block per worker
whatever the size of the store. Every check runs before any text or row is read.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from numpy import ndarray

from fse._backends import BACKEND_NAMES, get_backend
from fse._cpus import usable_cpus
from fse._encoder._batches import _named, batch_size, batch_source, check_sink, durable, flush
from fse._encoder._sources import opened
from fse._runtime import check_workers
from fse._store import ShardSink, VectorStore
from fse._store._search import score_blocks

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
    """The outputs as a caller gets them: labels from their indices, one column flattened
    where the head gives (n,)."""
    if output == "labels":
        return model.classes_[rows]
    if output != "proba" and rows.shape[1] == 1 and model._content.flat:
        return rows[:, 0].copy()
    return rows


def infer(model: Any, texts: Any, output: str, workers: Any, backend: Any, sink: Any) -> Any:
    """``output`` of every text of ``texts`` (or row of a store): an array, or None when
    ``sink`` took them (its ``flush`` runs after the last row)."""
    check_workers(workers)
    check_backend(backend)
    if sink is not None:
        check_output_sink(model, sink, output)
    if isinstance(texts, VectorStore):
        parts = []
        for first_row, block in _store_blocks(model, texts, output, workers, backend):
            if sink is None:
                parts.append(block)
            else:
                _commit(sink, first_row, block)
        return _joined(model, parts, output, sink)
    head = model._pipeline(backend, with_vectors=False)
    with opened(texts, model._tokenizer) as source:
        rows, _ = head.infer(source, output=RUNTIME[output], workers=workers, sink=sink)
    if sink is not None:
        flush(sink)
        return None
    return finish(model, rows, output)


def _joined(model: Any, parts: list, output: str, sink: Any) -> Any:
    if sink is not None:
        flush(sink)
        return None
    width = model._head.width(RUNTIME[output])
    dtype = np.int64 if output == "labels" else np.float32
    empty = np.empty((0,) if output == "labels" else (0, width), dtype=dtype)
    return finish(model, np.concatenate(parts) if parts else empty, output)


def _commit(sink: Any, first_row: int, block: ndarray) -> None:
    target = sink.target(first_row, len(block))
    target[...] = block
    sink.commit(first_row, target)


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
    if isinstance(items, VectorStore):
        if return_vectors:
            raise ValueError("a store holds the vectors already; read them with get_vectors")
        if sink is not None:
            check_output_sink(model, sink, output)
        _check_store(model, items)
        return _store_batches(model, items, output, workers, backend, sink)
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
        with (
            _named(k, row, batch_size(k, item)),
            batch_source(item, model._tokenizer) as (
                source,
                n,
            ),
        ):
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


def _check_store(model: Any, store: VectorStore) -> None:
    """A store gives the rows of its encoder as that encoder made them: the model's encoder,
    with the model's flag equal to the encoder's normalization."""
    if model.standalone:
        raise ValueError("a standalone model has no weights for the rows of a store")
    if store.encoder_fingerprint != model.encoder_fingerprint:
        raise ValueError(
            f"the store holds the rows of encoder {store.encoder_fingerprint}, but the model "
            f"reads those of {model.encoder_fingerprint}"
        )
    encoder_normalizes = "normalize@1" in model.encoder.steps
    if model.normalized != encoder_normalizes:
        raise ValueError(
            f"the head reads {'normalized' if model.normalized else 'raw'} rows, but the store "
            f"holds the {'normalized' if encoder_normalizes else 'raw'} rows of its encoder; "
            "predict from the texts instead"
        )


def _store_blocks(model: Any, store: VectorStore, output: str, workers: Any, backend: Any):
    """(first row, outputs) of every block of the store, in row order."""
    _check_store(model, store)
    host = get_backend(backend)
    host.capabilities().check_heads(host.name)
    head = model._head
    kind = RUNTIME[output]

    def outputs(first_row: int, z: ndarray) -> ndarray:
        if kind == "logits":
            return z
        out = np.empty(
            len(z) if kind == "labels" else (len(z), head.width(kind)),
            dtype=np.int64 if kind == "labels" else np.float32,
        )
        host.outputs(z, kind, head.link if kind == "proba" else None, out, first_row)
        return out

    with store._using():
        yield from score_blocks(
            host,
            store._shards,
            head.weight,
            head.bias,
            workers=check_workers(workers) or usable_cpus(),
            then=outputs,
        )


def _store_batches(model, store, output, workers, backend, sink) -> Iterator[PredictedBatch]:
    last = None
    for first_row, block in _store_blocks(model, store, output, workers, backend):
        n = len(block)
        if sink is None:
            yield PredictedBatch(first_row, n, finish(model, block, output), None, None)
            continue
        _commit(sink, first_row, block)
        last = durable(sink, first_row + n, last)
        yield PredictedBatch(first_row, n, None, None, last)
    if sink is not None:
        flush(sink)
