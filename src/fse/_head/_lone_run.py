#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``predict_batches`` of a single head (docs/architecture/runtime.md, "Runtime"; scores.md,
"A single head").

A single head is a group of one: it runs through the same loop and the same sink checks as
``fse.Heads`` (``_sink_run``), on the runtime of its own, already cached plan. Three keywords
reach the sinks of that run: ``vector_sink``, ``score_sink`` and ``doc_ids``. ``sink=`` keeps
the role and the behavior it has had since 2.0, whatever the sink is (``_run``): a call that
gives none of the three keywords runs exactly as it did, and every check of 2.0 is made in
the order of 2.0.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from fse._encoder._doc_ids import check_switch
from fse._runtime import LengthSettings, check_workers
from fse._store import VectorStore

from . import _run
from ._outputs import PredictedBatch
from ._positions_run import start as start_stream
from ._positions_run import stream_batches
from ._run import RUNTIME, check_backend, finish, refuse_store_lengths, rule
from ._sink_run import run_batches, start, start_lone


def batches(
    model: Any,
    items: Any,
    output: str,
    return_vectors: bool,
    workers: Any,
    backend: Any,
    sink: Any,
    vector_sink: Any = None,
    score_sink: Any = None,
    doc_ids: Any = False,
    settings: LengthSettings | None = None,
    positions: bool = False,
) -> Generator[Any, None, None]:
    """The checks of ``predict_batches`` now, then its batches as a generator. With a
    ``score_sink`` the run attaches to it here, after every argument is checked and before
    the first batch is read. With ``positions=True`` the batches are ``(texts, position)``
    and the sinks resume (``_positions_run``)."""
    method = f"{type(model).__name__}.predict_batches"
    check_workers(workers)
    check_backend(backend)
    check_switch(doc_ids, method)
    new = vector_sink is not None or score_sink is not None or doc_ids or positions
    if sink is not None and new:
        raise ValueError(
            "sink= keeps its role: the outputs, or with return_vectors=True the vectors. Give "
            "the vectors as vector_sink=, the scores as score_sink=, the document IDs as "
            "doc_ids=True and the positions of a stream as positions=True, without sink="
        )
    if not new:
        return _run.batches(
            model, items, output, return_vectors, workers, backend, sink, settings, method
        )
    if isinstance(items, VectorStore):
        refuse_store_lengths(settings, method)
        raise (
            ValueError(
                "vector_sink, score_sink and doc_ids take texts; the outputs of a store come from "
                "predict_batches(store) and go to sink="
            )
            if not return_vectors
            else ValueError("a store holds the vectors already; read them with get_vectors")
        )
    if return_vectors and model.standalone:
        raise ValueError("a standalone model has no vectors: its table is folded")
    kinds = (RUNTIME[output],)
    run = start_lone(
        model,
        kinds,
        return_vectors,
        vector_sink,
        score_sink,
        doc_ids,
        method,
        start_stream if positions else start,
    )
    pipeline = model._pipeline(backend, with_vectors=return_vectors, workers=workers).runtime

    def make(row: int, n: int, rows: list, vectors: Any, last: int | None, report: Any) -> Any:
        batch = PredictedBatch(row, n, finish(model, rows[0], output), vectors, last)
        return batch if report is None else (batch, report)

    if positions:
        return stream_batches(
            pipeline, iter(items), run, workers, method, make, settings, rule(model)
        )
    return run_batches(
        model._tokenizer,
        pipeline,
        iter(items),
        kinds,
        workers,
        run,
        doc_ids,
        method,
        make,
        settings,
        rule(model),
    )
