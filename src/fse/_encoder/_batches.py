#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Encoding into a sink and in batches: what ``Encoder.encode(sink=)`` and
``Encoder.encode_batches`` run (ARCHITECTURE.md, "Runtime").

A stream that never ends is encoded batch by batch, each batch a finite list (as a
consumer of a queue polls them), so the rows of a batch are finished when the next one
is read. With a sink, every batch is committed to it, numbered on from the rows before;
``durable_rows`` of a batch is the sink's, so a consumer confirms its input up to there.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, NamedTuple

from fse import _codes
from fse._runtime import Pipeline

from ._sources import opened


class EncodedBatch(NamedTuple):
    """One batch of ``encode_batches``: its rows ``first_row`` to ``first_row + n_rows -
    1`` of the run, as an array or ``QuantizedRows`` (None when a sink took them), and the
    rows of the run that are durable (the sink's ``durable_rows``; None without a sink or
    for a sink that promises nothing)."""

    first_row: int
    n_rows: int
    rows: Any
    durable_rows: int | None


def check_sink(sink: Any, dtype: str | None, fingerprint: str, dim: int) -> str:
    """The dtype of a run into ``sink``, checked against ``dtype`` and the encoder."""
    if not callable(getattr(sink, "target", None)) or not callable(getattr(sink, "commit", None)):
        raise TypeError(f"the sink must have target and commit methods, not {type(sink).__name__}")
    own = _codes.check_dtype(getattr(sink, "dtype", "float32"))
    if dtype is not None and _codes.check_dtype(dtype) != own:
        raise ValueError(f"dtype {dtype!r} differs from the dtype {own!r} of the sink; pass none")
    encoder = getattr(sink, "encoder", None)
    if encoder is not None and encoder != fingerprint:
        raise ValueError(f"the sink takes the rows of encoder {encoder}, not of {fingerprint}")
    if getattr(sink, "dim", None) != dim:
        raise ValueError(
            f"the sink must take rows of {dim} values, not {getattr(sink, 'dim', None)!r}"
        )
    return own


def run_into(pipeline: Pipeline, source: Any, sink: Any, workers: Any, first_row: int) -> None:
    """Commit the rows of ``source`` to ``sink`` from ``first_row``."""
    pipeline.infer(source, workers=workers, sink=sink, first_row=first_row)


def flush(sink: Any) -> None:
    """The sink's ``flush`` after the last commit of a successful run, if it has one."""
    method = getattr(sink, "flush", None)
    if callable(method):
        method()


def durable(sink: Any) -> int | None:
    """The sink's durable rows after a batch: published by ``publish`` if it has one."""
    publish = getattr(sink, "publish", None)
    if callable(publish):
        publish()
    rows = getattr(sink, "durable_rows", None)
    return rows if isinstance(rows, int) else None


def batches(
    pipeline: Pipeline,
    items: Iterable[Any],
    tokenizer: str,
    *,
    dtype: str,
    workers: Any,
    sink: Any,
    wrap: Any,
) -> Iterator[EncodedBatch]:
    """The EncodedBatch of every item of ``items``, a finite list of sentences (or token
    lists) each; ``wrap`` turns int8 rows into QuantizedRows."""
    row = 0
    for k, item in enumerate(items):
        if not isinstance(item, list | tuple):
            raise TypeError(
                f"batch {k} is a {type(item).__name__}; every batch is a list (or tuple) of sentences"
            )
        with opened(item, tokenizer) as source:
            if sink is None:
                rows = wrap(pipeline.infer(source, workers=workers, dtype=dtype))
                yield EncodedBatch(row, len(item), rows, None)
            else:
                run_into(pipeline, source, sink, workers, row)
                yield EncodedBatch(row, len(item), None, durable(sink))
        row += len(item)
    if sink is not None:
        flush(sink)
