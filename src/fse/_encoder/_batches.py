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
from contextlib import contextmanager
from typing import Any, NamedTuple

import numpy as np

from fse import _codes
from fse._runtime import Pipeline, wants_inverse_norms

from ._corpus import Corpus
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
    wants_inverse_norms(sink)
    for name in ("flush", "publish"):
        method = getattr(sink, name, None)
        if method is not None and not callable(method):
            raise TypeError(f"{name} of the sink must be a method, not {type(method).__name__}")
    rows = getattr(sink, "durable_rows", None)
    if rows is not None and (isinstance(rows, bool) or not isinstance(rows, int | np.integer)):
        raise TypeError(f"durable_rows of the sink must be an int, not {type(rows).__name__}")
    return own


@contextmanager
def _named(k: int, row: int, n: int) -> Iterator[None]:
    """A ValueError or TypeError of batch ``k`` names the batch and its rows of the run."""
    try:
        yield
    except (ValueError, TypeError) as error:
        where = f"batch {k} (rows {row} to {row + n - 1} of the run)"
        if type(error) not in (ValueError, TypeError):  # such as UnicodeEncodeError: kept
            error.add_note(f"in {where}")
            raise
        raise type(error)(f"{where}: {error}") from None


def run_into(pipeline: Pipeline, source: Any, sink: Any, workers: Any, first_row: int) -> None:
    """Commit the rows of ``source`` to ``sink`` from ``first_row``."""
    pipeline.infer(source, workers=workers, sink=sink, first_row=first_row)


def flush(sink: Any) -> None:
    """The sink's ``flush`` after the last commit of a successful run, if it has one."""
    _call(sink, "flush")


def durable(sink: Any, done: int, last: int | None) -> int | None:
    """The sink's durable rows after a batch, once ``publish`` (if it has one) ran: an int
    from ``last`` (the rows durable before) to ``done`` (the rows finished), or None for
    a sink that promises nothing."""
    _call(sink, "publish")
    rows = getattr(sink, "durable_rows", None)
    if rows is None:
        return None
    if isinstance(rows, bool) or not isinstance(rows, int | np.integer):
        raise TypeError(f"durable_rows of the sink must be an int, not {type(rows).__name__}")
    if not (last or 0) <= rows <= done:
        raise ValueError(
            f"durable_rows of the sink is {rows}; after {done} rows it must be between "
            f"{last or 0} (durable before) and {done}"
        )
    return int(rows)


def _call(sink: Any, name: str) -> None:
    method = getattr(sink, name, None)
    if method is None:
        return
    if not callable(method):
        raise TypeError(f"{name} of the sink must be a method, not {type(method).__name__}")
    method()


def batches(
    pipeline: Pipeline,
    items: Iterable[Any],
    tokenizer: str,
    *,
    dtype: str,
    workers: Any,
    sink: Any,
) -> Iterator[EncodedBatch]:
    """The EncodedBatch of every item of ``items``, a finite list of sentences (or token
    lists) each."""
    row, last = 0, None
    for k, item in enumerate(items):
        if not isinstance(item, list | tuple | Corpus):
            raise TypeError(
                f"batch {k} is a {type(item).__name__}; every batch is a list (or tuple) of "
                "sentences or a fse.Corpus"
            )
        n = len(item)
        with _named(k, row, n), opened(item, tokenizer) as source:
            if sink is None:
                found = EncodedBatch(
                    row, n, pipeline.infer(source, workers=workers, dtype=dtype), None
                )
            else:
                run_into(pipeline, source, sink, workers, row)
                last = durable(sink, row + n, last)
                found = EncodedBatch(row, n, None, last)
        yield found
        row += n
    if sink is not None:
        flush(sink)
