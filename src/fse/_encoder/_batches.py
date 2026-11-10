#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Encoding into a sink and in batches: what ``Encoder.encode(sink=)`` and
``Encoder.encode_batches`` run (docs/architecture/runtime.md, "Runtime").

A stream that never ends is encoded batch by batch, each batch a finite list, tuple or 1-D NumPy array (as a
consumer of a queue polls them), so the rows of a batch are finished when the next one
is read. With a sink, every batch is committed to it, numbered on from the rows before;
``durable_rows`` of a batch is the sink's, so a consumer confirms its input up to there.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from fse import _codes
from fse._runtime import (
    LengthCollector,
    LengthSettings,
    Pipeline,
    QuantizedRows,
    check_workers,
    streaming,
    wants_inverse_norms,
)
from fse._stream import StreamSource
from fse._text import is_arrow

from ._doc_ids import (
    check_id_mode,
    check_modes,
    check_switch,
    infer_ids,
    refuse_score_sink,
    split_pair,
)
from ._inputs import check_array
from ._resume import refuse_resume_sink
from ._sources import opened


class EncodedBatch(NamedTuple):
    """One batch of ``encode_batches``: its rows ``first_row`` to ``first_row + n_rows -
    1`` of the run, as an array or ``QuantizedRows`` (None when a sink took them), and the
    rows of the run that were durable when the batch was given (the sink's
    ``durable_rows`` after its ``publish``; None without a sink or for a sink that
    promises nothing). It is a snapshot: the last batch can report fewer rows than the
    run has, as the sink's ``flush`` runs after it. ``vectors`` is the same array as
    ``rows``, named as in ``PredictedBatch`` and ``HeadsBatch``."""

    first_row: int
    n_rows: int
    rows: NDArray[Any] | QuantizedRows | None
    durable_rows: int | None

    @property
    def vectors(self) -> NDArray[Any] | QuantizedRows | None:
        """An alias of ``rows``."""
        return self.rows


def check_sink(
    sink: Any,
    dtype: str | None,
    fingerprint: str,
    dim: int,
    hint: str = "",
    *,
    resumed: bool = False,
) -> str:
    """The dtype of a run into ``sink``, checked against ``dtype`` and the encoder; ``hint``
    ends the message about a wrong width. A sink made with ``resume=True`` is refused unless
    ``resumed`` says the call resumes (``Encoder.encode``)."""
    refuse_score_sink(sink)
    if not resumed:
        refuse_resume_sink(sink)
    if not callable(getattr(sink, "target", None)) or not callable(getattr(sink, "commit", None)):
        raise TypeError(f"the sink must have target and commit methods, not {type(sink).__name__}")
    own = _codes.check_dtype(getattr(sink, "dtype", "float32"))
    if dtype is not None and _codes.check_dtype(dtype) != own:
        raise ValueError(f"dtype {dtype!r} differs from the dtype {own!r} of the sink; pass none")
    takes = getattr(sink, "encoder_fingerprint", None)
    if takes is not None and takes != fingerprint:
        raise ValueError(f"the sink takes the rows of encoder {takes}, not of {fingerprint}")
    if getattr(sink, "dim", None) != dim:
        raise ValueError(
            f"the sink must take rows of {dim} values, not {getattr(sink, 'dim', None)!r}{hint}"
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


def batch_size(k: int, item: Any) -> int | None:
    """The rows of batch ``k``, a list, tuple or 1-D array; None for an Arrow string
    column, whose rows are known once its source is made (no ``len()`` is needed)."""
    if is_arrow(item):
        return None
    if not isinstance(item, list | tuple | np.ndarray):
        raise TypeError(
            f"batch {k} is a {type(item).__name__}; every batch is a list (or tuple) of "
            "sentences, a one-dimensional numpy text array, or an Arrow string column"
        )
    try:
        check_array(item)
    except TypeError as error:
        raise TypeError(f"batch {k}: {error}") from None
    return len(item)


@contextmanager
def batch_source(item: Any, tokenizer: str) -> Iterator[tuple[Any, int]]:
    """The source of one batch and its rows, which the source counts (no ``len()`` of the
    item). An Arrow column whose producer could not export comes as a stream; its texts
    are read now, as the stream would read them, so the batch knows its rows."""
    with opened(item, tokenizer) as source:
        if not isinstance(source, StreamSource):
            yield source, len(source)
            return
    with opened(list(item), tokenizer) as listed:
        yield listed, len(listed)


@contextmanager
def _named(k: int, row: int, n: int | None) -> Iterator[None]:
    """A ValueError or TypeError of batch ``k`` names the batch and its rows of the run
    (from ``row`` on, where its rows are not known yet)."""
    try:
        yield
    except (ValueError, TypeError) as error:
        rows = f"from row {row}" if n is None else f"rows {row} to {row + n - 1}"
        where = f"batch {k} ({rows} of the run)"
        if type(error) not in (ValueError, TypeError):  # such as UnicodeEncodeError: kept
            error.add_note(f"in {where}")
            raise
        raise type(error)(f"{where}: {error}") from None


def run_into(
    pipeline: Pipeline,
    source: Any,
    sink: Any,
    workers: Any,
    first_row: int,
    doc_ids: Any = None,
    lengths: LengthSettings | None = None,
    collector: LengthCollector | None = None,
) -> None:
    """Commit the rows of ``source`` to ``sink`` from ``first_row``; ``lengths`` and
    ``collector`` as in ``Pipeline.infer``."""
    check_id_mode(sink, doc_ids)
    if doc_ids is None:
        pipeline.infer(
            source,
            workers=workers,
            sink=sink,
            first_row=first_row,
            lengths=lengths,
            collector=collector,
        )
    else:
        infer_ids(pipeline, source, sink, workers, first_row, doc_ids, lengths, collector)


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


def check_positions(positions: Any, method: str = "encode_batches") -> None:
    """``positions`` of ``method`` is a bool."""
    if not isinstance(positions, bool):
        raise TypeError(f"{method}: positions must be a bool, not {type(positions).__name__}")


def prepare(
    sink: Any,
    dtype: str | None,
    doc_ids: Any,
    fingerprint: str,
    dim: int,
    workers: Any,
    *,
    positions: bool = False,
) -> str:
    """The dtype of an ``encode_batches`` run; checks the sink, the document-ID mode and the
    workers when the method is called, before any batch is read. A sink made with
    ``resume=True`` takes the batches of a run with ``positions=True`` only."""
    check_switch(doc_ids, "encode_batches")
    if positions and sink is None:
        raise ValueError("positions=True needs a sink made with resume=True")
    if sink is not None:
        dtype = check_sink(sink, dtype, fingerprint, dim, resumed=positions)
    else:
        dtype = _codes.check_dtype("float32" if dtype is None else dtype)
    check_modes([] if sink is None else [("sink", sink)], doc_ids, "encode_batches")
    check_workers(workers)
    return dtype


def _call(sink: Any, name: str) -> None:
    method = getattr(sink, name, None)
    if method is None:
        return
    if not callable(method):
        raise TypeError(f"{name} of the sink must be a method, not {type(method).__name__}")
    method()


@streaming
def batches(
    pipeline: Pipeline,
    items: Iterable[Any],
    tokenizer: str,
    *,
    dtype: str,
    workers: Any,
    sink: Any,
    doc_ids: bool = False,
    lengths: LengthSettings | None = None,
    rule: tuple[int | None, int | None] = (None, None),
) -> Generator[Any, None, None]:
    """The EncodedBatch of every item of ``items``, a finite list, tuple or 1-D NumPy array of sentences (or token
    lists) or an Arrow string column each; with ``lengths`` of a level, the pair of it and
    the ``LengthReport`` of its rows (``rule`` is the encoder's)."""
    row, last = 0, None
    for k, item in enumerate(items):
        ids = None
        if doc_ids:
            item, ids = split_pair(k, item, "encode_batches")
        with _named(k, row, batch_size(k, item)), batch_source(item, tokenizer) as (source, n):
            collector = None if lengths is None else lengths.collector(row, rule)
            if sink is None:
                rows = pipeline.infer(
                    source,
                    workers=workers,
                    dtype=dtype,
                    first_row=row,
                    lengths=lengths,
                    collector=collector,
                )
                found = EncodedBatch(row, n, rows, None)
            else:
                run_into(pipeline, source, sink, workers, row, ids, lengths, collector)
                last = durable(sink, row + n, last)
                found = EncodedBatch(row, n, None, last)
        yield (found, collector.report()) if collector is not None else found
        row += n
    if sink is not None:
        flush(sink)
