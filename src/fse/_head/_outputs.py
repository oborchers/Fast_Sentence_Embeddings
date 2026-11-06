#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""What a head gives: the batches of ``predict_batches`` and the identity of its outputs.

The fingerprint of a model alone does not tell its decisions from its probabilities of
the same width; the output identity does: the fingerprint of the model's fingerprint and
the name of the output. A sink that names an ``encoder_fingerprint`` takes the outputs of
that identity only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from numpy import ndarray

from fse._container import fingerprint
from fse._runtime import LengthReport

# The outputs a sink can take, by the name ``output_identity`` gives them.
SINK_OUTPUTS = ("decision", "proba", "values")


class PredictedBatch(NamedTuple):
    """One batch of ``predict_batches``: its rows ``first_row`` to ``first_row + n_rows -
    1`` of the run, their ``outputs`` (labels, probabilities, decisions or values; None
    when a sink took them), their ``vectors`` (only with ``return_vectors=True``, None
    when a sink took them) and the rows of the run that were durable when the batch was
    given (the sink's ``durable_rows``; None without a sink or for one that promises
    nothing)."""

    first_row: int
    n_rows: int
    outputs: Any
    vectors: Any
    durable_rows: int | None


def output_identity(model_fingerprint: str, output: str) -> str:
    """The identity of the rows ``output`` of a model: the fingerprint of the document
    ``{"model": its fingerprint, "output": output}``."""
    if output not in SINK_OUTPUTS:
        raise ValueError(f"output must be one of {', '.join(SINK_OUTPUTS)}, not {output!r}")
    return fingerprint({}, {"model": model_fingerprint, "output": output})


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class HeadsBatch:
    """One batch of ``Heads.predict_batches``: its rows ``first_row`` to ``first_row +
    n_rows - 1`` of the run, the ``outputs`` of every head by name (as ``predict`` of that
    head gives them, or the output asked for), with ``return_vectors=True`` the ``vectors``,
    the rows of the encoder (None if a ``vector_sink`` took them), and with a ``vector_sink``
    or a ``score_sink`` the ``durable_rows``: the rows of the run that were durable when the
    batch was given (the ``durable_rows`` of the sink after its ``publish``, the minimum over
    both sinks, each checked on its own; None without a sink, or if a sink promises nothing). It is a snapshot: the last batch can report
    fewer rows than the run has, as the sink's ``flush`` runs after it; the sink's own
    ``durable_rows`` has the final count.

    ``lengths`` is the ``fse.LengthReport`` of the rows of the batch if the call passed
    ``lengths="summary"`` or ``"per_row"``, else None.

    Read the fields by name: a batch is not iterable and does not unpack, so a field added in
    a later minor release (with a default) breaks no caller."""

    first_row: int
    n_rows: int
    outputs: Mapping[str, ndarray]
    vectors: ndarray | None
    durable_rows: int | None = None
    lengths: LengthReport | None = None
