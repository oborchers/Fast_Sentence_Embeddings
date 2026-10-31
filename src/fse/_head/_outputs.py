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

from typing import Any, NamedTuple

from fse._container import fingerprint

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
