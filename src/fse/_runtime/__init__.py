#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The runtime of train and infer: tasks, the executor, sinks and the pipeline.

ARCHITECTURE.md, "Runtime". Private: PR 5's Encoder builds an EncodeSpec and calls a
Pipeline; the public entry points come with it.
"""

from ._count import count_tokens
from ._pipeline import Pipeline, check_workers
from ._sinks import (
    ArrayRows,
    GramSink,
    GrowingRows,
    MemorySink,
    QuantizedRows,
    Rows,
    Sink,
    Stored,
    wants_inverse_norms,
)
from ._spec import EncodeSpec

__all__ = [
    "ArrayRows",
    "EncodeSpec",
    "GramSink",
    "GrowingRows",
    "MemorySink",
    "Pipeline",
    "QuantizedRows",
    "Rows",
    "Sink",
    "Stored",
    "check_workers",
    "count_tokens",
    "wants_inverse_norms",
]
