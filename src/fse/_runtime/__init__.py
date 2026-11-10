#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The runtime of encoding: tasks, the executor, sinks and the pipeline.

docs/architecture/runtime.md, "Runtime". Private: the Encoder builds an EncodeSpec and calls a
Pipeline; the public entry points are those of the Encoder.
"""

from ._head import OUTPUTS, HeadPipeline, make_head
from ._head_spec import HeadSpec
from ._heads import HeadsPipeline
from ._heads_plan import HeadsPlan, Member, plan_heads
from ._lengths import LengthCollector, LengthReport, LengthSettings, rule_of
from ._pipeline import Pipeline, check_sink_rows, check_workers
from ._pool import Pool, Workers, iteration, resolve_workers, streaming, workers_of
from ._sinks import (
    ArrayRows,
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
    "OUTPUTS",
    "ArrayRows",
    "EncodeSpec",
    "GrowingRows",
    "HeadPipeline",
    "HeadSpec",
    "HeadsPipeline",
    "HeadsPlan",
    "LengthCollector",
    "LengthReport",
    "LengthSettings",
    "Member",
    "MemorySink",
    "Pipeline",
    "Pool",
    "QuantizedRows",
    "Rows",
    "Sink",
    "Stored",
    "Workers",
    "check_sink_rows",
    "check_workers",
    "iteration",
    "make_head",
    "plan_heads",
    "resolve_workers",
    "rule_of",
    "streaming",
    "wants_inverse_norms",
    "workers_of",
]
