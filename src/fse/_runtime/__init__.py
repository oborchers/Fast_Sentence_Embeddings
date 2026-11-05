#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The runtime of encoding: tasks, the executor, sinks and the pipeline.

docs/architecture/runtime.md, "Runtime". Private: PR 5's Encoder builds an EncodeSpec and calls a
Pipeline; the public entry points come with it.
"""

from ._head import OUTPUTS, HeadPipeline, make_head
from ._head_spec import HeadSpec
from ._heads import HeadsPipeline
from ._heads_plan import HeadsPlan, Member, plan_heads
from ._pipeline import Pipeline, check_sink_rows, check_workers
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
    "Member",
    "MemorySink",
    "Pipeline",
    "QuantizedRows",
    "Rows",
    "Sink",
    "Stored",
    "check_sink_rows",
    "check_workers",
    "make_head",
    "plan_heads",
    "wants_inverse_norms",
]
