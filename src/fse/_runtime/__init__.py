#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The runtime of train and infer: tasks, the executor, sinks and the pipeline.

ARCHITECTURE.md, "Runtime". Private: PR 5's Encoder builds an EncodeSpec and calls a
Pipeline; the public entry points come with it.
"""

from ._pipeline import Pipeline
from ._sinks import ArrayRows, GramSink, GrowingRows, Sink, Stored
from ._spec import EncodeSpec

__all__ = ["ArrayRows", "EncodeSpec", "GramSink", "GrowingRows", "Pipeline", "Sink", "Stored"]
