#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The encoder of fse 2.0 (docs/architecture/overview.md, "Principle and stages")."""

from fse._runtime import QuantizedRows

from ._batches import EncodedBatch
from ._encoder import Encoder

__all__ = ["EncodedBatch", "Encoder", "QuantizedRows"]
