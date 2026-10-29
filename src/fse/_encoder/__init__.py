#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The encoder of fse 2.0 (ARCHITECTURE.md, "2.0 design")."""

from ._batches import EncodedBatch
from ._corpus import Corpus
from ._encoder import Encoder, QuantizedRows

__all__ = ["Corpus", "EncodedBatch", "Encoder", "QuantizedRows"]
