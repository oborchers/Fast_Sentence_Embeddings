#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The encoder of fse 2.0 and its recipes (ARCHITECTURE.md, "2.0 design")."""

from ._encoder import Encoder, QuantizedRows
from ._recipes import SIF, USIF, Average

__all__ = ["SIF", "USIF", "Average", "Encoder", "QuantizedRows"]
