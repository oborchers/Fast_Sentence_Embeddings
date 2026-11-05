#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Linear heads on finished encoders: fse.Classifier and fse.Regressor (docs/architecture/formats.md, "linear_head 1"). fse imports heads trained elsewhere and trains nothing."""

from ._classifier import Classifier
from ._heads import Heads
from ._outputs import HeadsBatch, PredictedBatch
from ._regressor import Regressor

__all__ = ["Classifier", "Heads", "HeadsBatch", "PredictedBatch", "Regressor"]
