#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Linear heads on finished encoders: fse.Classifier and fse.Regressor (docs/architecture/formats.md, "linear_head 1"). fse imports heads trained elsewhere and trains nothing."""

from ._classifier import Classifier
from ._outputs import PredictedBatch
from ._regressor import Regressor

__all__ = ["Classifier", "PredictedBatch", "Regressor"]
