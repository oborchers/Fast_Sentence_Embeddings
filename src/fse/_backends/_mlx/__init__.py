#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The mlx backend (``backend="mlx"``, ``pip install "fse[mlx]"``): the weighted mean in
a Metal kernel on Apple silicon, bit-identical to the cpu backend (ARCHITECTURE.md,
"MLX"). Importing it does not import MLX; making a backend does.
"""

from ._backend import MlxBackend, fallbacks

__all__ = ["MlxBackend", "fallbacks"]
