#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The mlx backend (``backend="mlx"``, ``pip install "fse[mlx]"``): the weighted mean in
a Metal kernel on Apple silicon, within the tolerance contract of device backends
(docs/architecture/backends.md, "MLX"). ``_load`` imports MLX at its top, and the device thread and the backend import
``_load``; only ``get_backend("mlx")`` imports the backend, so nothing else loads MLX.
"""
