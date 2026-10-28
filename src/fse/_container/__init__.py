#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Artifacts as named tensors with string metadata, and the containers that store them.

ARCHITECTURE.md, "Artifacts", "safetensors 1" and "NpyContainer 1", specifies the format.
A container writes a mapping of names to numpy arrays plus a flat ``str`` to ``str``
metadata map and opens it again, bit for bit. It adds the integrity keys itself: the
sha256 of every tensor (``fse.checksum.<name>``) and the fingerprint of the content
(``fse.fingerprint``), which does not depend on the container.

Only the standard library and numpy are used, and the modules import each other
relatively, so the 1.x export can copy this package; from fse they import only
``UnsupportedArtifactError`` of ``fse._contract``.
"""

from ._common import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    DTYPE_CODES,
    FINGERPRINT_KEY,
    INTEGER,
    CorruptArtifactError,
    TensorContainer,
    array_sha256,
    canonical_json,
    fingerprint,
    tensor_code,
)
from ._npy import NpyContainer
from ._safetensors import SafetensorsContainer

__all__ = [
    "CHECKSUM_PREFIX",
    "CONTAINERS",
    "CREATED_BY_KEY",
    "DTYPE_CODES",
    "FINGERPRINT_KEY",
    "INTEGER",
    "CorruptArtifactError",
    "NpyContainer",
    "SafetensorsContainer",
    "TensorContainer",
    "array_sha256",
    "canonical_json",
    "fingerprint",
    "tensor_code",
]

#: Every container this fse can write and read; the golden test runs over all of them.
CONTAINERS: dict[str, TensorContainer] = {
    NpyContainer.name: NpyContainer(),
    SafetensorsContainer.name: SafetensorsContainer(),
}
