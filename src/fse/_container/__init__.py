#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Artifacts as named tensors with string metadata, and the containers that store them.

ARCHITECTURE.md, "Artifacts", "safetensors 1" and "NpyContainer 1", specifies the format.
A container writes a mapping of names to numpy arrays plus a flat ``str`` to ``str``
metadata map and opens it again, bit for bit. It adds the integrity keys itself: the
xxh3-128 checksum of every tensor (``fse.checksum.<name>``) and the fingerprint of the
content (``fse.fingerprint``), which does not depend on the container.

Only the standard library, numpy and (in ``_hash``) xxhash are used, and the modules
import each other relatively, so another project can copy this package; from fse they
import only ``UnsupportedArtifactError`` of ``fse._contract``.
"""

from ._common import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    DTYPE_CODES,
    FINGERPRINT_FORM,
    FINGERPRINT_KEY,
    INTEGER,
    CorruptArtifactError,
    TensorContainer,
    canonical_json,
    checksum,
    fingerprint,
    is_fingerprint,
    tensor_code,
)
from ._foreign import read_foreign
from ._hash import Hasher
from ._npy import NpyContainer
from ._safetensors import SafetensorsContainer

__all__ = [
    "CHECKSUM_PREFIX",
    "CONTAINERS",
    "CREATED_BY_KEY",
    "DTYPE_CODES",
    "FINGERPRINT_FORM",
    "FINGERPRINT_KEY",
    "INTEGER",
    "CorruptArtifactError",
    "Hasher",
    "NpyContainer",
    "SafetensorsContainer",
    "TensorContainer",
    "canonical_json",
    "checksum",
    "fingerprint",
    "is_fingerprint",
    "read_foreign",
    "tensor_code",
]

#: Every container this fse can write and read; the golden test runs over all of them.
CONTAINERS: dict[str, TensorContainer] = {
    NpyContainer.name: NpyContainer(),
    SafetensorsContainer.name: SafetensorsContainer(),
}
