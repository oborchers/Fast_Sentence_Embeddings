#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""safetensors files of other writers (such as Model2Vec), read strictly.

The same checks of the structure as safetensors 1 (ARCHITECTURE.md): a header of at most
16 MiB within the file, a JSON object without duplicate keys or NaN, exactly dtype, shape
and data offsets per tensor, the dtypes of the format only, offsets that fit the shape,
tensors aligned to their type and covering the data without gaps up to the end of the
file. What other writers lack is not required: ``__metadata__`` may be absent and is
ignored (it must still map strings to strings), and there are no checksums. The file is
read whole into immutable bytes, through one handle of a regular file.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy import ndarray

from ._common import DTYPE_CODES, CorruptArtifactError, _open_regular
from ._safetensors import (
    _SAFETENSORS_LENGTH,
    _check_coverage,
    _header_json,
    _identity,
    _read_data,
    _tensor_entry,
)


def read_foreign(path: Path) -> dict[str, ndarray]:
    """The read-only tensors of a safetensors file of another writer."""
    path = Path(path)
    with _open_regular(path) as handle:
        before = _identity(handle)
        size = before[0]
        length, header = _header_json(handle, path, size)
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
        ):
            raise CorruptArtifactError(f"{path} has __metadata__ that is no map of strings")
        start = _SAFETENSORS_LENGTH + length
        if start % 8:
            raise CorruptArtifactError(
                f"the data of {path} starts at {start}, not at a multiple of 8"
            )
        entries = {name: _tensor_entry(path, name, entry) for name, entry in header.items()}
        _check_coverage(path, entries, start, size)
        data = _read_data(handle, path, (start, size - start), False, before)
    tensors = {}
    for name, (code, shape, begin, end) in entries.items():
        dtype = np.dtype(DTYPE_CODES[code])
        array = (
            np.frombuffer(data, dtype=dtype, count=math.prod(shape), offset=begin).reshape(shape)
            if end > begin
            else np.empty(shape, dtype=dtype)
        )
        array.setflags(write=False)
        tensors[name] = array
    return tensors
