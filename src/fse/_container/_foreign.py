#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Files of other writers (Model2Vec), read once and strictly.

``read_regular`` reads a regular file (never a link or FIFO) through one handle into
immutable bytes, within a size limit, and raises if it changes meanwhile, so the bytes a
caller checks are the bytes it parses. ``foreign_tensors`` parses a safetensors file of
another writer from such bytes with the structural checks of safetensors 1
(docs/architecture/formats.md, "safetensors 1"): a header of at most 16 MiB within the file, a JSON object without
duplicate keys or NaN, exactly dtype, shape and data offsets per tensor, the dtypes of the
format only, offsets that fit the shape, tensors aligned to their type and covering the
data without gaps up to the end of the file. What other writers lack is not required:
``__metadata__`` may be absent and is ignored (it must still map strings to strings), and
there are no checksums. ``foreign_header`` checks the header alone, from a handle, so a
caller can refuse a file by its tensors before its data are read.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy import ndarray

from ._common import DTYPE_CODES, CorruptArtifactError, _open_regular
from ._safetensors import (
    _SAFETENSORS_LENGTH,
    _check_coverage,
    _header_json,
    _identity,
    _tensor_entry,
)

#: name -> (dtype code, shape, begin, end) of the data of every tensor of a header.
Entries = dict[str, tuple[str, tuple[int, ...], int, int]]


def read_regular(path: Path, *, limit: int | None = None, inspect: Callable | None = None) -> bytes:
    """The bytes of the regular file ``path``, read once through one handle: a link, a
    FIFO or anything else raises, so does a file of more than ``limit`` bytes or one that
    changes while it is read. ``inspect(handle, size)`` may read and check a prefix of the
    file first (such as a header); the bytes are then read from the start."""
    path = Path(path)
    with _open_regular(path) as handle:
        before = _identity(handle)
        size = before[0]
        if limit is not None and size > limit:
            raise CorruptArtifactError(
                f"{path} has {size} bytes, more than the {limit} it may have"
            )
        if inspect is not None:
            inspect(handle, size)
            handle.seek(0)
        data = handle.read(size)
        if len(data) != size or handle.read(1) or _identity(handle) != before:
            raise CorruptArtifactError(f"{path} changed while it was read")
    return data


def foreign_header(handle, path: Path, size: int) -> tuple[int, Entries]:
    """The offset of the data and the checked tensors of a safetensors file of ``size``
    bytes, read from the start of ``handle`` (the header only)."""
    length, header = _header_json(handle, path, size)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
    ):
        raise CorruptArtifactError(f"{path} has __metadata__ that is no map of strings")
    start = _SAFETENSORS_LENGTH + length
    if start % 8:
        raise CorruptArtifactError(f"the data of {path} starts at {start}, not at a multiple of 8")
    entries = {name: _tensor_entry(path, name, entry) for name, entry in header.items()}
    _check_coverage(path, entries, start, size)
    return start, entries


def foreign_tensors(data: bytes, path: Path) -> dict[str, ndarray]:
    """The read-only tensors of a safetensors file of another writer, given as its bytes;
    ``path`` names it in errors."""
    start, entries = foreign_header(io.BytesIO(data), Path(path), len(data))
    tensors = {}
    for name, (code, shape, begin, end) in entries.items():
        dtype = np.dtype(DTYPE_CODES[code])
        array = (
            np.frombuffer(data, dtype=dtype, count=math.prod(shape), offset=start + begin).reshape(
                shape
            )
            if end > begin
            else np.empty(shape, dtype=dtype)
        )
        array.setflags(write=False)
        tensors[name] = array
    return tensors


def read_foreign(path: Path) -> dict[str, ndarray]:
    """The read-only tensors of a safetensors file of another writer."""
    return foreign_tensors(read_regular(path), path)
