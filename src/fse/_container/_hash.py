#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The one hash of the artifacts: xxh3-128 (ARCHITECTURE.md, "Identities").

Every checksum and fingerprint is ``xxh3-128:`` and the 32 lowercase hex digits of XXH3
128 with seed 0, in the canonical big-endian form, of bytes fed in order. This module only
streams and formats; reading arrays in bounded chunks belongs to the containers. It is the
only module of fse that imports a hash library (``hash-importers`` of the guardrails) and
imports nothing from fse, so it can be copied with the containers.

It also gives the two digests the Hugging Face Hub pins a file of a commit with: the sha256
of an LFS file and the git blob sha1 of any file; they identify a downloaded file against a
trusted commit and are no checksums of fse. Loading computes the one a file is pinned by
(``pin_digest``), pushing both (``file_digests``).

The data always go through the streaming object: python-xxhash releases the GIL in
``update`` (for inputs over 64 KiB from 4.0 on), never in its one-shot functions before 4.0.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

import numpy as np
import xxhash

ALGORITHM = "xxh3-128"
PREFIX = ALGORITHM + ":"
#: The length of every value, so the safetensors writer can reserve its header.
PLACEHOLDER = PREFIX + "0" * 32

_VALUE = re.compile(r"xxh3-128:[0-9a-f]{32}")
# The sha256 values of the pre-release 2.0 format: bare checksums and fingerprints.
_LEGACY = re.compile(r"(sha256:)?[0-9a-f]{64}")
# A value of another algorithm: a name that is not ours, a colon and hex digits.
_OTHER = re.compile(r"(?!xxh3-128:|sha256:)[a-z0-9][a-z0-9_-]*:[0-9a-f]+")


def _buffer(data: object) -> bytes | bytearray | memoryview:
    """The bytes of ``data``: bytes, bytearray, a C-contiguous memoryview or a C-contiguous
    little-endian (or single-byte) array; anything else raises, whatever xxhash accepts."""
    if isinstance(data, bytes | bytearray):
        return data
    if isinstance(data, memoryview):
        if not data.c_contiguous:
            raise ValueError("only C-contiguous memory can be hashed")
        if data.nbytes == 0:  # cast refuses a shape with a zero
            return b""
        return data.cast("B") if data.format != "B" or data.ndim != 1 else data
    if isinstance(data, np.ndarray):
        if not data.flags.c_contiguous:
            raise ValueError("only C-contiguous arrays can be hashed")
        if data.dtype.hasobject or data.dtype.str[0] == ">":
            raise ValueError(
                f"arrays of {data.dtype.str} cannot be hashed; the format is little-endian"
            )
        return memoryview(data.reshape(-1).view(np.uint8))
    raise TypeError(f"bytes or a numpy array can be hashed, not {type(data).__name__}")


class Hasher:
    """An incremental checksum of the bytes fed to ``update``, in order."""

    __slots__ = ("_state",)

    def __init__(self):
        self._state = xxhash.xxh3_128()

    def update(self, data: object):
        self._state.update(_buffer(data))

    def checksum(self) -> str:
        """The checksum of everything fed so far; more data may follow."""
        return PREFIX + self._state.hexdigest()


def checksum(data: object) -> str:
    """The checksum of ``data`` (see ``Hasher.update`` for what it takes)."""
    hasher = Hasher()
    hasher.update(data)
    return hasher.checksum()


#: The checksum of no bytes: that of every tensor without elements.
EMPTY = checksum(b"")


def classify(value: object) -> Literal["ok", "legacy", "unknown", "malformed"]:
    """Whether ``value`` is a checksum of this format, a sha256 one of the pre-release
    format, one of another algorithm, or none at all."""
    if not isinstance(value, str):
        return "malformed"
    if _VALUE.fullmatch(value):
        return "ok"
    if _LEGACY.fullmatch(value):
        return "legacy"
    if _OTHER.fullmatch(value):
        return "unknown"
    return "malformed"


PinKind = Literal["sha256", "git_sha1"]


def pin_digest(data: bytes | bytearray | memoryview, kind: PinKind) -> str:
    """The one digest ``kind`` of ``data`` as hex, hashed in place (no copy): the sha256,
    or the git blob sha1 (``sha1(b"blob <size>\\0" + data)``). Exactly one algorithm runs."""
    view = memoryview(data).cast("B")
    if kind == "sha256":
        return hashlib.sha256(view).hexdigest()
    if kind == "git_sha1":
        sha1 = hashlib.sha1(b"blob %d\0" % view.nbytes)  # the id git gives the content
        sha1.update(view)
        return sha1.hexdigest()
    raise ValueError(f"unknown digest kind {kind!r}")


def file_digests(handle, size: int, chunk: int = 2**22) -> tuple[str, str]:
    """The sha256 and the git blob sha1 (``sha1(b"blob <size>\\0" + data)``) of the
    ``size`` bytes read from ``handle``, both as hex; raises if the file is shorter."""
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(b"blob %d\0" % size)  # the id git gives the content, not security
    remaining = size
    while remaining:
        part = handle.read(min(chunk, remaining))
        if not part:
            raise ValueError(f"the file ended {remaining} bytes early")
        sha256.update(part)
        sha1.update(part)
        remaining -= len(part)
    return sha256.hexdigest(), sha1.hexdigest()
