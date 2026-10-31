#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The precompiled charsmap of a SentencePiece normalizer, as tokenizers reads it.

docs/architecture/tokenizers.md, "unigram@1". The blob of spm_precompiled 0.1.4: a little-endian u32
``size``, ``size / 4`` little-endian u32 units of a darts-clone double array, then the
values, UTF-8 strings each ended by NUL. ``find`` searches the array with the bytes of a
key and gives the value of the shortest key that is a prefix of them. The blob is checked
completely when it is read, because a broken one makes tokenizers panic: the header,
the values, and every node reachable from the root (shared nodes once, a cycle refused),
so no search can leave the array. Errors are ValueError; readers name the file.
"""

from __future__ import annotations

import numpy as np

_LABEL_MASK = (1 << 31) | 0xFF
_VALUE_MASK = (1 << 31) - 1
_LABELS = np.arange(1, 256, dtype=np.int64)


def _offset(unit):
    return (unit >> 10) << ((unit & (1 << 9)) >> 6)


class Charsmap:
    """A checked charsmap; ``blob`` is the decoded bytes of the file."""

    __slots__ = ("_cache", "_root", "_units", "_values", "blob")

    def __init__(self, blob: bytes):
        if not isinstance(blob, bytes):
            raise TypeError(f"a charsmap is bytes, not {type(blob).__name__}")
        if len(blob) < 4:
            raise ValueError(f"a charsmap needs at least 4 bytes, not {len(blob)}")
        size = int.from_bytes(blob[:4], "little")
        if size == 0 or size % 4 or 4 + size > len(blob):
            raise ValueError(
                f"the charsmap's array has {size} bytes, which is no positive multiple of 4 "
                f"within its {len(blob) - 4} bytes"
            )
        units = np.frombuffer(blob, dtype="<u4", count=size // 4, offset=4).astype(np.int64)
        values = blob[4 + size :]
        try:
            values.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"the values of the charsmap are no UTF-8: {error}") from None
        self._root = int(_offset(units[0]))
        _check(units, values, self._root)
        self.blob = blob
        self._units = units.tolist()
        self._values = values
        self._cache: dict[int, str] = {}

    def parts(self) -> tuple[np.ndarray, bytes, int]:
        """The units (uint32), the values and the root of the checked blob, as the native
        tokenizer takes them."""
        return np.array(self._units, dtype=np.uint32), self._values, self._root

    def find(self, key: bytes) -> str | None:
        """The value of the shortest key that is a prefix of ``key``, or None. The search
        ends at a byte 0, as tokenizers' does."""
        units, p = self._units, self._root
        for c in key:
            if c == 0:
                return None
            p ^= c
            unit = units[p]
            if unit & _LABEL_MASK != c:
                return None
            p ^= _offset(unit)
            if (unit >> 8) & 1:
                return self._value(units[p] & _VALUE_MASK)
        return None

    def _value(self, start: int) -> str:
        value = self._cache.get(start)
        if value is None:
            end = self._values.find(b"\0", start)
            value = self._values[start : len(self._values) if end < 0 else end].decode("utf-8")
            self._cache[start] = value
        return value


def _children(units: np.ndarray, values: bytes, base: int) -> list[int]:
    """The nodes below the node at ``base``, every probe and value checked."""
    if base | 0xFF >= len(units):
        raise ValueError(f"a node of the charsmap at {base} probes outside its {len(units)} units")
    pos = base ^ _LABELS
    child = pos[(units[pos] & _LABEL_MASK) == _LABELS]
    below = child ^ _offset(units[child])
    if (below >= len(units)).any():
        raise ValueError("a node of the charsmap points outside the array")
    for start in (units[below[((units[child] >> 8) & 1) == 1]] & _VALUE_MASK).tolist():
        if start > len(values) or (start < len(values) and 0x80 <= values[start] < 0xC0):
            raise ValueError(
                f"a value of the charsmap starts at {start}, which is no code point boundary "
                f"within its {len(values)} bytes of values"
            )
    return below.tolist()


def _check(units: np.ndarray, values: bytes, root: int) -> None:
    """Every node reachable from the root, once each, by depth-first search; a node on
    the current path reached again is a cycle."""
    done: set[int] = set()
    on_path = {root}
    stack = [(root, iter(_children(units, values, root)))]
    while stack:
        base, below = stack[-1]
        node = next(below, None)
        if node is None:
            stack.pop()
            on_path.discard(base)
            done.add(base)
        elif node in on_path:
            raise ValueError("the charsmap has a cycle")
        elif node not in done:
            on_path.add(node)
            stack.append((node, iter(_children(units, values, node))))
