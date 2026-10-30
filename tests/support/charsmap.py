"""Precompiled charsmaps for tests: a small builder of the darts-clone layout that
spm_precompiled 0.1.4 (and so tokenizers' ``normalizers.Precompiled``) reads.

Every node gets its own block of 256 units; ``share=True`` gives equal subtrees one
block, as darts-clone does for equal suffixes, so a node is reached on several paths.
"""

from __future__ import annotations

import struct


def _trie(mapping: dict[str, str]) -> tuple[dict, bytes]:
    values = bytearray()
    starts: dict[str, int] = {}  # equal values are stored once
    trie: dict = {}
    for key in sorted(mapping):
        node = trie
        for byte in key.encode():
            node = node.setdefault(byte, {})
        value = mapping[key]
        if value not in starts:
            starts[value] = len(values)
            values += value.encode() + b"\0"
        node[None] = starts[value]
    return trie, bytes(values)


def _signature(node: dict) -> tuple:
    return tuple(
        sorted(
            (label, child.get(None), _signature(child))
            for label, child in node.items()
            if label is not None
        )
    )


class _Builder:
    def __init__(self, share: bool):
        self.units = [0] * 256
        self.blocks: dict[tuple, int] = {}
        self.share = share

    def block(self, child: dict) -> tuple[int, bool]:
        """The block of ``child``'s children: a shared one, or a new one."""
        key = (*_signature(child), child.get(None))
        if self.share and key in self.blocks:
            return self.blocks[key], False
        below = len(self.units)
        self.units.extend([0] * 256)
        self.blocks[key] = below
        return below, True

    def place(self, node: dict, base: int) -> None:
        for label, child in node.items():
            if label is None:
                continue
            pos = base ^ label
            below, fresh = self.block(child)
            leaf = None in child
            self.units[pos] = label | (int(leaf) << 8) | ((pos ^ below) << 10)
            if leaf:
                self.units[below] = child[None] | (1 << 31)
            if fresh:
                self.place(child, below)


def build(mapping: dict[str, str], *, share: bool = False) -> bytes:
    """The blob of a charsmap that maps every key of ``mapping`` to its value."""
    trie, values = _trie(mapping)
    builder = _Builder(share)
    root = len(builder.units)
    builder.units.extend([0] * 256)
    builder.units[0] = root << 10
    builder.place(trie, root)
    units = builder.units
    return struct.pack("<I", 4 * len(units)) + struct.pack(f"<{len(units)}I", *units) + values


def units_of(blob: bytes) -> list[int]:
    """The units of a blob, to break it in tests."""
    (size,) = struct.unpack_from("<I", blob)
    return list(struct.unpack_from(f"<{size // 4}I", blob, 4))


def with_units(blob: bytes, units: list[int]) -> bytes:
    """``blob`` with other units (as many as before) and the same values."""
    (size,) = struct.unpack_from("<I", blob)
    return blob[:4] + struct.pack(f"<{len(units)}I", *units) + blob[4 + size :]
