#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The sign bits of a shard that a ShardSink writes (docs/architecture/store.md, "Sign bits").

With ``ShardSink(sign_bits=True)`` each shard being written holds one ``BitsColumn``: the
commit of the rows, which runs in the hash thread in row order, derives their bits from the
codes it was handed (the rule ``sign_bits@1`` of ``fse._reference_bits``) and keeps them in
memory, 1/32 of the float32 bytes of the shard (1/16 for float16, 1/8 for int8). They are
published only after the shard: ``publish`` runs in the publisher thread once the rows of the
shard are counted, writes ``.sign_bits-<first row>.writing-<hex>`` beside the shard and renames
it into place without replacing anything, with the durability of the sink (``_bits_write``).
A shard without its bits is a valid state, so a stop between the two leaves a valid store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy import ndarray

from fse import _reference_bits

from ._bits_format import width
from ._bits_write import publish_bits
from ._vectors_format import StoreInfo, open_shard


class BitsColumn:
    """The bits of the rows of one shard being written."""

    def __init__(
        self, store: Path, info: StoreInfo, first_row: int, capacity: int, created_by: str
    ):
        self.store, self.info, self.first_row, self.created_by = store, info, first_row, created_by
        self.bits: ndarray | None = np.empty((capacity, width(info.dim)), dtype=np.uint8)

    def commit(self, begin: int, codes: ndarray) -> None:
        """Derive the bits of the committed rows from ``begin`` (in the shard) from ``codes``,
        the rows as stored."""
        assert self.bits is not None
        self.bits[begin : begin + len(codes)] = _reference_bits.sign_bits(codes)

    def publish(self, rows: int, *, synced: bool) -> bool:
        """Publish the bits of the first ``rows`` rows of the shard, which is published:
        True if this call published them, False if a valid artifact was there first. The
        artifact is bound to the shard as it is on disk."""
        assert self.bits is not None
        shard, tensors = open_shard(self.store, self.info, self.first_row)
        try:
            if shard.rows != rows:
                raise RuntimeError(f"{shard.name} holds {shard.rows} rows, not the {rows} written")
            return publish_bits(
                self.store, self.info, shard, self.bits[:rows], synced=synced, by=self.created_by
            )
        finally:
            tensors.clear()  # release the mappings

    def release(self) -> None:
        self.bits = None
