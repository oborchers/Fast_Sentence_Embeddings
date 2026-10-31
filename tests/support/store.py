"""Small stores and valid manifest edits for recovery and repair tests."""

import secrets

import numpy as np

from fse._container import CHECKSUM_PREFIX, FINGERPRINT_KEY, NpyContainer, checksum, tensor_code
from fse._container._npy import write_manifest
from fse._store._layout import StoreInfo, create_store, shard_metadata, shard_name
from support.encoder import finished

ENCODER = finished(dim=4)


def make_store(path, *, dtype="float32", sizes=(4, 4, 2), tagged=()):
    """Write known, distinct rows, then tag the requested shard starts as unsynced."""
    info = StoreInfo(ENCODER.fingerprint, dtype, ENCODER.output_dim)
    create_store(path, info)
    first = 0
    for rows in sizes:
        add_shard(path, info, first, rows)
        if first in tagged:
            tag_shard(path / shard_name(first))
        first += rows
    return info


def add_shard(path, info, first, rows):
    """One structurally valid shard with its checksums and fingerprint."""
    codes = np.arange(first * info.dim + 1, (first + rows) * info.dim + 1).reshape(rows, info.dim)
    tensors = {
        "vectors": codes.astype(info.dtype),
        "inv_norms": np.full(rows, 0.25, np.float32),
    }
    if info.dtype == "int8":
        tensors["scales"] = np.full(rows, 0.5, np.float32)
    NpyContainer().write(
        path / shard_name(first), tensors, shard_metadata(info, first, rows, "test")
    )


def tag_shard(path, value="none"):
    """Rewrite only the manifest, recomputing its fingerprint for the durability tag."""
    change_metadata(path, {"fse.durability": value})


def change_metadata(path, changes):
    """Change metadata through the real manifest writer and recompute its fingerprint."""
    tensors, metadata = NpyContainer().open(path, mmap=False)
    metadata = {
        key: val
        for key, val in metadata.items()
        if key != FINGERPRINT_KEY and not key.startswith(CHECKSUM_PREFIX)
    }
    metadata.update(changes)
    entries = {
        name: (tensor_code(tensor), tensor.shape, checksum(tensor))
        for name, tensor in tensors.items()
    }
    tmp = path / f".manifest.json.tagging-{secrets.token_hex(4)}"
    write_manifest(path, entries, metadata, tmp)


def append_rows(path, *, value=99, rows=4, durability="full"):
    """Append distinguishable rows through the real full writer, returning their start."""
    from fse import ShardSink
    from fse._store._layout import read_store

    info = read_store(path)
    with ShardSink(path, ENCODER, dtype=info.dtype, shard_rows=rows, durability=durability) as sink:
        first = sink.first_row
        target = sink.target(0, rows)
        target[0][...] = value
        if info.dtype == "int8":
            target[1][...] = 1
        target[-1][...] = 0.25
        sink.commit(0, target)
    return first
