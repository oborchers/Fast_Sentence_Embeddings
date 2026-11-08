"""Stores with signed, multi-byte rows and the oracle of their sign bits, for the tests of the
companion artifact ``sign_bits-<first row>``."""

import numpy as np

from fse import _reference_bits
from fse._container import NpyContainer
from fse._store._layout import StoreInfo, create_store, shard_metadata, shard_name
from support.encoder import finished

DIM = 10  # two bytes per row, six padding bits
ENCODER = finished(dim=DIM)


def signed_vectors(rows, dtype, seed):
    """Distinct signed rows with the edge cases of the rule in the first row: +0.0, -0.0 and a
    positive subnormal (not for int8, where 0 stays 0)."""
    rng = np.random.default_rng(seed)
    if dtype == "int8":
        values = rng.integers(-127, 128, (rows, DIM)).astype(np.int8)
        values[0, :3] = (0, -1, 1)
        return values
    values = rng.standard_normal((rows, DIM)).astype(dtype)
    tiny = np.float32(2.0**-149) if dtype == "float32" else np.float16(2.0**-24)
    values[0, :3] = (np.array(0.0, dtype), np.array(-0.0, dtype), tiny)
    return values


def add_signed_shard(path, info, first, rows, seed):
    """The shard ``first`` of ``rows`` signed rows of ``seed``; returns its rows."""
    vectors = signed_vectors(rows, info.dtype, seed)
    tensors = {"vectors": vectors, "inv_norms": np.full(rows, 0.25, np.float32)}
    if info.dtype == "int8":
        tensors["scales"] = np.full(rows, 0.5, np.float32)
    NpyContainer().write(
        path / shard_name(first), tensors, shard_metadata(info, first, rows, "test")
    )
    return vectors


def signed_store(path, *, dtype="float32", sizes=(5, 3), seed=0):
    """A store of ``sizes`` shards of signed rows; returns its info and the rows of every
    shard by first row."""
    info = StoreInfo(ENCODER.fingerprint, dtype, DIM)
    create_store(path, info)
    first, rows_of = 0, {}
    for number, rows in enumerate(sizes):
        rows_of[first] = add_signed_shard(path, info, first, rows, seed + number)
        first += rows
    return info, rows_of


def expected_bits(vectors):
    """The oracle of the rule: one bit per positive value, from a loop over Python ints."""
    out = np.zeros((len(vectors), -(-vectors.shape[1] // 8)), np.uint8)
    for r, row in enumerate(vectors):
        for d, value in enumerate(row):
            pattern = (
                int(value) if value.dtype == np.int8 else int(value.view(f"i{value.itemsize}"))
            )
            if pattern > 0:
                out[r, d // 8] |= 0x80 >> (d % 8)
    return out


def stored_bits(store):
    """The bits mapped for each shard of an opened store, by first row (None: no artifact)."""
    return {shard.first_row: shard.sign_bits for shard, _ in store._shards}


def check_against_reference(vectors):
    assert np.array_equal(expected_bits(vectors), _reference_bits.sign_bits(vectors))


def tree(path):
    """All entries and file bytes below ``path``."""
    return {
        str(p.relative_to(path)): p.read_bytes() if p.is_file() else None
        for p in sorted(path.rglob("*"))
    }
