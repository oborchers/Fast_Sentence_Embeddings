"""Writing the sign bits of a published shard: derived from the mapped vectors, hashed on the way,
published whole or not at all."""

import numpy as np
import pytest

import fse
from fse._container import NpyContainer, _npy, _npy_inplace
from fse._store import _bits_write
from fse._store._bits_format import bits_name, open_bits
from fse._store._layout import read_store, shard_name
from fse._store._recovery import snapshot
from support.bits import expected_bits, signed_store, tree


def opened(path):
    info = read_store(path)
    return info, snapshot(path, info, verify=True).shards


def test_the_bits_are_derived_from_the_vectors_in_blocks(tmp_path, monkeypatch):
    path = tmp_path / "store"
    _, rows_of = signed_store(path, sizes=(9,))
    _, shards = opened(path)
    (shard, tensors) = shards[0]
    monkeypatch.setattr(_bits_write, "READ_BYTES", 3 * 10 * 4)  # three rows a block
    bits = _bits_write.derived(shard, tensors["vectors"])
    assert np.array_equal(bits, expected_bits(rows_of[0]))
    assert bits.dtype == np.uint8


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_the_rule_is_the_integer_pattern_whatever_the_float_environment(tmp_path, dtype):
    """+0.0, -0.0 and a positive subnormal: 0, 0 and 1, whatever the thread's FTZ and DAZ
    (the rule reads the pattern, never compares floats)."""
    path = tmp_path / "store"
    _, rows_of = signed_store(path, dtype=dtype, sizes=(4,))
    _, shards = opened(path)
    shard, tensors = shards[0]
    bits = _bits_write.derived(shard, tensors["vectors"])
    first = int(bits[0, 0])
    # +0.0 and -0.0 (codes 0 and -1 for int8) give 0; the positive subnormal (code 1) gives 1
    want = (0, 0, 1)
    assert tuple((first >> (7 - d)) & 1 for d in range(3)) == want
    assert np.array_equal(bits, expected_bits(rows_of[0]))


def test_vectors_that_changed_since_the_shard_was_written_publish_nothing(tmp_path):
    path = tmp_path / "store"
    signed_store(path, sizes=(5, 3))
    file = path / shard_name(0) / "vectors.npy"
    data = bytearray(file.read_bytes())
    data[-1] ^= 0x40  # an untagged shard is not hashed when it is opened
    file.write_bytes(bytes(data))
    with pytest.raises(fse.CorruptArtifactError, match="do not match their checksum"):
        fse.add_sign_bits(path, workers=1)
    assert not list(path.glob("*sign_bits-*"))


def test_a_second_writer_that_was_faster_is_not_an_error_if_its_artifact_is_valid(tmp_path):
    path = tmp_path / "store"
    signed_store(path, sizes=(5,))
    info, shards = opened(path)
    shard, tensors = shards[0]
    args = (path, info, shard, tensors["vectors"])
    assert _bits_write.write_bits(*args, synced=True, by="one") is True
    first = tree(path / bits_name(0))
    assert _bits_write.write_bits(*args, synced=True, by="two") is False
    assert tree(path / bits_name(0)) == first
    assert not list(path.glob(".sign_bits-*"))


def test_a_faster_writer_with_a_damaged_artifact_is_an_error(tmp_path):
    path = tmp_path / "store"
    signed_store(path, sizes=(5,))
    info, shards = opened(path)
    shard, tensors = shards[0]
    args = (path, info, shard, tensors["vectors"])
    _bits_write.write_bits(*args, synced=True, by="one")
    (path / bits_name(0) / "manifest.json").write_text("{}")
    with pytest.raises(fse.CorruptArtifactError):
        _bits_write.write_bits(*args, synced=True, by="two")
    assert not list(path.glob(".sign_bits-*"))


def test_a_failed_publication_removes_the_temporary_directory(tmp_path, monkeypatch):
    path = tmp_path / "store"
    signed_store(path, sizes=(5,))
    info, shards = opened(path)
    shard, tensors = shards[0]

    def refuse(*_args, **_kwargs):
        raise OSError(5, "no space")

    monkeypatch.setattr(_bits_write, "publish_directory", refuse)
    with pytest.raises(OSError, match="no space"):
        _bits_write.write_bits(path, info, shard, tensors["vectors"], synced=True, by="x")
    assert not list(path.glob("*sign_bits-*"))


def test_a_failed_sync_after_the_rename_is_reported_and_the_bits_stay(tmp_path, monkeypatch):
    path = tmp_path / "store"
    signed_store(path, sizes=(5,))
    info, shards = opened(path)
    shard, tensors = shards[0]
    real = _bits_write.publish_directory

    def unconfirmed(tmp, final, **_kwargs):
        real(tmp, final, sync_parent=False)
        raise fse.PublishedNotDurableError("the rename is not confirmed durable")

    monkeypatch.setattr(_bits_write, "publish_directory", unconfirmed)
    with pytest.raises(fse.PublishedNotDurableError):
        _bits_write.write_bits(path, info, shard, tensors["vectors"], synced=True, by="x")
    assert open_bits(path, info, shard, verify=True) is not None
    assert not list(path.glob(".sign_bits-*"))


def test_unsynced_bits_sync_no_directory_and_are_tagged(tmp_path, monkeypatch):
    path = tmp_path / "store"
    signed_store(path, sizes=(5, 3))
    info, shards = opened(path)
    synced = []
    for module in (_npy, _npy_inplace):
        real = module._sync_directory

        def spy(directory, *args, _real=real, **kwargs):
            synced.append(directory)
            return _real(directory, *args, **kwargs)

        monkeypatch.setattr(module, "_sync_directory", spy)
    (shard, tensors), (other, other_tensors) = shards
    _bits_write.write_bits(path, info, shard, tensors["vectors"], synced=False, by="x")
    assert synced == []
    assert NpyContainer().metadata(path / bits_name(0))["fse.durability"] == "none"
    _bits_write.write_bits(path, info, other, other_tensors["vectors"], synced=True, by="y")
    assert path in synced  # the rename is synced in the store directory
    assert "fse.durability" not in NpyContainer().metadata(path / bits_name(5))
    assert not list(path.glob(".sign_bits-*"))


def test_hash_and_bits_are_made_from_one_copy_of_the_bytes(tmp_path, monkeypatch):
    path = tmp_path / "store"
    signed_store(path, sizes=(5,))
    _, shards = opened(path)
    shard, tensors = shards[0]
    vectors = np.array(tensors["vectors"])  # writable, as another process could change a map
    seen = []
    real = _bits_write._reference_bits.sign_bits

    def spy(block):
        seen.append(np.shares_memory(block, vectors))
        return real(block)

    monkeypatch.setattr(_bits_write._reference_bits, "sign_bits", spy)
    _bits_write.derived(shard, vectors)
    assert seen == [False]


def test_created_by_names_fse():
    assert _bits_write.created_by().startswith("fse")
