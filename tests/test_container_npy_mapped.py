"""Writing an NpyContainer artifact in place, as the ShardSink of fse._store does."""

import errno
import os

import numpy as np
import pytest

from fse._container import NpyContainer
from fse._container import _npy_mapped as mapped
from fse._container._common import array_sha256, tensor_code
from fse._container._npy import write_manifest


def _write_in_place(directory, rows, keep=None):
    """``vectors.npy`` of ``rows``, written through two mappings, then the manifest."""
    directory.mkdir()
    file = directory / "vectors.npy"
    fd, offset = mapped.create(file, rows.dtype, rows.shape)
    try:
        row_bytes = rows.shape[1] * rows.dtype.itemsize
        half = len(rows) // 2
        for begin, end in ((0, half), (half, len(rows))):
            mapped.allocate(fd, offset + begin * row_bytes, offset + end * row_bytes)
            mapping, view = mapped.map_range(
                fd,
                offset,
                begin * row_bytes,
                end * row_bytes,
                rows.dtype,
                (end - begin, rows.shape[1]),
            )
            view[...] = rows[begin:end]
            mapping.flush()
            del view
            mapping.close()
        if keep is not None:
            mapped.shrink(fd, offset, rows.dtype, (keep, rows.shape[1]))
            rows = rows[:keep]
        mapped.sync_file(fd)
    finally:
        os.close(fd)
    entries = {"vectors": (tensor_code(rows), rows.shape, array_sha256(rows))}
    write_manifest(directory, entries, {"fse.kind": "test"}, directory / ".manifest.writing")
    return rows


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8])
def test_an_artifact_written_in_place_is_one_of_the_container(tmp_path, dtype):
    rows = (np.arange(5 * 37 * 3) % 100).reshape(-1, 3).astype(dtype)
    _write_in_place(tmp_path / "shard", rows)
    tensors, metadata = NpyContainer().open(tmp_path / "shard", mmap=True)
    np.testing.assert_array_equal(tensors["vectors"], rows)
    NpyContainer().verify(tmp_path / "shard")
    np.testing.assert_array_equal(np.load(tmp_path / "shard" / "vectors.npy", mmap_mode="r"), rows)
    NpyContainer().write(tmp_path / "plain", {"vectors": rows}, {"fse.kind": "test"})
    _, plain = NpyContainer().open(tmp_path / "plain", mmap=False)
    assert metadata == plain  # the same checksum and fingerprint


def test_a_shard_that_ends_early_keeps_its_data_offset(tmp_path):
    rows = np.arange(400, dtype=np.float32).reshape(100, 4)
    kept = _write_in_place(tmp_path / "shard", rows, keep=30)
    file = tmp_path / "shard" / "vectors.npy"
    assert os.path.getsize(file) == len(mapped.header(np.float32, (100, 4))) + kept.nbytes
    np.testing.assert_array_equal(np.load(file), rows[:30])
    tensors, _ = NpyContainer().open(tmp_path / "shard", mmap=False)
    np.testing.assert_array_equal(tensors["vectors"], rows[:30])


def test_headers_are_padded_to_the_alignment():
    head = mapped.header(np.float16, (1_000_000, 300))
    assert len(head) % 64 == 0
    assert head.endswith(b"\n")
    assert len(mapped.header(np.float16, (0, 300), len(head))) == len(head)
    with pytest.raises(ValueError, match="does not fit in 64 bytes"):
        mapped.header(np.float16, (10**60, 10**60), 64)


def test_create_never_reuses_a_file(tmp_path):
    (tmp_path / "vectors.npy").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        mapped.create(tmp_path / "vectors.npy", np.float32, (2, 2))


def test_a_full_disk_raises_before_a_worker_writes(tmp_path, monkeypatch):
    fd, offset = mapped.create(tmp_path / "vectors.npy", np.float32, (4, 4))

    def full(*_args):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(mapped, "_write_at", full)
    monkeypatch.setattr(os, "posix_fallocate", full, raising=False)
    try:
        with pytest.raises(OSError, match="No space left"):
            mapped.allocate(fd, offset, offset + 64)
    finally:
        os.close(fd)


def test_publishing_never_replaces_a_directory(tmp_path):
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    (tmp_path / "a" / "x").write_bytes(b"1")
    (tmp_path / "b" / "y").write_bytes(b"2")
    with pytest.raises(FileExistsError):
        mapped.publish_directory(tmp_path / "a", tmp_path / "b")  # not empty
    with pytest.raises(FileExistsError):
        mapped.publish_directory(tmp_path / "a", tmp_path / "c")  # empty
    assert (tmp_path / "a" / "x").exists()
    assert (tmp_path / "b" / "y").exists()
    mapped.publish_directory(tmp_path / "a", tmp_path / "d")
    assert (tmp_path / "d" / "x").read_bytes() == b"1"
    assert not (tmp_path / "a").exists()
