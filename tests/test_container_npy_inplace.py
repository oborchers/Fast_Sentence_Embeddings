"""Writing an NpyContainer artifact in place, as the ShardSink of fse._store does."""

import ctypes
import errno
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fse._container import NpyContainer, PublishedNotDurableError
from fse._container import _npy_inplace as inplace
from fse._container._common import checksum, tensor_code
from fse._container._npy import write_manifest


def _write_in_place(directory, rows, keep=None):
    """``vectors.npy`` of ``rows``, written in two ranges at their offsets, then the manifest."""
    directory.mkdir()
    fd, offset = inplace.create(directory / "vectors.npy", rows.dtype, rows.shape)
    try:
        row_bytes = rows.shape[1] * rows.dtype.itemsize
        half = len(rows) // 2
        for begin, end in ((half, len(rows)), (0, half)):  # in any order
            inplace.write_at(fd, memoryview(rows[begin:end]), offset + begin * row_bytes)
        if keep is not None:
            inplace.shrink(fd, offset, rows.dtype, (keep, rows.shape[1]))
            rows = rows[:keep]
        inplace.sync_file(fd)
    finally:
        os.close(fd)
    entries = {"vectors": (tensor_code(rows), rows.shape, checksum(rows))}
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
    assert os.path.getsize(file) == len(inplace.header(np.float32, (100, 4))) + kept.nbytes
    np.testing.assert_array_equal(np.load(file), rows[:30])
    tensors, _ = NpyContainer().open(tmp_path / "shard", mmap=False)
    np.testing.assert_array_equal(tensors["vectors"], rows[:30])


def test_headers_are_padded_to_the_alignment():
    head = inplace.header(np.float16, (1_000_000, 300))
    assert len(head) % 64 == 0
    assert head.endswith(b"\n")
    assert len(inplace.header(np.float16, (0, 300), len(head))) == len(head)
    with pytest.raises(ValueError, match="does not fit in 64 bytes"):
        inplace.header(np.float16, (10**60, 10**60), 64)


def test_create_never_reuses_a_file(tmp_path):
    (tmp_path / "vectors.npy").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        inplace.create(tmp_path / "vectors.npy", np.float32, (2, 2))


@pytest.mark.parametrize("write", ["native", "seek"])
def test_write_at_writes_everything_and_raises_on_a_full_disk(tmp_path, monkeypatch, write):
    fd, offset = inplace.create(tmp_path / "vectors.npy", np.uint8, (1, 100))
    real = inplace._write_some if write == "native" else inplace._seek_and_write
    try:
        monkeypatch.setattr(  # short writes
            inplace, "_write_some", lambda f, data, at: real(f, data[:7], at)
        )
        inplace.write_at(fd, memoryview(bytes(range(100))), offset)
        os.lseek(fd, offset, os.SEEK_SET)
        assert os.read(fd, 100) == bytes(range(100))

        def full(*_args):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(inplace, "_write_some", full)
        with pytest.raises(OSError, match="No space left"):
            inplace.write_at(fd, memoryview(b"x"), offset)
    finally:
        os.close(fd)


def test_headers_survive_short_writes_and_no_progress_raises(tmp_path, monkeypatch):
    real = inplace._write_some
    monkeypatch.setattr(inplace, "_write_some", lambda f, data, at: real(f, data[:3], at))
    fd, offset = inplace.create(tmp_path / "vectors.npy", np.float32, (2, 3))
    os.close(fd)
    assert np.load(tmp_path / "vectors.npy").shape == (2, 3)  # the whole header
    assert offset % 64 == 0
    monkeypatch.setattr(inplace, "_write_some", lambda *_args: 0)
    with pytest.raises(OSError, match="made no progress"):
        inplace.create(tmp_path / "other.npy", np.float32, (2, 3))


@pytest.mark.parametrize("libc_api", ["native", "syscall"])
def test_publishing_never_replaces_a_directory(tmp_path, monkeypatch, libc_api):
    if libc_api == "syscall":
        if sys.platform != "linux":
            pytest.skip("the direct renameat2 syscall is Linux only")
        # Exercise musl's missing wrapper on glibc too, with the actual kernel syscall.
        libc = ctypes.CDLL(None, use_errno=True)
        monkeypatch.setattr(
            inplace.ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace(syscall=libc.syscall)
        )
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    (tmp_path / "a" / "x").write_bytes(b"1")
    (tmp_path / "b" / "y").write_bytes(b"2")
    with pytest.raises(FileExistsError):
        inplace.publish_directory(tmp_path / "a", tmp_path / "b")  # not empty
    with pytest.raises(FileExistsError):
        inplace.publish_directory(tmp_path / "a", tmp_path / "c")  # empty
    assert (tmp_path / "a" / "x").exists()
    assert (tmp_path / "b" / "y").exists()
    inplace.publish_directory(tmp_path / "a", tmp_path / "d")
    assert (tmp_path / "d" / "x").read_bytes() == b"1"
    assert not (tmp_path / "a").exists()


def test_a_failed_sync_after_the_rename_keeps_the_published_directory(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x").write_bytes(b"1")

    def fail(path, **_kwargs):
        raise OSError(errno.EIO, "no sync", str(path))

    monkeypatch.setattr(inplace, "_sync_directory", fail)
    with pytest.raises(PublishedNotDurableError, match="is published, but syncing") as raised:
        inplace.publish_directory(tmp_path / "a", tmp_path / "b")
    assert isinstance(raised.value, OSError)
    assert isinstance(raised.value.__cause__, OSError)
    assert (tmp_path / "b" / "x").read_bytes() == b"1"  # published, complete, kept
    assert not (tmp_path / "a").exists()


def test_a_failed_rename_publishes_nothing(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()

    def fail(_tmp, path):
        raise OSError(errno.EXDEV, "no rename", str(path))

    monkeypatch.setattr(inplace, "_rename_exclusive", fail)
    with pytest.raises(OSError, match="no rename") as raised:
        inplace.publish_directory(tmp_path / "a", tmp_path / "b")
    assert not isinstance(raised.value, PublishedNotDurableError)
    assert (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()


def test_an_unknown_linux_syscall_architecture_fails_without_renaming(tmp_path, monkeypatch):
    monkeypatch.setattr(
        inplace.os, "uname", lambda: SimpleNamespace(machine="unknown"), raising=False
    )
    with pytest.raises(OSError, match="no supported exclusive rename syscall"):
        inplace._rename_linux(SimpleNamespace(), tmp_path / "a", tmp_path / "b")


@pytest.mark.skipif(os.name == "nt", reason="Windows uses MoveFileExW")
@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EINVAL])
def test_a_linux_syscall_error_is_reported(tmp_path, monkeypatch, error):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "data").write_bytes(b"kept")

    def unavailable(*_args):
        ctypes.set_errno(error)
        return -1

    monkeypatch.setattr(inplace.sys, "platform", "linux")
    monkeypatch.setattr(
        inplace.os, "uname", lambda: SimpleNamespace(machine="x86_64"), raising=False
    )
    monkeypatch.setattr(
        inplace.ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace(syscall=unavailable)
    )
    with pytest.raises(OSError, match=rf"\[Errno {error}\]") as raised:
        inplace._rename_exclusive(tmp_path / "a", tmp_path / "b")
    assert raised.value.errno == error
    assert (tmp_path / "a" / "data").read_bytes() == b"kept"
    assert not (tmp_path / "b").exists()
