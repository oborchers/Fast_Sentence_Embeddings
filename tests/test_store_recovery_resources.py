"""Mappings belong to a returned store, never to a failed open's traceback."""

import gc
import os
import shutil
import weakref
from pathlib import Path

import numpy as np
import pytest

from fse import ShardSink, UnsupportedArtifactError, VectorStore
from fse._container import CorruptArtifactError, NpyContainer, _npy
from fse._store import _recovery, _vectorstore
from fse._store._layout import Shard, shard_name
from support.store import ENCODER, add_shard, make_store, tag_shard


@pytest.mark.parametrize("entry", ["file", "dangling_link", "directory_link"])
@pytest.mark.parametrize("opener", ["reader", "sink"])
def test_invalid_shard_entries_keep_corruption_context(tmp_path, entry, opener):
    path = tmp_path / "store"
    make_store(path, sizes=(4,))
    bad = path / shard_name(4)
    if entry == "file":
        bad.write_bytes(b"not a shard")
    else:
        target = tmp_path / "target"
        if entry == "directory_link":
            target.mkdir()
        try:
            bad.symlink_to(target, target_is_directory=True)
        except OSError:
            if os.name == "nt":
                pytest.skip("this Windows runner cannot create symbolic links")
            raise
    factory = VectorStore.open if opener == "reader" else lambda store: ShardSink(store, ENCODER)
    with pytest.raises(CorruptArtifactError, match=r"first row 4.*fse\.repair_store"):
        factory(path)


def _track_maps(monkeypatch):
    refs = []
    real = NpyContainer.open

    def opened(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        refs.extend(weakref.ref(tensor) for tensor in result[0].values())
        return result

    monkeypatch.setattr(NpyContainer, "open", opened)
    return refs


@pytest.mark.parametrize("failure", ["checksum", "unsupported", "gap", "backend", "sink"])
def test_a_retained_exception_does_not_keep_mappings_open(tmp_path, monkeypatch, failure):
    path = tmp_path / "store"
    info = make_store(path, tagged=(4,))
    if failure == "checksum":
        with (path / shard_name(4) / "vectors.npy").open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"\xff")
    elif failure == "unsupported":
        tag_shard(path / shard_name(4), "future")
    elif failure == "gap":
        add_shard(path, info, 12, 1)
    elif failure == "backend":

        def unavailable(_name):
            raise RuntimeError("backend unavailable")

        monkeypatch.setattr(_vectorstore, "get_backend", unavailable)
    refs = _track_maps(monkeypatch)
    factory = (
        (lambda: ShardSink(path, ENCODER)) if failure == "sink" else lambda: VectorStore.open(path)
    )
    with pytest.raises(
        (CorruptArtifactError, UnsupportedArtifactError, ValueError, RuntimeError)
    ) as caught:
        factory()
    assert caught.value.__traceback__ is not None
    assert refs
    gc.collect()
    assert all(ref() is None for ref in refs)
    shutil.rmtree(path)  # also proves the resource release on real Windows
    assert str(caught.value)  # keep the exception and traceback alive through the removal


def test_verification_uses_the_original_maps_without_reopening_paths(tmp_path, monkeypatch):
    path = tmp_path / "store"
    make_store(path, dtype="int8", tagged=(4, 8))
    maps, blocked, hashed = [], set(), []
    real_open, real_read, real_file = _recovery.open_shard, _npy._read_manifest, _npy._open_regular
    real_hash = _recovery.checksum

    def opened(store, info, row):
        shard, tensors = real_open(store, info, row)
        if shard.unsynced:
            maps.extend(tensors.values())
            blocked.add(store / shard.name)
        return shard, tensors

    def manifest(store):
        assert Path(store) not in blocked, "manifest reopened after open_shard"
        return real_read(store)

    def file(file):
        assert Path(file).parent not in blocked, "tensor path reopened after open_shard"
        return real_file(file)

    def checksum(tensor):
        assert any(tensor is original for original in maps)
        hashed.append(id(tensor))
        return real_hash(tensor)

    monkeypatch.setattr(_recovery, "open_shard", opened)
    monkeypatch.setattr(_recovery, "checksum", checksum)
    monkeypatch.setattr(_npy, "_read_manifest", manifest)
    monkeypatch.setattr(_npy, "_open_regular", file)
    with VectorStore.open(path) as store:
        assert len(store) == 10
    assert sorted(hashed) == sorted(id(tensor) for tensor in maps)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_value_errors_beyond_the_first_block_are_detected(monkeypatch, dtype):
    monkeypatch.setattr(_recovery, "_VALUE_BYTES", 8)
    codes = np.ones((5, 4), dtype=dtype)
    codes[-1, -1] = -128 if dtype == "int8" else np.nan
    with pytest.raises(CorruptArtifactError):
        _recovery.check_values(Shard("test", 0, 5), {"vectors": codes})


def test_float_value_checks_bound_temporary_arrays(monkeypatch):
    monkeypatch.setattr(_recovery, "_VALUE_BYTES", 64)
    real = np.isfinite
    sizes = []

    def finite(array):
        sizes.append(array.nbytes)
        return real(array)

    monkeypatch.setattr(_recovery.np, "isfinite", finite)
    _recovery.check_values(
        Shard("test", 0, 100),
        {"vectors": np.ones((100, 4), np.float32), "inv_norms": np.ones(100, np.float32)},
    )
    assert len(sizes) > 2
    assert max(sizes) <= 64
