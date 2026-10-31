"""Power loss admits independent writeback, unlike a process crash in the page cache."""

import os
import shutil
import tokenize

import numpy as np
import pytest

from fse import ShardSink, VectorStore
from fse._container import _npy
from fse._store._layout import StoreInfo, create_store
from support.durability import PowerLoss, materialize
from support.encoder import finished

POSIX = pytest.mark.skipif(os.name == "nt", reason="the POSIX persistence models")
ENCODER = finished(dim=4)


def _record(root, monkeypatch, platform, dtype, *, broken=False):
    info = StoreInfo(ENCODER.fingerprint, dtype, ENCODER.output_dim)
    create_store(root, info)
    expected = []
    with monkeypatch.context() as patch:
        trace = PowerLoss(root, patch, platform)
        if broken:
            real = _npy._sync_directory

            def plain(path, *, full=True):
                real(path, full=False if path.name.startswith(".shard-") else full)

            patch.setattr(_npy, "_sync_directory", plain)
        with ShardSink(root, ENCODER, dtype=dtype, shard_rows=4) as sink:
            # Flush each batch before the next reservation. Every observed confirmation
            # belongs to the preceding trace, with no scheduling-dependent crash points.
            for first, count in ((0, 4), (4, 4), (8, 2)):
                target = sink.target(first, count)
                target[0][...] = np.arange(first * 4 + 1, (first + count) * 4 + 1).reshape(count, 4)
                if dtype == "int8":
                    target[1][...] = 0.5
                target[-1][...] = 0.25
                expected.append(target[0].copy())
                sink.commit(first, target)
                assert sink.flush() == first + count
                trace.confirm(sink.durable_rows)
    return trace, np.concatenate(expected)


def _assert_prefix(root, state, confirmed, expected):
    materialize(root, state)
    try:
        with VectorStore.open(root) as store:
            assert len(store) >= confirmed
            # Compare stored codes as bits, including float16 and int8, without decoding.
            parts = [tensors["vectors"] for _, tensors in store._shards]
            got = np.concatenate(parts) if parts else expected[:0]
            assert got.tobytes() == expected[: len(got)].tobytes()
            store.verify()  # all tensors, not only the vector rows
    finally:
        shutil.rmtree(root)


@POSIX
@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_full_writer_survives_every_admitted_power_loss(tmp_path, monkeypatch, platform, dtype):
    trace, expected = _record(tmp_path / "store", monkeypatch, platform, dtype)
    states = 0
    for state, confirmed in trace.states():
        _assert_prefix(tmp_path / "crash", state, confirmed, expected)
        states += 1
    assert states >= 4  # empty, one shard, two shards, the sealed final shard
    assert trace.rows == len(expected) == 10


@POSIX
def test_model_detects_missing_full_sync_before_shard_rename(tmp_path, monkeypatch):
    trace, expected = _record(tmp_path / "store", monkeypatch, "darwin", "float32", broken=True)
    # Require silent wrong rows, not just a missing manifest or an open error.
    for state, _ in trace.states():
        root = tmp_path / "crash"
        materialize(root, state)
        try:
            with VectorStore.open(root) as store:
                if store.get_vectors().tobytes() != expected[: len(store)].tobytes():
                    return
        except (ValueError, tokenize.TokenError):
            # Numpy can report a torn .npy header as TokenError. Only a successful
            # open returning wrong rows is a witness for this mutation oracle.
            pass
        finally:
            shutil.rmtree(root)
    pytest.fail("the darwin model missed silently damaged rows after the persisted rename")
