"""Document IDs participate in full durability, verification and whole-shard repair."""

import json
import os
import shutil

import numpy as np
import pytest

import fse
from fse._container import NpyContainer, _npy_inplace
from fse._store import _id_writer
from fse._store._layout import StoreInfo, create_store
from support.document_ids import column, write_store
from support.durability import PowerLoss, materialize
from support.encoder import finished
from support.store_sync import Recorder


@pytest.mark.parametrize("durability", ["full", "none"])
def test_bad_ids_quarantine_the_whole_tail(tmp_path, durability):
    write_store(tmp_path / "s", list(range(12)), shard_rows=4, durability=durability)
    file = tmp_path / "s" / "shard-0000000000000004" / "doc_ids.npy"
    damaged = np.load(file, mmap_mode="r+")
    damaged[0] = 99
    damaged.flush()
    del damaged
    if durability == "none":
        with pytest.raises(fse.CorruptArtifactError, match="doc_ids"):
            fse.VectorStore.open(tmp_path / "s")
    else:
        with (
            fse.VectorStore.open(tmp_path / "s") as store,
            pytest.raises(fse.CorruptArtifactError, match="doc_ids"),
        ):
            store.verify()
    assert fse.repair_store(tmp_path / "s") == 4
    assert fse.repair_store(tmp_path / "s") == 4
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert list(column(store, "doc_ids")) == [0, 1, 2, 3]


def test_stats_are_recomputed_even_with_valid_checksums(tmp_path):
    write_store(tmp_path / "s", [1, 2, 3], shard_rows=4)
    directory = tmp_path / "s" / "shard-0000000000000000"
    tensors, metadata = NpyContainer().open(directory, mmap=False)
    metadata["fse.doc_ids_max"] = "4"
    metadata = {
        k: v
        for k, v in metadata.items()
        if not k.startswith("fse.checksum.") and k != "fse.fingerprint"
    }
    NpyContainer().write(tmp_path / "replacement", tensors, metadata)
    shutil.copyfile(tmp_path / "replacement" / "manifest.json", directory / "manifest.json")
    with (
        fse.VectorStore.open(tmp_path / "s") as store,
        pytest.raises(fse.CorruptArtifactError, match="statistics"),
    ):
        store.verify()
    assert fse.repair_store(tmp_path / "s") == 0


def test_future_resume_aborts_repair_without_mutation(tmp_path):
    write_store(tmp_path / "s", [1, 2, 3, 4], shard_rows=2)
    directory = tmp_path / "s" / "shard-0000000000000002"
    tensors, metadata = NpyContainer().open(directory, mmap=False)
    metadata["fse.resume"] = "source_rows@1"
    metadata = {
        k: v
        for k, v in metadata.items()
        if not k.startswith("fse.checksum.") and k != "fse.fingerprint"
    }
    NpyContainer().write(tmp_path / "replacement", tensors, metadata)
    shutil.copyfile(tmp_path / "replacement" / "manifest.json", directory / "manifest.json")
    first = tmp_path / "s" / "shard-0000000000000000" / "doc_ids.npy"
    with first.open("r+b") as stream:
        stream.truncate(4)
    before = sorted(p.name for p in (tmp_path / "s").iterdir())
    with pytest.raises(fse.UnsupportedArtifactError, match="source_rows@1"):
        fse.repair_store(tmp_path / "s")
    assert sorted(p.name for p in (tmp_path / "s").iterdir()) == before


def test_ids_are_synced_before_publication(tmp_path, monkeypatch):
    encoder = finished()
    root = tmp_path / "s"
    with fse.ShardSink(root, encoder, doc_ids=True) as sink:
        recorder = Recorder(monkeypatch, full=True)
        encoder.encode(["w1", "w2"], doc_ids=[1, 2], sink=sink)
    events = recorder.named(root)
    assert events.index("fsync doc_ids.npy") < events.index("rename")
    # Windows publishes with MoveFileExW and does not synchronize directories.
    assert events[-1] == ("rename" if os.name == "nt" else "full store")


@pytest.mark.skipif(os.name == "nt", reason="POSIX power-loss model")
@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_power_loss_keeps_ids_and_vectors_together(tmp_path, monkeypatch, platform):
    root, encoder = tmp_path / "s", finished()
    create_store(root, StoreInfo(encoder.fingerprint, "float32", encoder.output_dim, True, 2))
    expected = np.array([5, 5, -1, 2, 2, 0, 5], dtype="int64")
    with monkeypatch.context() as patch:
        trace = PowerLoss(root, patch, platform)
        patch.setattr(_id_writer, "write_at", _npy_inplace.write_at)
        with fse.ShardSink(root, encoder, doc_ids=True, shard_rows=4) as sink:
            for first, count in ((0, 4), (4, 3)):
                target = sink.target(first, count)
                target[0][...] = 1
                target[-1][...] = 1
                sink.commit(first, target, doc_ids=expected[first : first + count])
                sink.flush()
                trace.confirm(sink.durable_rows)
    for index, (state, confirmed) in enumerate(trace.states()):
        target = tmp_path / f"crash-{index}"
        materialize(target, state)
        with fse.VectorStore.open(target) as store:
            assert len(store) >= confirmed
            store.verify()
            if len(store):
                assert np.array_equal(column(store, "doc_ids"), expected[: len(store)])
                assert np.all(column(store, "vectors") == 1)
        shutil.rmtree(target)


def test_header_versions_and_modes_are_consistent(tmp_path):
    write_store(tmp_path / "s", [1])
    file = tmp_path / "s" / "store.json"
    metadata = json.loads(file.read_text())
    metadata["store_version"] = 1
    file.write_text(json.dumps(metadata))
    with pytest.raises(fse.CorruptArtifactError, match="versions differ"):
        fse.VectorStore.open(tmp_path / "s")
