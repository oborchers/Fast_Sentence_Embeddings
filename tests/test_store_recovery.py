"""Unsynced shards are verified on every open, and gaps never relink older rows."""

import json

import numpy as np
import pytest

from fse import ShardSink, UnsupportedArtifactError, VectorStore
from fse._container import CorruptArtifactError
from fse._store._layout import open_shard, shard_name
from support.store import ENCODER, add_shard, make_store, tag_shard


def _flip_data(file):
    with file.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_tag_is_part_of_version_one_and_checksums_come_from_the_manifest(tmp_path, dtype):
    path = tmp_path / "store"
    info = make_store(path, dtype=dtype, tagged=(4,))
    shard, tensors = open_shard(path, info, 4)
    assert shard.unsynced
    assert set(shard.checksums) == set(tensors)
    assert not open_shard(path, info, 0)[0].unsynced
    raw = (path / "store.json").read_bytes()
    assert json.loads(raw)["store_version"] == 1
    with VectorStore.open(path) as store:
        assert len(store) == 10
    assert (path / "store.json").read_bytes() == raw


@pytest.mark.parametrize("value", ["full", "flush", "", "NONE", "future"])
def test_unknown_durability_values_are_unsupported(tmp_path, value):
    path = tmp_path / "store"
    make_store(path, sizes=(4,))
    tag_shard(path / shard_name(0), value)
    with pytest.raises(UnsupportedArtifactError, match="durability"):
        VectorStore.open(path)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("tensor", ["vectors", "inv_norms"])
def test_tagged_tensor_damage_is_detected_before_rows_are_served(tmp_path, dtype, tensor):
    path = tmp_path / "store"
    make_store(path, dtype=dtype, tagged=(4,))
    _flip_data(path / shard_name(4) / f"{tensor}.npy")
    with pytest.raises(CorruptArtifactError, match=r"first row 4.*fse.repair_store"):
        VectorStore.open(path)


def test_tagged_int8_scales_are_hashed(tmp_path):
    path = tmp_path / "store"
    make_store(path, dtype="int8", tagged=(4,))
    _flip_data(path / shard_name(4) / "scales.npy")
    with pytest.raises(CorruptArtifactError, match=r"scales.*fse.repair_store"):
        VectorStore.open(path)


@pytest.mark.parametrize("damage", ["truncate", "manifest", "zero_vectors", "torn_header"])
def test_damaged_shards_name_the_row_and_repair_action(tmp_path, damage):
    path = tmp_path / "store"
    make_store(path, tagged=(4,))
    shard = path / shard_name(4)
    if damage == "manifest":
        (shard / "manifest.json").unlink()
    elif damage == "truncate":
        with (shard / "inv_norms.npy").open("r+b") as handle:
            handle.truncate(128)
    else:
        with (shard / "vectors.npy").open("r+b") as handle:
            handle.seek(128 if damage == "zero_vectors" else 32)
            handle.write(b"\0" * (64 if damage == "zero_vectors" else 16))
    with pytest.raises(CorruptArtifactError, match=r"first row 4.*fse.repair_store"):
        VectorStore.open(path)


@pytest.mark.parametrize("opener", ["reader", "sink"])
@pytest.mark.parametrize("orphan", [2, 12])
def test_every_open_refuses_an_orphan_inside_or_after_the_chain(tmp_path, opener, orphan):
    path = tmp_path / "store"
    info = make_store(path, sizes=(4, 4))
    add_shard(path, info, orphan, 1)
    with pytest.raises(CorruptArtifactError, match="fse.repair_store"):
        if opener == "reader":
            VectorStore.open(path)
        else:
            ShardSink(path, ENCODER)


def test_a_missing_shard_with_a_later_shard_is_a_gap(tmp_path):
    path = tmp_path / "store"
    info = make_store(path, sizes=(4,))
    add_shard(path, info, 8, 2)
    with pytest.raises(CorruptArtifactError, match="gap.*row 4.*fse.repair_store"):
        VectorStore.open(path)


def test_noncanonical_names_and_hidden_entries_are_ignored(tmp_path):
    path = tmp_path / "store"
    make_store(path, sizes=(4,))
    for name in ("shard-4", "shard-00000000000000004", "shard-+000000000000004", ".broken-x"):
        (path / name).mkdir()
    (path / "notes").write_text("ignored")
    with VectorStore.open(path) as store:
        assert len(store) == 4


def test_publication_between_walk_and_listing_keeps_the_snapshot(tmp_path, monkeypatch):
    from fse._store import _recovery

    path = tmp_path / "store"
    info = make_store(path, sizes=(4,))
    real = _recovery.canonical_shards

    def listed(store):
        add_shard(path, info, 4, 4)
        add_shard(path, info, 8, 4)
        return real(store)

    monkeypatch.setattr(_recovery, "canonical_shards", listed)
    with VectorStore.open(path) as store:
        assert len(store) == 4


@pytest.mark.parametrize("tagged", [(), (4,), (0, 4, 8)])
def test_every_tagged_shard_is_hashed_on_every_open(tmp_path, monkeypatch, tagged):
    from fse._store import _recovery

    path = tmp_path / "store"
    make_store(path, tagged=tagged)
    calls = []
    real = _recovery.checksum

    def hashed(tensor):
        calls.append(tensor.nbytes)
        return real(tensor)

    monkeypatch.setattr(_recovery, "checksum", hashed)
    for _ in range(2):
        calls.clear()
        with VectorStore.open(path) as store:
            assert len(store) == 10
        assert len(calls) == 2 * len(tagged)


def test_full_append_refuses_an_unsynced_prefix_without_hashing(tmp_path, monkeypatch):
    from fse._store import _recovery

    path = tmp_path / "store"
    make_store(path, tagged=(4,))
    calls = []
    monkeypatch.setattr(_recovery, "checksum", lambda tensor: calls.append(tensor))
    with pytest.raises(ValueError, match=r"durability='none' from row 4.*durability='full'"):
        ShardSink(path, ENCODER)
    assert not calls


def test_full_append_to_an_untagged_store_keeps_store_json(tmp_path):
    path = tmp_path / "store"
    make_store(path, sizes=(4,))
    raw = (path / "store.json").read_bytes()
    with ShardSink(path, ENCODER) as sink:
        assert sink.first_row == 4
        target = sink.target(0, 1)
        target[0][...] = np.ones((1, 4), np.float32)
        target[-1][...] = 0.5
        sink.commit(0, target)
    with VectorStore.open(path) as store:
        assert len(store) == 5
    assert (path / "store.json").read_bytes() == raw
