"""Strict durability policy, publication counts and immutable store identity."""

import json

import pytest
from pydantic import ValidationError

import fse
from fse._store._layout import shard_name
from support.encoder import finished
from support.runtime import texts


@pytest.mark.parametrize("value", ["flush", "fsync", "", None, 1, True, b"none"])
def test_durability_is_a_strict_literal_before_store_creation(tmp_path, value):
    path = tmp_path / "store"
    with pytest.raises(ValidationError):
        fse.ShardSink(path, finished(), durability=value)
    assert not path.exists()


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("durability", ["full", "none"])
def test_publication_counts_tags_and_store_identity(tmp_path, dtype, durability):
    encoder = finished()
    path = tmp_path / "store"
    sink = fse.ShardSink(path, encoder, dtype=dtype, shard_rows=8, durability=durability)
    identity = (path / "store.json").read_bytes()
    assert sink.published_rows == sink.durable_rows == 0
    with sink:
        encoder.encode(texts(20), sink=sink, workers=1)
    assert sink.published_rows == 20
    assert sink.durable_rows == (20 if durability == "full" else 0)
    assert (path / "store.json").read_bytes() == identity
    assert json.loads(identity)["store_version"] == 1
    for first in (0, 8, 16):
        metadata = json.loads((path / shard_name(first) / "manifest.json").read_bytes())["metadata"]
        assert metadata.get("fse.durability") == ("none" if durability == "none" else None)
    with fse.VectorStore.open(path) as store:
        assert len(store) == 20
        store.verify()
    if durability == "none":
        with pytest.raises(ValueError, match="durability='none'"):
            fse.ShardSink(path, encoder, dtype=dtype)
    with fse.ShardSink(path, encoder, dtype=dtype, durability=durability) as appended:
        assert appended.first_row == 20
        encoder.encode(texts(3), sink=appended)
    assert appended.published_rows == 3
    assert appended.durable_rows == (3 if durability == "full" else 0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_none_appends_after_a_full_prefix_without_rewriting_or_promoting_it(tmp_path, dtype):
    encoder = finished()
    path = tmp_path / "store"
    with fse.ShardSink(path, encoder, dtype=dtype, shard_rows=4) as full:
        encoder.encode(texts(4), sink=full)
    identity = (path / "store.json").read_bytes()
    first_manifest = path / shard_name(0) / "manifest.json"
    first_bytes = first_manifest.read_bytes()
    with fse.ShardSink(path, encoder, dtype=dtype, shard_rows=4, durability="none") as none:
        assert none.first_row == 4
        encoder.encode(texts(6), sink=none)
    assert (none.published_rows, none.durable_rows) == (6, 0)
    assert (full.published_rows, full.durable_rows) == (4, 4)
    assert (path / "store.json").read_bytes() == identity
    assert first_manifest.read_bytes() == first_bytes
    with fse.VectorStore.open(path) as store:
        assert len(store) == 10
        assert [(s.first_row, s.unsynced) for s, _ in store._shards] == [
            (0, False),
            (4, True),
            (8, True),
        ]
        store.verify()
    with pytest.raises(ValueError, match="from row 4"):
        fse.ShardSink(path, encoder, dtype=dtype)
