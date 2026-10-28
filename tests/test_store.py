"""The package of the stores of sentence vectors."""

import fse
from fse import _store


def test_the_package_exports_the_shard_sink():
    assert _store.__all__ == ["SearchResult", "ShardSink", "VectorStore"]
    assert fse.ShardSink is _store.ShardSink
    assert fse.VectorStore is _store.VectorStore
