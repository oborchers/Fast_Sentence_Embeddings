"""Real none publications and prefix recovery over replayed persistence states."""

import numpy as np

import fse
from fse._store import _shard
from fse._store._layout import StoreInfo, create_store
from support.durability import PowerLoss
from support.store import ENCODER, append_rows


def record_none(root, monkeypatch, platform, dtype="float32", *, tag=True):
    info = StoreInfo(ENCODER.fingerprint, dtype, ENCODER.output_dim)
    create_store(root, info)
    expected = []
    with monkeypatch.context() as patch:
        trace = PowerLoss(root, patch, platform)
        if not tag:
            real = _shard.shard_metadata

            def untagged(*args, **kwargs):
                metadata = real(*args, **kwargs)
                del metadata["fse.durability"]
                return metadata

            patch.setattr(_shard, "shard_metadata", untagged)
        with fse.ShardSink(root, ENCODER, dtype=dtype, shard_rows=4, durability="none") as sink:
            for first, count in ((0, 4), (4, 4), (8, 2)):
                target = sink.target(first, count)
                target[0][...] = np.arange(first * 4 + 1, (first + count) * 4 + 1).reshape(count, 4)
                if dtype == "int8":
                    target[1][...] = 0.5
                target[-1][...] = 0.25
                expected.append(target[0].copy())
                sink.commit(first, target)
                assert sink.flush() == 0
                assert sink.published_rows == first + count
                assert sink.durable_rows == 0
                trace.confirm(0)
    return trace, np.concatenate(expected)


def codes(store, empty):
    arrays = [tensors["vectors"] for _, tensors in store._shards]
    return np.concatenate(arrays) if arrays else empty[:0]


def check_prefix(root, expected):
    """Return an intact prefix length, or None for strictly diagnosed corruption."""
    try:
        with fse.VectorStore.open(root) as store:
            assert codes(store, expected).tobytes() == expected[: len(store)].tobytes()
            store.verify()
            return len(store)
    except fse.CorruptArtifactError:
        return None


def recover_and_append(root, expected):
    kept = fse.repair_store(root)
    assert check_prefix(root, expected) == kept
    assert append_rows(root, durability="none") == kept
    with fse.VectorStore.open(root) as store:
        found = codes(store, expected)
        assert len(store) == kept + 4
        assert found[:kept].tobytes() == expected[:kept].tobytes()
        assert (found[kept:] == 99).all()
        store.verify()
