"""A corrupt tail and the append-before-rerun oracle shared by repair crash tests."""

import numpy as np
import pytest

import fse
from fse._store._layout import shard_name
from support.store import append_rows, make_store


def damaged_store(path):
    make_store(path, sizes=(4, 4, 4, 4))
    (path / shard_name(4) / "manifest.json").unlink()


def after_crash(path):
    """Try a real open and append before rerunning repair, never hiding a gap first."""
    first = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    try:
        with fse.VectorStore.open(path) as store:
            before = store.get_vectors()
            assert before.tobytes() == first.tobytes()
    except fse.CorruptArtifactError:
        with pytest.raises(fse.CorruptArtifactError):
            append_rows(path)
        assert fse.repair_store(path) == 4
        assert append_rows(path) == 4
    else:
        assert append_rows(path) == 4
        assert fse.repair_store(path) == 8
    with fse.VectorStore.open(path) as store:
        assert len(store) == 8
        assert store.get_vectors(stop=4).tobytes() == first.tobytes()
        assert (store.get_vectors(start=4) == 99).all()
        store.verify()
    assert fse.repair_store(path) == 8
    moved = [p.name for group in path.glob(".broken-*") for p in group.iterdir()]
    assert sorted(moved) == [shard_name(row) for row in (4, 8, 12)]
