"""The ID stream shrinks, hashes and closes with its vector shard."""

import numpy as np
import pytest

from fse._container import checksum
from fse._store._id_writer import IdColumn


def test_shrink_and_digest(tmp_path):
    writer = IdColumn(tmp_path, 100)
    values = np.array([5, 5, -1], dtype="int64")
    writer.write(0, values)
    writer.commit(values)
    assert writer.finish(3, synced=True) == ("I64", (3,), checksum(values))
    assert np.array_equal(np.load(tmp_path / "row_ids.npy"), values)
    assert writer.stats.metadata()["fse.row_ids_sorted"] == "false"
    writer.close()


def test_no_publication_before_id_write(tmp_path):
    with pytest.raises(RuntimeError, match="not written"):
        IdColumn(tmp_path, 10).finish(1, synced=False)
