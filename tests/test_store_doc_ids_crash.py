"""Real process exits cannot publish a vector prefix with different document IDs."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fse
from support.document_ids import column

CHILD = """
import os, sys, threading
from pathlib import Path
import numpy as np
import fse
from fse._store import _id_writer, _shard
from fse._container._npy_inplace import create, write_at
sys.path.insert(0, sys.argv[4])
from support.encoder import finished

root, point, policy = sys.argv[1:4]
first = threading.Event()
real_write, real_publish = _id_writer.IdColumn.write, _shard.publish_directory
def write(self, begin, ids):
    if point == "ids" and "shard-0000000000000004" in self.path.parent.name:
        assert first.wait(30)
        self.fd, self.offset = create(self.path, np.dtype("<i8"), (self.capacity,))
        write_at(self.fd, memoryview(ids).cast("B")[:3], self.offset)
        os._exit(3)
    real_write(self, begin, ids)
def publish(tmp, path, **kw):
    real_publish(tmp, path, **kw)
    if path.name == "shard-0000000000000000":
        first.set()
    if point == "published" and path.name == "shard-0000000000000004":
        os._exit(3)
_id_writer.IdColumn.write = write
_shard.publish_directory = publish
encoder = finished()
with fse.ShardSink(root, encoder, doc_ids=True, shard_rows=4, durability=policy) as sink:
    encoder.encode(["w1"]*12, doc_ids=list(range(12)), sink=sink)
"""


@pytest.mark.parametrize("policy", ["full", "none"])
@pytest.mark.parametrize(("point", "rows"), [("ids", 4), ("published", 8)])
def test_process_crash_prefix(tmp_path, policy, point, rows):
    root = tmp_path / "s"
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(root), point, policy, str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3, result.stderr
    with fse.VectorStore.open(root) as store:
        assert len(store) == rows
        assert np.array_equal(column(store, "doc_ids"), np.arange(rows))
        store.verify()
