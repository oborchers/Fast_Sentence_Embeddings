"""A process that dies while it writes a store leaves a valid prefix (process crashes;
power loss depends on the syncs, whose order the shard writer follows)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fse
from fse._store._layout import StoreInfo, shards
from support.encoder import unfolded
from support.runtime import texts

ENCODER = unfolded()
TESTS = Path(__file__).parent

CHILD = """
import os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[3])
import fse
from fse._store import _shard
from support.encoder import unfolded
from support.runtime import texts

point, calls, real = sys.argv[2], [0], _shard.publish_directory
real_write, real_shrink = _shard.ShardWriter.write, _shard.shrink

def write(self, begin, codes):
    if point == "write" and self.first_row == 64:
        first = Path(sys.argv[1]) / _shard.shard_name(0)
        deadline = time.monotonic() + 30
        while not first.exists() and time.monotonic() < deadline:
            time.sleep(0.01)  # the first shard is published by its own thread
        _shard.write_at(self.fd, memoryview(codes).cast("B")[: codes.nbytes // 2], self.offset)
        os._exit(3)  # halfway through a write of the second shard
    real_write(self, begin, codes)

def shrink(*args):
    if point == "shrink":
        os._exit(3)  # finishing the last shard, which ends early
    real_shrink(*args)

def publish(tmp, path):
    calls[0] += 1
    if point == "before" and calls[0] == 2:
        os._exit(3)  # the second shard is written and synced, not renamed
    real(tmp, path)
    if point == "after" and calls[0] == 2:
        os._exit(3)  # the second shard is published, the sink never learns it

_shard.publish_directory = publish
_shard.ShardWriter.write, _shard.shrink = write, shrink
encoder = unfolded()
with fse.ShardSink(sys.argv[1], encoder, shard_rows=64) as sink:
    encoder.encode(texts(300), sink=sink, workers=2)
"""


def _rows(path):
    info = StoreInfo(ENCODER.fingerprint, "float32", ENCODER.output_dim)
    found = shards(path, info)
    return [(s.first_row, s.rows) for s, _ in found], np.concatenate(
        [t["vectors"] for _, t in found]
    )


@pytest.mark.parametrize(
    ("point", "published"), [("write", 64), ("before", 64), ("after", 128), ("shrink", 256)]
)
def test_a_crash_leaves_a_prefix_that_appends_continue(tmp_path, point, published):
    store = tmp_path / "s"
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(store), point, str(TESTS)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 3, done.stderr
    found, _ = _rows(store)
    assert sum(n for _, n in found) == published
    leftovers = [p.name for p in store.iterdir() if p.name.startswith(".")]
    # Shards still being written stay behind as hidden directories, which readers ignore.
    assert leftovers
    assert all(name.startswith(".shard-") and ".writing-" in name for name in leftovers)
    want = ENCODER.encode(texts(300))
    with fse.ShardSink(store, ENCODER, shard_rows=64) as sink:
        assert sink.first_row == published
        ENCODER.encode(texts(300)[published:], sink=sink)
    found, codes = _rows(store)
    np.testing.assert_array_equal(codes, want)
    assert found[-1][0] + found[-1][1] == 300
