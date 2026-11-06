"""A process that dies while it writes a score store leaves a valid prefix that a new run
appends to (process crashes; power loss depends on the syncs of the shared shard writer)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fse
from support import score_heads as sh
from support import scores as sc

TESTS = Path(__file__).parent

CHILD = """
import os, sys, threading
sys.path.insert(0, sys.argv[3])
import fse
from fse._store import _columns, _score_shard, _shard
from support import score_heads as sh

point, calls = sys.argv[2], [0]
real_publish, real_write = _shard.publish_directory, _score_shard.ScoreShardWriter.write
real_shrink = _columns.shrink
first_published = threading.Event()

def write(self, begin, rows, ids=None):
    if point == "write" and self.first_row == 64:
        assert first_published.wait(30), "first shard was not published"
        self.scores.ensure()
        _columns.write_at(
            self.scores.fd, memoryview(rows).cast("B")[: rows.nbytes // 2], self.scores.offset
        )
        os._exit(3)  # halfway through a write of the second shard
    real_write(self, begin, rows, ids)

def shrink(*args):
    if point == "shrink":
        os._exit(3)  # finishing the last shard, which ends early
    real_shrink(*args)

def publish(tmp, path, **kwargs):
    calls[0] += 1
    if point == "before" and calls[0] == 2:
        os._exit(3)  # the second shard is written and synced, not renamed
    real_publish(tmp, path, **kwargs)
    if path.name == _shard.shard_name(0):
        first_published.set()
    if point == "after" and calls[0] == 2:
        os._exit(3)  # the second shard is published, the sink never learns it

_shard.publish_directory = publish
_score_shard.ScoreShardWriter.write, _columns.shrink = write, shrink
heads = sh.group()
texts = sh.texts(300)
with fse.ScoreSink(sys.argv[1], heads, shard_rows=64, durability=sys.argv[4]) as sink:
    for batch in heads.predict_batches([texts[i : i + 50] for i in range(0, 300, 50)], score_sink=sink, workers=2):
        pass
"""


@pytest.mark.parametrize("durability", ["full", "none"])
@pytest.mark.parametrize(
    ("point", "published"), [("write", 64), ("before", 64), ("after", 128), ("shrink", 256)]
)
def test_a_crash_leaves_a_prefix_that_a_new_run_appends_to(tmp_path, point, published, durability):
    store = tmp_path / "s"
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(store), point, str(TESTS), durability],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 3, done.stderr
    _, scores, _ = sc.read_back(store)
    assert len(scores) == published
    leftovers = [p.name for p in store.iterdir() if p.name.startswith(".")]
    assert leftovers  # shards still being written stay behind as hidden directories
    assert all(name.startswith(".shard-") and ".writing-" in name for name in leftovers)
    heads, texts = sh.group(), sh.texts(300)
    stored = {"quality": "values", "toxic": "decision", "topic": "decision"}
    want = heads.predict(texts, output=stored)
    with fse.ScoreStore.open(store) as reader:
        reader.verify()
        assert (
            reader.get_scores("topic", "decision").tobytes()
            == np.asarray(want["topic"][:published]).tobytes()
        )
    if durability == "none":
        with pytest.raises(ValueError, match="durability='none'"):
            fse.ScoreSink(store, heads)
    with fse.ScoreSink(store, heads, shard_rows=64, durability=durability) as sink:
        assert sink.first_row == published
        list(heads.predict_batches([texts[published:]], score_sink=sink))
    with fse.ScoreStore.open(store) as reader:
        assert len(reader) == 300
        reader.verify()
        assert (
            reader.get_scores("topic", "decision").tobytes() == np.asarray(want["topic"]).tobytes()
        )


def test_a_full_disk_while_writing_scores_raises_and_leaves_a_prefix(tmp_path, monkeypatch):
    heads, texts = sh.group(), sh.texts(40)
    from fse._store import _columns

    calls = [0]
    original = _columns.write_at

    def full(fd, data, offset):
        calls[0] += 1
        if calls[0] > 3:
            raise OSError(28, "No space left on device")
        return original(fd, data, offset)

    monkeypatch.setattr(_columns, "write_at", full)
    with (
        pytest.raises(OSError, match="No space left"),
        fse.ScoreSink(tmp_path / "s", heads, shard_rows=8) as sink,
    ):
        list(
            heads.predict_batches(
                [texts[i : i + 8] for i in range(0, 40, 8)], score_sink=sink, workers=1
            )
        )
    monkeypatch.setattr(_columns, "write_at", original)
    rows = sc.read_back(tmp_path / "s")[1] if list((tmp_path / "s").glob("shard-*")) else []
    assert len(rows) < 40
    assert fse.repair_store(tmp_path / "s") == len(rows)
