"""The gate of a stream: a killed run continues at the position its store reports, bit for bit
as one run would (A: the bytes, with the same batches and a stop between whole shards; B: the
rows, with other batches and workers), and no record is missing or written twice."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fse
from fse._container import PublishedNotDurableError
from fse._store import _resume_format as sync_module
from fse._store._sink import ShardSink as SinkClass
from support import positions as ps
from support.resume import ENCODER, SHARD_ROWS, canonical, sentences

SENTENCES = sentences(640)  # ten shards of 64 rows
TESTS = str(Path(__file__).resolve().parent)

CHILD = r"""
import sys, threading
sys.path.insert(0, sys.argv[1])
from fse._store import _shard
from support import positions as ps
from support.resume import ENCODER, SHARD_ROWS, sentences

path, hook, nth = sys.argv[2], sys.argv[3], int(sys.argv[4])
count, lock = [0], threading.Lock()

def reached():
    with lock:
        count[0] += 1
        hit = count[0] == nth
    if hit:
        print("reached", flush=True)
        threading.Event().wait()  # this thread stays here; the parent ends the process

real_publish = _shard.publish_directory
def publish(tmp, destination, **kwargs):
    if hook == "before_publish":
        reached()
    real_publish(tmp, destination, **kwargs)
    if hook == "after_publish":
        reached()
_shard.publish_directory = publish

if hook == "commit":
    real = _shard.ShardWriter.commit
    def commit(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        reached()
        return result
    _shard.ShardWriter.commit = commit

texts = sentences(640)
with ps.sink(path) as sink:
    for _ in ENCODER.encode_batches(ps.stream(texts, SHARD_ROWS, ps.int_start(path)), sink=sink, positions=True):
        pass
print("finished", flush=True)
"""


def _run_until_killed(path, hook, nth):
    """Run the child until it reaches the ``nth`` event of ``hook``, then SIGKILL it."""
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, TESTS, str(path), hook, str(nth)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = child.stdout.readline().strip()  # the child blocks at the hook until it is killed
        assert line == "reached", f"the child said {line!r}: {child.stderr.read()}"
    finally:
        child.kill()  # SIGKILL on POSIX, TerminateProcess on Windows
        child.communicate()
    assert child.returncode != 0


def _valid_prefix(path):
    """A killed store opens, verifies and holds the first whole shards of the run, and the
    position it reports names the batch its last shard ends in."""
    with fse.VectorStore.open(path) as store:
        store.verify()
        rows = len(store)
        assert rows % SHARD_ROWS == 0
        want = ENCODER.encode(SENTENCES[:rows], workers=1)
        np.testing.assert_array_equal(store.get_vectors().view(np.uint32), want.view(np.uint32))
        if rows:
            assert store.source_position == (f"o{rows - SHARD_ROWS}", SHARD_ROWS)
    return rows


@pytest.fixture
def whole(tmp_path):
    ps.write(tmp_path / "whole", ps.stream(SENTENCES, SHARD_ROWS))
    return canonical(tmp_path / "whole")


@pytest.mark.parametrize(
    ("hook", "nth"),
    [("before_publish", 3), ("after_publish", 3), ("commit", 4), ("after_publish", 1)],
)
def test_a_killed_stream_continues_with_the_bytes_of_one_run(tmp_path, whole, hook, nth):
    _run_until_killed(tmp_path / "s", hook, nth)
    if (tmp_path / "s" / "store.json").exists():
        _valid_prefix(tmp_path / "s")
    ps.write(tmp_path / "s", ps.stream(SENTENCES, SHARD_ROWS, start=ps.int_start(tmp_path / "s")))
    assert canonical(tmp_path / "s") == whole


def test_a_stream_killed_again_and_again_continues_with_the_bytes_of_one_run(tmp_path, whole):
    seen = []
    for hook, nth in (("after_publish", 2), ("before_publish", 2), ("commit", 3), ("commit", 2)):
        _run_until_killed(tmp_path / "s", hook, nth)
        seen.append(_valid_prefix(tmp_path / "s"))
    assert seen == sorted(seen)  # the store only grows
    ps.write(tmp_path / "s", ps.stream(SENTENCES, SHARD_ROWS, start=ps.int_start(tmp_path / "s")))
    assert canonical(tmp_path / "s") == whole


@pytest.mark.parametrize("size", [5, 64, 100, 700])
@pytest.mark.parametrize("workers", [1, 3])
def test_a_killed_stream_continues_with_the_rows_of_one_run(tmp_path, size, workers):
    _run_until_killed(tmp_path / "s", "commit", 4)
    kept = _valid_prefix(tmp_path / "s")
    results, found = ps.resume(tmp_path / "s", SENTENCES, size, workers=workers)
    assert found[0] == f"o{kept - SHARD_ROWS}" if kept else True
    codes = ps.vectors(tmp_path / "s")
    want = ENCODER.encode(SENTENCES, workers=1)
    assert len(codes) == len(SENTENCES)  # no missing and no doubled record
    np.testing.assert_array_equal(codes.view(np.uint32), want.view(np.uint32))
    assert sum(r.n_rows for r in results) == len(SENTENCES) - kept


# --- errors, and the sync of the store directory when a resume attaches ---------------------


class StopRunError(RuntimeError):
    pass


def test_an_error_after_some_commits_leaves_a_valid_prefix_and_an_anchor_inside_it(
    tmp_path, monkeypatch
):
    real, calls = SinkClass.commit, []

    def commit(self, first_row, rows, **kwargs):
        if len(calls) == 2:  # the third commit fails: two batches of 100 are done, no more
            raise StopRunError("stop")
        calls.append(first_row)
        return real(self, first_row, rows, **kwargs)

    monkeypatch.setattr(SinkClass, "commit", commit)
    with pytest.raises(StopRunError):
        ps.write(tmp_path / "s", ps.stream(SENTENCES, 100))
    monkeypatch.undo()
    kept = _valid_prefix_any(tmp_path / "s")
    assert 0 < kept < len(SENTENCES)
    anchors = ps.anchors(tmp_path / "s")
    assert all(row < first + rows for first, rows, _, row in anchors)  # R < E for every shard
    ps.resume(tmp_path / "s", SENTENCES, 70, workers=3)
    np.testing.assert_array_equal(
        ps.vectors(tmp_path / "s").view(np.uint32),
        ENCODER.encode(SENTENCES, workers=1).view(np.uint32),
    )


def _valid_prefix_any(path):
    with fse.VectorStore.open(path) as store:
        store.verify()
        rows = len(store)
        want = ENCODER.encode(SENTENCES[:rows], workers=1)
        np.testing.assert_array_equal(store.get_vectors().view(np.uint32), want.view(np.uint32))
    return rows


def test_an_error_in_flush_leaves_a_valid_prefix(tmp_path, monkeypatch):
    def flush(_self):
        raise StopRunError("flush")

    monkeypatch.setattr(SinkClass, "flush", flush)
    with pytest.raises(StopRunError):
        ps.write(tmp_path / "s", ps.stream(SENTENCES, 100))
    monkeypatch.undo()
    _valid_prefix_any(tmp_path / "s")
    ps.resume(tmp_path / "s", SENTENCES, 100)
    assert len(ps.vectors(tmp_path / "s")) == len(SENTENCES)


class Events:
    """The syncs of the store directory and the shards made, in order."""

    def __init__(self, monkeypatch):
        self.log = []
        real_sync, real_writer = sync_module._sync_directory, SinkClass._new_writer

        def sync(path, **kwargs):
            self.log.append(f"sync {Path(path).name}")
            real_sync(path, **kwargs)

        def writer(sink, row):
            self.log.append("writer")
            return real_writer(sink, row)

        monkeypatch.setattr(sync_module, "_sync_directory", sync)
        monkeypatch.setattr(SinkClass, "_new_writer", writer)


def test_a_resumed_stream_under_full_syncs_the_store_once_before_it_reserves(tmp_path, monkeypatch):
    ps.write(tmp_path / "s", ps.stream(SENTENCES, 100), stop=3)
    events = Events(monkeypatch)
    ps.resume(tmp_path / "s", SENTENCES, 100)
    assert events.log[:3] == ["sync s", f"sync {tmp_path.name}", "writer"]  # the store, its parent
    assert events.log.count("sync s") == 1


def test_a_resumed_stream_under_none_does_not_sync(tmp_path, monkeypatch):
    ps.write(tmp_path / "s", ps.stream(SENTENCES, 100), stop=3, durability="none")
    events = Events(monkeypatch)
    ps.resume(tmp_path / "s", SENTENCES, 100, durability="none")
    assert "sync s" not in events.log


def test_a_failed_sync_is_named_before_any_record_is_reserved(tmp_path, monkeypatch):
    ps.write(tmp_path / "s", ps.stream(SENTENCES, 100), stop=3)

    def failing(_path, **_kwargs):
        raise OSError("disk")

    monkeypatch.setattr(sync_module, "_sync_directory", failing)
    with (
        ps.sink(tmp_path / "s") as target,
        pytest.raises(PublishedNotDurableError, match="not confirmed"),
    ):
        ENCODER.encode_batches(ps.stream(SENTENCES, 100), sink=target, positions=True)
    assert target.published_rows == 0
