"""The gates of a resumable store: a killed run continues bit for bit as one run would
(gate A: the bytes; gate B: the rows and the progress), and a resume confirms the entries of
the store directory before it counts any row as durable."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fse
from fse import ShardSink
from fse._container import PublishedNotDurableError, _npy_inplace
from fse._store import _sink as sink_module
from fse._store._sink import ShardSink as SinkClass
from support.resume import ENCODER, SHARD_ROWS, canonical, drop_tail, rows_of, sentences, write
from support.runtime import small_tasks

SENTENCES = sentences(640)  # ten shards of 64 rows
TESTS = str(Path(__file__).resolve().parent)

CHILD = r"""
import sys, threading
sys.path.insert(0, sys.argv[1])
from fse import ShardSink
from fse._store import _shard
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

for name in ("write", "commit"):
    if hook == name:
        real = getattr(_shard.ShardWriter, name)
        def wrapped(self, *args, _real=real, _name=name, **kwargs):
            if _name == "write":
                reached()
            result = _real(self, *args, **kwargs)
            if _name == "commit":
                reached()
            return result
        setattr(_shard.ShardWriter, name, wrapped)

with ShardSink(path, ENCODER, shard_rows=SHARD_ROWS, resume=True) as sink:
    ENCODER.encode(sentences(640), workers=1, sink=sink)
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


def _prefix_is_valid(path):
    """A killed store opens, verifies and holds the first whole shards of the run."""
    with fse.VectorStore.open(path) as store:
        store.verify()
        rows = len(store)
        assert rows % SHARD_ROWS == 0
        want = ENCODER.encode(SENTENCES[:rows], workers=1)
        np.testing.assert_array_equal(store.get_vectors().view(np.uint32), want.view(np.uint32))
    return rows


@pytest.fixture
def whole(tmp_path):
    write(tmp_path / "whole", SENTENCES)
    return canonical(tmp_path / "whole")


@pytest.mark.parametrize(
    ("hook", "nth"),
    [
        ("before_publish", 3),
        ("after_publish", 3),
        ("write", 4),
        ("commit", 4),
        ("after_publish", 1),
    ],
)
def test_a_killed_run_continues_with_the_bytes_of_one_run(tmp_path, whole, hook, nth):
    _run_until_killed(tmp_path / "s", hook, nth)
    if (tmp_path / "s" / "store.json").exists():
        _prefix_is_valid(tmp_path / "s")
    write(tmp_path / "s", SENTENCES)
    assert canonical(tmp_path / "s") == whole


def test_a_run_killed_again_and_again_continues_with_the_bytes_of_one_run(tmp_path, whole):
    seen = []
    for hook, nth in (("after_publish", 2), ("before_publish", 2), ("write", 3), ("commit", 2)):
        _run_until_killed(tmp_path / "s", hook, nth)
        seen.append(_prefix_is_valid(tmp_path / "s"))
    assert seen == sorted(seen)  # the store only grows
    write(tmp_path / "s", SENTENCES)
    assert canonical(tmp_path / "s") == whole


# --- gate B: rows and progress --------------------------------------------------------------


def _want_rows():
    return ENCODER.encode(SENTENCES, workers=1).view(np.uint32)


def _same_rows(path):
    codes, _ = rows_of(path)
    assert len(codes) == len(SENTENCES)  # no missing and no doubled row
    np.testing.assert_array_equal(codes.view(np.uint32), _want_rows())


class StopRunError(RuntimeError):
    pass


@pytest.mark.parametrize("workers", [1, 3])
def test_an_error_after_some_commits_leaves_a_valid_prefix(tmp_path, monkeypatch, workers):
    small_tasks(monkeypatch, chunk_min=64)
    real, calls = SinkClass.commit, []

    def commit(self, first_row, rows, **kwargs):
        if len(calls) == 1:  # the second task fails
            raise StopRunError("stop")
        calls.append(first_row)
        return real(self, first_row, rows, **kwargs)

    monkeypatch.setattr(SinkClass, "commit", commit)
    with pytest.raises(StopRunError):
        write(tmp_path / "s", SENTENCES, workers=workers)
    monkeypatch.undo()
    with fse.VectorStore.open(tmp_path / "s") as store:
        store.verify()
        kept = len(store)
    assert 0 < kept < len(SENTENCES)  # a shortened shard is possible
    small_tasks(monkeypatch, chunk_min=128)  # other task boundaries on the replay
    write(tmp_path / "s", SENTENCES, workers=2)
    _same_rows(tmp_path / "s")


def test_an_error_in_flush_leaves_a_valid_prefix(tmp_path, monkeypatch):
    def flush(_self):
        raise StopRunError("flush")

    monkeypatch.setattr(SinkClass, "flush", flush)
    with pytest.raises(StopRunError):
        write(tmp_path / "s", SENTENCES)
    monkeypatch.undo()
    _prefix_is_valid_or_whole(tmp_path / "s")
    write(tmp_path / "s", SENTENCES, workers=3)
    _same_rows(tmp_path / "s")


def _prefix_is_valid_or_whole(path):
    with fse.VectorStore.open(path) as store:
        store.verify()
        np.testing.assert_array_equal(
            store.get_vectors().view(np.uint32),
            ENCODER.encode(SENTENCES[: len(store)]).view(np.uint32),
        )


def test_a_resume_after_a_shortened_shard_gives_the_same_rows(tmp_path):
    write(tmp_path / "s", SENTENCES[:100])  # not this source: a prefix store cannot resume it
    with pytest.raises(ValueError, match="another source or partition"):
        write(tmp_path / "s", SENTENCES)
    # a shortened shard of the right source: stop after 100 rows by sealing
    path = tmp_path / "t"
    with ShardSink(path, ENCODER, shard_rows=SHARD_ROWS, resume=True) as sink:
        real = sink.commit

        def commit(first_row, rows, **kwargs):
            real(first_row, rows, **kwargs)
            if first_row + len(rows[0]) >= 100:
                sink.seal()
                raise StopRunError("stop")

        sink.commit = commit
        with pytest.raises(StopRunError):
            ENCODER.encode(SENTENCES, workers=1, sink=sink)
    _prefix_is_valid_or_whole(path)
    write(path, SENTENCES, workers=3)
    _same_rows(path)


def test_the_tags_of_the_first_run_stay(tmp_path):
    write(tmp_path / "s", SENTENCES, durability="none")
    drop_tail(tmp_path / "s", 4)
    write(tmp_path / "s", SENTENCES, durability="none", workers=3)
    _same_rows(tmp_path / "s")
    for entry in sorted(p for p in (tmp_path / "s").iterdir() if p.name.startswith("shard-")):
        metadata = json.loads((entry / "manifest.json").read_text())["metadata"]
        assert metadata["fse.durability"] == "none"


# --- the sync of the store directory when a resume attaches ---------------------------------


class Events:
    """The syncs of the store directory and the shards made, in order."""

    def __init__(self, monkeypatch):
        self.log = []
        real_sync, real_writer = sink_module._sync_directory, SinkClass._new_writer

        def sync(path, **kwargs):
            self.log.append(f"sync {Path(path).name}")
            real_sync(path, **kwargs)

        def writer(sink, row):
            self.log.append("writer")
            return real_writer(sink, row)

        monkeypatch.setattr(sink_module, "_sync_directory", sync)
        monkeypatch.setattr(SinkClass, "_new_writer", writer)


def test_a_resume_under_full_syncs_the_store_once_before_it_reserves(tmp_path, monkeypatch):
    write(tmp_path / "s", SENTENCES)
    drop_tail(tmp_path / "s", 3)
    events = Events(monkeypatch)
    write(tmp_path / "s", SENTENCES)
    assert events.log[:3] == ["sync s", f"sync {tmp_path.name}", "writer"]  # the store, its parent
    assert events.log.count("sync s") == 1


def test_a_resume_under_none_does_not_sync(tmp_path, monkeypatch):
    write(tmp_path / "s", SENTENCES, durability="none")
    drop_tail(tmp_path / "s", 3)
    events = Events(monkeypatch)
    write(tmp_path / "s", SENTENCES, durability="none")
    assert "sync s" not in events.log


def test_a_failed_sync_stops_the_resume_before_any_reservation(tmp_path, monkeypatch):
    write(tmp_path / "s", SENTENCES)
    drop_tail(tmp_path / "s", 3)
    before = canonical(tmp_path / "s")
    events = Events(monkeypatch)

    def failing(*_args, **_kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(sink_module, "_sync_directory", failing)
    with pytest.raises(PublishedNotDurableError, match="not confirmed durable"):
        write(tmp_path / "s", SENTENCES)
    assert "writer" not in events.log
    assert canonical(tmp_path / "s") == before


def test_a_rename_that_was_visible_but_not_synced_is_confirmed_by_the_resume(tmp_path, monkeypatch):
    """The power-loss model: the process ended after the rename of a shard and before the sync
    of its directory (the first run ended with PublishedNotDurableError). The next run
    attaches, syncs the directory and only then counts the shard as kept."""
    store = tmp_path / "s"
    real = _npy_inplace._sync_directory

    def failing_once(path, **kwargs):
        if Path(path) == store and not hasattr(failing_once, "failed"):
            failing_once.failed = True
            raise OSError(5, "Input/output error")
        real(path, **kwargs)

    monkeypatch.setattr(_npy_inplace, "_sync_directory", failing_once)
    with pytest.raises(PublishedNotDurableError):
        write(store, SENTENCES)
    monkeypatch.undo()
    assert (store / "shard-0000000000000000").is_dir()  # visible, untagged, not confirmed
    events = Events(monkeypatch)
    sink = write(store, SENTENCES)
    assert events.log[0] == "sync s"
    assert sink.first_row >= SHARD_ROWS  # the shard counts as kept
    assert sink.durable_rows == sink.published_rows  # and the new rows are durable
    _same_rows(store)


def test_a_resume_syncs_the_parent_of_a_store_an_earlier_run_made(tmp_path, monkeypatch):
    """The first run ended after the rename of the whole store and before the sync of its
    parent: the next run confirms the entry of the store too."""
    write(tmp_path / "s", SENTENCES)
    drop_tail(tmp_path / "s", 3)
    seen = []
    real = sink_module._sync_directory

    def failing_parent(path, **kwargs):
        seen.append(Path(path))
        if Path(path) == tmp_path:
            raise OSError(5, "Input/output error")
        real(path, **kwargs)

    monkeypatch.setattr(sink_module, "_sync_directory", failing_parent)
    with pytest.raises(PublishedNotDurableError, match="not confirmed durable"):
        write(tmp_path / "s", SENTENCES)
    assert seen == [tmp_path / "s", tmp_path]


def test_a_closed_sink_and_bad_workers_are_refused_before_the_store_is_made(tmp_path):
    sink = ShardSink(tmp_path / "s", ENCODER, shard_rows=SHARD_ROWS, resume=True)
    with pytest.raises(ValueError, match="workers"):
        ENCODER.encode(SENTENCES, sink=sink, workers=0)
    sink.close()
    with pytest.raises(RuntimeError, match="closed"):
        ENCODER.encode(SENTENCES, sink=sink)
    assert not (tmp_path / "s").exists()


@pytest.mark.parametrize("workers", [0, True, -1])
def test_workers_are_checked_where_no_row_is_left_to_run(tmp_path, workers):
    write(tmp_path / "s", SENTENCES)
    with (
        ShardSink(tmp_path / "s", ENCODER, shard_rows=SHARD_ROWS, resume=True) as sink,
        pytest.raises((ValueError, TypeError)),
    ):
        ENCODER.encode(SENTENCES, sink=sink, workers=workers)
    with pytest.raises((ValueError, TypeError)):
        write(tmp_path / "e", [], workers=workers)


def test_ids_blank_lines_and_a_missing_final_newline_in_a_part(tmp_path):
    rows = 40_000
    source = [s if i % 7 else "" for i, s in enumerate(sentences(rows))]
    path = tmp_path / "t.txt"
    path.write_text("\n".join(source), encoding="utf-8")  # no final newline
    ids = np.arange(5_000_000, 5_000_000 + rows, dtype=np.int64)
    kwargs = {"doc_ids": True, "ids": ids, "shard": (2, 3), "rows": 4096}
    write(tmp_path / "whole", path, **kwargs)
    write(tmp_path / "part", path, **kwargs)
    drop_tail(tmp_path / "part", 1, rows=4096)
    write(tmp_path / "part", path, **kwargs)
    assert canonical(tmp_path / "part") == canonical(tmp_path / "whole")
    first = 32_768
    codes, _ = rows_of(tmp_path / "part")
    want = ENCODER.encode(source[first:], workers=2)
    np.testing.assert_array_equal(codes.view(np.uint32), want.view(np.uint32))
    with fse.VectorStore.open(tmp_path / "part") as store:
        np.testing.assert_array_equal(store.get_doc_ids(), ids[first:])
