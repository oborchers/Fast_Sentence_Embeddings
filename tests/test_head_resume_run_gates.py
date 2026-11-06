"""The gates of two resumable stores: a killed run continues bit for bit as one run would
(gate A: the bytes; gate B: the rows and the progress), also when it dies in the first phase,
in which only the shorter store is written."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fse
from fse._head import _resume_run
from support.resume_heads import (
    ENCODER,
    HEADS,
    SHARD_ROWS,
    cut,
    run,
    score_sink,
    scores_of,
    sentences,
    stores,
    vector_sink,
    vectors_of,
)
from support.runtime import small_tasks

SENTENCES = sentences(640)  # ten shards of 64 rows
TESTS = str(Path(__file__).resolve().parent)

CHILD = r"""
import sys, threading
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from fse._store import _shard
from support.resume_heads import HEADS, sentences, vector_sink, score_sink

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

if hook == "write":
    original = _shard.ShardWriter.write
    def wrapped(self, *args, **kwargs):
        reached()
        return original(self, *args, **kwargs)
    _shard.ShardWriter.write = wrapped

Path(path).mkdir(parents=True, exist_ok=True)
with vector_sink(path + "/v") as vectors, score_sink(path + "/s") as scores:
    HEADS.predict(sentences(640), return_vectors=True, vector_sink=vectors,
                  score_sink=scores, workers=2)
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


def _prefixes_are_valid(path, *, whole_shards=True):
    """Both killed stores open, verify and hold the first whole shards of the run."""
    rows = []
    for name in ("v", "s"):
        if not (path / name / "store.json").exists():
            rows.append(0)
            continue
        opened = fse.VectorStore if name == "v" else fse.ScoreStore
        with opened.open(path / name) as store:
            store.verify()
            rows.append(len(store))
        assert not whole_shards or rows[-1] % SHARD_ROWS == 0
    vectors = vectors_of(path / "v") if rows[0] else None
    if vectors is not None:
        want = ENCODER.encode(SENTENCES[: rows[0]], workers=1)
        np.testing.assert_array_equal(vectors.view(np.uint32), want.view(np.uint32))
    return tuple(rows)


@pytest.fixture(scope="module")
def whole(tmp_path_factory):
    path = tmp_path_factory.mktemp("whole")
    run(path, SENTENCES)
    return stores(path)


# --- gate A: the bytes ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hook", "nth"),
    [
        ("before_publish", 3),
        ("after_publish", 3),
        ("write", 4),
        ("after_publish", 1),
    ],
)
def test_two_killed_stores_continue_with_the_bytes_of_one_run(tmp_path, whole, hook, nth):
    _run_until_killed(tmp_path / "s", hook, nth)
    _prefixes_are_valid(tmp_path / "s")
    run(tmp_path / "s", SENTENCES)
    assert stores(tmp_path / "s") == whole


@pytest.mark.parametrize(("vector_shards", "score_shards"), [(6, 3), (3, 6)])
def test_a_run_killed_early_continues_with_the_bytes_of_one_run(
    tmp_path, whole, vector_shards, score_shards
):
    """The kill falls at the first publication of the child. Which store has grown by then
    depends on the threads (the second phase can start while the hook holds the publisher),
    so only the prefixes and the end are checked; the first phase alone is the next test."""
    directory = tmp_path / "s"
    run(directory, SENTENCES)
    cut(directory, vector_shards, score_shards)
    before = _prefixes_are_valid(directory)
    _run_until_killed(directory, "after_publish", 1)
    after = _prefixes_are_valid(directory)
    assert all(a >= b for a, b in zip(after, before, strict=True))
    run(directory, SENTENCES)
    assert stores(directory) == whole


@pytest.mark.parametrize(("vector_shards", "score_shards"), [(6, 3), (3, 6)])
def test_a_run_that_stops_after_its_first_phase_has_written_the_shorter_store_only(
    tmp_path, monkeypatch, whole, vector_shards, score_shards
):
    directory = tmp_path / "s"
    run(directory, SENTENCES)
    cut(directory, vector_shards, score_shards)
    real = _resume_run._run_phase
    phases_run = []

    def stopping(run_, phase):
        if phases_run:
            raise StopRunError("after the first phase")
        phases_run.append(phase)
        real(run_, phase)

    monkeypatch.setattr(_resume_run, "_run_phase", stopping)
    with pytest.raises(StopRunError):
        run(directory, SENTENCES)
    monkeypatch.undo()
    longer = max(vector_shards, score_shards) * SHARD_ROWS
    assert _prefixes_are_valid(directory) == (longer, longer)  # the shorter one caught up
    run(directory, SENTENCES)
    assert stores(directory) == whole


def test_a_run_killed_again_and_again_continues_with_the_bytes_of_one_run(tmp_path, whole):
    directory = tmp_path / "s"
    seen = []
    for hook, nth in (
        ("after_publish", 2),
        ("before_publish", 2),
        ("write", 3),
        ("after_publish", 1),
    ):
        _run_until_killed(directory, hook, nth)
        seen.append(_prefixes_are_valid(directory))
    for column in zip(*seen, strict=True):
        assert list(column) == sorted(column)  # a store only grows
    run(directory, SENTENCES)
    assert stores(directory) == whole


# --- gate B: rows and progress -----------------------------------------------------------


class StopRunError(RuntimeError):
    pass


def _same_rows(directory, whole):
    """The rows of the stores are those of the uninterrupted run: none missing, none doubled."""
    vectors, scores = vectors_of(directory / "v"), scores_of(directory / "s")[0]
    assert len(vectors) == len(scores) == len(SENTENCES)
    np.testing.assert_array_equal(
        vectors.view(np.uint32), ENCODER.encode(SENTENCES, workers=1).view(np.uint32)
    )
    np.testing.assert_array_equal(scores.view(np.uint32), scores_of(whole / "s")[0].view(np.uint32))


@pytest.fixture(scope="module")
def whole_directory(tmp_path_factory):
    path = tmp_path_factory.mktemp("whole_rows")
    run(path, SENTENCES)
    return path


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize(("vector_shards", "score_shards"), [(0, 0), (6, 3), (3, 6)])
def test_an_error_after_some_commits_leaves_valid_prefixes(
    tmp_path, monkeypatch, whole_directory, workers, vector_shards, score_shards
):
    small_tasks(monkeypatch, chunk_min=64)
    if vector_shards or score_shards:
        run(tmp_path, SENTENCES)
        cut(tmp_path, vector_shards, score_shards)
    real, calls = fse.ScoreSink.commit, []

    def commit(self, first_row, rows, **kwargs):
        if len(calls) == 1:  # the second task fails
            raise StopRunError("stop")
        calls.append(first_row)
        return real(self, first_row, rows, **kwargs)

    monkeypatch.setattr(fse.ScoreSink, "commit", commit)
    with pytest.raises(StopRunError):
        run(tmp_path, SENTENCES, workers=workers)
    monkeypatch.undo()
    _prefixes_are_valid(tmp_path, whole_shards=False)  # a shortened shard is possible
    small_tasks(monkeypatch, chunk_min=128)  # other task boundaries on the replay
    run(tmp_path, SENTENCES, workers=2)
    _same_rows(tmp_path, whole_directory)


def test_an_error_in_flush_leaves_valid_prefixes(tmp_path, whole_directory):
    real = fse.ShardSink.flush

    def failing(_self):
        raise StopRunError("flush")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fse.ShardSink, "flush", failing)
        with pytest.raises(StopRunError):
            run(tmp_path, SENTENCES)
    assert fse.ShardSink.flush is real
    _prefixes_are_valid(tmp_path, whole_shards=False)
    run(tmp_path, SENTENCES, workers=3)
    _same_rows(tmp_path, whole_directory)


def test_a_resume_after_a_shortened_shard_gives_the_same_rows(
    tmp_path, monkeypatch, whole_directory
):
    """A run that stopped after 100 rows and sealed its shards: the next run goes on after them."""
    small_tasks(monkeypatch, chunk_min=64)
    with (
        vector_sink(tmp_path / "v") as vectors,
        score_sink(tmp_path / "s") as scores,
    ):
        real = scores.commit

        def commit(first_row, rows, **kwargs):
            real(first_row, rows, **kwargs)
            if first_row + len(rows) >= 100:
                scores.seal()
                vectors.seal()
                raise StopRunError("stop")

        scores.commit = commit
        with pytest.raises(StopRunError):
            HEADS.predict(
                SENTENCES, return_vectors=True, vector_sink=vectors, score_sink=scores, workers=1
            )
    rows = _prefixes_are_valid(tmp_path, whole_shards=False)
    assert 0 < sum(rows) < 2 * len(SENTENCES)
    run(tmp_path, SENTENCES, workers=3)
    _same_rows(tmp_path, whole_directory)


def test_the_tags_of_the_first_run_stay(tmp_path):
    run(tmp_path / "a", SENTENCES, durability="none")
    run(tmp_path / "b", SENTENCES, durability="none")
    cut(tmp_path / "b", 6, 3)
    run(tmp_path / "b", SENTENCES, durability="none", workers=3)
    for name in ("v", "s"):
        for entry in sorted((tmp_path / "b" / name).glob("shard-*")):
            metadata = json.loads((entry / "manifest.json").read_text())["metadata"]
            assert metadata["fse.durability"] == "none"
