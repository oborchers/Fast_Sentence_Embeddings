"""A process that dies while a single head writes two stores in one run leaves two valid
prefixes; the last confirmation is within both, and a new run with document IDs appends to
them (the run is that of a group of one: tests/test_head_sink_run_crash.py; a power loss is
a matter of the shard writers, which this run does not change)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fse
from support import score_heads as sh
from support import sink_run as sr

TESTS = Path(__file__).parent

CHILD = """
import os, sys, threading
sys.path.insert(0, sys.argv[3])
import numpy as np
import fse
from fse._store import _columns, _shard
from support import score_heads as sh

root, point, durability = sys.argv[1], sys.argv[2], sys.argv[4]
calls, lock = [0], threading.Lock()
real_publish, real_shrink = _shard.publish_directory, _columns.shrink
real_target, real_commit = fse.ShardSink.target, fse.ScoreSink.commit
real_score_publish = fse.ScoreSink.publish


def target(self, first_row, n_rows):
    if point == "score_reserved" and first_row == 100:
        os._exit(3)  # the scores of the task are reserved, the vectors are not
    return real_target(self, first_row, n_rows)


def commit(self, first_row, rows, **kwargs):
    real_commit(self, first_row, rows, **kwargs)
    if point == "score_committed" and first_row == 100:
        os._exit(3)  # the scores are committed, the vectors are not


def publish(tmp, path, **kwargs):
    with lock:
        calls[0] += 1
        n = calls[0]
    if point == "before" and n == 5:
        os._exit(3)  # a shard is written and synced, not renamed
    real_publish(tmp, path, **kwargs)
    if point == "after" and n == 5:
        os._exit(3)  # a shard is published, no sink learns it


def shrink(*args):
    if point == "shrink":
        os._exit(3)
    real_shrink(*args)


def score_publish(self):
    if point == "between" and calls[0] >= 3:
        os._exit(3)  # the vector sink published, the score sink did not yet
    real_score_publish(self)


fse.ShardSink.target, fse.ScoreSink.commit = target, commit
_shard.publish_directory, _columns.shrink = publish, shrink
fse.ScoreSink.publish = score_publish
heads = sh.classifier(sh.encoder(), 3, seed=4)
texts = sh.texts(300)
ids = np.arange(300, dtype=np.int64) * 3 + 1
enc = heads.encoder
batches = [(texts[i : i + 50], ids[i : i + 50]) for i in range(0, 300, 50)]
with (
    fse.ShardSink(root + "/v", enc, shard_rows=64, durability=durability, doc_ids=True) as v,
    fse.ScoreSink(root + "/s", heads, name="m", shard_rows=64, durability=durability, doc_ids=True) as s,
):
    for batch in heads.predict_batches(
        batches, return_vectors=True, vector_sink=v, score_sink=s, doc_ids=True, workers=2
    ):
        print(batch.durable_rows, flush=True)
"""

POINTS = ["score_reserved", "score_committed", "before", "after", "between"]


def _run(root, **kwargs):
    clf, texts = sh.classifier(sh.encoder(), 3, seed=4), sh.texts(300)
    ids = np.arange(300, dtype=np.int64) * 3 + 1
    with (
        fse.ShardSink(root / "v", clf.encoder, shard_rows=64, doc_ids=True) as v,
        fse.ScoreSink(root / "s", clf, name="m", shard_rows=64, doc_ids=True) as s,
    ):
        batches = [(texts[i : i + 50], ids[i : i + 50]) for i in range(0, 300, 50)]
        list(
            clf.predict_batches(
                batches,
                return_vectors=True,
                vector_sink=v,
                score_sink=s,
                doc_ids=True,
                workers=2,
                **kwargs,
            )
        )


def _reference(tmp_path):
    ref = tmp_path / "ref"
    ref.mkdir()
    _run(ref)
    return {"v": sr.tensors(ref / "v"), "s": sr.tensors(ref / "s")}


@pytest.mark.parametrize("durability", ["full", "none"])
@pytest.mark.parametrize("point", POINTS)
def test_a_crash_leaves_two_valid_prefixes_and_a_new_run_with_ids_appends(
    tmp_path, point, durability
):
    reference = _reference(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(root), point, str(TESTS), durability],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 3, done.stderr
    confirmed = [int(line) for line in done.stdout.split() if line.isdigit()]
    last = confirmed[-1] if confirmed else 0
    lengths = []
    for name, tensor, store in (("v", "vectors", fse.VectorStore), ("s", "scores", fse.ScoreStore)):
        path = root / name
        got = sr.tensors(path) if (path / "store.json").exists() else {}
        rows = len(got.get(tensor, ()))
        for key in sr.ROW_TENSORS & got.keys():
            assert got[key].tobytes() == reference[name][key][:rows].tobytes(), (name, key)
        if (path / "store.json").exists():
            with store.open(path) as opened:
                opened.verify()
        lengths.append(rows)
    assert last <= min(lengths)
    if durability == "none":
        assert last == 0  # nothing is confirmed without syncs
        return  # a store written under none is not appended to without repair
    _append(root, lengths, last)


def _append(root, lengths, start):
    clf, texts = sh.classifier(sh.encoder(), 3, seed=4), sh.texts(300)
    ids = np.arange(300, dtype=np.int64) * 3 + 1
    with (
        fse.ShardSink(root / "v", clf.encoder, shard_rows=64, doc_ids=True) as v,
        fse.ScoreSink(root / "s", clf, name="m", shard_rows=64, doc_ids=True) as s,
    ):
        assert (v.first_row, s.first_row) == tuple(lengths)
        list(
            clf.predict_batches(
                [(texts[start:], ids[start:])],
                return_vectors=True,
                vector_sink=v,
                score_sink=s,
                doc_ids=True,
            )
        )
    _, vector_ids = sr.read_vectors(root / "v")
    _, score_ids = sr.read_scores(root / "s")
    assert vector_ids[-(300 - start) :].tolist() == ids[start:].tolist()
    assert score_ids[-(300 - start) :].tolist() == ids[start:].tolist()
