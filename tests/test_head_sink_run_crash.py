"""A process that dies while it writes two stores in one run leaves two valid prefixes; the
last confirmation is within both, and a new run with document IDs appends to them (process
crashes; a power loss is a matter of the shard writers, which this run does not change)."""

import json
import os
import signal
import subprocess
import sys
import time
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
heads = sh.group()
texts = sh.texts(300)
ids = np.arange(300, dtype=np.int64) * 3 + 1
enc = heads["quality"].encoder
batches = [(texts[i : i + 50], ids[i : i + 50]) for i in range(0, 300, 50)]
with (
    fse.ShardSink(root + "/v", enc, shard_rows=64, durability=durability, doc_ids=True) as v,
    fse.ScoreSink(root + "/s", heads, shard_rows=64, durability=durability, doc_ids=True) as s,
):
    for batch in heads.predict_batches(
        batches, return_vectors=True, vector_sink=v, score_sink=s, doc_ids=True, workers=2
    ):
        print(batch.durable_rows, flush=True)
"""

POINTS = ["score_reserved", "score_committed", "before", "after", "shrink", "between"]


@pytest.mark.parametrize("durability", ["full", "none"])
@pytest.mark.parametrize("point", POINTS)
def test_a_crash_leaves_two_valid_prefixes_and_a_new_run_with_ids_appends(
    tmp_path, point, durability
):
    heads, texts = sh.group(), sh.texts(300)
    ids = np.arange(300, dtype=np.int64) * 3 + 1
    reference = sr.reference(tmp_path, heads, texts, batch=50, shard_rows=64, ids=ids)
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
    lengths = _lengths(root, reference)
    assert last <= min(lengths)
    if durability == "none":
        assert last == 0  # nothing is confirmed without syncs
    hidden = _hidden(root / "v") + _hidden(root / "s")
    assert all(".shard-" in name and ".writing-" in name for name in hidden)
    if durability == "none":
        return  # a store written under none is not appended to without repair
    _append(root, heads, texts, ids, lengths)


def _hidden(path):
    return [p.name for p in path.iterdir() if p.name.startswith(".")] if path.exists() else []


def _lengths(root, reference):
    out = {}
    for name, tensor in (("v", "vectors"), ("s", "scores")):
        path = root / name
        got = sr.tensors(path) if (path / "store.json").exists() else {}
        n = len(got.get(tensor, ()))
        for key in sr.ROW_TENSORS & got.keys():
            assert got[key].tobytes() == reference[name][key][:n].tobytes(), (name, key)
        out[name] = n
    return out["v"], out["s"]


def _append(root, heads, texts, ids, lengths):
    """Run on from the last confirmed row with IDs: allowed whatever the two lengths are."""
    enc = heads["quality"].encoder
    start = min(lengths)
    with (
        fse.ShardSink(root / "v", enc, shard_rows=64, doc_ids=True) as v,
        fse.ScoreSink(root / "s", heads, shard_rows=64, doc_ids=True) as s,
    ):
        assert (v.first_row, s.first_row) == lengths
        list(
            heads.predict_batches(
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


SIGINT_CHILD = """
import json, os, signal, sys, time
from pathlib import Path
import fse
from fse._store import _shard
from fse._store._layout import shard_name
from support import score_heads as sh

signal.signal(signal.SIGINT, signal.default_int_handler)
root, events, hooked = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
heads = sh.group()
texts = sh.texts(300)
v = fse.ShardSink(root / "v", heads["quality"].encoder, shard_rows=64)
s = fse.ScoreSink(root / "s", heads, shard_rows=64)
sinks = {"v": v, "s": s}
real = _shard.publish_directory


def wait_for(done, what):
    deadline = time.monotonic() + 30
    while not done():
        if time.monotonic() > deadline:
            print(json.dumps({"timeout": what}), flush=True)
            os._exit(3)
        time.sleep(0.01)


def publish(tmp, path, **kwargs):
    if path.parent == root / hooked and path.name == shard_name(256):
        installed = lambda: signal.getsignal(signal.SIGINT) is not signal.default_int_handler
        wait_for(installed, "the deferral")
        (events / "ready").touch()
        wait_for(lambda: sinks[hooked]._publisher.failed, "the SIGINT")
    real(tmp, path, **kwargs)


_shard.publish_directory = publish
batches = heads.predict_batches(
    [texts[i : i + 50] for i in range(0, 300, 50)],
    return_vectors=True, vector_sink=v, score_sink=s, workers=2,
)
for _ in range(6):
    next(batches)  # the last rows are committed, not flushed
order = ["s", "v"] if hooked == "s" else ["v", "s"]
raised = 0
for name in order:
    try:
        sinks[name].close()
    except KeyboardInterrupt:
        raised += 1
print(json.dumps({
    "raised": raised,
    "hidden": sorted(p.name for r in ("v", "s") for p in (root / r).iterdir() if p.name.startswith(".")),
}))
"""


@pytest.mark.skipif(os.name == "nt", reason="no SIGINT to another process on Windows")
@pytest.mark.parametrize("hooked", ["s", "v"])
def test_a_real_sigint_during_either_close_leaves_two_valid_folders(tmp_path, hooked):
    events = tmp_path / "events"
    events.mkdir()
    root = tmp_path / "run"
    root.mkdir()
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")]),
    }
    child = subprocess.Popen(
        [sys.executable, "-c", SIGINT_CHILD, str(root), str(events), hooked],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        deadline = time.monotonic() + 60
        while not (events / "ready").exists():
            assert child.poll() is None, "the child ended early"
            assert time.monotonic() < deadline, "no ready event within 60 s"
            time.sleep(0.01)
        child.send_signal(signal.SIGINT)
        out, err = child.communicate(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    assert child.returncode == 0, err
    result = json.loads(out.strip().splitlines()[-1])
    assert result == {"raised": 1, "hidden": []}
    heads = sh.group()
    reference = sr.reference(tmp_path, heads, sh.texts(300), batch=50, shard_rows=64)
    lengths = _lengths(root, reference)
    assert min(lengths) >= 256  # the published prefix stays
    fse.VectorStore.open(root / "v").close()
