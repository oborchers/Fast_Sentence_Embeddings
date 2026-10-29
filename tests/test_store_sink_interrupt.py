"""An interrupt (Ctrl+C) while a ShardSink closes: every thread ends, the running work
is finished, nothing after it, and one KeyboardInterrupt is raised after the clean-up."""

import os

import numpy as np
import pytest

from fse._store import ShardSink
from fse._store._layout import StoreInfo, shard_name, shards
from support.encoder import unfolded

ENCODER = unfolded()
TIMEOUT = 10  # seconds; every wait fails the test when it runs out


def _commit(sink, first, n, value=0.0):
    target = sink.target(first, n)
    target[0][...] = value
    target[1][...] = 0
    sink.commit(first, target)


def _publishing(tmp_path, monkeypatch):
    """A sink with shard 1 published, the publication of shard 2 running until every
    thread is abandoned, and shard 3 written but not full; close goes on at once."""
    import threading

    from fse._store import _shard

    sink = ShardSink(tmp_path / "s", ENCODER, shard_rows=4)
    _commit(sink, 0, 4)
    assert sink.publish() == 4  # a published shard
    entered, release, real_publish = threading.Event(), threading.Event(), _shard.publish_directory
    held = []  # checked after close: an assert inside it would be masked

    def publish(tmp, path):
        if path.name == shard_name(4):
            entered.set()
            held.append(release.wait(TIMEOUT))  # the publisher runs this while close stops
        real_publish(tmp, path)

    monkeypatch.setattr(_shard, "publish_directory", publish)
    _commit(sink, 4, 4, 1.0)  # shard 2, full: handed to the publisher
    assert entered.wait(TIMEOUT)
    _commit(sink, 8, 2, 2.0)  # shard 3, not full: still a temporary directory
    sink._hash.settle()  # its rows are committed and written; only the publisher runs
    sink._io.settle()
    real_abandon = sink._publisher.abandon

    def abandon():
        real_abandon()
        release.set()  # the running piece goes on once every thread is abandoned

    sink._publisher.abandon = abandon
    sink._settle = sink._drain = lambda: None  # close goes on while the publisher runs
    return sink, release, held


@pytest.mark.usefixtures("python_sigint")
@pytest.mark.parametrize("at", ["hash", "io", "publisher"])
def test_an_interrupt_at_each_stop_of_close_finishes_the_running_work_only(
    tmp_path, monkeypatch, at
):
    import signal

    from fse._store import _shard

    sink, release, held = _publishing(tmp_path, monkeypatch)
    seen = []  # checked after close: an assert inside it would be masked
    threads = (sink._hash, sink._io, sink._publisher)
    thread = dict(zip(("hash", "io", "publisher"), threads, strict=True))[at]
    real_stop, real_abort, aborted = thread.stop, _shard.ShardWriter.abort, []

    def stop():
        signal.raise_signal(signal.SIGINT)  # Ctrl+C as close stops this thread
        seen.append(("stop", len(sink._interrupts), [t.failed for t in threads]))
        real_stop()

    def abort(self):
        aborted.append(self.first_row)
        signal.raise_signal(signal.SIGINT)  # and again during the clean-up
        seen.append(("abort", len(sink._interrupts)))
        real_abort(self)

    thread.stop = stop
    monkeypatch.setattr(_shard.ShardWriter, "abort", abort)
    try:
        with pytest.raises(KeyboardInterrupt):
            sink.close()
    finally:
        release.set()
    # Both interrupts were handled at once and only noted; the first abandoned all three
    # threads together, and the publish that ran went on only after that.
    assert seen == [("stop", 1, [True, True, True]), ("abort", 2)]
    assert held == [True]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert all(t._ended.is_set() for t in threads)
    assert aborted == [8]  # shard 3 was dropped, its directory removed
    assert not [p for p in (tmp_path / "s").iterdir() if p.name.startswith(".")]
    assert sink.durable_rows == 8  # the running publish of shard 2 was finished
    found = shards(tmp_path / "s", StoreInfo(ENCODER.fingerprint, "float32", ENCODER.output_dim))
    assert [(s.first_row, s.rows) for s, _ in found] == [(0, 4), (4, 4)]
    np.testing.assert_array_equal(found[1][1]["vectors"], np.ones((4, ENCODER.output_dim)))
    sink.close()  # closed: nothing more to do
    assert sink.durable_rows == 8


_CHILD = """
import json, os, signal, sys, time
from pathlib import Path
from fse._store import ShardSink, _shard
from fse._store._layout import shard_name
from support.encoder import unfolded

signal.signal(signal.SIGINT, signal.default_int_handler)  # also when started ignoring it
store, events = Path(sys.argv[1]), Path(sys.argv[2])
encoder = unfolded()
sink = ShardSink(store, encoder, shard_rows=4)
real = _shard.publish_directory


def wait_for(done, what):
    deadline = time.monotonic() + 30
    while not done():
        if time.monotonic() > deadline:
            print(json.dumps({"timeout": what}), flush=True)
            os._exit(3)  # from the publisher thread: end the child at once
        time.sleep(0.01)


def publish(tmp, path):
    if path.name == shard_name(4):
        # Close has deferred SIGINT; the parent sends it now, then once more.
        installed = lambda: signal.getsignal(signal.SIGINT) is not signal.default_int_handler
        wait_for(installed, "the deferral")
        (events / "first").touch()
        wait_for(lambda: sink._publisher.failed, "the first SIGINT")  # it abandoned all
        (events / "second").touch()
        wait_for(lambda: len(sink._interrupts) == 2, "the second SIGINT")  # only noted
    real(tmp, path)


_shard.publish_directory = publish
for first, n in ((0, 4), (4, 4), (8, 2)):
    target = sink.target(first, n)
    target[0][...] = first
    target[1][...] = 0
    sink.commit(first, target)
    if first == 0:
        sink.publish()
raised = 0
try:
    sink.close()
except KeyboardInterrupt:
    raised += 1
threads = (sink._hash, sink._io, sink._publisher)
print(json.dumps({
    "raised": raised,
    "interrupts": len(sink._interrupts),
    "durable": sink.durable_rows,
    "ended": all(t._ended.is_set() for t in threads),
    "hidden": sorted(p.name for p in store.iterdir() if p.name.startswith(".")),
}))
"""


@pytest.mark.skipif(os.name == "nt", reason="no SIGINT to another process on Windows")
def test_a_real_sigint_during_close_is_deferred_until_the_clean_up_is_done(tmp_path):
    import json
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    events = tmp_path / "events"
    events.mkdir()
    support = str(Path(__file__).parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([support, os.environ.get("PYTHONPATH", "")])}
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(tmp_path / "s"), str(events)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        for name in ("first", "second"):
            deadline = time.monotonic() + 30
            while not (events / name).exists():
                assert child.poll() is None, "the child ended early"
                assert time.monotonic() < deadline, f"no {name} event within 30 s"
                time.sleep(0.01)
            child.send_signal(signal.SIGINT)  # only once the child waits for it
        out, err = child.communicate(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    assert child.returncode == 0, err
    result = json.loads(out.strip().splitlines()[-1])
    assert result == {"raised": 1, "interrupts": 2, "durable": 8, "ended": True, "hidden": []}
