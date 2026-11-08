"""The check that a test leaves no sink writer thread: it names a thread that outlives its
test, accepts one that ends a moment later, and does not blame a test for an earlier leak."""

import threading

from support.sink_threads import left_since, sink_threads
from support.waiting import until


def _thread(name, release):
    thread = threading.Thread(target=release.wait, name=name, daemon=True)
    thread.start()
    return thread


def test_a_thread_that_outlives_the_test_is_named():
    release = threading.Event()
    before = sink_threads()
    thread = _thread("fse-shard-write", release)
    try:
        assert left_since(before, timeout=0.05) == [thread]
    finally:
        release.set()
        thread.join(10)


def test_a_thread_that_ends_a_moment_later_is_accepted(monkeypatch):
    release = threading.Event()
    before = sink_threads()
    thread = _thread("fse-shard-hash", release)
    seen_alive = []
    real = sink_threads

    def looking():  # the first look finds the thread alive, then it is let go
        found = real()
        if not seen_alive:
            seen_alive.append(thread in found)
            release.set()
        return found

    monkeypatch.setattr("support.sink_threads.sink_threads", looking)
    try:
        assert left_since(before, timeout=10) == []
    finally:
        release.set()
        thread.join(10)
    assert seen_alive == [True]


def test_a_leak_from_before_is_in_the_base_set_and_no_new_one_follows():
    release = threading.Event()
    earlier = _thread("fse-shard-publish", release)
    try:
        before = sink_threads()
        assert earlier in before
        assert left_since(before, timeout=0.05) == []
    finally:
        release.set()
        earlier.join(10)


def test_other_threads_are_not_sink_threads():
    release = threading.Event()
    before = sink_threads()
    thread = _thread("fse-worker-0", release)
    try:
        assert left_since(before, timeout=0.05) == []
    finally:
        release.set()
        thread.join(10)


def test_until_gives_the_state_at_the_deadline():
    assert until(lambda: True, timeout=0)
    assert not until(lambda: False, timeout=0.02)
