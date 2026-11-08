"""Sign-bit backfills share pool threads, permits and call lifetimes."""

import inspect
import threading

import pytest
from pydantic import ValidationError

import fse
from fse._store import _add_bits, _bits_write, _search_run
from fse._store._bits_format import bits_name
from support.bits import signed_store, tree
from support.concurrent import Readers
from support.pool import no_fse_threads_left  # noqa: F401
from support.waiting import arrived, blocked_in_wait


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("sizes", [(5,), (5, 3, 2)])
def test_backfill_on_pool_writes_identical_bytes(tmp_path, monkeypatch, dtype, sizes):
    own, shared = tmp_path / "own", tmp_path / "shared"
    for path in (own, shared):
        signed_store(path, dtype=dtype, sizes=sizes)
    assert fse.add_sign_bits(own, workers=2) == len(sizes)
    real = _bits_write.write_bits
    names = []

    def write(*args, **kwargs):
        names.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(_bits_write, "write_bits", write)
    with fse.Pool(2, max_tasks=1) as pool:
        assert fse.add_sign_bits(shared, pool=pool) == len(sizes)
        assert fse.add_sign_bits(shared, pool=pool) == 0
        assert not pool.closed
    for row in (0, 5, 8)[: len(sizes)]:
        assert tree(own / bits_name(row)) == tree(shared / bits_name(row))
    assert names
    # One shard follows the shared runner too, but its task runs inline inside the pool.
    if len(sizes) > 1:
        assert all(name.startswith("fse-worker") for name in names)


@pytest.mark.parametrize("kind", ["type", "closed", "too_many"])
def test_pool_is_validated_before_reading_store(tmp_path, monkeypatch, kind):
    def no_read(*_args, **_kwargs):
        raise AssertionError("store was read before pool validation")

    monkeypatch.setattr(_add_bits, "read_store", no_read)
    with fse.Pool(2) as pool:
        kwargs = {"pool": pool}
        if kind == "type":
            kwargs["pool"] = object()
            error = (TypeError, ValidationError)
        elif kind == "closed":
            pool.close()
            error = RuntimeError
        else:
            kwargs["workers"] = 3
            error = ValueError
        with pytest.raises(error):
            fse.add_sign_bits(tmp_path, **kwargs)
    assert list(inspect.signature(fse.add_sign_bits).parameters)[-1] == "pool"


@pytest.mark.parametrize("sizes", [(5,), (5, 3)])
def test_nested_backfill_refused_before_any_write(tmp_path, sizes):
    path = tmp_path / "store"
    signed_store(path, sizes=sizes)
    with fse.Pool(2) as pool:

        def nested():
            with pool.inside():
                return fse.add_sign_bits(path, pool=pool)

        future = pool.submit(nested)
        with pytest.raises(RuntimeError, match="same pool"):
            future.result(timeout=10)
        assert not list(path.glob("*sign_bits-*"))
        assert fse.add_sign_bits(path, workers=1, pool=pool) == len(sizes)


def test_backfill_waits_for_a_shared_place(tmp_path, monkeypatch):
    path = tmp_path / "store"
    signed_store(path, sizes=(5, 3))
    names = []
    real = _bits_write.write_bits

    def write(*args, **kwargs):
        names.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(_bits_write, "write_bits", write)
    with fse.Pool(2, max_tasks=1) as pool:
        pool.acquire()
        reached = blocked_in_wait(pool._permits._cond, "backfill")
        readers = Readers({"backfill": lambda: fse.add_sign_bits(path, pool=pool)}).start()
        try:
            assert arrived(reached, pool._permits._cond)
            assert not names
        finally:
            pool.release()
            readers.wait()
        assert readers.join() == {"backfill": 2}
        assert pool.try_acquire()
        pool.release()


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_failure_settles_own_writers_but_not_an_independent_call(tmp_path, monkeypatch, error_type):
    path = tmp_path / "store"
    signed_store(path, sizes=(5, 3, 2, 4))
    condition = threading.Condition()
    release = False
    started = threading.Event()
    waiting = []
    owners = []
    real_write = _bits_write._write
    real_init = _search_run._Own.__init__

    def init(self, pool):
        real_init(self, pool)
        owners.append((self.ended, blocked_in_wait(self.ended, "backfill")))

    def write(tmp, *args, **kwargs):
        if "0000000000000000" in tmp.name:
            assert started.wait(10)
            assert arrived(waiting[0], condition)
            tmp.mkdir()
            raise error_type("failed writer")
        if "0000000000000005" in tmp.name:
            waiting.append(blocked_in_wait(condition, threading.current_thread().name))
            with condition:
                started.set()
                assert condition.wait_for(lambda: release, 10)
        return real_write(tmp, *args, **kwargs)

    monkeypatch.setattr(_search_run._Own, "__init__", init)
    monkeypatch.setattr(_bits_write, "_write", write)
    with fse.Pool(4) as pool:
        independent = threading.Event()
        done = threading.Event()
        other_waits = []

        def other():
            with pool.inside():
                other_waits.append(blocked_in_wait(done._cond, threading.current_thread().name))
                independent.set()
                assert done.wait(10)

        future = pool.submit(other)
        assert independent.wait(10)
        assert arrived(other_waits[0], done._cond)
        readers = Readers(
            {"backfill": lambda: fse.add_sign_bits(path, workers=2, pool=pool)}
        ).start()
        try:
            assert started.wait(10)
            cond, event = owners[0]
            assert arrived(event, cond)
        finally:
            with condition:
                release = True
                condition.notify_all()
            readers.wait()
        try:
            with pytest.raises(error_type, match="failed writer"):
                readers.join()
            assert not future.done()  # failed backfill did not wait for the independent task
            assert not list(path.glob(".sign_bits-*"))
            assert pool.try_acquire()
            pool.release()
        finally:
            done.set()
            future.result(timeout=10)
    monkeypatch.undo()
    assert fse.add_sign_bits(path, workers=1) >= 1
