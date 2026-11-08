"""Text planning uses the same pool as encoding, heads, streams and searches."""

import threading

import numpy as np
import pytest

import fse
from fse._runtime import _executor, _pipeline
from fse._store import _search_run
from fse._text import ArrowSource, _scan
from support.concurrent import Readers, exact
from support.encoder import finished
from support.pool import make_head, no_fse_threads_left  # noqa: F401
from support.waiting import arrived, blocked_in_wait

TEXT = "w1 w2 " * 40


def source_file(tmp_path):
    path = tmp_path / "text"
    path.write_text((TEXT + "\n") * 5000)
    return path


def public_result(encoder, head, path, call, pool):
    if call == "encode":
        return encoder.encode(path, workers=3, pool=pool)
    if call == "head":
        return head.predict_proba(path, workers=3, pool=pool)
    if call == "heads":
        return fse.Heads(a=head).predict(path, workers=3, pool=pool)
    pa = pytest.importorskip("pyarrow")
    batch = pa.array([TEXT] * 5000)
    return [batch.rows for batch in encoder.encode_batches([batch], workers=3, pool=pool)]


@pytest.mark.parametrize("call", ["encode", "head", "heads", "batches"])
def test_public_paths_pass_pool_to_scan(tmp_path, monkeypatch, call):
    path = source_file(tmp_path)
    encoder = finished(dim=12)
    head = make_head(encoder)
    calls, names = [], []
    real_run = _pipeline.run
    scan_class = ArrowSource if call == "batches" else _scan._Buffered
    real_scan = scan_class._scanner

    def run(*args, **kwargs):
        calls.append(kwargs.get("pool"))
        return real_run(*args, **kwargs)

    def scanner(self, *args):
        fn = real_scan(self, *args)

        def scan(span):
            names.append(threading.current_thread().name)
            return fn(span)

        return scan

    def no_executor(**_kwargs):
        raise AssertionError("scan started extra threads")

    wanted = exact(public_result(encoder, head, path, call, None))
    monkeypatch.setattr(_pipeline, "run", run)
    monkeypatch.setattr(scan_class, "_scanner", scanner)
    monkeypatch.setattr(_scan, "ThreadPoolExecutor", no_executor)
    with fse.Pool(3, max_tasks=1) as pool:
        assert exact(public_result(encoder, head, path, call, pool)) == wanted
        assert calls
        assert all(p is pool for p in calls)
        assert names
        assert all(name.startswith("fse-worker") for name in names)
    if call == "batches":
        calls.clear()
        names.clear()
        assert (
            exact(public_result(encoder, head, path, call, None)) == wanted
        )  # its iteration owns one pool for scans too
        assert calls
        assert all(p is not None for p in calls)
        assert all(name.startswith("fse-worker") for name in names)


def test_encoding_from_a_task_on_the_same_pool_is_refused(tmp_path):
    path = source_file(tmp_path)
    encoder = finished()
    with fse.Pool(2, max_tasks=1) as pool:

        def nested():
            with pool.inside():
                return encoder.encode(path, pool=pool)

        with pytest.raises(RuntimeError, match="same pool"):
            pool.submit(nested).result(timeout=10)
        assert encoder.encode(path, pool=pool).shape == (5000, 6)


def test_encode_and_search_overlap_on_six_and_two_workers(tmp_path, monkeypatch):
    path = source_file(tmp_path)
    encoder = finished()
    store_path = tmp_path / "store"
    with fse.ShardSink(store_path, encoder, shard_rows=10) as sink:
        encoder.encode(["w1", "w2"] * 40, sink=sink)
    condition = threading.Condition()
    release = False
    started = threading.Event()
    waiting = []
    real = _scan._Buffered._scanner

    def scanner(self, *args):
        fn = real(self, *args)

        def scan(span):
            if span[0] == 0:
                waiting.append(blocked_in_wait(condition, threading.current_thread().name))
                with condition:
                    started.set()
                    assert condition.wait_for(lambda: release, 10)
            return fn(span)

        return scan

    def no_executor(**_kwargs):
        raise AssertionError("encode or search started extra threads")

    monkeypatch.setattr(_scan._Buffered, "_scanner", scanner)
    monkeypatch.setattr(_scan, "ThreadPoolExecutor", no_executor)
    monkeypatch.setattr(_search_run, "ThreadPoolExecutor", no_executor)
    # Shard writers of other tests can still live in this worker process. Only the
    # threads created for these two calls belong to this pool's thread budget.
    existing = set(threading.enumerate())
    with fse.VectorStore.open(store_path) as store, fse.Pool(8, start=True) as pool:
        query = np.ones(6, dtype=np.float32)
        wanted = exact(store.search(query, 5, mode="exact", workers=1))
        readers = Readers({"encode": lambda: encoder.encode(path, workers=6, pool=pool)}).start()
        try:
            assert started.wait(10)
            assert arrived(waiting[0], condition)
            assert exact(store.search(query, 5, mode="exact", workers=2, pool=pool)) == wanted
            threads = [
                t for t in threading.enumerate() if t not in existing and t.name.startswith("fse-")
            ]
            assert len(threads) == 8
            assert all(t.name.startswith("fse-worker") for t in threads)
        finally:
            with condition:
                release = True
                condition.notify_all()
            readers.wait()
        assert readers.join()["encode"].shape == (5000, 6)


@pytest.mark.parametrize("max_tasks", [1, 4])
def test_scan_interrupt_settles_tasks_and_gives_back_places(tmp_path, monkeypatch, max_tasks):
    path = source_file(tmp_path)
    encoder = finished()
    condition = threading.Condition()
    release = False
    started = threading.Event()
    waiting = []
    real_scan = _scan._Buffered._scanner
    real_drain = _executor._Run._wait_for_own
    draining = threading.Event()

    def drain(self):
        draining.set()  # the caller reached error cleanup while its sibling is still held
        return real_drain(self)

    def scanner(self, *args):
        fn = real_scan(self, *args)

        def scan(span):
            if span[0] == 0:
                if max_tasks == 1:
                    raise KeyboardInterrupt("scan interrupted")
                assert started.wait(10)
                assert arrived(waiting[0], condition)
                raise KeyboardInterrupt("scan interrupted")
            if not started.is_set():
                waiting.append(blocked_in_wait(condition, threading.current_thread().name))
                with condition:
                    started.set()
                    assert condition.wait_for(lambda: release, 10)
            return fn(span)

        return scan

    monkeypatch.setattr(_scan._Buffered, "_scanner", scanner)
    monkeypatch.setattr(_executor._Run, "_wait_for_own", drain)
    with fse.Pool(3, max_tasks=max_tasks) as pool:
        readers = Readers({"encode": lambda: encoder.encode(path, pool=pool)}).start()
        try:
            if max_tasks > 1:
                assert started.wait(10)
                assert arrived(waiting[0], condition)
            assert draining.wait(10)
        finally:
            with condition:
                release = True
                condition.notify_all()
            readers.wait()
        with pytest.raises(KeyboardInterrupt, match="scan interrupted"):
            readers.join()
        for _ in range(max_tasks):
            assert pool.try_acquire()
        for _ in range(max_tasks):
            pool.release()
        monkeypatch.undo()
        assert encoder.encode(path, pool=pool).shape == (5000, 6)
