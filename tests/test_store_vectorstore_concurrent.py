"""Many threads reading one ``fse.VectorStore`` at once: search, rescore, score, ``take_*``,
``get_*`` and ``iter_batches`` give the bits of a serial run, a refresh meanwhile reaches only
later calls, and ``close`` raises while a reader holds its snapshot (docs/architecture/
store.md, "Concurrency")."""

import gc
import threading

import numpy as np
import pytest

import fse
from fse import VectorStore
from fse._store import _prefilter
from support import prefilter_search as pf
from support.concurrent import Gate, Readers, exact, hold_every_read, repeated
from support.encoder import finished
from support.take_store import critical_rows

DIM = 5
ROWS = 42
SHARD_ROWS = 7  # six shards: every search and score runs several tasks in a pool
WORKERS = 4
DTYPES = ["float32", "float16", "int8"]


def stored(n, dtype, seed):
    """Stored values (and int8 scales) with -0.0, subnormals and ties, of a moderate scale
    so that no linear score overflows."""
    codes, scales = critical_rows(n, dtype, seed=seed, dim=DIM)
    if dtype == "int8":
        scales = (np.random.default_rng(seed).random(n) + 0.01).astype(np.float32)
    codes[n // 2 :: 5] = codes[0]  # equal rows: ties by position
    return codes, scales


def write(path, dtype, first, n, seed):
    """Append ``n`` rows with document IDs to the store ``path`` (created by the first)."""
    codes, scales = stored(n, dtype, seed)
    docs = (np.arange(first, first + n, dtype=np.int64) * 7) % 9
    with fse.ShardSink(
        path, finished(dim=DIM), dtype=dtype, shard_rows=SHARD_ROWS, doc_ids=True
    ) as sink:
        target = sink.target(0, n)
        target[0][...] = codes
        if scales is not None:
            target[1][...] = scales
        target[-1][...] = 0.25
        sink.commit(0, target, doc_ids=docs)


def queries(n, seed=1):
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)


Q4, Q8, W3 = queries(4), queries(8, seed=2), queries(3, seed=3)
CANDIDATES = np.array(
    [[3, 9, -1, 9, 40], [0, 1, 2, 3, 4], [41, 20, 13, -1, -1], [5, 5, 5, 38, 17]], np.int64
)
TAKE = np.array([[41, 0, 7], [7, 7, 30]], np.int64)

# Every public read of a store, each with the arguments that make it run several tasks.
OPERATIONS = {
    "search_exact": lambda s: s.search(Q4, 5, mode="exact", workers=WORKERS),
    "search_auto": lambda s: s.search(Q8, 6, workers=WORKERS),  # exact up to 8 queries
    "search_single": lambda s: s.search(Q4[0], 7, mode="exact", workers=WORKERS),
    "search_rows": lambda s: s.search(Q4, 5, mode="exact", rows=list(range(3, 40, 2))),
    "search_doc_ids": lambda s: s.search(Q4, 5, mode="exact", doc_ids=[1, 4, 8]),
    "search_per_doc": lambda s: s.search(Q4, 3, mode="exact", max_per_doc=1, workers=WORKERS),
    "rescore": lambda s: s.rescore(Q4, CANDIDATES, 3, workers=WORKERS),
    "score_exact": lambda s: s.score(W3, [0.5, -1.0, 0.0], mode="exact", workers=WORKERS),
    "score_auto": lambda s: s.score(W3[0], workers=WORKERS),
    "take_vectors": lambda s: s.take_vectors(TAKE),
    "take_doc_ids": lambda s: s.take_doc_ids(TAKE),
    "get_vectors": lambda s: s.get_vectors(),
    "get_doc_ids": lambda s: s.get_doc_ids(start=2),
    "iter_batches": lambda s: list(s.iter_batches(block_rows=4)),
    "verify": lambda s: s.verify(),
}
IN_USE = "the store is in use (a search, score or read is running); close it when it is done"
CLOSED = "the VectorStore is closed"


def serial(store):
    return {name: exact(read(store)) for name, read in OPERATIONS.items()}


@pytest.fixture(params=DTYPES)
def made(request, tmp_path):
    """The path of a store of ``ROWS`` rows and its dtype."""
    path = tmp_path / "store"
    write(path, request.param, 0, ROWS, seed=0)
    return path, request.param


def test_many_threads_read_with_the_bits_of_a_serial_run(made):
    path, _ = made
    with VectorStore.open(path) as store:
        want = serial(store)
        start = threading.Barrier(2 * len(OPERATIONS), timeout=60)

        def reader(name):
            def run():
                start.wait()
                return [exact(OPERATIONS[name](store)) for _ in range(3)]

            return run

        jobs = {f"{name}-{i}": reader(name) for name in OPERATIONS for i in range(2)}
        got = Readers(jobs).start().join()
        assert store._active == 0
    for job, outcomes in got.items():
        assert outcomes == [want[job.rsplit("-", 1)[0]]] * 3, job


def test_readers_hold_their_snapshot_through_a_refresh_and_close_raises(made, monkeypatch):
    path, dtype = made
    names = list(OPERATIONS)
    gate = Gate(names)
    hold_every_read(gate, monkeypatch, OPERATIONS)
    with VectorStore.open(path) as store:
        before = serial(store)
        readers = Readers({name: repeated(read, store, 3) for name, read in OPERATIONS.items()})
        readers.start()
        try:
            assert gate.all_arrived()
            assert store._active == len(names)  # every reader counted, its snapshot taken
            write(path, dtype, ROWS, 2 * SHARD_ROWS + 3, seed=9)
            assert store.refresh() == 2 * SHARD_ROWS + 3  # readers do not block a refresh
            with pytest.raises(RuntimeError, match="in use"):
                store.close()
        finally:
            gate.open()
            readers.wait()
        got = readers.join()
        after = serial(store)
        assert store._active == 0
    for name in names:
        first, *later = got[name]
        assert first == before[name], name  # the held call kept the rows it started with
        assert later == [after[name]] * 2, name  # later calls see the refresh
    with VectorStore.open(path) as fresh:
        assert serial(fresh) == after


def test_reads_under_refreshes_and_a_close_match_a_snapshot_in_order(made):
    """Readers loop while a writer appends and the store refreshes three times, and a close
    meanwhile either raises (a read runs) or ends the reads: every outcome is that of one
    of the snapshots, and a reader never goes back to an older one."""
    path, dtype = made
    with VectorStore.open(path) as store:
        snapshots = [VectorStore.open(path)]
        start = threading.Barrier(len(OPERATIONS) + 1, timeout=60)
        refreshed = threading.Event()  # set also when the refresher fails
        ended = []  # the errors that ended a reader, or the close

        def reader(read):
            def run():
                start.wait()
                outcomes = []
                while len(outcomes) < 12 or not refreshed.wait(0):
                    try:
                        outcomes.append(exact(read(store)))
                    except RuntimeError as error:
                        ended.append(str(error))
                        break
                return outcomes

            return run

        def refresher():
            try:
                start.wait()
                for round_ in range(3):
                    write(path, dtype, len(snapshots[-1]), SHARD_ROWS + round_, seed=20 + round_)
                    snapshots.append(VectorStore.open(path))
                    assert store.refresh() == SHARD_ROWS + round_
            finally:
                refreshed.set()  # the readers end also after a failure
            try:
                store.close()
            except RuntimeError as error:
                ended.append(str(error))

        jobs = {name: reader(read) for name, read in OPERATIONS.items()}
        jobs["refresher"] = refresher
        try:
            got = Readers(jobs).start().join()
            assert store._active == 0
            assert set(ended) <= {CLOSED, IN_USE}
            for name, read in OPERATIONS.items():
                states = [exact(read(snapshot)) for snapshot in snapshots]
                seen = [states.index(outcome) for outcome in got[name]]  # raises if none
                assert seen == sorted(seen), name
        finally:
            for snapshot in snapshots:
                snapshot.close()


def test_the_int8_prefilter_in_many_threads_gives_the_bits_without_it(tmp_path, monkeypatch):
    path = tmp_path / "int8"
    pf.write(path, pf.codes_like(600, 32), shard_rows=100)
    pf.enable(monkeypatch)  # the variant is not enabled in the library
    taken = []
    real = _prefilter.search

    def counted(*args, **kwargs):
        found = real(*args, **kwargs)
        taken.append(found is not None)
        return found

    monkeypatch.setattr(_prefilter, "search", counted)
    calls = {
        "exact": lambda s: s.search(pf.queries(4, 32), 10, mode="exact", workers=WORKERS),
        "auto": lambda s: s.search(pf.queries(8, 32, seed=2), 5, workers=WORKERS),
        "single": lambda s: s.search(pf.queries(1, 32, seed=3)[0], 20, mode="exact"),
        "rows": lambda s: s.search(pf.queries(2, 32), 7, mode="exact", rows=list(range(50, 550))),
    }
    with VectorStore.open(path) as store:
        with pf.without_prefilter(monkeypatch):
            want = {name: exact(call(store)) for name, call in calls.items()}
        start = threading.Barrier(2 * len(calls), timeout=60)

        def reader(call):
            def run():
                start.wait()
                return [exact(call(store)) for _ in range(3)]

            return run

        jobs = {f"{name}-{i}": reader(call) for name, call in calls.items() for i in range(2)}
        got = Readers(jobs).start().join()
    # Three calls (exact, auto, single) took the prefilter, the rows= call did not (DR-340); the
    # order of the calls is not fixed, their number is: 2 jobs and 3 rounds of each call.
    assert taken.count(True) == 3 * 2 * 3
    assert taken.count(False) == 1 * 2 * 3
    for job, outcomes in got.items():
        assert outcomes == [want[job.rsplit("-", 1)[0]]] * 3, job


def test_blas_in_many_threads_stays_close_to_the_contract(made):
    """``mode="blas"`` (and ``"auto"`` above 8 queries) promises no bits and no ids: every
    concurrent call gives a valid result whose scores are close to the exact ones."""
    path, _ = made
    many = queries(12, seed=4)
    calls = {
        "search_blas": lambda s: s.search(many, 6, mode="blas", workers=WORKERS),
        "search_auto": lambda s: s.search(many, 6, workers=WORKERS),
        "score_blas": lambda s: s.score(many, mode="blas", workers=WORKERS),
        "score_auto": lambda s: s.score(many, workers=WORKERS),
    }
    with VectorStore.open(path) as store:
        hits = store.search(many, 6, mode="exact")
        linear = store.score(many, mode="exact")
        start = threading.Barrier(2 * len(calls), timeout=60)

        def reader(call):
            def run():
                start.wait()
                return [call(store) for _ in range(3)]

            return run

        jobs = {f"{name}-{i}": reader(call) for name, call in calls.items() for i in range(2)}
        got = Readers(jobs).start().join()
        for job, outcomes in got.items():
            for outcome in outcomes:
                if job.startswith("score"):
                    assert (outcome.dtype, outcome.shape) == (np.float32, linear.shape)
                    np.testing.assert_allclose(outcome, linear, rtol=1e-4, atol=1e-4)
                    continue
                assert outcome.ids.dtype == np.int64
                assert outcome.ids.shape == outcome.scores.shape == (12, 6)
                assert ((outcome.ids >= 0) & (outcome.ids < ROWS)).all()
                assert all(len(set(row)) == 6 for row in outcome.ids.tolist())
                assert (np.diff(outcome.scores, axis=1) <= 0).all()
                np.testing.assert_allclose(outcome.scores, hits.scores, rtol=1e-5, atol=1e-5)
                rescored = store.rescore(many, outcome.ids, 6)  # the exact cosines of its ids
                np.testing.assert_allclose(outcome.scores, rescored.scores, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("read", ["get_vectors", "get_doc_ids"])
def test_a_close_during_the_final_copy_of_get_leaves_the_result_whole(made, monkeypatch, read):
    """``get_*`` joins its parts after the guard: a close then succeeds, and the parts keep
    the mapped rows readable until the copy ends."""
    path, _ = made
    store = VectorStore.open(path)
    want = exact(OPERATIONS[read](store))
    gate = Gate([read])
    gate.hold(monkeypatch, VectorStore, "_row_parts", after=True)  # the guard is released
    readers = Readers({read: repeated(OPERATIONS[read], store, 1)}).start()
    try:
        assert gate.all_arrived()
        assert store._active == 0
        store.close()
        gc.collect()
    finally:
        gate.open()
        readers.wait()
    assert readers.join() == {read: [want]}
    with pytest.raises(RuntimeError, match="closed"):
        OPERATIONS[read](store)
