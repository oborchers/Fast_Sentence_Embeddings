"""``pool=`` on ``VectorStore.search``, ``score`` and ``rescore`` and on the store mode of a
head (docs/architecture/store.md, "Concurrency"): the bits of a call without a pool,
the tasks on the pool's threads and no threads of the call's own, the checks of ``workers``
and of a closed pool, errors and early ends that give every place back, and many calls at
once on one pool with fewer places than threads."""

import inspect
import threading

import numpy as np
import pytest

import fse
from fse import Pool, VectorStore
from fse._store import _prefilter, _search, _search_run
from support import prefilter_search as pf
from support.concurrent import Readers, exact
from support.encoder import finished
from support.pool import no_fse_threads_left, settled  # noqa: F401
from support.take_store import critical_rows

DIM = 5
ROWS = 42
SHARD_ROWS = 7  # six shards: every call but a one-query rescore runs several tasks
DTYPES = ["float32", "float16", "int8"]
F32 = np.float32


def write(path, dtype, n=ROWS, seed=0):
    """A store of ``n`` rows with -0.0, subnormals, ties and document IDs."""
    codes, scales = critical_rows(n, dtype, seed=seed, dim=DIM)
    if dtype == "int8":
        scales = (np.random.default_rng(seed).random(n) + 0.01).astype(F32)
    codes[n // 2 :: 5] = codes[0]
    docs = (np.arange(n, dtype=np.int64) * 7) % 9
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
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(F32)


Q4, W3 = queries(4), queries(3, seed=3)
CANDIDATES = np.array(
    [[3, 9, -1, 9, 40], [0, 1, 2, 3, 4], [41, 20, 13, -1, -1], [5, 5, 5, 38, 17]], np.int64
)
CALLS = {
    "search_exact": lambda s, **kw: s.search(Q4, 5, mode="exact", **kw),
    "search_auto": lambda s, **kw: s.search(Q4, 6, **kw),  # exact up to 8 queries
    "search_single": lambda s, **kw: s.search(Q4[0], 7, mode="exact", **kw),
    "search_rows": lambda s, **kw: s.search(Q4, 5, mode="exact", rows=list(range(3, 40, 2)), **kw),
    "search_doc_ids": lambda s, **kw: s.search(Q4, 5, mode="exact", doc_ids=[1, 4, 8], **kw),
    "search_per_doc": lambda s, **kw: s.search(Q4, 3, mode="exact", max_per_doc=1, **kw),
    "rescore": lambda s, **kw: s.rescore(Q4, CANDIDATES, 3, **kw),
    "rescore_per_doc": lambda s, **kw: s.rescore(Q4, CANDIDATES, 3, max_per_doc=1, **kw),
    "rescore_one": lambda s, **kw: s.rescore(Q4[0], CANDIDATES[0], 3, **kw),  # one task
    "score_exact": lambda s, **kw: s.score(W3, [0.5, -1.0, 0.0], mode="exact", **kw),
    "score_single": lambda s, **kw: s.score(W3[0], **kw),
}
ONE_TASK = {"rescore_one"}


class Submits:
    """Counts the tasks given to a pool, and fails a call that makes threads of its own."""

    def __init__(self, monkeypatch):
        self.count = 0
        real = Pool._submit

        def submit(pool, *args):
            self.count += 1
            return real(pool, *args)

        def own_threads(*_args, **_kwargs):
            raise AssertionError("a call with a pool made threads of its own")

        monkeypatch.setattr(Pool, "_submit", submit)
        monkeypatch.setattr(_search_run, "ThreadPoolExecutor", own_threads)


def free_places(pool):
    return pool._permits._value  # the count of its BoundedSemaphore


@pytest.fixture(params=DTYPES)
def store(request, tmp_path):
    write(tmp_path / "store", request.param)
    with VectorStore.open(tmp_path / "store") as opened:
        yield opened


@pytest.mark.parametrize("workers", [None, 2])
def test_a_pool_gives_the_bits_of_a_call_without_one_on_its_threads(store, workers, monkeypatch):
    want = {name: exact(call(store, workers=4)) for name, call in CALLS.items()}
    with Pool(4) as pool:
        submits = Submits(monkeypatch)
        for name, call in CALLS.items():
            before = submits.count
            assert exact(call(store, pool=pool, workers=workers)) == want[name], name
            ran = submits.count - before
            assert (ran == 0) if name in ONE_TASK else (ran > 0), name
        assert free_places(pool) == pool.max_tasks
        assert not pool.closed  # a borrowed pool stays the caller's
        assert exact(CALLS["search_exact"](store, pool=pool)) == want["search_exact"]
    assert store._active == 0


def test_blas_on_a_pool_stays_close_to_the_contract(tmp_path):
    """``mode="blas"`` promises no bits: on a pool it gives a valid result close to exact."""
    write(tmp_path / "store", "float32")
    many = queries(12, seed=4)
    with VectorStore.open(tmp_path / "store") as store, Pool(3) as pool:
        hits = store.search(many, 6, mode="blas", pool=pool)
        linear = store.score(many, mode="blas", pool=pool)
        assert hits.ids.shape == (12, 6)
        assert hits.ids.dtype == np.int64
        np.testing.assert_allclose(
            hits.scores, store.search(many, 6, mode="exact").scores, atol=1e-5
        )
        np.testing.assert_allclose(linear, store.score(many, mode="exact"), rtol=1e-5, atol=1e-5)


def test_the_workers_of_a_call_and_a_closed_pool_are_checked_before_any_row(tmp_path):
    write(tmp_path / "store", "float32")
    with VectorStore.open(tmp_path / "store") as store:
        with Pool(2) as pool:
            for call in (CALLS["search_exact"], CALLS["rescore"], CALLS["score_exact"]):
                with pytest.raises(ValueError, match=r"workers \(3\) must not exceed pool"):
                    call(store, pool=pool, workers=3)
        for call in (CALLS["search_exact"], CALLS["rescore"], CALLS["score_exact"]):
            with pytest.raises(RuntimeError, match="the pool is closed"):
                call(store, pool=pool)
            with pytest.raises((TypeError, ValueError), match="pool"):
                call(store, pool=object())
        assert store._active == 0


def test_an_error_in_a_task_on_a_pool_gives_every_place_back(tmp_path, monkeypatch):
    write(tmp_path / "store", "float32")
    real = _search._block_cosines
    calls = []

    def third_fails(*args):
        calls.append(1)
        if len(calls) == 3:
            raise ValueError("block three")
        return real(*args)

    with VectorStore.open(tmp_path / "store") as store, Pool(2, max_tasks=2) as pool:
        want = exact(CALLS["search_exact"](store))
        monkeypatch.setattr(_search, "_block_cosines", third_fails)
        with pytest.raises(ValueError, match="block three"):
            store.search(Q4, 5, mode="exact", pool=pool)
        monkeypatch.setattr(_search, "_block_cosines", real)
        assert free_places(pool) == 2
        assert exact(CALLS["search_exact"](store, pool=pool)) == want
        assert store._active == 0


def test_the_int8_prefilter_and_the_rows_path_run_on_the_pool(tmp_path, monkeypatch):
    path = tmp_path / "int8"
    pf.write(path, pf.codes_like(600, 32), shard_rows=100)
    pf.enable(monkeypatch)  # the variant is not enabled in the library
    taken = []
    real = _prefilter.search

    def counted(*args, **kwargs):
        found = real(*args, **kwargs)
        taken.append(found is not None)
        return found

    many, rows = pf.queries(4, 32), list(range(50, 550))
    with VectorStore.open(path) as store:
        with pf.without_prefilter(monkeypatch):
            want = [exact(store.search(many, 10, mode="exact", rows=r)) for r in (None, rows)]
        monkeypatch.setattr(_prefilter, "search", counted)
        with Pool(3, max_tasks=2) as pool:
            submits = Submits(monkeypatch)
            got = [
                exact(store.search(many, 10, mode="exact", rows=r, pool=pool)) for r in (None, rows)
            ]
            assert submits.count > 0
            assert free_places(pool) == 2
    assert taken == [True, False]  # DR-340: a rows= call takes the previous path, on the pool too
    assert got == want


def test_a_prefilter_that_gives_up_hands_the_pool_to_the_exact_search(tmp_path, monkeypatch):
    row = pf.codes_like(1, 8, seed=3)
    pf.write(tmp_path / "s", np.repeat(row, 68_000, axis=0), shard_rows=17_000)
    pf.enable(monkeypatch)
    monkeypatch.setattr(_search, "_MIN_ROWS", 4000)
    monkeypatch.setattr(_search, "_MAX_ROWS", 4000)
    monkeypatch.setattr(_prefilter, "_BUDGET_MIN", 3000)
    many = pf.queries(2, 8)
    with fse.VectorStore.open(tmp_path / "s") as store, Pool(3, max_tasks=2) as pool:
        with pf.without_prefilter(monkeypatch):
            want = exact(store.search(many, 10, mode="exact"))
        got, records = pf.run(store, many, 10, pool=pool)
        assert pf.prefilter_record(records).fse_fallback == "budget"
        assert exact(got) == want
        assert free_places(pool) == 2
        assert exact(store.search(many, 10, mode="exact", pool=pool)) == want


def classifier():
    rng = np.random.default_rng(5)
    return fse.Classifier.from_arrays(
        finished(dim=DIM),
        rng.standard_normal((4, DIM)).astype(F32),
        rng.standard_normal(4).astype(F32),
        labels=list("abcd"),
        link="softmax",
    )


def regressor():
    rng = np.random.default_rng(6)
    return fse.Regressor.from_arrays(
        finished(dim=DIM), rng.standard_normal((2, DIM)).astype(F32), [0.5, -0.25]
    )


def test_the_store_mode_of_a_head_runs_on_the_pool_it_is_given(store, monkeypatch):
    """The head took ``pool=`` and dropped it; now its blocks run on the pool, with the
    bits of a call without one."""
    model = classifier()
    want = {
        "labels": exact(model.predict(store)),
        "proba": exact(model.predict_proba(store)),
        "batches": [exact(b.outputs) for b in model.predict_batches(store, output="proba")],
        "decision": exact(model.decision_function(store)),
        "values": exact(regressor().predict(store)),
    }
    with Pool(3) as pool:
        submits = Submits(monkeypatch)
        assert exact(model.predict(store, pool=pool)) == want["labels"]
        assert exact(model.predict_proba(store, pool=pool, workers=2)) == want["proba"]
        batches = model.predict_batches(store, output="proba", pool=pool)
        assert [exact(b.outputs) for b in batches] == want["batches"]
        assert exact(model.decision_function(store, pool=pool)) == want["decision"]
        assert exact(regressor().predict(store, pool=pool)) == want["values"]
        assert submits.count > 0
        assert free_places(pool) == pool.max_tasks
    assert store._active == 0


@pytest.mark.parametrize("how", ["close", "collect", "error"])
def test_a_store_stream_that_ends_early_frees_the_borrowed_pool(store, how):
    model = classifier()
    want = exact(model.predict_proba(store))
    with Pool(2, max_tasks=2) as pool:
        stream = model.predict_batches(store, output="proba", pool=pool)
        first = next(stream)
        if how == "close":
            stream.close()
        elif how == "collect":
            del stream
        else:
            with pytest.raises(ZeroDivisionError):
                stream.throw(ZeroDivisionError())
        assert first.first_row == 0
        assert free_places(pool) == 2
        assert exact(model.predict_proba(store, pool=pool)) == want
    assert settled(0)


def test_a_store_stream_can_search_on_its_own_pool_between_two_blocks(store):
    """The blocks a stream holds take no place: a call on the same pool between two of them
    finds places as soon as the running tasks of the stream end."""
    model = classifier()
    want = exact(CALLS["search_exact"](store))
    with Pool(2, max_tasks=1) as pool:
        for _ in model.predict_batches(store, output="proba", pool=pool):
            assert exact(CALLS["search_exact"](store, pool=pool)) == want
        assert free_places(pool) == 1


def test_many_calls_at_once_on_one_pool_with_fewer_places_than_threads(store):
    """Twelve threads search, rescore, score and predict on one pool of three threads and
    two places: every outcome has the bits of a serial call, no call made threads of its
    own, and the pool runs on afterwards."""
    model = classifier()
    jobs = dict(CALLS)
    jobs["predict"] = lambda s, **kw: model.predict_proba(s, **kw)
    want = {name: exact(call(store)) for name, call in jobs.items()}
    with Pool(3, max_tasks=2) as pool:
        start = threading.Barrier(len(jobs), timeout=60)
        seen: set[str] = set()

        def reader(call):
            def run():
                start.wait()
                found = [exact(call(store, pool=pool)) for _ in range(3)]
                seen.update(t.name for t in threading.enumerate())
                return found

            return run

        got = Readers({name: reader(call) for name, call in jobs.items()}).start().join()
        assert free_places(pool) == 2
        assert exact(CALLS["rescore"](store, pool=pool)) == want["rescore"]
    for name, outcomes in got.items():
        assert outcomes == [want[name]] * 3, name
    assert not [n for n in seen if n.startswith(("fse-search", "fse-score", "fse-rescore"))]
    assert len([n for n in seen if n.startswith("fse-worker")]) <= 3
    assert store._active == 0


@pytest.mark.parametrize("method", ["search", "score", "rescore"])
def test_pool_is_the_last_keyword_of_the_store_methods(method):
    parameters = inspect.signature(getattr(VectorStore, method)).parameters
    last = list(parameters.values())[-1]
    assert last.name == "pool"
    assert last.kind is inspect.Parameter.KEYWORD_ONLY
    assert last.default is None
