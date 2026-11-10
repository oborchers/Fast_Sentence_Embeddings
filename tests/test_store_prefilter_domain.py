"""The domain of the int8 prefilter, the state of its main thread and the backends that
enable it (docs/architecture/store.md, "Search")."""

from types import SimpleNamespace

import numpy as np
import pytest

import fse
from fse import _native
from fse._backends import _cpu
from fse._backends._cpu import CpuBackend
from fse._store import _prefilter
from support import prefilter_search as ps

DIM = 64
N = 3000


def _store(tmp_path, n=N, dim=DIM, **kwargs):
    codes = ps.codes_like(n, dim, seed=3)
    ps.write(tmp_path / "s", codes, **kwargs)
    return codes


def test_empty_pieces_are_not_kept_and_many_pieces_are_joined():
    state = _prefilter._Survivors(1, 2, np.array([0.1]))
    empty = SimpleNamespace(
        status="ok",
        heap=np.array([[0.9, 0.8]]),
        rows=(np.empty(0, np.int64),),
        upper=(np.empty(0),),
    )
    for _ in range(1000):
        assert state.add(empty, 0, 10) is None
    assert state.rows == [[]]
    assert state.read == 10_000
    one = SimpleNamespace(
        status="ok",
        heap=np.array([[0.9, 0.8]]),
        rows=(np.array([1], np.int64),),
        upper=(np.array([0.85]),),
    )
    for first in range(1000):
        assert state.add(one, first * 10, 10) is None
        assert len(state.rows[0]) <= _prefilter._FRAGMENTS + 1
    assert state.count == [1000]


def test_the_running_threshold_cuts_the_survivors_before_the_budget_gives_up():
    got = SimpleNamespace(
        status="ok",
        heap=np.array([[0.9, 0.8]]),
        rows=(np.arange(5, dtype=np.int64),),
        upper=(np.array([1.0, 0.95, 0.5, 0.4, 0.3]),),
    )
    state = _prefilter._Survivors(1, 2, np.array([0.1]))
    state.budget = 3
    assert state.add(got, 100, 5) is None
    assert state.count == [2]
    assert state.read == 5
    offsets, positions = state.csr()
    assert offsets.tolist() == [0, 2]
    assert positions.tolist() == [100, 101]
    state.release()
    assert state.rows == []


@pytest.mark.parametrize(("status", "reason"), [("unsafe", "unsafe_row"), ("overflow", "overflow")])
def test_a_status_of_a_pass_names_the_fallback(status, reason):
    state = _prefilter._Survivors(1, 2, np.array([0.1]))
    assert state.add(SimpleNamespace(status=status), 0, 5) == reason
    assert state.read == 0


def _shards(rows, dim, dtype=np.int8):
    return [(SimpleNamespace(rows=rows), {"vectors": np.empty((0, dim), dtype)})]


def _applies(*, q=1, k=10, dim=256, rows=65_536, dtype=np.int8, **changes):
    backend = SimpleNamespace(prefilter_variant=lambda: "portable")
    arguments = {"exact": True, "grouped": False, "filtered": False} | changes
    queries = np.zeros((q, dim), np.float32)
    return _prefilter.applies(backend, _shards(rows, dim, dtype), queries, k, rows, **arguments)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({}, True),
        ({"k": 100}, True),
        ({"k": 101}, False),
        ({"q": 8}, True),
        ({"q": 9}, False),
        ({"dim": 255}, False),
        ({"dim": 256}, True),
        ({"dim": 1024}, True),
        ({"dim": 1025}, False),
        ({"rows": 65_535}, False),
        ({"rows": 65_536}, True),
        ({"dtype": np.float32}, False),
        ({"dtype": np.float16}, False),
        ({"exact": False}, False),
        ({"grouped": True}, False),
        ({"filtered": True}, False),
    ],
)
def test_the_domain_has_these_limits(change, expected):
    assert _applies(**change) is expected


def test_no_shards_and_no_variant_are_outside_the_domain():
    queries = np.zeros((1, 256), np.float32)
    plain = {"exact": True, "grouped": False, "filtered": False}
    none = SimpleNamespace(prefilter_variant=lambda: None)
    some = SimpleNamespace(prefilter_variant=lambda: "portable")
    assert not _prefilter.applies(none, _shards(70_000, 256), queries, 10, 70_000, **plain)
    assert not _prefilter.applies(some, [], queries, 10, 70_000, **plain)


@pytest.mark.parametrize(("rows", "dim", "entered"), [(65_536, 256, True), (65_535, 256, False)])
def test_a_store_at_the_limit_of_rows_enters_the_prefilter_or_not(
    tmp_path, monkeypatch, rows, dim, entered
):
    ps.enable(monkeypatch, rows=None, dim=None)
    ps.write(tmp_path / "s", ps.codes_like(rows, dim, seed=3), shard_rows=20_000)
    qs = ps.queries(1, dim)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, want, records = ps.both(monkeypatch, store, qs, 10, workers=2)
        assert bool(ps.events(records, "store.prefilter")) is entered
        assert ps.same(got, want)
        if entered:
            assert ps.prefilter_record(records).fse_fallback == "none"


def test_modes_that_are_not_exact_and_stores_that_are_not_int8_never_enter(tmp_path, monkeypatch):
    ps.enable(monkeypatch)
    docs = np.arange(N, dtype=np.int64) % 40
    ps.write(tmp_path / "s", ps.codes_like(N, DIM, seed=3), docs=docs)
    qs = ps.queries(9, DIM)
    with fse.VectorStore.open(tmp_path / "s") as store:
        for arguments, q in (({"mode": "blas"}, 3), ({"mode": "auto"}, 9), ({"max_per_doc": 1}, 3)):
            with ps.captured() as records:
                store.search(qs[:q], 5, **arguments)
            assert not ps.events(records, "store.prefilter"), arguments
        got, want, records = ps.both(monkeypatch, store, qs[:3], 5, max_per_doc=1)
        assert not ps.events(records, "store.prefilter")  # the grouped search is its own path
        assert ps.same(got, want)
        with ps.captured() as records:
            store.search(qs[:8], 5, mode="auto")  # up to 8 queries is the exact path
        assert ps.events(records, "store.prefilter")
    float_store = tmp_path / "f"
    with fse.ShardSink(float_store, ps.finished(dim=DIM), dtype="float32", shard_rows=700) as sink:
        target = sink.target(0, N)
        target[0][...] = np.random.default_rng(1).standard_normal((N, DIM))
        target[-1][...] = 0.125
        sink.commit(0, target)
    with fse.VectorStore.open(float_store) as store, ps.captured() as records:
        store.search(qs[:2], 5, mode="exact")
    assert not ps.events(records, "store.prefilter")


def _shipped() -> bool:
    """Whether this machine runs the variant the library ships: NEON with the dot product."""
    return CpuBackend().isa == "neon" and _native.prefilter_can_run("neon-sdot")


def test_the_library_ships_neon_sdot_only_and_the_numpy_backend_never_enters(tmp_path, monkeypatch):
    assert frozenset({"neon-sdot"}) == _cpu._PREFILTER_ENABLED  # avx2 stays off until DR-288
    monkeypatch.setattr(_prefilter, "MIN_ROWS", 1)
    monkeypatch.setattr(_prefilter, "MIN_DIM", 1)
    _store(tmp_path)
    with fse.VectorStore.open(tmp_path / "s", backend="numpy") as store:
        _, records = ps.run(store, ps.queries(2, DIM), 5)
    assert not ps.events(records, "store.prefilter")


def test_the_shipped_variant_is_chosen_without_rows_and_never_with_rows(tmp_path, monkeypatch):
    """DR-340: on ARM with the dot product a search without ``rows=`` enters the prefilter
    (also over a view of several stores and shards); with ``rows=`` or ``doc_ids=`` it never
    does; where the variant cannot run (x86) nothing enters. The bits are the previous ones."""
    monkeypatch.setattr(_prefilter, "MIN_ROWS", 1)
    monkeypatch.setattr(_prefilter, "MIN_DIM", 1)
    docs = np.arange(N, dtype=np.int64) % 40
    ps.write(tmp_path / "a", ps.codes_like(N, DIM, seed=3), shard_rows=700, docs=docs)
    ps.write(tmp_path / "b", ps.codes_like(N, DIM, seed=4), shard_rows=900, docs=docs)
    qs = ps.queries(3, DIM)
    selections = (
        ({}, _shipped()),
        ({"workers": 3}, _shipped()),
        ({"rows": list(range(100, 2500))}, False),
        ({"doc_ids": [1, 2, 3]}, False),
    )
    for path in (tmp_path / "a", [tmp_path / "a", tmp_path / "b"]):
        opener = (
            fse.VectorStore.open if isinstance(path, type(tmp_path)) else fse.VectorStore.open_view
        )
        with opener(path) as store:
            assert (store._backend.prefilter_variant() == "neon-sdot") is _shipped()
            for arguments, entered in selections:
                got, want, records = ps.both(monkeypatch, store, qs, 6, **arguments)
                assert bool(ps.events(records, "store.prefilter")) is entered, arguments
                assert ps.same(got, want)


def test_the_numpy_backend_with_a_variant_matches_the_cpu_backend(tmp_path, monkeypatch):
    _store(tmp_path)
    ps.enable(monkeypatch)
    from fse._backends import _numpy

    monkeypatch.setattr(_numpy.NumpyBackend, "prefilter_variant", lambda _self: "portable")
    qs = ps.queries(2, DIM)
    with fse.VectorStore.open(tmp_path / "s", backend="numpy") as a:
        got, records = ps.run(a, qs, 5)
        assert ps.prefilter_record(records).fse_fallback == "none"
    with fse.VectorStore.open(tmp_path / "s", backend="cpu") as b:
        want, _ = ps.run(b, qs, 5)
    assert ps.same(got, want)
