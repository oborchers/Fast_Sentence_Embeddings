"""The int8 prefilter in the exact search: the bits of the search without it, the fallbacks of
the whole call, the limits of its domain (docs/architecture/store.md, "Search")."""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

import fse
from fse import _reference
from fse._store import _prefilter, _search
from support import prefilter_search as ps
from support import prefilter_witnesses as witnesses

DIM = 64
N = 3000


def _store(tmp_path, n=N, dim=DIM, **kwargs):
    codes = ps.codes_like(n, dim, seed=3)
    ps.write(tmp_path / "s", codes, **kwargs)
    return codes


def _both(monkeypatch, store, qs, k, **arguments):
    """The outcome with the prefilter, the outcome without, and the records of the first."""
    got, records = ps.run(store, qs, k, **arguments)
    with ps.without_prefilter(monkeypatch):
        want, _ = ps.run(store, qs, k, **arguments)
    return got, want, records


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("q", [1, 3, 8])
@pytest.mark.parametrize("k", [1, 10, 37])
def test_the_bits_are_those_of_the_search_without_it_and_of_the_reference(
    tmp_path, monkeypatch, workers, q, k
):
    codes = _store(tmp_path)
    ps.enable(monkeypatch)
    qs = ps.queries(q, DIM)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, want, records = _both(monkeypatch, store, qs, k, workers=workers)
        record = ps.prefilter_record(records)
        assert record.fse_fallback == "none"
        assert record.fse_rows == N
        assert record.fse_queries == q
        assert record.fse_k == k
        assert record.fse_workers == workers
        assert 0 < record.fse_pairs <= q * N
        assert ps.same(got, want)
        assert len(ps.events(records, "store.search")) == 1  # the event stays once per call
    inverse = _reference.inverse_norms(codes)
    cosines = _reference.cosines(codes, inverse, qs)
    for j in range(q):
        top = _reference.top_k(cosines[:, j], k)
        np.testing.assert_array_equal(got.ids[j], top)
        np.testing.assert_array_equal(ps.bits(got.scores[j]), ps.bits(cosines[top, j]))


def test_the_survivors_skip_rows_and_do_not_depend_on_the_threads(tmp_path, monkeypatch):
    _store(tmp_path)
    ps.enable(monkeypatch)
    qs = ps.queries(1, DIM)
    pairs = set()
    with fse.VectorStore.open(tmp_path / "s") as store:
        for workers in (1, 2, 4):
            got, records = ps.run(store, qs, 3, workers=workers)
            pairs.add(ps.prefilter_record(records).fse_pairs)
            if workers == 1:
                first = got
            assert ps.same(got, first)
    assert len(pairs) == 1
    assert pairs.pop() < N // 4


def test_blocks_of_every_size_give_the_same_survivors(tmp_path, monkeypatch):
    _store(tmp_path)
    ps.enable(monkeypatch)
    qs = ps.queries(2, DIM)
    pairs = set()
    with fse.VectorStore.open(tmp_path / "s") as store:
        for size in (4000, 700, 129):
            with monkeypatch.context() as patch:
                patch.setattr(_search, "_MIN_ROWS", size)
                patch.setattr(_search, "_MAX_ROWS", size)
                got, records = ps.run(store, qs, 5)
            pairs.add(ps.prefilter_record(records).fse_pairs)
            if size == 4000:
                first = got
            assert ps.same(got, first)
    assert len(pairs) == 1


@pytest.mark.parametrize("selection", ["range", "scattered", "unsorted_duplicates", "missing"])
@pytest.mark.parametrize("workers", [1, 4])
def test_rows_selections_give_the_bits_of_the_search_without_it(
    tmp_path, monkeypatch, selection, workers
):
    _store(tmp_path)
    ps.enable(monkeypatch)
    rng = np.random.default_rng(9)
    rows = {
        "range": np.arange(500, 2400),
        "scattered": np.sort(rng.choice(N, 1500, replace=False)),
        "unsorted_duplicates": np.concatenate([rng.permutation(N)[:900], np.arange(100, 300)]),
        "missing": np.concatenate([np.arange(0, 700), np.arange(2100, 2200)]),
    }[selection]
    qs = ps.queries(4, DIM)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, want, records = _both(monkeypatch, store, qs, 6, rows=rows.tolist(), workers=workers)
        record = ps.prefilter_record(records)
        assert record.fse_fallback == "none"
        assert record.fse_rows == len(np.unique(rows))
        assert ps.same(got, want)
        assert len(ps.events(records, "store.search")) == 1


def test_a_view_and_a_refreshed_store_give_the_bits_of_the_search_without_it(tmp_path, monkeypatch):
    ps.enable(monkeypatch)
    codes = ps.codes_like(2 * N, DIM, seed=4)
    ps.write(tmp_path / "a", codes[:N], shard_rows=500)
    ps.write(tmp_path / "b", codes[N:], shard_rows=900)
    qs = ps.queries(3, DIM, seed=5)
    with fse.VectorStore.open_view([tmp_path / "a", tmp_path / "b"]) as view:
        got, want, records = _both(monkeypatch, view, qs, 8, workers=3)
        assert ps.prefilter_record(records).fse_rows == 2 * N
        assert ps.same(got, want)
        got, want, records = _both(monkeypatch, view, qs, 8, rows=list(range(N - 50, N + 50)))
        assert ps.prefilter_record(records).fse_rows == 100
        assert ps.same(got, want)
    with fse.VectorStore.open(tmp_path / "a") as store:
        extra = ps.codes_like(1200, DIM, seed=6)
        with fse.ShardSink(tmp_path / "a", ps.finished(dim=DIM), dtype="int8") as sink:
            target = sink.target(0, len(extra))
            target[0][...], target[1][...] = extra, 0.5
            target[-1][...] = _reference.inverse_norms(extra)
            sink.commit(0, target)
        before, _ = ps.run(store, qs, 8)
        assert store.refresh() == 1200
        got, want, records = _both(monkeypatch, store, qs, 8)
        assert ps.prefilter_record(records).fse_rows == N + 1200
        assert ps.same(got, want)
        assert not ps.same(got, before)


def test_ties_zero_rows_and_extreme_codes(tmp_path, monkeypatch):
    ps.enable(monkeypatch)
    rng = np.random.default_rng(11)
    codes = ps.codes_like(N, DIM, seed=7)
    codes[100:140] = codes[5]  # 40 equal rows around the k-th value
    codes[200:230] = 0  # zero rows: inverse norm 0, estimate +-0.0
    codes[300:310] = -128
    codes[310:320] = 127
    codes[320:330] = rng.choice(np.array([-128, 127], np.int8), (10, DIM))
    ps.write(tmp_path / "s", codes, shard_rows=640)
    qs = np.concatenate([ps.queries(3, DIM, seed=8), codes[5:6].astype(np.float32)])
    with fse.VectorStore.open(tmp_path / "s") as store:
        for k in (1, 10, 35, 100):
            got, want, records = _both(monkeypatch, store, qs, k, workers=3)
            assert ps.prefilter_record(records).fse_fallback == "none"
            assert ps.same(got, want)


def test_extreme_codes_at_the_widest_row(tmp_path, monkeypatch):
    """d = 1024 with rows of only -128 (the largest sum of squares), only 127, a mixture of
    both and a row along the query: the integer sums reach their bounds."""
    ps.enable(monkeypatch, rows=1, dim=None)
    rng = np.random.default_rng(13)
    codes = ps.codes_like(1500, 1024, seed=5)
    codes[10:20] = -128
    codes[20:30] = 127
    codes[30:40] = rng.choice(np.array([-128, 127], np.int8), (10, 1024))
    ps.write(tmp_path / "s", codes, shard_rows=400)
    qs = np.concatenate([ps.queries(3, 1024), codes[35:36].astype(np.float32)])
    with fse.VectorStore.open(tmp_path / "s") as store:
        for k in (1, 10, 40):
            got, want, records = _both(monkeypatch, store, qs, k, workers=2)
            assert ps.prefilter_record(records).fse_fallback == "none"
            assert ps.same(got, want)


def test_equal_rows_across_the_block_edge_keep_their_order(tmp_path, monkeypatch):
    """Rows 65,535 and 65,536 are equal and the best rows: the first is the end of a block
    and the second the start of the next, so their order comes from the positions."""
    ps.enable(monkeypatch, rows=None, dim=None)
    codes = ps.codes_like(70_000, 256, seed=3, duplicates=False)
    codes[65_535] = codes[65_536] = codes[7]
    ps.write(tmp_path / "s", codes, shard_rows=30_000)
    qs = codes[7:8].astype(np.float32)
    with fse.VectorStore.open(tmp_path / "s") as store:
        for workers in (1, 3):
            got, want, records = _both(monkeypatch, store, qs, 4, workers=workers)
            assert ps.prefilter_record(records).fse_fallback == "none"
            assert ps.same(got, want)
            assert got.ids[0, :3].tolist() == [7, 65_535, 65_536]


@pytest.mark.parametrize("with_docs", [False, True])
def test_a_store_with_document_ids_labels_its_hits_as_before(tmp_path, monkeypatch, with_docs):
    ps.enable(monkeypatch)
    docs = np.random.default_rng(2).integers(0, 50, N).astype(np.int64) if with_docs else None
    ps.write(tmp_path / "s", ps.codes_like(N, DIM, seed=3), docs=docs)
    qs = ps.queries(2, DIM)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, want, records = _both(monkeypatch, store, qs, 5)
        assert ps.prefilter_record(records).fse_fallback == "none"
        assert ps.same(got, want)
        assert hasattr(got, "doc_ids") == with_docs
        one, want_one, _ = _both(monkeypatch, store, qs[0], 5)  # a single vector
        assert one.ids.shape == (5,)
        assert ps.same(one, want_one)


def test_a_query_out_of_the_proven_range_takes_the_search_without_it(tmp_path, monkeypatch):
    _store(tmp_path)
    ps.enable(monkeypatch)
    qs = np.zeros((2, DIM), np.float32)
    qs[0, :2] = [2.0**-74, 2.0**-75]  # its norm squares to a subnormal; normalized: ||u||^2 1.25
    qs[1] = ps.queries(1, DIM)[0]
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, want, records = _both(monkeypatch, store, qs, 5)
        record = ps.prefilter_record(records)
        assert record.fse_fallback == "query_range"
        assert record.fse_pairs == 0
        assert record.fse_rows == 0
        assert ps.same(got, want)
        assert len(ps.events(records, "store.search")) == 1  # the search of today logs it


@pytest.mark.parametrize(
    "damage",
    ["nan", "inf", "negative", "double", "zero_row_inf", "half"],
)
def test_an_unsafe_row_takes_the_search_without_it_with_its_bits_and_error(
    tmp_path, monkeypatch, damage
):
    codes = ps.codes_like(N, DIM, seed=3)
    inverse = _reference.inverse_norms(codes).copy()
    at = 1700
    if damage == "nan":
        inverse[at] = np.nan
    elif damage == "inf":
        inverse[at] = np.inf
    elif damage == "negative":
        inverse[at] = -inverse[at]
    elif damage == "double":
        inverse[at] *= 2
    elif damage == "half":
        inverse[at] *= 0.5  # a norm that is too small is safe: the bound holds
    else:
        codes[at] = 0
        inverse[at] = np.inf
    ps.write(tmp_path / "s", codes, inverse=inverse)
    ps.enable(monkeypatch)
    qs = ps.queries(2, DIM)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, want, records = _both(monkeypatch, store, qs, 5)
        record = ps.prefilter_record(records)
        assert record.fse_fallback == ("none" if damage == "half" else "unsafe_row")
        assert record.fse_pairs == 0 or damage == "half"
        assert ps.same(got, want)
        if damage in ("nan", "inf", "zero_row_inf"):
            assert isinstance(got, tuple)  # the error of today, with the same text


def test_a_block_that_keeps_too_many_rows_overflows_to_the_search_without_it(tmp_path, monkeypatch):
    row = ps.codes_like(1, DIM, seed=3)
    ps.write(tmp_path / "s", np.repeat(row, 5000, axis=0), shard_rows=5000)
    ps.enable(monkeypatch)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, want, records = _both(monkeypatch, store, ps.queries(2, DIM), 10)
        assert ps.prefilter_record(records).fse_fallback == "overflow"
        assert ps.same(got, want)


def test_survivors_beyond_the_budget_take_the_search_without_it(tmp_path, monkeypatch):
    row = ps.codes_like(1, DIM, seed=3)
    ps.write(tmp_path / "s", np.repeat(row, 68_000, axis=0), shard_rows=17_000)
    ps.enable(monkeypatch)
    monkeypatch.setattr(_search, "_MIN_ROWS", 4000)
    monkeypatch.setattr(_search, "_MAX_ROWS", 4000)
    monkeypatch.setattr(_prefilter, "_BUDGET_MIN", 3000)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, want, records = _both(monkeypatch, store, ps.queries(2, DIM), 10)
        record = ps.prefilter_record(records)
        assert record.fse_fallback == "budget"
        assert record.fse_pairs == 0
        assert ps.same(got, want)


@pytest.mark.parametrize("workers", [1, 3])
def test_the_adversarial_pair_keeps_its_exact_top_row(tmp_path, monkeypatch, workers):
    """A row A with a smaller estimate than B and a larger exact score survives only the
    full slack and the threshold minus the slack; the witness of the reference meta-tests."""
    stored, inverse, query, _ = witnesses.pair_store()
    ps.write(tmp_path / "s", stored, inverse=inverse, shard_rows=5)
    ps.enable(monkeypatch)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, want, records = _both(monkeypatch, store, query, 1, workers=workers)
        assert ps.prefilter_record(records).fse_fallback == "none"
        assert ps.same(got, want)
        assert got.ids.tolist() == [[5]]


def test_a_tie_at_the_floor_stays_a_survivor(tmp_path, monkeypatch):
    """Two rows with ``fl(a_A + E) == fl(a_B - E)``: the comparison is inclusive, so both are
    survivors (the canonical set), and B, the better row, is the hit."""
    stored, inverse, query = witnesses.exact_tie_store()
    ps.write(tmp_path / "s", stored, inverse=inverse, shard_rows=1)
    ps.enable(monkeypatch)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, want, records = _both(monkeypatch, store, query, 1)
        assert ps.prefilter_record(records).fse_pairs == 2
        assert ps.same(got, want)
        assert got.ids.tolist() == [[1]]


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
    arguments = {"exact": True, "grouped": False, "kind": "rows", "filtered": False} | changes
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
        ({"filtered": True}, True),
        ({"filtered": True, "kind": "doc_ids"}, False),
    ],
)
def test_the_domain_has_these_limits(change, expected):
    assert _applies(**change) is expected


def test_no_shards_and_no_variant_are_outside_the_domain():
    queries = np.zeros((1, 256), np.float32)
    plain = {"exact": True, "grouped": False, "kind": "rows", "filtered": False}
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
        got, want, records = _both(monkeypatch, store, qs, 10, workers=2)
        assert bool(ps.events(records, "store.prefilter")) is entered
        assert ps.same(got, want)
        if entered:
            assert ps.prefilter_record(records).fse_fallback == "none"


def test_modes_that_are_not_exact_and_stores_that_are_not_int8_never_enter(tmp_path, monkeypatch):
    ps.enable(monkeypatch)
    _store(tmp_path)
    qs = ps.queries(9, DIM)
    with fse.VectorStore.open(tmp_path / "s") as store:
        for arguments, q in (({"mode": "blas"}, 3), ({"mode": "auto"}, 9), ({"max_per_doc": 1}, 3)):
            with ps.captured() as records, contextlib.suppress(ValueError):
                store.search(qs[:q], 5, **arguments)  # max_per_doc needs document IDs
            assert not ps.events(records, "store.prefilter")
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


def test_a_backend_without_a_variant_and_the_default_list_never_enter(tmp_path, monkeypatch):
    _store(tmp_path)
    monkeypatch.setattr(_prefilter, "MIN_ROWS", 1)
    monkeypatch.setattr(_prefilter, "MIN_DIM", 1)
    qs = ps.queries(2, DIM)
    for backend in ("cpu", "numpy"):  # nothing is enabled: _PREFILTER_ENABLED is empty
        with fse.VectorStore.open(tmp_path / "s", backend=backend) as store:
            _, records = ps.run(store, qs, 5)
            assert not ps.events(records, "store.prefilter")


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
