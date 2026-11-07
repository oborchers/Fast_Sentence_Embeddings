"""The int8 prefilter in the exact search: the bits of the search without it, the fallbacks of
the whole call, the limits of its domain (docs/architecture/store.md, "Search")."""

import threading

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
        got, want, records = ps.both(monkeypatch, store, qs, k, workers=workers)
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
        got, want, records = ps.both(monkeypatch, store, qs, 6, rows=rows.tolist(), workers=workers)
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
        got, want, records = ps.both(monkeypatch, view, qs, 8, workers=3)
        assert ps.prefilter_record(records).fse_rows == 2 * N
        assert ps.same(got, want)
        got, want, records = ps.both(monkeypatch, view, qs, 8, rows=list(range(N - 50, N + 50)))
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
        got, want, records = ps.both(monkeypatch, store, qs, 8)
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
            got, want, records = ps.both(monkeypatch, store, qs, k, workers=3)
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
            got, want, records = ps.both(monkeypatch, store, qs, k, workers=2)
            assert ps.prefilter_record(records).fse_fallback == "none"
            assert ps.same(got, want)


def test_equal_rows_across_the_block_edge_keep_their_order(tmp_path, monkeypatch):
    """Rows 65,535 and 65,536 are equal and the best rows: the first is the end of a block
    and the second the start of the next, so their order comes from the positions."""
    ps.enable(monkeypatch, rows=None, dim=None)
    codes = ps.codes_like(70_000, 256, seed=3, duplicates=False)
    codes[65_535] = codes[65_536] = codes[7]
    ps.write(tmp_path / "s", codes, shard_rows=70_000)  # one shard: the edge is a block edge
    qs = codes[7:8].astype(np.float32)
    with fse.VectorStore.open(tmp_path / "s") as store:
        for workers in (1, 3):
            got, want, records = ps.both(monkeypatch, store, qs, 4, workers=workers)
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
        got, want, records = ps.both(monkeypatch, store, qs, 5)
        assert ps.prefilter_record(records).fse_fallback == "none"
        assert ps.same(got, want)
        assert hasattr(got, "doc_ids") == with_docs
        one, want_one, _ = ps.both(monkeypatch, store, qs[0], 5)  # a single vector
        assert one.ids.shape == (5,)
        assert ps.same(one, want_one)


def test_a_query_out_of_the_proven_range_takes_the_search_without_it(tmp_path, monkeypatch):
    _store(tmp_path)
    ps.enable(monkeypatch)
    qs = np.zeros((2, DIM), np.float32)
    qs[0, :2] = [2.0**-74, 2.0**-75]  # its norm squares to a subnormal; normalized: ||u||^2 1.25
    qs[1] = ps.queries(1, DIM)[0]
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, want, records = ps.both(monkeypatch, store, qs, 5)
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
        got, want, records = ps.both(monkeypatch, store, qs, 5)
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
        got, want, records = ps.both(monkeypatch, store, ps.queries(2, DIM), 10)
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
        got, want, records = ps.both(monkeypatch, store, ps.queries(2, DIM), 10)
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
        got, want, records = ps.both(monkeypatch, store, query, 1, workers=workers)
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
        got, want, records = ps.both(monkeypatch, store, query, 1)
        assert ps.prefilter_record(records).fse_pairs == 2
        assert ps.same(got, want)
        assert got.ids.tolist() == [[1]]


ENVS = [{"flush": True}, {"round_up": True}, {"flush": True, "round_up": True}]


@pytest.mark.parametrize("env", ENVS, ids=lambda e: "-".join(e))
def test_a_caller_with_another_float_environment_gets_the_same_bits(tmp_path, monkeypatch, env):
    """The cut of the survivors runs in the calling thread: under flush to zero or rounding
    toward +inf it keeps every collected row, and the result is still that of the exact path."""
    from fse import _native

    _store(tmp_path)
    ps.enable(monkeypatch)
    qs = ps.queries(3, DIM)
    with fse.VectorStore.open(tmp_path / "s") as store:
        _, want, _ = ps.both(monkeypatch, store, qs, 6)
        assert _prefilter._default_environment()
        saved = _native._test_fpenv_set(**env)
        try:
            assert not _prefilter._default_environment()
            got, records = ps.run(store, qs, 6, workers=2)
        finally:
            _native._test_fpenv_restore(saved)
        assert _prefilter._default_environment()
        assert ps.prefilter_record(records).fse_fallback == "none"
        assert ps.same(got, want)


def test_a_fallback_leaves_no_search_thread_running(tmp_path, monkeypatch):
    row = ps.codes_like(1, DIM, seed=3)
    ps.write(tmp_path / "s", np.repeat(row, 68_000, axis=0), shard_rows=17_000)
    ps.enable(monkeypatch)
    monkeypatch.setattr(_search, "_MIN_ROWS", 4000)
    monkeypatch.setattr(_search, "_MAX_ROWS", 4000)
    monkeypatch.setattr(_prefilter, "_BUDGET_MIN", 3000)
    with fse.VectorStore.open(tmp_path / "s") as store:
        _, records = ps.run(store, ps.queries(2, DIM), 10, workers=3)
    assert ps.prefilter_record(records).fse_fallback == "budget"
    assert not [t for t in threading.enumerate() if t.name.startswith("fse-search")]
