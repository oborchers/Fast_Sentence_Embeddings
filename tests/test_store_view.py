"""``VectorStore.open_view``: stores read as one, with the bits and the order of ties of a
single store that holds the same rows."""

import gc
import pickle
import sys
import weakref

import numpy as np
import pytest

import fse
from fse import VectorStore
from fse._container import CorruptArtifactError, NpyContainer
from fse._store import _view
from support.logs import expect_log
from support.views import SIZES, vector_parts, write_vectors

pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning")
QUERIES = np.random.default_rng(5).normal(size=(3, 17)).astype(np.float32)


def bits(array):
    return np.asarray(array).view(np.uint8).tobytes()


@pytest.fixture(params=["float32", "float16", "int8"])
def pair(request, tmp_path):
    whole, parts = vector_parts(tmp_path, dtype=request.param)
    with VectorStore.open(whole) as one, VectorStore.open_view(parts) as view:
        yield one, view


# --- the same as one store -----------------------------------------------------------


def test_what_the_view_holds(pair):
    one, view = pair
    assert len(view) == len(one) == sum(SIZES)
    assert (view.dim, view.dtype, view.has_doc_ids) == (one.dim, one.dtype, one.has_doc_ids)
    assert view.encoder_fingerprint == one.encoder_fingerprint
    assert view.path is None


@pytest.mark.parametrize("query", [QUERIES, QUERIES[0]])
def test_search_gives_the_bits_and_the_ties_of_one_store(pair, query):
    one, view = pair
    k = len(one)  # every row is ranked, so every tie across a boundary is decided
    got, want = view.search(query, k, mode="exact"), one.search(query, k, mode="exact")
    assert bits(got.ids) == bits(want.ids)
    assert bits(got.scores) == bits(want.scores)
    assert bits(got.doc_ids) == bits(want.doc_ids)


def test_search_with_filters_and_one_row_per_document(pair):
    one, view = pair
    ids = one.get_doc_ids()
    for options in (
        {"doc_ids": [int(ids[0]), int(ids[7])]},
        {"rows": [30, 2, 12, 13, 21, 22]},
        {"max_per_doc": 1},
        {"max_per_doc": 1, "rows": list(range(5, 35))},
    ):
        got = view.search(QUERIES, 6, mode="exact", **options)
        want = one.search(QUERIES, 6, mode="exact", **options)
        assert bits(got.ids) == bits(want.ids)
        assert bits(got.scores) == bits(want.scores)
        assert bits(got.doc_ids) == bits(want.doc_ids)


def test_blas_mode_runs_over_the_view(pair):
    one, view = pair  # BLAS blocks differ with the shards, so only the values are compared
    got, want = view.search(QUERIES, 5, mode="blas"), one.search(QUERIES, 5, mode="blas")
    np.testing.assert_allclose(got.scores, want.scores, atol=1e-3)


def test_score_and_the_reads_of_rows(pair):
    one, view = pair
    weights = np.random.default_rng(1).normal(size=(2, 17)).astype(np.float32)
    assert bits(view.score(weights, [0.5, -0.25], mode="exact")) == bits(
        one.score(weights, [0.5, -0.25], mode="exact")
    )
    assert bits(view.get_vectors()) == bits(one.get_vectors())
    assert bits(view.get_vectors(start=11, stop=40)) == bits(one.get_vectors(start=11, stop=40))
    assert view.get_doc_ids(start=5, stop=36).tolist() == one.get_doc_ids(start=5, stop=36).tolist()
    rows = np.array([[39, 0], [13, 12], [22, 22]], dtype=np.int64)
    assert bits(view.take_vectors(rows)) == bits(one.take_vectors(rows))
    assert view.take_doc_ids(rows).tolist() == one.take_doc_ids(rows).tolist()


def raw(batch):
    return batch.vectors.codes if hasattr(batch.vectors, "codes") else batch.vectors


def test_iter_batches_run_over_the_joined_rows(pair):
    one, view = pair
    batches = list(view.iter_batches(block_rows=4, start=3, stop=37))
    assert batches[0].store_row == 3
    assert sum(b.n_rows for b in batches) == 34
    want = list(one.iter_batches(block_rows=1000, start=3, stop=37))
    assert bits(np.concatenate([raw(b) for b in batches])) == bits(
        np.concatenate([raw(b) for b in want])
    )
    assert [b.store_row for b in batches] == sorted(b.store_row for b in batches)
    assert (
        np.concatenate([b.doc_ids for b in batches]).tolist()
        == one.get_doc_ids(start=3, stop=37).tolist()
    )


def test_a_store_without_document_ids(tmp_path):
    whole, parts = vector_parts(tmp_path, doc_ids=False)
    with VectorStore.open(whole) as one, VectorStore.open_view(parts) as view:
        assert not view.has_doc_ids
        assert bits(view.search(QUERIES, 8, mode="exact").ids) == bits(
            one.search(QUERIES, 8, mode="exact").ids
        )
        with pytest.raises(ValueError, match="no doc_ids"):
            view.get_doc_ids()


def test_an_empty_store_among_the_members(tmp_path):
    whole, parts = vector_parts(tmp_path, sizes=(6, 7))
    empty = tmp_path / "empty"
    write_vectors(empty, [], np.empty(0, np.int64))
    with VectorStore.open(whole) as one, VectorStore.open_view([parts[0], empty, parts[1]]) as view:
        assert len(view) == len(one)
        assert bits(view.search(QUERIES, 13, mode="exact").ids) == bits(
            one.search(QUERIES, 13, mode="exact").ids
        )
        member, row = view.locate([0, 5, 6, 12])
        assert member.tolist() == [0, 0, 2, 2]
        assert row.tolist() == [0, 5, 0, 6]


# --- paths, members, source, locate ----------------------------------------------------


def test_paths_and_members(tmp_path):
    whole, parts = vector_parts(tmp_path)
    with VectorStore.open_view(parts) as view:
        assert view.paths == tuple(parts)
        assert [m["path"] for m in view.members] == parts
        assert [m["first_row"] for m in view.members] == [0, 13, 22]
        assert [m["rows"] for m in view.members] == list(SIZES)
        assert all(m["source"] is None for m in view.members)
        assert view.source is None
        with pytest.raises(TypeError):
            view.members[0]["rows"] = 1
    with VectorStore.open(whole) as one:
        assert one.paths == (whole,)
        assert [dict(m) for m in one.members] == [
            {"path": whole, "first_row": 0, "rows": sum(SIZES), "source": None}
        ]


def test_locate_names_the_store_and_the_row(tmp_path):
    _, parts = vector_parts(tmp_path)
    with VectorStore.open_view(parts) as view:
        member, row = view.locate(np.array([[0, 12], [13, 21], [22, 39]], dtype=np.int64))
        assert member.dtype == row.dtype == np.int64
        assert member.tolist() == [[0, 0], [1, 1], [2, 2]]
        assert row.tolist() == [[0, 12], [0, 8], [0, 17]]
        for bad in ([-1], [40], [0, 1000]):
            with pytest.raises(ValueError, match="within the 40 rows"):
                view.locate(bad)
        empty = view.locate([])
        assert empty[0].shape == empty[1].shape == (0,)
        with pytest.raises(TypeError):
            view.locate(np.array([1.0]))
        for opened, paths in ((view.members[0]["path"], view.paths),):
            assert opened == paths[0]


def test_locate_of_a_store_that_is_no_view(tmp_path):
    whole, _ = vector_parts(tmp_path)
    with VectorStore.open(whole) as one:
        member, row = one.locate([3, 39])
        assert member.tolist() == [0, 0]
        assert row.tolist() == [3, 39]
        member[0] = 9  # a new array each time
        assert one.locate([3])[0].tolist() == [0]
        with pytest.raises(ValueError, match="40 rows"):
            one.locate([40])
    with pytest.raises(RuntimeError, match="closed"):
        one.locate([1])


# --- what is refused -------------------------------------------------------------------


def test_what_the_stores_must_share(tmp_path):
    _, parts = vector_parts(tmp_path, sizes=(6, 7))
    half = tmp_path / "half"
    write_vectors(half, ["w1 w2"] * 3, np.arange(3), dtype="float16")
    plain = tmp_path / "plain"
    write_vectors(plain, ["w1 w2"] * 3, None)
    other = tmp_path / "other"
    with fse.ShardSink(other, finished_other(), shard_rows=4) as sink:
        finished_other().encode(["w1 w2"] * 3, sink=sink, workers=1)
    for bad, text in ((half, "dtype"), (plain, "has_doc_ids"), (other, "encoder")):
        with pytest.raises(ValueError, match=text):
            VectorStore.open_view([parts[0], bad])
        with pytest.raises(ValueError, match=text):
            VectorStore.open_view([bad, parts[0]])


def finished_other():
    from support.encoder import finished

    return finished(dim=17, seed=9)


def test_what_a_list_of_paths_must_be(tmp_path):
    _, parts = vector_parts(tmp_path, sizes=(4, 5))
    with pytest.raises(TypeError, match="list or tuple"):
        VectorStore.open_view(parts[0])
    with pytest.raises(TypeError, match="list or tuple"):
        VectorStore.open_view(str(parts[0]))
    with pytest.raises(ValueError, match="at least one"):
        VectorStore.open_view([])
    with pytest.raises(TypeError, match="str or path"):
        VectorStore.open_view([parts[0], 3])
    with pytest.raises(ValueError, match="listed twice"):
        VectorStore.open_view([parts[0], parts[1], str(parts[0])])

    with pytest.raises(TypeError, match="encoder must be an Encoder"):
        VectorStore.open_view(parts, encoder=3)
    with pytest.raises(ValueError, match="backend"):
        VectorStore.open_view(parts, backend="tpu")
    with VectorStore.open_view(tuple(parts)) as view:
        assert len(view) == 9


@pytest.mark.skipif(sys.platform == "win32", reason="a symbolic link needs a privilege there")
def test_a_link_to_a_listed_store_is_listed_twice(tmp_path):
    _, parts = vector_parts(tmp_path, sizes=(4, 5))
    link = tmp_path / "link"
    link.symlink_to(parts[0], target_is_directory=True)
    with pytest.raises(ValueError, match="listed twice"):
        VectorStore.open_view([parts[0], link])


def test_a_view_of_one_store_and_of_the_encoder(tmp_path):
    whole, parts = vector_parts(tmp_path, sizes=(4, 5))
    with VectorStore.open_view([whole]) as view:
        assert len(view) == 9
        assert view.paths == (whole,)
    with VectorStore.open_view(parts, encoder=fse_encoder()) as view:
        assert len(view) == 9
    with pytest.raises(ValueError, match="encoder"):
        VectorStore.open_view(parts, encoder=finished_other())


def fse_encoder():
    from support.encoder import finished

    return finished(dim=17)


@pytest.mark.parametrize("failing", ["missing", "unfit"])
def test_a_failed_open_closes_the_members_and_holds_no_mappings(tmp_path, monkeypatch, failing):
    _, parts = vector_parts(tmp_path, sizes=(4, 5))
    bad = tmp_path / "bad"
    if failing == "unfit":
        write_vectors(bad, ["w1 w2"] * 3, None, dtype="int8")
    refs, closed = [], []
    real_open, real_close = NpyContainer.open, VectorStore.close

    def opened(self, *args, **kwargs):
        result = real_open(self, *args, **kwargs)
        refs.extend(weakref.ref(tensor) for tensor in result[0].values())
        return result

    def closing(self):
        closed.append(self)
        real_close(self)

    monkeypatch.setattr(NpyContainer, "open", opened)
    monkeypatch.setattr(VectorStore, "close", closing)
    with pytest.raises((ValueError, CorruptArtifactError)) as caught:
        VectorStore.open_view([parts[0], parts[1], bad])
    gc.collect()  # the error and its traceback are still held: only the cleared frames free them
    assert len(closed) == (2 if failing == "missing" else 3)
    assert refs
    assert all(ref() is None for ref in refs)
    assert str(caught.value)


# --- life cycle ----------------------------------------------------------------------


def test_a_view_has_no_refresh_and_close_closes_the_stores(tmp_path):
    _, parts = vector_parts(tmp_path, sizes=(4, 5))
    view = VectorStore.open_view(parts)
    members = view._view.stores
    with pytest.raises(ValueError, match="does not refresh"):
        view.refresh()
    view.close()
    assert all(m._closed for m in members)
    assert view._shards == []
    for call in (view.refresh, view.verify, lambda: view.search(QUERIES[0], 1)):
        with pytest.raises(RuntimeError, match="closed"):
            call()
    view.close()  # closing twice does no harm


def test_close_waits_for_a_running_read(tmp_path):
    _, parts = vector_parts(tmp_path, sizes=(4, 5))
    with VectorStore.open_view(parts) as view:
        with view._using(), pytest.raises(RuntimeError, match="in use"):
            view.close()
        assert not view._closed
        assert not any(m._closed for m in view._view.stores)


def test_verify_checks_each_store_over_its_path(tmp_path):
    _, parts = vector_parts(tmp_path, sizes=(4, 5))
    with (
        VectorStore.open_view(parts) as view,
        expect_log("store.verified", rows=4),
        expect_log("store.verified", rows=5),
    ):
        view.verify()


def test_verify_finds_damage_in_a_store(tmp_path):
    _, parts = vector_parts(tmp_path, sizes=(4, 5))
    victim = sorted(parts[1].glob("shard-*/vectors.npy"))[0]
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0x55
    victim.write_bytes(bytes(data))  # before the view maps it: Windows cannot write a mapped file
    with VectorStore.open_view(parts) as view, pytest.raises(CorruptArtifactError):
        view.verify()


def test_the_view_is_logged(tmp_path):
    _, parts = vector_parts(tmp_path, sizes=(4, 5))
    with expect_log("store.view_opened", stores=2, rows=9, dtype="float32", mode="list"):
        VectorStore.open_view(parts).close()


def test_the_pieces_the_view_leans_on(tmp_path):
    _, parts = vector_parts(tmp_path, sizes=(4, 5))
    with VectorStore.open_view(parts) as view:
        assert view._view.mode == _view.LIST_ORDER
        assert view._view.offsets == (0, 4)
        with pytest.raises(TypeError, match="pickled"):
            pickle.dumps(view)
