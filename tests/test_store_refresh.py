"""``VectorStore.refresh`` and ``ScoreStore.refresh``: the shards published since the
snapshot, added in place; the snapshots of running calls stay valid."""

import gc
import shutil
import threading
import weakref

import numpy as np
import pytest

import fse
from fse import ScoreStore, VectorStore
from fse._container import FINGERPRINT_KEY, CorruptArtifactError, NpyContainer
from fse._scores import _scorestore
from fse._store import _search, _vectorstore
from fse._store._kind import Shard
from fse._store._layout import create_store, shard_name
from support import scores as sc
from support.logs import expect_log
from support.store import add_shard, append_rows, make_store, tag_shard

pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning")


@pytest.fixture
def vectors(tmp_path):
    path = tmp_path / "store"
    make_store(path, sizes=(4, 4))
    with VectorStore.open(path) as store:
        yield store


@pytest.fixture
def scores(tmp_path):
    path = tmp_path / "scores"
    sc.make_scores(path, sizes=(4, 4))
    with ScoreStore.open(path) as store:
        yield store


def _new_scores(store, first, rows, **kwargs):
    sc.add_shard(store.path, store._info, first, rows, **kwargs)


# --- what a refresh does --------------------------------------------------------------


def test_refresh_adds_the_published_shards_in_place(vectors):
    assert len(vectors) == 8
    first = append_rows(vectors.path, value=7, rows=3)
    assert first == 8
    with expect_log("store.refreshed", rows=11, added_rows=3, added_shards=1):
        assert vectors.refresh() == 3
    assert len(vectors) == 11
    assert vectors.refresh() == 0
    assert len(vectors) == 11
    with VectorStore.open(vectors.path) as fresh:
        np.testing.assert_array_equal(vectors.get_vectors(), fresh.get_vectors())
        query = np.ones(vectors.dim, np.float32)
        a, b = vectors.search(query, 5, mode="exact"), fresh.search(query, 5, mode="exact")
        np.testing.assert_array_equal(a.ids, b.ids)
        np.testing.assert_array_equal(a.scores.view(np.uint32), b.scores.view(np.uint32))
        np.testing.assert_array_equal(vectors.take_vectors([10, 0]), fresh.take_vectors([10, 0]))


def test_score_refresh_adds_the_published_shards_in_place(scores):
    _new_scores(scores, 8, 3)
    with expect_log("scores.refreshed", rows=11, added_rows=3, added_shards=1):
        assert scores.refresh() == 3
    assert len(scores) == 11
    assert scores.refresh() == 0
    with ScoreStore.open(scores.path) as fresh:
        for ours, theirs in zip(scores.iter_batches(), fresh.iter_batches(), strict=True):
            assert ours.store_row == theirs.store_row
            for name in ours.outputs:
                np.testing.assert_array_equal(ours.outputs[name], theirs.outputs[name])


def test_refresh_reaches_the_shards_a_writer_publishes_one_after_another(vectors):
    for rows in (2, 1, 5):
        append_rows(vectors.path, rows=rows)
    assert vectors.refresh() == 8
    assert len(vectors) == 16
    assert vectors.refresh() == 0


def test_an_empty_store_takes_its_first_shard(tmp_path):
    path = tmp_path / "store"
    info = make_store(path, sizes=())
    with VectorStore.open(path) as store:
        assert len(store) == 0
        assert store.refresh() == 0
        add_shard(path, info, 0, 3)
        assert store.refresh() == 3
        assert len(store) == 3


def test_an_empty_score_store_takes_its_first_shard_and_its_thresholds(tmp_path):
    path = tmp_path / "scores"
    info = sc.make_scores(path, sizes=())
    with ScoreStore.open(path) as store:
        assert store.refresh() == 0
        old = store.predict_batches()
        sc.add_shard(path, info, 0, 3, limit=0.2)
        assert store.refresh() == 3
        assert list(old) == []  # rows and thresholds of the call come from the same state
        (batch,) = store.predict_batches()
        assert batch.n_rows == 3


def test_only_the_new_tagged_shards_are_hashed(tmp_path):
    path = tmp_path / "store"
    info = make_store(path, sizes=(4, 4), tagged=(0, 4))
    with VectorStore.open(path) as store:
        add_shard(path, info, 8, 4)
        tag_shard(path / shard_name(8))
        with expect_log("store.refreshed", unsynced_shards=1) as records:
            store.refresh()
        tensors = store._shards[-1][1]
    (record,) = records
    assert record.fse_verified_bytes == sum(t.nbytes for t in tensors.values()) > 0


def test_the_last_shard_has_its_manifest_fingerprint(vectors):
    shard = vectors._shards[-1][0]
    expected = NpyContainer().metadata(vectors.path / shard.name)[FINGERPRINT_KEY]
    assert shard.fingerprint == expected
    assert Shard("a", 0, 1, fingerprint="x") == Shard("a", 0, 1)  # not part of the identity


# --- what it refuses ------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["gap", "overlap"])
def test_a_gap_or_an_orphan_raises_and_the_snapshot_stays(tmp_path, kind):
    path = tmp_path / "store"
    info = make_store(path, sizes=(4, 4))
    with VectorStore.open(path) as store:
        add_shard(path, info, 12 if kind == "gap" else 6, 2)
        with pytest.raises(CorruptArtifactError, match=r"fse\.repair_store"):
            store.refresh()
        assert len(store) == 8
        assert store.search(np.ones(store.dim, np.float32), 3).ids.shape == (3,)


def test_a_break_of_the_chain_raises(scores):
    sc.add_shard(scores.path, scores._info, 8, 2, limit=0.9)  # other thresholds
    with pytest.raises(CorruptArtifactError, match="thresholds differ"):
        scores.refresh()
    assert len(scores) == 8


def test_a_new_shard_that_does_not_match_the_store_raises(vectors):
    add_shard(vectors.path, vectors._info, 8, 2)
    with (vectors.path / shard_name(8) / "vectors.npy").open("r+b") as handle:
        handle.truncate(10)
    with pytest.raises(CorruptArtifactError):
        vectors.refresh()
    assert len(vectors) == 8


def test_a_changed_last_shard_is_detected(vectors):
    tag_shard(vectors.path / shard_name(4))  # a repair or a rewrite below the snapshot
    add_shard(vectors.path, vectors._info, 8, 2)
    with pytest.raises(RuntimeError, match="changed below its snapshot; open it again"):
        vectors.refresh()
    assert len(vectors) == 8


def test_a_missing_last_shard_is_detected(scores):
    shutil.rmtree(scores.path / shard_name(4))
    with pytest.raises(RuntimeError, match="changed below its snapshot"):
        scores.refresh()


def test_another_store_in_the_place_is_detected(tmp_path):
    path = tmp_path / "store"
    make_store(path, dtype="float32")
    with VectorStore.open(path) as store:
        shutil.rmtree(path)
        make_store(path, dtype="float16")
        with pytest.raises(RuntimeError, match="changed below its snapshot"):
            store.refresh()
        assert len(store) == 10


@pytest.mark.parametrize("failure", ["gap", "changed"])
def test_a_failed_refresh_holds_no_mappings(tmp_path, monkeypatch, failure):
    path = tmp_path / "store"
    info = make_store(path, sizes=(4, 4))
    with VectorStore.open(path) as store:
        if failure == "gap":
            add_shard(path, info, 8, 2)
            add_shard(path, info, 14, 2)
        else:
            tag_shard(path / shard_name(4))
        refs = []
        real = NpyContainer.open

        def opened(self, *args, **kwargs):
            result = real(self, *args, **kwargs)
            refs.extend(weakref.ref(tensor) for tensor in result[0].values())
            return result

        monkeypatch.setattr(NpyContainer, "open", opened)
        with pytest.raises((CorruptArtifactError, RuntimeError)) as caught:
            store.refresh()
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert str(caught.value)


def test_a_closed_store_cannot_refresh(vectors, scores):
    vectors.close()
    scores.close()
    with pytest.raises(RuntimeError, match="closed"):
        vectors.refresh()
    with pytest.raises(RuntimeError, match="closed"):
        scores.refresh()


# --- snapshots of running calls -------------------------------------------------------


def _stalled(monkeypatch, module, name):
    """Make ``module.name`` stop when called until ``release`` is set; the returned event
    says it stopped, and the shards it was given are in ``seen``."""
    real = getattr(module, name)
    reached, release, seen = threading.Event(), threading.Event(), []

    def stalled(*args, **kwargs):
        seen.append(args)
        reached.set()
        assert release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, stalled)
    return reached, release, seen


def test_a_running_score_keeps_its_snapshot(vectors, monkeypatch):
    reached, release, _ = _stalled(monkeypatch, _search, "score")
    out = {}

    def run():
        out["scores"] = vectors.score(np.ones(vectors.dim, np.float32))

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert reached.wait(10)
        append_rows(vectors.path)
        assert vectors.refresh() == 4  # a running call does not block a refresh
    finally:
        release.set()
        thread.join(10)
    assert out["scores"].shape == (8,)
    assert vectors.score(np.ones(vectors.dim, np.float32)).shape == (12,)


def test_a_running_search_keeps_the_rows_ids_and_scores_of_its_snapshot(tmp_path, monkeypatch):
    from support.take_store import critical_rows, write_rows

    path = tmp_path / "store"
    codes, _ = critical_rows(14, "float32")
    ids = np.arange(14, dtype=np.int64) + 100
    write_rows(path, "float32", codes, shard_rows=7, doc_ids=ids)
    with VectorStore.open(path) as store:
        reached, release, _ = _stalled(monkeypatch, _search, "search")
        out = {}

        def run():
            out["hits"] = store.search(codes[3], 20, mode="exact")

        thread = threading.Thread(target=run)
        thread.start()
        try:
            assert reached.wait(10)
            more, _ = critical_rows(7, "float32", seed=5)
            with fse.ShardSink(path, encoder_of(codes), shard_rows=7, doc_ids=True) as sink:
                target = sink.target(0, 7)
                target[0][...] = more
                target[-1][...] = 0.25
                sink.commit(0, target, doc_ids=np.arange(7, dtype=np.int64) + 500)
            assert store.refresh() == 7
        finally:
            release.set()
            thread.join(10)
        hits = out["hits"]
        assert len(hits.ids) == 14
        assert hits.ids.max() < 14
        assert set(hits.doc_ids) == set(ids)
        assert len(store.search(codes[3], 30, mode="exact").ids) == 21


def encoder_of(codes):
    from support.encoder import finished

    return finished(dim=codes.shape[1])


def test_a_generator_made_before_a_refresh_reads_the_rows_of_its_call(vectors):
    fresh = vectors.iter_batches(block_rows=2)
    started = vectors.iter_batches(block_rows=2)
    first = next(started)
    append_rows(vectors.path)
    assert vectors.refresh() == 4
    assert sum(len(b.vectors) for b in fresh) == 8
    assert first.store_row == 0
    assert sum(len(b.vectors) for b in started) == 6
    assert sum(len(b.vectors) for b in vectors.iter_batches()) == 12
    with pytest.raises(ValueError, match="not within the 12 rows"):
        vectors.iter_batches(stop=13)


def test_a_score_generator_made_before_a_refresh_reads_the_rows_of_its_call(scores):
    before = scores.iter_batches(block_rows=2)
    predicted = scores.predict_batches(block_rows=2)
    _new_scores(scores, 8, 4)
    assert scores.refresh() == 4
    assert sum(b.n_rows for b in before) == 8
    assert sum(b.n_rows for b in predicted) == 8
    assert sum(b.n_rows for b in scores.iter_batches()) == 12
    assert scores.get_scores("quality", "values").shape == (12,)


def test_a_refresh_does_not_let_close_unmap_rows_it_reads(vectors, monkeypatch):
    reached, release, _ = _stalled(monkeypatch, _vectorstore, "refreshed")
    thread = threading.Thread(target=vectors.refresh)
    thread.start()
    try:
        assert reached.wait(10)
        with pytest.raises(RuntimeError, match="in use"):
            vectors.close()
    finally:
        release.set()
        thread.join(10)
    vectors.close()


def test_a_score_store_closed_during_a_refresh_stays_closed(scores, monkeypatch):
    _new_scores(scores, 8, 2)
    reached, release, _ = _stalled(monkeypatch, _scorestore, "refreshed")
    errors = []

    def run():
        try:
            scores.refresh()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert reached.wait(10)
        scores.close()
    finally:
        release.set()
        thread.join(10)
    assert errors == ["the ScoreStore is closed"]
    with pytest.raises(RuntimeError, match="closed"):
        len(list(scores.iter_batches()))


def test_two_refreshes_at_once_add_the_shards_once(vectors, monkeypatch):
    append_rows(vectors.path)
    reached, release, _ = _stalled(monkeypatch, _vectorstore, "refreshed")
    results = []
    first = threading.Thread(target=lambda: results.append(vectors.refresh()))
    second = threading.Thread(target=lambda: results.append(vectors.refresh()))
    first.start()
    try:
        assert reached.wait(10)
        second.start()
    finally:
        release.set()
        first.join(10)
        second.join(10)
    assert sorted(results) == [0, 4]
    assert len(vectors) == 12


def test_create_store_then_open_and_refresh_roundtrip(tmp_path):
    path = tmp_path / "scores"
    info = sc.default_info()
    create_store(path, info)
    with ScoreStore.open(path) as store:
        assert store.refresh() == 0
