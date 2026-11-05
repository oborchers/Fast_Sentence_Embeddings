"""ScoreStore: opening, what it holds, copies, verification and closing."""

import numpy as np
import pytest

import fse
from fse._container import CorruptArtifactError
from fse._store._layout import shard_name
from support import score_heads as sh
from support import scores as sc
from support.logs import captured
from support.store import make_store

STORED = {"quality": "values", "toxic": "proba", "topic": "decision"}


@pytest.fixture
def store_path(tmp_path):
    sc.write_store(tmp_path / "s", sh.group(), sh.texts(40), store=STORED, doc_ids=True)
    return tmp_path / "s"


def test_open_reports_what_the_store_holds(store_path):
    heads = sh.group()
    with fse.ScoreStore.open(store_path) as store:
        assert len(store) == 40
        assert list(store.heads) == ["quality", "toxic", "topic"]
        assert store.encoder_fingerprint == heads.encoder_fingerprint
        assert store.has_doc_ids
        toxic = store.heads["toxic"]
        assert (toxic.stored, toxic.path, toxic.columns) == ("proba", "folded", (1, 4))
        assert toxic.labels == ("a", "b", "c")
        assert store.heads["quality"].stored == "values"
        with pytest.raises(TypeError):
            store.heads["x"] = 1


def test_open_with_the_heads_checks_names_order_and_models(store_path):
    heads = sh.group()
    fse.ScoreStore.open(store_path, heads=heads).close()
    enc = sh.encoder()
    cases = [
        (
            fse.Heads(topic=heads["topic"], quality=heads["quality"], toxic=heads["toxic"]),
            "holds the heads",
        ),
        (fse.Heads(quality=heads["quality"]), "holds the heads"),
        (
            fse.Heads(
                quality=sh.regressor(enc, seed=1), toxic=heads["toxic"], topic=heads["topic"]
            ),
            "another model",
        ),
    ]
    for other, message in cases:
        with pytest.raises(ValueError, match=message):
            fse.ScoreStore.open(store_path, heads=other)
    with pytest.raises(TypeError, match=r"fse\.Heads"):
        fse.ScoreStore.open(store_path, heads=object())


def test_open_checks_the_encoder_of_the_heads(store_path):
    other = sh.group(sh.encoder(seed=4))
    with pytest.raises(ValueError, match=r"holds the heads|another"):
        fse.ScoreStore.open(store_path, heads=other)


def test_the_constructor_and_pickling_are_refused(store_path):
    with pytest.raises(TypeError, match=r"ScoreStore\.open"):
        fse.ScoreStore()
    with fse.ScoreStore.open(store_path) as store, pytest.raises(TypeError, match="pickled"):
        store.__reduce__()


def test_a_vector_store_and_a_missing_path_are_refused(tmp_path):
    make_store(tmp_path / "v", sizes=(2,))
    with pytest.raises(fse.UnsupportedArtifactError, match=r"fse\.VectorStore"):
        fse.ScoreStore.open(tmp_path / "v")
    with pytest.raises(CorruptArtifactError):
        fse.ScoreStore.open(tmp_path / "missing")
    with pytest.raises(ValueError, match="backend"):
        fse.ScoreStore.open(tmp_path / "v", backend="gpu")


def test_get_scores_gives_a_copy_of_what_was_stored_and_nothing_else(store_path):
    with fse.ScoreStore.open(store_path) as store:
        proba = store.get_scores("toxic", "proba")
        assert proba.shape == (40, 3)
        assert proba.flags.writeable
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        assert store.get_scores("topic", "decision", start=5, stop=9).shape == (4, 3)
        assert store.get_scores("quality", "values").shape == (40, 1)
        for head, wrong in (("toxic", "decision"), ("topic", "proba"), ("quality", "decision")):
            with pytest.raises(ValueError, match=f"head {head!r} stored"):
                store.get_scores(head, wrong)
        with pytest.raises(ValueError, match="holds the heads"):
            store.get_scores("nope", "values")
        with pytest.raises(ValueError, match="not within the 40 rows"):
            store.get_scores("toxic", "proba", start=3, stop=99)


def test_a_flat_head_gives_one_dimension(tmp_path):
    enc = sh.encoder()
    heads = fse.Heads(flat=sh.regressor(enc, vector=True), flag=sh.classifier(enc, 1))
    sc.write_store(tmp_path / "s", heads, sh.texts(10))
    with fse.ScoreStore.open(tmp_path / "s") as store:
        assert store.get_scores("flat", "values").shape == (10,)
        assert store.get_scores("flag", "decision").shape == (10,)


def test_document_ids_are_copies_and_need_a_store_with_them(store_path, tmp_path):
    with fse.ScoreStore.open(store_path) as store:
        ids = store.get_doc_ids()
        assert ids.dtype == np.int64
        assert ids.tolist() == list(range(40))
        assert store.get_doc_ids(start=38).tolist() == [38, 39]
    sc.write_store(tmp_path / "plain", sh.group(), sh.texts(5))
    with fse.ScoreStore.open(tmp_path / "plain") as store:
        assert not store.has_doc_ids
        with pytest.raises(ValueError, match="no doc_ids"):
            store.get_doc_ids()


def test_an_empty_store_is_readable(tmp_path):
    heads = sh.group()
    with fse.ScoreSink(tmp_path / "s", heads) as sink:
        sink._attach(dict(heads.paths()))
    with fse.ScoreStore.open(tmp_path / "s") as store:
        assert len(store) == 0
        assert list(store.iter_batches()) == []
        assert list(store.predict_batches()) == []
        assert store.get_scores("topic", "decision").shape == (0, 3)
        assert store.get_scores("quality", "values").shape == (0, 1)


def test_verify_hashes_the_shards_and_reports(store_path):
    with captured() as records, fse.ScoreStore.open(store_path) as store:
        store.verify()
    opened = [r for r in records if r.fse_event == "scores.opened"]
    verified = [r for r in records if r.fse_event == "scores.verified"]
    assert (opened[0].fse_rows, opened[0].fse_shards, opened[0].fse_heads) == (40, 3, 3)
    assert opened[0].fse_columns == 7
    assert (verified[0].fse_rows, verified[0].fse_shards) == (40, 3)


def test_verify_finds_a_changed_score_that_open_does_not_hash(store_path):
    with (store_path / shard_name(0) / "scores.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\xff")
    with fse.ScoreStore.open(store_path) as store, pytest.raises(CorruptArtifactError):
        store.verify()


def test_a_closed_store_gives_nothing_more_but_views_stay_readable(store_path):
    store = fse.ScoreStore.open(store_path)
    batches = store.iter_batches(block_rows=10)
    first = next(batches)
    view = first.outputs["topic"]
    store.close()
    store.close()  # closing twice is fine
    assert view.shape == (10, 3)
    assert np.isfinite(view).all()
    with pytest.raises(RuntimeError, match="closed"):
        next(batches)
    for call in (
        store.iter_batches,
        store.predict_batches,
        store.verify,
        store.get_doc_ids,
        lambda: store.get_scores("topic", "decision"),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            call()


def test_only_the_shards_published_at_open_are_read(store_path):
    store = fse.ScoreStore.open(store_path)
    heads = sh.group()
    with fse.ScoreSink(store_path, heads, store=STORED, doc_ids=True) as sink:
        list(heads.predict_batches([(sh.texts(5), np.arange(5))], score_sink=sink, doc_ids=True))
    assert len(store) == 40
    with fse.ScoreStore.open(store_path) as again:
        assert len(again) == 45
    store.close()
