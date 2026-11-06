"""Document IDs through a single head: the vector store and the score store get the same
IDs, and a wrong input fails with the type and text of the shared validator."""

import numpy as np
import pytest

import fse
from support import lone
from support import score_heads as sh


@pytest.fixture
def clf():
    return sh.classifier(lone.enc(), 3, seed=4)


def _ids_run(tmp_path, clf, batch_list):
    with (
        fse.ShardSink(tmp_path / "v", clf.encoder, doc_ids=True, shard_rows=8) as vectors,
        fse.ScoreSink(tmp_path / "s", clf, name="m", doc_ids=True, shard_rows=8) as scores,
    ):
        list(
            clf.predict_batches(
                batch_list,
                return_vectors=True,
                vector_sink=vectors,
                score_sink=scores,
                doc_ids=True,
                workers=2,
            )
        )


def test_both_stores_hold_the_ids_of_the_batches(tmp_path, clf):
    batch_list = lone.batches(ids=True)
    _ids_run(tmp_path, clf, batch_list)
    expected = np.concatenate([np.asarray(ids) for _, ids in batch_list])
    with fse.VectorStore.open(tmp_path / "v") as store:
        assert store.get_doc_ids().tolist() == expected.tolist()
    with fse.ScoreStore.open(tmp_path / "s", heads=clf, name="m") as store:
        assert store.get_doc_ids().tolist() == expected.tolist()


@pytest.mark.parametrize(
    ("item", "error", "pattern"),
    [
        ("w1 w2", TypeError, "Classifier.predict_batches: batch 1 must be a"),
        (("w1 w2", None), TypeError, "Classifier.predict_batches: batch 1 must give its doc_ids"),
        ((["w1", "w2"], [1]), ValueError, r"batch 1 \(rows 13 to 14 of the run\): doc_ids has 1"),
    ],
)
def test_a_wrong_batch_names_the_method_and_the_batch(tmp_path, clf, item, error, pattern):
    first = lone.batches(ids=True)[0]
    with (
        fse.ScoreSink(tmp_path / "s", clf, name="m", doc_ids=True) as sink,
        pytest.raises(error, match=pattern),
    ):
        list(clf.predict_batches([first, item], score_sink=sink, doc_ids=True))


def test_a_has_doc_ids_that_is_no_bool_fails_with_ids_and_is_ignored_without(clf):
    class Odd(fse.MemorySink):
        has_doc_ids = 1

    sink = Odd(clf.encoder.output_dim)
    with pytest.raises(TypeError, match="has_doc_ids of the vector_sink must be True"):
        clf.predict_batches(iter(()), return_vectors=True, vector_sink=sink, doc_ids=True)
    got = list(clf.predict_batches(lone.batches(13)[:1], return_vectors=True, vector_sink=sink))
    assert got[0].vectors is None
    assert sink.result().shape == (13, clf.encoder.output_dim)


def test_a_foreign_sink_with_ids_runs_like_in_2_0_as_sink(clf):
    class Foreign(fse.MemorySink):
        has_doc_ids = True

    sink = Foreign(clf.encoder.output_dim)
    got = list(clf.predict_batches(lone.batches(13)[:1], return_vectors=True, sink=sink))
    assert got[0].n_rows == 13
    assert sink.result().shape == (13, clf.encoder.output_dim)


def test_stores_of_unequal_length_need_ids(tmp_path, clf):
    with fse.ShardSink(tmp_path / "v", clf.encoder, shard_rows=8) as vectors:
        list(clf.encoder.encode_batches(lone.batches(10), sink=vectors))
    with (
        fse.ShardSink(tmp_path / "v", clf.encoder, shard_rows=8) as vectors,
        fse.ScoreSink(tmp_path / "s", clf, name="m", shard_rows=8) as scores,
    ):
        with pytest.raises(ValueError, match="position links the two stores"):
            clf.predict_batches(
                _poisoned(), return_vectors=True, vector_sink=vectors, score_sink=scores
            )
        assert vectors._reserved == 0
    with fse.ScoreStore.open(tmp_path / "s", heads=clf, name="m") as store:
        assert len(store) == 0  # the attach made one valid, empty store (scores.md)


def test_the_check_after_the_attach_leaves_one_valid_empty_score_store(tmp_path, clf):
    """The score store exists and holds rows; the vector store is new: the rows differ."""
    with fse.ScoreSink(tmp_path / "s", clf, name="m", shard_rows=8) as scores:
        list(clf.predict_batches(lone.batches(10), score_sink=scores))
    with (
        fse.ShardSink(tmp_path / "v", clf.encoder, shard_rows=8) as vectors,
        fse.ScoreSink(tmp_path / "s", clf, name="m", shard_rows=8) as scores,
        pytest.raises(ValueError, match="position links the two stores"),
    ):
        clf.predict_batches(
            _poisoned(), return_vectors=True, vector_sink=vectors, score_sink=scores
        )
    with fse.ScoreStore.open(tmp_path / "s", heads=clf, name="m") as store:
        assert len(store) == 10


def _poisoned():
    yield from ()
    raise AssertionError("a batch was read")
