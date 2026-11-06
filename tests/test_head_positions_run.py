"""predict_batches(positions=True): arguments, modes and the rows of a resumed stream into the
stores of vectors and of scores."""

import numpy as np
import pytest
from pydantic import ValidationError

import fse
from fse import ScoreSink, ShardSink
from fse._store._layout import read_store
from support import positions_heads as ph
from support.resume_heads import ENCODER, HEADS, score_sink, vector_sink

TEXTS = ph.TEXTS
BATCH = 50


def _quality():
    return HEADS["quality"]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """An uninterrupted stream into both stores, and the outputs of one batch of all texts."""
    directory = tmp_path_factory.mktemp("whole")
    results = ph.run(directory, ph.stream(TEXTS, 100))
    # with vectors asked for, the heads on raw means take the unfolded path: predict_batches
    (whole,) = HEADS.predict_batches([TEXTS], return_vectors=True)
    return ph.stores(directory), results, whole.outputs


# --- arguments ---------------------------------------------------------------------------


@pytest.mark.parametrize("positions", [1, "yes", None])
def test_positions_is_a_bool(positions):
    with pytest.raises(ValidationError, match="positions"):
        HEADS.predict_batches([], positions=positions)
    with pytest.raises(ValidationError, match="positions"):
        _quality().predict_batches([], positions=positions)


def test_positions_need_a_sink():
    with pytest.raises(ValueError, match="positions=True needs a vector_sink or a score_sink"):
        HEADS.predict_batches([], positions=True)
    with pytest.raises(ValueError, match="positions=True needs a vector_sink or a score_sink"):
        _quality().predict_batches([], positions=True)


def test_the_sinks_must_resume(tmp_path):
    with (
        ScoreSink(tmp_path / "s", HEADS) as sink,
        pytest.raises(ValueError, match="score_sink to be made with resume=True"),
    ):
        HEADS.predict_batches([], score_sink=sink, positions=True)
    with (
        ShardSink(tmp_path / "v", ENCODER) as sink,
        pytest.raises(ValueError, match="vector_sink to be made with resume=True"),
    ):
        HEADS.predict_batches([], vector_sink=sink, return_vectors=True, positions=True)
    assert not (tmp_path / "s").exists()


def test_two_sinks_resume_both_or_neither(tmp_path):
    with (
        vector_sink(tmp_path / "v") as vectors,
        ScoreSink(tmp_path / "s", HEADS) as scores,
        pytest.raises(ValueError, match="score_sink to be made with resume=True"),
    ):
        HEADS.predict_batches(
            [], vector_sink=vectors, score_sink=scores, return_vectors=True, positions=True
        )


def test_a_resumable_sink_takes_positions_only(tmp_path):
    with score_sink(tmp_path / "s") as sink, pytest.raises(ValueError, match="resume=True"):
        HEADS.predict_batches([], score_sink=sink)
    with vector_sink(tmp_path / "v") as sink, pytest.raises(ValueError, match="resume=True"):
        HEADS.predict_batches([], vector_sink=sink, return_vectors=True)


def test_sink_keeps_its_role(tmp_path):
    with (
        ShardSink(tmp_path / "s", ENCODER) as sink,
        pytest.raises(ValueError, match="sink= keeps its role"),
    ):
        _quality().predict_batches([], sink=sink, positions=True)


def test_vectors_need_the_path_that_gives_them(tmp_path):
    with (
        vector_sink(tmp_path / "v") as sink,
        pytest.raises(ValueError, match="vector_sink takes the vectors of return_vectors=True"),
    ):
        HEADS.predict_batches([], vector_sink=sink, positions=True)


def test_two_stores_need_two_folders(tmp_path):
    with (
        vector_sink(tmp_path / "v") as vectors,
        score_sink(tmp_path / "v" / "s") as scores,
        pytest.raises(ValueError, match="two folders apart"),
    ):
        HEADS.predict_batches(
            [], vector_sink=vectors, score_sink=scores, return_vectors=True, positions=True
        )


# --- modes ----------------------------------------------------------------------------------


def test_a_stream_does_not_resume_a_finite_source(tmp_path):
    ph.run(tmp_path / "w", ph.stream(TEXTS, BATCH), vectors=False, stop=2)
    with fse.ScoreStore.open(tmp_path / "w" / "s") as store:
        assert store.source_position is not None
    sink = score_sink(tmp_path / "w" / "s")
    with sink, pytest.raises(ValueError, match="mode source_position@1"):
        HEADS.predict(TEXTS, score_sink=sink)


def test_a_finite_run_does_not_resume_a_stream(tmp_path):
    with score_sink(tmp_path / "s") as sink:
        HEADS.predict(TEXTS, score_sink=sink)
    with (
        score_sink(tmp_path / "s") as sink,
        pytest.raises(ValueError, match="mode source_rows@1"),
    ):
        list(HEADS.predict_batches([(TEXTS, "o0")], score_sink=sink, positions=True))
    assert read_store(tmp_path / "s").resume.mode == "source_rows@1"


# --- the rows -------------------------------------------------------------------------------


def test_a_stream_gives_the_rows_of_one_batch(reference):
    (vectors, scores, _), results, expected = reference
    assert vectors.shape[0] == scores.shape[0] == len(TEXTS)
    for name in HEADS:
        assert np.array_equal(ph.gathered(results, name), expected[name])
    assert [r.first_row for r in results] == [0, 100, 200, 300]


@pytest.mark.parametrize("size", [1, 29, 50, 64, 300])
@pytest.mark.parametrize("by", ["v", "s"])
def test_a_resumed_stream_gives_the_rows_of_one_run(tmp_path, reference, size, by):
    (vectors, scores, _), _, expected = reference
    ph.run(tmp_path, ph.stream(TEXTS, 100), stop=3)
    results, _ = ph.resume(tmp_path, size, by=by)
    got = ph.stores(tmp_path)
    assert np.array_equal(got[0], vectors)
    assert np.array_equal(got[1], scores)
    # what each batch gave is what predict gives for those records
    skipped = results[0].first_row if results[0].n_rows else 0
    assert skipped >= 0
    for name in HEADS:
        rows = ph.gathered(results, name)
        assert np.array_equal(rows, expected[name][len(expected[name]) - len(rows) :])
