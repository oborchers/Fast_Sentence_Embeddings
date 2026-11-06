"""Resuming a run of heads: arguments, modes, the identity of a source and the unit parts of
a resumed run (the rows of two stores of different lengths are in test_head_resume_run_phases)."""

import numpy as np
import pytest

import fse
from fse import ScoreSink, ShardSink
from fse._head._resume_run import _Discard, _Shifted
from fse._store._layout import read_store
from support import score_heads as sh
from support.resume_heads import (
    ENCODER,
    HEADS,
    SHARD_ROWS,
    Poisoned,
    run,
    score_sink,
    sentences,
    vector_sink,
)

SENTENCES = sentences(320)


def _quality(heads=HEADS):
    return heads["quality"]


# --- arguments: raised before the source is read -------------------------------------------


@pytest.mark.parametrize("shard", [(0,), [0, 1], (True, 2), (0.0, 2), "01", (0, 1, 2), 3])
def test_a_shard_must_be_a_pair_of_ints(tmp_path, shard):
    with (
        score_sink(tmp_path / "s") as sink,
        pytest.raises(TypeError, match="shard must be a tuple"),
    ):
        HEADS.predict(Poisoned(), score_sink=sink, shard=shard)
    assert not (tmp_path / "s").exists()


@pytest.mark.parametrize("shard", [(0, 0), (2, 2), (-1, 2), (0, -1)])
def test_a_shard_must_name_a_part(tmp_path, shard):
    with score_sink(tmp_path / "s") as sink, pytest.raises(ValueError, match="shard must be"):
        HEADS.predict(Poisoned(), score_sink=sink, shard=shard)


@pytest.mark.parametrize(
    "keywords",
    [{"shard": (0, 1)}, {"return_vectors": True}, {"doc_ids": [1, 2]}],
)
def test_the_resume_keywords_need_a_sink(keywords):
    with pytest.raises(ValueError, match="need a vector_sink or a score_sink made with resume"):
        HEADS.predict(Poisoned(), **keywords)


def test_a_sink_without_resume_is_refused(tmp_path):
    with (
        ScoreSink(tmp_path / "s", HEADS) as sink,
        pytest.raises(ValueError, match="score_sink must be made with resume=True"),
    ):
        HEADS.predict(Poisoned(), score_sink=sink)
    with (
        ShardSink(tmp_path / "v", ENCODER) as sink,
        pytest.raises(ValueError, match="vector_sink must be made with resume=True"),
    ):
        HEADS.predict(Poisoned(), vector_sink=sink, return_vectors=True)
    assert not (tmp_path / "s").exists()  # a sink made without resume made its store itself
    assert not any((tmp_path / "v").glob("shard-*"))


def test_two_sinks_resume_both_or_neither(tmp_path):
    with (
        vector_sink(tmp_path / "v") as vectors,
        ScoreSink(tmp_path / "s", HEADS) as scores,
        pytest.raises(ValueError, match="score_sink must be made with resume=True"),
    ):
        HEADS.predict(Poisoned(), vector_sink=vectors, score_sink=scores, return_vectors=True)
    assert not any((tmp_path / "v").glob("*"))


def test_vectors_need_the_path_that_gives_them(tmp_path):
    with (
        vector_sink(tmp_path / "v") as sink,
        pytest.raises(ValueError, match="vector_sink takes the vectors of return_vectors=True"),
    ):
        HEADS.predict(Poisoned(), vector_sink=sink)


@pytest.mark.parametrize("kind", ["generator", "iterator"])
def test_a_source_that_cannot_be_read_again_is_refused(tmp_path, kind):
    source = (t for t in SENTENCES) if kind == "generator" else iter(SENTENCES)
    with score_sink(tmp_path / "s") as sink, pytest.raises(TypeError, match=r"read again|stream"):
        HEADS.predict(source, score_sink=sink)
    assert not (tmp_path / "s").exists()


def test_the_sinks_need_two_folders_apart(tmp_path):
    with (
        vector_sink(tmp_path / "v") as vectors,
        score_sink(tmp_path / "v" / "inner") as scores,
        pytest.raises(ValueError, match="two folders apart"),
    ):
        HEADS.predict(Poisoned(), vector_sink=vectors, score_sink=scores, return_vectors=True)


def test_ids_must_come_with_sinks_that_store_them(tmp_path):
    with (
        score_sink(tmp_path / "s") as sink,
        pytest.raises(ValueError, match="doc_ids must be supplied exactly"),
    ):
        HEADS.predict(Poisoned(), score_sink=sink, doc_ids=[1, 2, 3])
    with score_sink(tmp_path / "t", doc_ids=True) as sink:
        with pytest.raises(ValueError, match="doc_ids must be supplied exactly"):
            HEADS.predict(Poisoned(), score_sink=sink)
        with pytest.raises(TypeError, match="not a bool"):
            HEADS.predict(Poisoned(), score_sink=sink, doc_ids=True)


def test_the_sink_must_be_made_for_the_heads(tmp_path):
    other = fse.Heads(quality=_quality(), topic=HEADS["topic"])
    with (
        score_sink(tmp_path / "s", heads=other) as sink,
        pytest.raises(ValueError, match="the sink was made for the heads"),
    ):
        HEADS.predict(Poisoned(), score_sink=sink)


def test_bad_workers_and_a_closed_sink_are_refused_before_the_store_is_made(tmp_path):
    with score_sink(tmp_path / "s") as sink, pytest.raises(ValueError, match="workers"):
        HEADS.predict(SENTENCES, score_sink=sink, workers=0)
    with pytest.raises(RuntimeError, match="closed"):
        HEADS.predict(SENTENCES, score_sink=sink)
    assert not (tmp_path / "s").exists()


def test_a_sink_that_resumes_takes_no_predict_batches(tmp_path):
    with (
        score_sink(tmp_path / "s") as scores,
        pytest.raises(ValueError, match="resume=True takes the rows of predict"),
    ):
        HEADS.predict_batches([SENTENCES], score_sink=scores)
    with (
        vector_sink(tmp_path / "v") as vectors,
        pytest.raises(ValueError, match="resume=True takes the rows of"),
    ):
        HEADS.predict_batches([SENTENCES], return_vectors=True, vector_sink=vectors)
    assert not (tmp_path / "s").exists()
    assert not (tmp_path / "v").exists()


def test_a_sink_takes_one_source(tmp_path):
    with score_sink(tmp_path / "s") as sink:
        HEADS.predict(SENTENCES, score_sink=sink)
        with pytest.raises(RuntimeError, match="took a source already"):
            HEADS.predict(SENTENCES, score_sink=sink)


# --- the entries of one head ------------------------------------------------------------


def test_a_regressor_takes_sink_or_the_new_keywords(tmp_path):
    model = HEADS["quality"]
    with (
        ScoreSink(tmp_path / "s", model, name="quality", resume=True) as sink,
        pytest.raises(ValueError, match="sink= keeps its role"),
    ):
        model.predict(Poisoned(), sink=object(), score_sink=sink)


def test_a_single_head_resumes_into_a_sink_of_one_head(tmp_path):
    for name, model in (("topic", HEADS["topic"]), ("quality", HEADS["quality"])):
        path = tmp_path / name
        with ScoreSink(path, model, name=name, shard_rows=SHARD_ROWS, resume=True) as sink:
            assert model.predict(SENTENCES, score_sink=sink, workers=2) is None
        with ScoreSink(path, model, name=name, shard_rows=SHARD_ROWS, resume=True) as sink:
            assert model.predict(SENTENCES, score_sink=sink) is None  # nothing left to run
        with fse.ScoreStore.open(path) as store:
            store.verify()
            assert len(store) == len(SENTENCES)


def test_a_single_head_refuses_a_sink_of_other_heads(tmp_path):
    with (
        score_sink(tmp_path / "s") as sink,
        pytest.raises(ValueError, match="a single head writes a sink of one head"),
    ):
        HEADS["topic"].predict(Poisoned(), score_sink=sink)


def test_a_single_head_writes_vectors_too(tmp_path):
    model = HEADS["topic"]
    with (
        vector_sink(tmp_path / "v") as vectors,
        ScoreSink(
            tmp_path / "s", model, name="topic", shard_rows=SHARD_ROWS, resume=True
        ) as scores,
    ):
        model.predict(
            SENTENCES, return_vectors=True, vector_sink=vectors, score_sink=scores, workers=2
        )
    want = ENCODER.encode(SENTENCES)
    got = fse.VectorStore.open(tmp_path / "v")
    np.testing.assert_array_equal(got.get_vectors().view(np.uint32), want.view(np.uint32))
    got.close()


# --- modes ----------------------------------------------------------------------------------


def test_a_resumable_score_store_is_opened_with_resume(tmp_path):
    run(tmp_path, SENTENCES, vectors=False)
    with pytest.raises(ValueError, match=r"is resumable \(mode source_rows@1\); open it with"):
        ScoreSink(tmp_path / "s", HEADS)


def test_a_plain_score_store_is_no_resumable_store(tmp_path):
    with ScoreSink(tmp_path / "s", HEADS, shard_rows=SHARD_ROWS) as sink:
        list(HEADS.predict_batches([SENTENCES], score_sink=sink))
    with pytest.raises(ValueError, match="is not a resumable store"):
        score_sink(tmp_path / "s")


def test_a_new_resumable_store_is_made_when_a_source_comes(tmp_path):
    with score_sink(tmp_path / "s"):
        pass
    assert not (tmp_path / "s").exists()
    run(tmp_path, SENTENCES, vectors=False)
    assert read_store(tmp_path / "s").resume is not None


def test_the_paths_of_the_heads_are_part_of_the_store(tmp_path):
    run(tmp_path, SENTENCES, vectors=True)
    with (
        score_sink(tmp_path / "s") as sink,
        pytest.raises(ValueError, match="a head has the same bits only on the same path"),
    ):
        HEADS.predict(SENTENCES, score_sink=sink)


# --- the identity of the source -------------------------------------------------------------


def test_another_source_or_part_is_refused_before_any_row(tmp_path):
    run(tmp_path, SENTENCES[:100], vectors=False)
    before = sorted(p.name for p in (tmp_path / "s").iterdir())
    changed = [*SENTENCES[:100]]
    changed[40] = changed[40] + " extra"
    for source, shard in ((changed, None), (SENTENCES[:99], None), (SENTENCES[:100], (1, 2))):
        with (
            score_sink(tmp_path / "s") as sink,
            pytest.raises(ValueError, match="another source or partition"),
        ):
            HEADS.predict(source, score_sink=sink, shard=shard)
    assert sorted(p.name for p in (tmp_path / "s").iterdir()) == before


def test_a_mismatch_of_one_store_makes_no_store_for_the_other(tmp_path):
    run(tmp_path, SENTENCES[:100], vectors=True)
    other = tmp_path / "other"
    other.mkdir()
    with (
        vector_sink(tmp_path / "v") as vectors,
        score_sink(other / "s") as scores,
        pytest.raises(ValueError, match="another source or partition"),
    ):
        HEADS.predict(SENTENCES[:99], return_vectors=True, vector_sink=vectors, score_sink=scores)
    assert not (other / "s").exists()  # the new store was not made


def test_other_heads_are_refused(tmp_path):
    run(tmp_path, SENTENCES, vectors=False)
    enc = sh.encoder()
    other = fse.Heads(
        quality=sh.regressor(enc, seed=1),
        toxic=HEADS["toxic"],
        topic=HEADS["topic"],
    )
    assert other["quality"].fingerprint != HEADS["quality"].fingerprint
    with pytest.raises(ValueError, match="holds head 'quality' with model"):
        score_sink(tmp_path / "s", heads=other)


# --- Discard and Shifted --------------------------------------------------------------------


def test_a_discarding_sink_keeps_nothing():
    sink = _Discard(5)
    first = sink.target(0, 7)
    assert first.shape == (7, 5)
    assert first.dtype == np.float32
    sink.commit(0, first)
    sink.commit(7, first, doc_ids=np.arange(7))
    assert sink.target(7, 3) is not first
    assert (sink.dtype, sink.inverse_norms, sink.dim) == ("float32", False, 5)


class _Recorder:
    dim = 3
    marker = "kept"

    def __init__(self):
        self.calls = []

    def target(self, first_row, n_rows):
        self.calls.append(("target", first_row, n_rows))
        return np.zeros((n_rows, 3), np.float32)

    def commit(self, first_row, rows, **keywords):
        self.calls.append(("commit", first_row, len(rows), keywords))


def test_a_shifted_sink_moves_the_rows_and_keeps_the_rest():
    inner = _Recorder()
    sink = _Shifted(inner, -10)
    rows = sink.target(14, 4)
    sink.commit(14, rows)
    sink.commit(18, rows, doc_ids="ids")
    assert inner.calls == [
        ("target", 4, 4),
        ("commit", 4, 4, {}),
        ("commit", 8, 4, {"doc_ids": "ids"}),
    ]
    assert sink.marker == "kept"
    assert sink.dim == 3


# --- findings of the review: wrong values and used sinks ----------------------------------


@pytest.mark.parametrize(
    ("keyword", "error"),
    [
        ("shard", TypeError),
        ("score_sink", ValueError),
        ("vector_sink", ValueError),
        ("doc_ids", ValueError),
    ],
)
@pytest.mark.parametrize("entry", ["heads", "classifier", "regressor"])
def test_a_false_value_of_a_new_keyword_is_checked_not_ignored(entry, keyword, error):
    model = {"heads": HEADS, "classifier": HEADS["topic"], "regressor": HEADS["quality"]}[entry]
    with pytest.raises(error):
        model.predict(Poisoned(), **{keyword: False})


def test_invalid_document_ids_are_refused_before_a_text_is_read(tmp_path):
    with (
        score_sink(tmp_path / "s", doc_ids=True) as sink,
        pytest.raises((TypeError, ValueError)),
    ):
        HEADS.predict(Poisoned(), score_sink=sink, doc_ids=np.array([1.5, 2.5]))
    assert not (tmp_path / "s").exists()


def test_a_sink_that_took_a_source_is_refused_before_a_text_is_read(tmp_path):
    with score_sink(tmp_path / "s") as used:
        HEADS.predict(SENTENCES, score_sink=used)
    partner = tmp_path / "partner"
    with vector_sink(partner) as vectors, score_sink(tmp_path / "t") as fresh:
        fresh_used = score_sink(tmp_path / "u")
        HEADS.predict(SENTENCES, score_sink=fresh_used)
        with pytest.raises(RuntimeError, match="took a source already"):
            HEADS.predict(
                Poisoned(), return_vectors=True, vector_sink=vectors, score_sink=fresh_used
            )
        fresh_used.close()
        del fresh
    assert not partner.exists()  # the new partner store was not made


def test_a_store_that_would_outgrow_its_root_is_refused_before_it_is_made(tmp_path, monkeypatch):
    from fse._scores import _schema
    from fse._store._layout import root_text

    with score_sink(tmp_path / "s") as sink:
        schema = sink.score_schema.with_paths({n: "folded" for n in HEADS})
        plain = len(root_text(schema.info(False))) + 1
        monkeypatch.setattr(_schema, "SCORE_STORE_BYTES", plain + 20)  # fits without the keys
        with pytest.raises(ValueError, match=r"store\.json of"):
            HEADS.predict(SENTENCES, score_sink=sink)
    assert not (tmp_path / "s").exists()
