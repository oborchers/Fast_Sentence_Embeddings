"""A resumed run of heads into a vector store and a score store that hold different numbers
of rows: the rows are those of an uninterrupted run, bit for bit."""

import numpy as np
import pytest

import fse
from fse._head import _resume_run
from fse._head._resume_run import _Discard
from fse._runtime import HeadsPipeline
from fse._store._resume_format import MODE
from support import score_heads as sh
from support import sink_run as sr
from support.logs import captured
from support.resume import canonical, drop_tail
from support.resume_heads import (
    HEADS,
    SHARD_ROWS,
    cut,
    run,
    score_sink,
    scores_of,
    sentences,
    stores,
    vector_sink,
    vectors_of,
)

SENTENCES = sentences(640)  # ten shards of 64 rows


@pytest.fixture(scope="module")
def whole(tmp_path_factory):
    path = tmp_path_factory.mktemp("whole")
    assert run(path, SENTENCES) is None
    return path


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """The two stores of predict_batches without resume: the bits fse gives a run today."""
    path = tmp_path_factory.mktemp("reference")
    _, vectors, _, scores, _ = sr.run(path, HEADS, SENTENCES, shard_rows=SHARD_ROWS, workers=2)
    return vectors, scores


def test_the_stores_of_a_resumed_run_hold_the_bits_of_predict_batches(whole, reference):
    vectors, scores = reference
    np.testing.assert_array_equal(vectors_of(whole / "v").view(np.uint32), vectors.view(np.uint32))
    np.testing.assert_array_equal(scores_of(whole / "s")[0].view(np.uint32), scores.view(np.uint32))


@pytest.mark.parametrize(
    ("vector_shards", "score_shards"),
    [(6, 3), (3, 6), (0, 4), (4, 0), (10, 5), (5, 10), (1, 9), (9, 1), (5, 5), (0, 0), (10, 10)],
)
def test_two_stores_of_any_lengths_continue_with_the_bytes_of_one_run(
    tmp_path, whole, vector_shards, score_shards
):
    run(tmp_path, SENTENCES)
    cut(tmp_path, vector_shards, score_shards)
    run(tmp_path, SENTENCES, workers=3)  # other workers: the same rows
    assert stores(tmp_path) == stores(whole)


def test_a_complete_pair_of_stores_is_left_alone(tmp_path, whole):
    run(tmp_path, SENTENCES)
    before = stores(tmp_path)
    assert run(tmp_path, SENTENCES) is None
    assert stores(tmp_path) == before == stores(whole)


@pytest.mark.parametrize("name", ["v", "s"])
def test_one_store_alone_resumes(tmp_path, name):
    only = {"vectors": name == "v", "scores": name == "s"}
    run(tmp_path / "whole", SENTENCES, **only)  # scores alone take the folded path
    run(tmp_path / "cut", SENTENCES, **only)
    drop_tail(tmp_path / "cut" / name, 4)
    run(tmp_path / "cut", SENTENCES, **only)
    assert canonical(tmp_path / "cut" / name) == canonical(tmp_path / "whole" / name)


def test_the_stores_may_have_other_grids(tmp_path):
    """Shards of 32 rows in one store and of 64 in the other: the prefixes differ in rows."""

    def go(directory):
        with (
            vector_sink(directory / "v", rows=64) as vectors,
            score_sink(directory / "s", rows=32) as scores,
        ):
            HEADS.predict(
                SENTENCES, return_vectors=True, vector_sink=vectors, score_sink=scores, workers=2
            )

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    go(tmp_path / "a")
    go(tmp_path / "b")
    drop_tail(tmp_path / "b" / "v", 3, rows=64)  # 192 rows
    drop_tail(tmp_path / "b" / "s", 9, rows=32)  # 288 rows
    go(tmp_path / "b")
    assert stores(tmp_path / "a") == stores(tmp_path / "b")


def test_stores_of_other_durability_resume(tmp_path):
    """One store under full durability and one under none: the rows are the same, the
    shards of the second carry the tag of an unsynced publication."""

    def go(directory):
        with (
            vector_sink(directory / "v", durability="none") as vectors,
            score_sink(directory / "s") as scores,
        ):
            HEADS.predict(
                SENTENCES, return_vectors=True, vector_sink=vectors, score_sink=scores, workers=2
            )

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    go(tmp_path / "a")
    go(tmp_path / "b")
    drop_tail(tmp_path / "b" / "v", 7)
    drop_tail(tmp_path / "b" / "s", 4)
    go(tmp_path / "b")
    np.testing.assert_array_equal(
        vectors_of(tmp_path / "a" / "v").view(np.uint32),
        vectors_of(tmp_path / "b" / "v").view(np.uint32),
    )
    np.testing.assert_array_equal(
        scores_of(tmp_path / "a" / "s")[0].view(np.uint32),
        scores_of(tmp_path / "b" / "s")[0].view(np.uint32),
    )


def test_scores_with_vectors_dropped_take_the_path_with_vectors(tmp_path):
    """A score store written with return_vectors=True and no vector store: the vectors are
    computed and dropped, so the heads take the path the store was written on."""
    run(tmp_path, SENTENCES, vectors=False, return_vectors=True)
    with pytest.raises(ValueError, match="a head has the same bits only on the same path"):
        run(tmp_path, SENTENCES, vectors=False)
    run(tmp_path, SENTENCES, vectors=False, return_vectors=True)  # complete: left alone


# --- the part of a source -----------------------------------------------------------------


def test_the_parts_of_a_source_give_the_rows_of_the_whole(tmp_path):
    big = sentences(33_000)  # three blocks of 16,384 rows: the parts are [0, 16384) and the rest
    run(tmp_path / "whole", big, rows=4096)
    for index in (0, 1):
        run(tmp_path / f"p{index}", big, shard=(index, 2), rows=4096)
        cut(tmp_path / f"p{index}", 2, 1, rows=4096)  # a stop inside the part
        run(tmp_path / f"p{index}", big, shard=(index, 2), rows=4096)
    joined = np.concatenate([scores_of(tmp_path / f"p{i}" / "s")[0] for i in (0, 1)])
    np.testing.assert_array_equal(
        joined.view(np.uint32), scores_of(tmp_path / "whole" / "s")[0].view(np.uint32)
    )
    vectors = np.concatenate([vectors_of(tmp_path / f"p{i}" / "v") for i in (0, 1)])
    np.testing.assert_array_equal(
        vectors.view(np.uint32), vectors_of(tmp_path / "whole" / "v").view(np.uint32)
    )
    assert len(joined) == len(big)


# --- document IDs ---------------------------------------------------------------------------


def test_document_ids_follow_the_rows_of_both_phases(tmp_path):
    ids = np.arange(len(SENTENCES), dtype=np.int64) * 3 + 7
    run(tmp_path, SENTENCES, doc_ids=ids)
    cut(tmp_path, 7, 2)
    run(tmp_path, SENTENCES, doc_ids=ids)
    with fse.VectorStore.open(tmp_path / "v") as store:
        np.testing.assert_array_equal(store.get_doc_ids(), ids)
    _, score_ids = scores_of(tmp_path / "s")
    np.testing.assert_array_equal(score_ids, ids)


def test_other_document_ids_are_another_source(tmp_path):
    ids = np.arange(len(SENTENCES), dtype=np.int64)
    run(tmp_path, SENTENCES, doc_ids=ids)
    cut(tmp_path, 3, 3)
    with pytest.raises(ValueError, match="another source or partition"):
        run(tmp_path, SENTENCES, doc_ids=ids + 1)


# --- lengths --------------------------------------------------------------------------------


@pytest.mark.parametrize(("vector_shards", "score_shards"), [(6, 3), (3, 6), (4, 4)])
def test_one_report_covers_the_rows_of_both_phases(tmp_path, vector_shards, score_shards):
    run(tmp_path, SENTENCES)
    cut(tmp_path, vector_shards, score_shards)
    start = min(vector_shards, score_shards) * SHARD_ROWS
    result, report = run(tmp_path, SENTENCES, lengths="per_row")
    assert result is None
    assert report.first_row == start
    assert report.n_rows == len(SENTENCES) - start
    assert len(report.tokens) == report.n_rows


def test_the_report_of_a_complete_pair_is_empty(tmp_path):
    run(tmp_path, SENTENCES)
    _, report = run(tmp_path, SENTENCES, lengths="summary")
    assert report.n_rows == 0
    assert report.first_row == len(SENTENCES)


def test_an_error_of_the_run_names_the_row_of_the_source(tmp_path, monkeypatch):
    run(tmp_path, SENTENCES)
    cut(tmp_path, 6, 3)

    def broken(*_args, **_kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(_resume_run.SinkRun, "infer", broken)
    with pytest.raises(RuntimeError, match="stop") as caught:
        run(tmp_path, SENTENCES)
    assert any("resumed at row 192 of the source" in n for n in caught.value.__notes__)
    monkeypatch.undo()
    run(tmp_path, SENTENCES)  # the prefix the failed run left is valid


# --- logging --------------------------------------------------------------------------------


def test_the_records_of_a_resumed_run_name_what_was_kept(tmp_path):
    run(tmp_path, SENTENCES)
    cut(tmp_path, 6, 3)
    with captured() as records:
        run(tmp_path, SENTENCES)
    opened = next(r for r in records if r.fse_event == "scores.sink_opened")
    assert opened.fse_resume == MODE
    assert opened.fse_kept_rows == opened.fse_rows == 3 * SHARD_ROWS
    assert opened.fse_source_rows == len(SENTENCES)
    assert opened.fse_source_fingerprint.startswith("xxh3-128:")
    closed = next(r for r in records if r.fse_event == "scores.sink_closed")
    assert closed.fse_skipped_rows == 3 * SHARD_ROWS
    assert all("/" not in r.getMessage() for r in records if r.fse_event.startswith("scores.sink"))


def test_a_plain_run_logs_no_resume_fields(tmp_path):
    with (
        captured() as records,
        fse.ScoreSink(tmp_path / "s", HEADS, shard_rows=SHARD_ROWS) as sink,
    ):
        list(HEADS.predict_batches([SENTENCES[:100]], score_sink=sink))
    for record in records:
        assert not hasattr(record, "fse_resume")
        assert not hasattr(record, "fse_skipped_rows")


# --- the outputs of the heads are not held ----------------------------------------------------


def test_the_main_outputs_and_the_dropped_vectors_go_to_sinks_that_keep_nothing(
    tmp_path, monkeypatch
):
    """Every phase gives the runtime a discarding sink for the output of each head (and for
    the vectors that are asked for but no store takes), so no result array is made."""
    seen = []
    real = HeadsPipeline.infer

    def spy(self, *args, **kwargs):
        arrays, vectors = real(self, *args, **kwargs)
        seen.append((kwargs["sinks"], kwargs.get("vector_sink"), arrays, vectors))
        return arrays, vectors

    monkeypatch.setattr(HeadsPipeline, "infer", spy)
    run(tmp_path, SENTENCES, vectors=False, return_vectors=True)
    assert len(seen) == 1
    sinks, vector_sink, arrays, vectors = seen[0]
    assert len(sinks) == len(HEADS)
    assert all(isinstance(s, _Discard) for s in sinks)
    assert isinstance(vector_sink, _Discard)
    assert arrays == [None] * len(HEADS)  # nothing was made for the result
    assert vectors is None


# --- stored probabilities, normalized heads --------------------------------------------------


@pytest.mark.parametrize("vectors", [False, True])
def test_stored_probabilities_of_normalized_and_binary_heads_keep_the_bits(tmp_path, vectors):
    """The kind the runtime computes follows what a head stores: a resumed run, cut in both
    stores, gives the scores of predict_batches into one score store."""
    enc = sh.encoder()
    heads = fse.Heads(
        topic=sh.classifier(enc, 3, seed=4, normalized=True), flag=sh.classifier(enc, 1, seed=5)
    )
    with fse.ScoreSink(tmp_path / "ref", heads, stored="proba", shard_rows=SHARD_ROWS) as sink:
        list(heads.predict_batches([SENTENCES], return_vectors=vectors, score_sink=sink, workers=2))
    want = scores_of(tmp_path / "ref")[0]
    run(
        tmp_path / "r",
        SENTENCES,
        heads=heads,
        vectors=vectors,
        return_vectors=vectors,
        stored="proba",
    )
    if vectors:
        cut(tmp_path / "r", 6, 3)
    else:
        drop_tail(tmp_path / "r" / "s", 3)
    run(
        tmp_path / "r",
        SENTENCES,
        heads=heads,
        vectors=vectors,
        return_vectors=vectors,
        stored="proba",
        workers=3,
    )
    got = scores_of(tmp_path / "r" / "s")[0]
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
