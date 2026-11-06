"""The inputs of a stream of heads: a single head, document IDs, Arrow columns, lengths and
the durability of the stores."""

import numpy as np
import pytest

import fse
from fse import ScoreSink
from support import positions_heads as ph
from support import score_heads as sh
from support.positions import stream, stream_ids
from support.resume_heads import HEADS, SHARD_ROWS, score_sink, scores_of, vector_sink, vectors_of
from support.runtime import ArrowChunks

TEXTS = ph.TEXTS[:300]
TOXIC = HEADS["toxic"]
QUALITY = HEADS["quality"]


def _lone(path, model, name, **kwargs):
    return ScoreSink(path, model, name=name, shard_rows=SHARD_ROWS, resume=True, **kwargs)


def _stream_into(model, sink, batches, stop=None, **kwargs):
    results = []
    for found in model.predict_batches(batches, score_sink=sink, positions=True, **kwargs):
        results.append(found)
        if stop is not None and len(results) == stop:
            break
    return results


@pytest.mark.parametrize(("name", "model"), [("toxic", TOXIC), ("quality", QUALITY)])
def test_a_single_head_resumes_its_stream(tmp_path, name, model):
    with _lone(tmp_path / "whole", model, name) as sink:
        whole = _stream_into(model, sink, stream(TEXTS, 50))
    with _lone(tmp_path / "s", model, name) as sink:
        _stream_into(model, sink, stream(TEXTS, 50), stop=3)
    with _lone(tmp_path / "s", model, name) as sink:
        position, skip = sink.source_position
        assert (position, skip) == ("o100", 50)
        results = _stream_into(model, sink, stream(TEXTS, 41, start=100))
    assert np.array_equal(scores_of(tmp_path / "s")[0], scores_of(tmp_path / "whole")[0])
    assert (results[0].first_row, results[0].n_rows) == (41, 0)
    computed = np.concatenate([r.outputs for r in results])
    assert np.array_equal(computed, np.concatenate([r.outputs for r in whole])[150:])


def test_a_single_head_with_vectors_resumes_both_stores(tmp_path):
    def go(path, size, start=0, stop=None):
        path.mkdir(parents=True, exist_ok=True)
        with vector_sink(path / "v") as vectors, _lone(path / "s", QUALITY, "quality") as scores:
            return list(
                QUALITY.predict_batches(
                    stream(TEXTS, size, start),
                    vector_sink=vectors,
                    score_sink=scores,
                    return_vectors=True,
                    positions=True,
                )
            )[:stop]

    go(tmp_path / "whole", 50)
    go(tmp_path / "s", 50, stop=3)
    with _lone(tmp_path / "s" / "s", QUALITY, "quality") as sink:
        start = int(sink.source_position[0][1:])
    go(tmp_path / "s", 30, start)
    assert np.array_equal(vectors_of(tmp_path / "s" / "v"), vectors_of(tmp_path / "whole" / "v"))
    assert np.array_equal(
        scores_of(tmp_path / "s" / "s")[0], scores_of(tmp_path / "whole" / "s")[0]
    )


def test_the_batches_of_a_single_head_are_texts_and_positions(tmp_path):
    with _lone(tmp_path / "s", QUALITY, "quality") as sink:
        batches = QUALITY.predict_batches([TEXTS[:5]], score_sink=sink, positions=True)
        with pytest.raises(TypeError, match=r"batch 0 must be a \(texts, position\) tuple"):
            list(batches)
    with _lone(tmp_path / "t", QUALITY, "quality") as sink:
        batches = QUALITY.predict_batches([(TEXTS[:5], "")], score_sink=sink, positions=True)
        with pytest.raises(ValueError, match=r"batch 0.*1 to 65536 bytes"):
            list(batches)
    assert not (tmp_path / "t").exists()


# --- document IDs, Arrow, lengths -------------------------------------------------------------


def test_document_ids_follow_the_records_of_both_stores(tmp_path):
    ids = np.arange(300, dtype=np.int64) * 5 + 3
    texts = TEXTS

    def go(path, batches, stop=None):
        path.mkdir(parents=True, exist_ok=True)
        results = []
        with (
            vector_sink(path / "v", doc_ids=True) as vectors,
            score_sink(path / "s", doc_ids=True) as scores,
        ):
            for found in HEADS.predict_batches(
                batches,
                vector_sink=vectors,
                score_sink=scores,
                return_vectors=True,
                doc_ids=True,
                positions=True,
            ):
                results.append(found)
                if stop is not None and len(results) == stop:
                    break
        return results

    go(tmp_path / "whole", stream_ids(texts, ids, 60))
    go(tmp_path / "s", stream_ids(texts, ids, 60), stop=3)
    with vector_sink(tmp_path / "s" / "v", doc_ids=True) as sink:
        start = int(sink.source_position[0][1:])
    go(tmp_path / "s", stream_ids(texts, ids, 45, start))
    with fse.VectorStore.open(tmp_path / "s" / "v") as store:
        assert np.array_equal(store.get_doc_ids(), ids)
    assert np.array_equal(scores_of(tmp_path / "s" / "s")[1], ids)
    assert np.array_equal(vectors_of(tmp_path / "s" / "v"), vectors_of(tmp_path / "whole" / "v"))
    assert np.array_equal(
        scores_of(tmp_path / "s" / "s")[0], scores_of(tmp_path / "whole" / "s")[0]
    )


def test_arrow_batches_are_cut_where_the_stores_hold_the_records(tmp_path):
    def arrow(start, size):
        return [
            (ArrowChunks(TEXTS[row : row + size], b"vu", 40), f"o{row}")
            for row in range(start, 300, size)
        ]

    ph.run(tmp_path / "whole", stream(TEXTS, 50))
    ph.run(tmp_path / "s", arrow(0, 50), stop=3)
    with score_sink(tmp_path / "s" / "s") as sink:
        start = int(sink.source_position[0][1:])
    ph.run(tmp_path / "s", arrow(start, 70))
    got, want = ph.stores(tmp_path / "s"), ph.stores(tmp_path / "whole")
    assert all(np.array_equal(a, b) for a, b in zip(got[:2], want[:2], strict=True))


@pytest.mark.parametrize("level", ["summary", "per_row"])
def test_a_report_covers_the_records_computed(tmp_path, level):
    ph.run(tmp_path, stream(TEXTS, 50), stop=3)
    with score_sink(tmp_path / "s") as sink:
        start = int(sink.source_position[0][1:])
    results = ph.run(tmp_path, stream(TEXTS, 40, start), lengths=level)
    assert [r.lengths.n_rows for r in results] == [r.n_rows for r in results]
    assert results[0].n_rows == 0
    assert results[0].lengths.first_row == results[0].first_row


# --- durability -------------------------------------------------------------------------------


def test_nothing_is_durable_without_durability(tmp_path):
    ph.run(tmp_path, stream(TEXTS, 50), durability="none", stop=3)
    results, _ = ph.resume(tmp_path, 50, durability="none", texts=TEXTS)
    assert {r.durable_rows for r in results} == {0}


def test_one_store_without_durability_confirms_nothing(tmp_path):
    ph.run(tmp_path, stream(TEXTS, 50), stop=3)
    with (
        vector_sink(tmp_path / "v") as vectors,
        score_sink(tmp_path / "s", durability="none") as scores,
    ):
        start = int(vectors.source_position[0][1:])
        results = list(
            HEADS.predict_batches(
                stream(TEXTS, 50, start),
                vector_sink=vectors,
                score_sink=scores,
                return_vectors=True,
                positions=True,
            )
        )
    assert {r.durable_rows for r in results} == {0}


def test_the_stream_may_end_without_a_batch(tmp_path):
    with score_sink(tmp_path / "s") as sink:
        assert list(HEADS.predict_batches(iter([]), score_sink=sink, positions=True)) == []
    assert not (tmp_path / "s").exists()


# --- the root of a score store -----------------------------------------------------------------


def _big(labels):
    names = [f"{i:05d}" + "x" * 95 for i in range(labels)]
    return sh.classifier(sh.encoder(), labels, labels=names)


def test_a_resuming_sink_leaves_room_for_the_largest_position(tmp_path):
    model = _big(9500)  # the heads fit alone, but not with a position of 64 KiB
    with ScoreSink(tmp_path / "plain", model, name="big") as sink:
        assert sink.durability == "full"
    with pytest.raises(ValueError, match=r"position of 65536 bytes need a store\.json of"):
        ScoreSink(tmp_path / "s", model, name="big", resume=True)
    assert not (tmp_path / "s").exists()


def test_a_position_that_does_not_fit_the_heads_is_refused_before_the_store_is_made(tmp_path):
    model = _big(9000)
    with ScoreSink(tmp_path / "s", model, name="big", resume=True) as sink:
        batches = model.predict_batches([(TEXTS[:5], "é" * 32768)], score_sink=sink, positions=True)
        with pytest.raises(ValueError, match="the root of the store needs"):
            list(batches)
    assert not (tmp_path / "s").exists()
