"""The vectors and the scores of a run into two stores are the bits of the two separate runs."""

import numpy as np
import pytest

import fse
from support import score_heads as sh
from support import sink_run as sr
from support.encoder import finished


def _heads(dtype="float32", normalize=False, head_normalized=False):
    enc = finished(dtype, fasttext=True, normalize=normalize)
    toxic = sh.classifier(
        enc,
        3,
        link="multilabel",
        labels=["a", "b", "c"],
        thresholds=[0.3, 0.5, 0.7],
        normalized=head_normalized,
    )
    topic = sh.classifier(enc, 3, seed=4, normalized=head_normalized)
    return fse.Heads(quality=sh.regressor(enc), toxic=toxic, topic=topic)


def _separate(tmp_path, heads, texts, ids, *, batch, workers, shard_rows, dtype, store=None):
    """The vector store of encode_batches and the score store of a run without a vector sink."""
    batches = sr.batches_of(texts, batch, ids)
    with fse.ShardSink(
        tmp_path / "rv",
        heads["quality"].encoder,
        dtype=dtype,
        shard_rows=shard_rows,
        doc_ids=ids is not None,
    ) as sink:
        list(
            heads["quality"].encoder.encode_batches(
                batches, sink=sink, workers=workers, doc_ids=ids is not None
            )
        )
    with fse.ScoreSink(
        tmp_path / "rs", heads, store=store, shard_rows=shard_rows, doc_ids=ids is not None
    ) as sink:
        list(
            heads.predict_batches(
                batches,
                score_sink=sink,
                return_vectors=True,
                workers=workers,
                doc_ids=ids is not None,
            )
        )


def _together(tmp_path, heads, texts, ids, *, batch, workers, shard_rows, dtype, store=None):
    with (
        fse.ShardSink(
            tmp_path / "v",
            heads["quality"].encoder,
            dtype=dtype,
            shard_rows=shard_rows,
            doc_ids=ids is not None,
        ) as vectors,
        fse.ScoreSink(
            tmp_path / "s", heads, store=store, shard_rows=shard_rows, doc_ids=ids is not None
        ) as scores,
    ):
        return list(
            heads.predict_batches(
                sr.batches_of(texts, batch, ids),
                return_vectors=True,
                vector_sink=vectors,
                score_sink=scores,
                workers=workers,
                doc_ids=ids is not None,
            )
        )


def _check(
    tmp_path,
    heads,
    texts,
    ids=None,
    *,
    batch=13,
    workers=2,
    shard_rows=7,
    dtype="float32",
    store=None,
):
    arguments = {
        "batch": batch,
        "workers": workers,
        "shard_rows": shard_rows,
        "dtype": dtype,
        "store": store,
    }
    _separate(tmp_path, heads, texts, ids, **arguments)
    got = _together(tmp_path, heads, texts, ids, **arguments)
    sr.same_tensors(sr.tensors(tmp_path / "v"), sr.tensors(tmp_path / "rv"))
    sr.same_tensors(sr.tensors(tmp_path / "s"), sr.tensors(tmp_path / "rs"))
    return got


@pytest.mark.parametrize("table", ["float32", "float16", "int8"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_the_table_and_the_sink_dtype_are_independent(tmp_path, table, dtype):
    _check(tmp_path, _heads(table), sh.texts(53), dtype=dtype)


@pytest.mark.parametrize(
    ("normalize", "head_normalized"), [(True, False), (False, True), (True, True)]
)
def test_an_encoder_and_a_head_that_differ_in_normalization(tmp_path, normalize, head_normalized):
    _check(tmp_path, _heads("float16", normalize, head_normalized), sh.texts(40))


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("shard_rows", [3, 16, 64])
def test_tasks_over_shard_boundaries_and_workers(tmp_path, workers, shard_rows):
    _check(tmp_path, _heads("float16"), sh.texts(70), workers=workers, shard_rows=shard_rows)


def test_the_stored_output_is_independent_of_the_returned_one(tmp_path):
    heads, texts = _heads("float16"), sh.texts(40)
    store = {"quality": "values", "toxic": "proba", "topic": "decision"}
    _check(tmp_path, heads, texts, store=store)


def test_returned_outputs_are_those_of_a_run_with_vectors_and_without_sinks(tmp_path):
    heads, texts = _heads("float16"), sh.texts(40)
    got = _check(tmp_path, heads, texts)
    reference = list(heads.predict_batches(sr.batches_of(texts, 13), return_vectors=True))
    for name in heads:
        joined = np.concatenate([np.asarray(b.outputs[name]) for b in got])
        expected = np.concatenate([np.asarray(b.outputs[name]) for b in reference])
        np.testing.assert_array_equal(joined, expected)
    assert all(b.vectors is None for b in got)


def test_the_ids_of_both_stores_are_the_input_and_the_tensors_match(tmp_path):
    heads, texts = _heads("float16"), sh.texts(45)
    ids = np.arange(45, dtype=np.int64) * 7 - 11
    _check(tmp_path, heads, texts, ids)
    assert sr.tensors(tmp_path / "v")["doc_ids"].tolist() == ids.tolist()
    assert sr.tensors(tmp_path / "s")["doc_ids"].tolist() == ids.tolist()


def test_empty_batches_and_empty_texts(tmp_path):
    heads = _heads("float16")
    texts = ["", "w1 w2", "unknownword", "w3"]
    batches = [[], texts, [], texts[:1]]
    with (
        fse.ShardSink(tmp_path / "v", heads["quality"].encoder, shard_rows=4) as vectors,
        fse.ScoreSink(tmp_path / "s", heads, shard_rows=4) as scores,
    ):
        got = list(
            heads.predict_batches(
                batches, return_vectors=True, vector_sink=vectors, score_sink=scores
            )
        )
    assert [b.n_rows for b in got] == [0, 4, 0, 1]
    assert sr.tensors(tmp_path / "v")["vectors"].shape[0] == 5
    assert sr.tensors(tmp_path / "s")["scores"].shape[0] == 5


def test_the_unfolded_path_differs_from_the_folded_one_where_the_fold_rounds(tmp_path):
    heads, texts = _heads("float16"), sh.texts(60)
    _together(tmp_path, heads, texts, None, batch=60, workers=1, shard_rows=64, dtype="float32")
    with fse.ScoreSink(tmp_path / "folded", heads, shard_rows=64) as sink:
        list(heads.predict_batches([texts], score_sink=sink))
    both = sr.tensors(tmp_path / "s")["scores"]
    folded = sr.tensors(tmp_path / "folded")["scores"]
    assert both.shape == folded.shape
    assert sr.bits(both).tobytes() != sr.bits(folded).tobytes()
