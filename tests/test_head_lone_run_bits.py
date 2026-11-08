"""A single head writes the stores a group of that head writes, byte for byte (class correctness)."""

import numpy as np
import pytest

import fse
from support import lone

CASES = [
    pytest.param(model, stored, id=f"{name}-{stored}")
    for name, model, options in lone.kinds()
    for stored in options
]
MODELS = {name: model for name, model, _ in lone.kinds()}


def _group_run(tmp_path, model, stored, vectors, ids, workers, shard_rows, batch_list):
    heads = fse.Heads(**{lone.NAME: model})
    kwargs = {"shard_rows": shard_rows, "doc_ids": ids}
    with fse.ScoreSink(tmp_path / "gs", heads, stored=stored, **kwargs) as scores:
        extra = {}
        if vectors:
            extra["vector_sink"] = fse.ShardSink(tmp_path / "gv", model.encoder, **kwargs)
        with scores:
            got = list(
                heads.predict_batches(
                    batch_list,
                    score_sink=scores,
                    return_vectors=vectors,
                    doc_ids=ids,
                    workers=workers,
                    **extra,
                )
            )
        if vectors:
            extra["vector_sink"].close()
    return got


def _lone_run(
    tmp_path, model, stored, vectors, ids, workers, shard_rows, batch_list, backend="cpu"
):
    kwargs = {"shard_rows": shard_rows, "doc_ids": ids}
    extra = {}
    if vectors:
        extra["vector_sink"] = fse.ShardSink(tmp_path / "lv", model.encoder, **kwargs)
    with fse.ScoreSink(tmp_path / "ls", model, name=lone.NAME, stored=stored, **kwargs) as scores:
        got = list(
            model.predict_batches(
                batch_list,
                score_sink=scores,
                return_vectors=vectors,
                doc_ids=ids,
                workers=workers,
                backend=backend,
                **extra,
            )
        )
    if vectors:
        extra["vector_sink"].close()
    return got


def _compare(tmp_path, model, stored, vectors, ids=False, workers=2, shard_rows=7):
    batch_list = lone.batches(ids=ids)
    group = _group_run(tmp_path, model, stored, vectors, ids, workers, shard_rows, batch_list)
    single = _lone_run(tmp_path, model, stored, vectors, ids, workers, shard_rows, batch_list)
    lone.same_store(tmp_path / "ls", tmp_path / "gs")
    if vectors:
        lone.same_store(tmp_path / "lv", tmp_path / "gv")
    assert [(b.first_row, b.n_rows, b.durable_rows) for b in single] == [
        (b.first_row, b.n_rows, b.durable_rows) for b in group
    ]
    for one, other in zip(single, group, strict=True):
        assert np.asarray(one.outputs).tobytes() == np.asarray(other.outputs[lone.NAME]).tobytes()
        assert (one.vectors is None) == (other.vectors is None)
    with fse.ScoreStore.open(tmp_path / "ls", heads=model, name=lone.NAME) as store:
        assert store.heads[lone.NAME].path == model.path(return_vectors=vectors)
        assert store.heads[lone.NAME].stored == stored


@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize(("model", "stored"), CASES)
def test_the_stores_equal_those_of_a_group_of_one(tmp_path, model, stored, vectors):
    _compare(tmp_path, model, stored, vectors)


@pytest.mark.parametrize("ids", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("shard_rows", [3, 16])
@pytest.mark.parametrize("name", ["softmax", "targets"])
def test_ids_workers_and_shard_sizes_do_not_change_the_bytes(
    tmp_path, name, ids, workers, shard_rows
):
    model = MODELS[name]
    stored = "values" if name == "targets" else "decision"
    _compare(tmp_path, model, stored, True, ids=ids, workers=workers, shard_rows=shard_rows)


@pytest.mark.parametrize("name", ["softmax", "targets", "multilabel"])
def test_the_tensors_equal_those_of_the_numpy_backend(tmp_path, name):
    model = MODELS[name]
    stored = "values" if name == "targets" else "decision"
    batch_list = lone.batches()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _lone_run(tmp_path / "a", model, stored, True, False, 2, 7, batch_list)
    _lone_run(tmp_path / "b", model, stored, True, False, 1, 7, batch_list, backend="numpy")
    from support import sink_run as sr

    sr.same_tensors(sr.tensors(tmp_path / "a" / "ls"), sr.tensors(tmp_path / "b" / "ls"))
    sr.same_tensors(sr.tensors(tmp_path / "a" / "lv"), sr.tensors(tmp_path / "b" / "lv"))


@pytest.mark.parametrize("name", ["softmax", "flat"])
def test_a_single_head_and_a_group_of_one_append_to_each_other(tmp_path, name):
    model = MODELS[name]
    stored = "values" if name == "flat" else "decision"
    batch_list = lone.batches(20)
    heads = fse.Heads(**{lone.NAME: model})
    kwargs = {"stored": stored, "shard_rows": 7}
    for first in ("lone", "group"):
        path = tmp_path / first
        for kind in (first, "group" if first == "lone" else "lone"):
            if kind == "lone":
                with fse.ScoreSink(path, model, name=lone.NAME, **kwargs) as sink:
                    list(model.predict_batches(batch_list, score_sink=sink))
            else:
                with fse.ScoreSink(path, heads, **kwargs) as sink:
                    list(heads.predict_batches(batch_list, score_sink=sink))
        with fse.ScoreStore.open(path, heads=model, name=lone.NAME) as store:
            assert len(store) == 40
            twice = np.asarray(store.get_scores(lone.NAME, stored.replace("decision", "decision")))
            assert np.array_equal(twice[:20].view("u4"), twice[20:].view("u4"))


@pytest.mark.parametrize("name", ["softmax", "flat"])
def test_a_standalone_head_writes_scores_without_vectors(tmp_path, name):
    model = lone.standalone(MODELS[name], tmp_path)
    stored = "values" if name == "flat" else "decision"
    _compare(tmp_path, model, stored, False)
    with pytest.raises(ValueError, match="standalone model has no vectors"):
        model.predict_batches(iter(()), return_vectors=True, vector_sink=object())
