"""All-at-once head predictions with the vectors of their encoder."""

from contextlib import ExitStack
from dataclasses import fields

import numpy as np
import pytest

import fse
from support import lengths as ls
from support import score_heads as sh
from support.pool import no_fse_threads_left, settled  # noqa: F401
from support.resume_heads import Poisoned
from support.runtime import small_tasks


def same(got, want):
    if hasattr(want, "items"):
        assert list(got) == list(want)
        for name in want:
            same(got[name], want[name])
    else:
        assert got.dtype == want.dtype
        if want.dtype == np.float32:
            got, want = got.view(np.uint32), want.view(np.uint32)
        np.testing.assert_array_equal(got, want)


def model(kind, encoder):
    if kind == "group":
        return fse.Heads(
            raw=sh.classifier(encoder),
            norm=sh.classifier(encoder, normalized=True),
            multi=sh.classifier(encoder, link="multilabel"),
            regression=sh.regressor(encoder),
        )
    if kind == "classifier":
        return sh.classifier(encoder)
    if kind == "multilabel":
        return sh.classifier(encoder, link="multilabel")
    return sh.regressor(encoder, vector=kind == "flat")


KINDS = ["group", "classifier", "multilabel", "flat", "column"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("source", ["list", "generator", "file", "empty"])
def test_predict_matches_batches_and_encoder_bits(tmp_path, monkeypatch, kind, normalize, source):
    small_tasks(monkeypatch)
    enc = sh.encoder(normalize=normalize)
    head = model(kind, enc)
    data = sh.texts(200) if source != "empty" else []
    batch = next(head.predict_batches([data], return_vectors=True, workers=1))
    want = batch.outputs
    for workers in (1, 3):
        items = data
        if source == "generator":
            items = (text for text in data)
        if source == "file":
            items = tmp_path / "texts.txt"
            items.write_text("".join(t + "\n" for t in data))
        result, vectors = head.predict(items, return_vectors=True, workers=workers)
        same(result, want)
        same(vectors, batch.vectors)
        same(vectors, enc.encode(data, workers=1))


@pytest.mark.parametrize("output", [None, "labels", "proba", "decision", {"raw": "decision"}])
def test_group_output_selection_has_batch_bits(output):
    enc = sh.encoder()
    heads = fse.Heads(raw=sh.classifier(enc), norm=sh.classifier(enc, normalized=True))
    data = sh.texts()
    batch = next(heads.predict_batches([data], output=output, return_vectors=True))
    result, vectors = heads.predict(data, output=output, return_vectors=True)
    same(result, batch.outputs)
    same(vectors, batch.vectors)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("level", ["summary", "per_row"])
@pytest.mark.parametrize("count", [False, True])
def test_flat_tuple_has_the_reference_length_report(monkeypatch, kind, level, count):
    small_tasks(monkeypatch)
    enc = ls.encoder(max_chars=80)
    head = model(kind, enc)
    result, vectors, report = head.predict(
        iter(ls.TEXTS), return_vectors=True, lengths=level, count_truncated=count, workers=3
    )
    ls.check_report(
        report, ls.expected(ls.TEXTS, max_chars=80), count=count, per_row=level == "per_row"
    )
    plain, plain_vectors = head.predict(ls.TEXTS, return_vectors=True, workers=1)
    same(result, plain)
    same(vectors, plain_vectors)
    without_vectors = head.predict(ls.TEXTS, lengths=level, count_truncated=count)
    assert len(without_vectors) == 2
    for field in fields(report):
        np.testing.assert_equal(getattr(report, field.name), getattr(without_vectors[1], field.name))


@pytest.mark.parametrize("kind", KINDS)
def test_truncation_reports_the_first_failing_row(monkeypatch, kind):
    small_tasks(monkeypatch)
    head = model(kind, ls.encoder())
    expected = ls.expected(ls.TEXTS)
    first = next(i for i, row in enumerate(expected) if row["token_truncated"])
    for workers in (1, 3):
        with pytest.raises(fse.TruncationError) as error:
            head.predict(ls.TEXTS, return_vectors=True, on_truncation="raise", workers=workers)
        assert error.value.row == first


@pytest.mark.parametrize("kind", ["group", "classifier", "flat"])
def test_standalone_refused_before_reading(tmp_path, kind):
    enc = sh.encoder()
    one = sh.regressor(enc) if kind == "flat" else sh.classifier(enc)
    path = tmp_path / "head.safetensors"
    one.save(path, standalone=True)
    standalone = type(one).load(path)
    head = fse.Heads(one=standalone) if kind == "group" else standalone
    with pytest.raises(ValueError, match="its table is folded"):
        head.predict(Poisoned(), return_vectors=True)


@pytest.mark.parametrize("kind", ["group", "classifier", "flat"])
def test_store_input_refused(tmp_path, kind):
    enc = sh.encoder()
    with fse.ShardSink(tmp_path / "v", enc) as sink:
        enc.encode(sh.texts(3), sink=sink)
    with fse.VectorStore.open(tmp_path / "v") as store:
        with pytest.raises(ValueError, match="get_vectors|group of heads reads texts"):
            model(kind, enc).predict(store, return_vectors=True)


def test_regressor_value_sink_with_vectors_refused_before_reading():
    with pytest.raises(ValueError, match="sink= takes the values"):
        sh.regressor(sh.encoder()).predict(Poisoned(), sink=object(), return_vectors=True)


@pytest.mark.parametrize("kind", ["group", "classifier", "flat"])
@pytest.mark.parametrize("option", [{"doc_ids": [1]}, {"shard": (0, 1)}])
def test_source_coordinates_still_need_a_resume_sink(kind, option):
    with pytest.raises(ValueError, match="need a vector_sink or a score_sink"):
        model(kind, sh.encoder()).predict(Poisoned(), return_vectors=True, **option)


@pytest.mark.parametrize("kind", ["group", "classifier", "flat"])
@pytest.mark.parametrize("sinks", ["score", "vector", "both"])
@pytest.mark.parametrize("lengths", [None, "summary"])
def test_resume_returns_stay_unchanged(tmp_path, kind, sinks, lengths):
    enc = sh.encoder()
    head = model(kind, enc)
    options = {}
    with ExitStack() as stack:
        if sinks != "vector":
            options["score_sink"] = stack.enter_context(
                fse.ScoreSink(tmp_path / "s", head, resume=True, name="head")
                if kind != "group"
                else fse.ScoreSink(tmp_path / "s", head, resume=True)
            )
        if sinks != "score":
            options["vector_sink"] = stack.enter_context(
                fse.ShardSink(tmp_path / "v", enc, resume=True)
            )
        got = head.predict(sh.texts(20), return_vectors=True, lengths=lengths, **options)
        if lengths is None:
            assert got is None
        else:
            assert got[0] is None
            assert got[1].n_rows == 20


@pytest.mark.parametrize("kind", ["group", "classifier", "flat"])
def test_reuses_pool_after_success_error_and_nested_call(monkeypatch, kind):
    small_tasks(monkeypatch)
    head = model(kind, sh.encoder())
    data = sh.texts(200)
    want, vectors = head.predict(data, return_vectors=True, workers=1)
    with fse.Pool(3, max_tasks=2) as pool:
        got, found = head.predict(data, return_vectors=True, pool=pool)
        same(got, want)
        same(found, vectors)
        with pytest.raises((TypeError, ValueError)):
            head.predict([*data, 7], return_vectors=True, pool=pool)

        def nested():
            with pool.inside():
                return head.predict(data, return_vectors=True, pool=pool)

        with pytest.raises(RuntimeError, match="same pool"):
            pool.submit(nested).result(timeout=10)
        same(head.predict(data, return_vectors=True, pool=pool)[0], want)
        assert not pool.closed
    assert settled(0)
