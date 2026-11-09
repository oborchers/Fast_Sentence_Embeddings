"""Caller-facing head types, checked without importing optional integrations."""

from collections.abc import Generator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_type

import numpy as np
from numpy.typing import NDArray

import fse

TEXTS = ["a b", "c", ""]


class _SpecialClassifier(fse.Classifier):
    pass


class _SpecialRegressor(fse.Regressor):
    pass


def _encoder() -> fse.Encoder:
    return fse.Encoder.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"])


def test_head_factory_types(tmp_path: Path) -> None:
    encoder = _encoder()
    classifier = _SpecialClassifier.from_arrays(
        encoder, np.ones((2, 3), np.float32), link="multilabel"
    )
    regressor = _SpecialRegressor.from_arrays(encoder, np.ones(3, np.float32))
    assert_type(classifier, _SpecialClassifier)
    assert_type(regressor, _SpecialRegressor)
    assert type(classifier) is _SpecialClassifier
    assert type(regressor) is _SpecialRegressor
    cquantized = classifier.quantize("int8")
    rquantized = regressor.quantize("int8")
    assert_type(cquantized, _SpecialClassifier)
    assert_type(rquantized, _SpecialRegressor)
    assert type(cquantized) is _SpecialClassifier
    assert type(rquantized) is _SpecialRegressor
    adjusted = classifier.with_thresholds(0.6)
    assert_type(adjusted, _SpecialClassifier)
    assert type(adjusted) is _SpecialClassifier
    cpath, rpath = tmp_path / "c.safetensors", tmp_path / "r.safetensors"
    assert_type(classifier.save(cpath), None)
    assert_type(regressor.save(rpath), None)
    cloaded = _SpecialClassifier.load(cpath)
    rloaded = _SpecialRegressor.load(rpath)
    assert_type(cloaded, _SpecialClassifier)
    assert_type(rloaded, _SpecialRegressor)
    assert type(cloaded) is _SpecialClassifier
    assert type(rloaded) is _SpecialRegressor
    cf = classifier.save_pretrained(tmp_path / "c", license="mit")
    rf = regressor.save_pretrained(tmp_path / "r", license="mit")
    cpretrained = _SpecialClassifier.from_pretrained(cf)
    rpretrained = _SpecialRegressor.from_pretrained(rf)
    assert_type(cpretrained, _SpecialClassifier)
    assert_type(rpretrained, _SpecialRegressor)
    assert type(cpretrained) is _SpecialClassifier
    assert type(rpretrained) is _SpecialRegressor
    # These integrations accept foreign objects. Their runtime behavior has its own tests.
    if TYPE_CHECKING:
        foreign: Any = object()
        assert_type(fse.Classifier.from_sklearn(encoder, foreign), fse.Classifier)
        assert_type(fse.Regressor.from_sklearn(encoder, foreign), fse.Regressor)
        assert_type(fse.Classifier.from_model2vec(foreign), fse.Classifier)
        assert_type(_SpecialClassifier.from_sklearn(encoder, foreign), _SpecialClassifier)
        assert_type(_SpecialRegressor.from_sklearn(encoder, foreign), _SpecialRegressor)
        assert_type(_SpecialClassifier.from_model2vec(foreign), _SpecialClassifier)


def test_classifier_predictions_and_scores() -> None:
    model = fse.Classifier.from_arrays(_encoder(), np.eye(3, dtype=np.float32))
    labels = model.predict(TEXTS)
    assert_type(labels, NDArray[Any])
    assert isinstance(labels, np.ndarray)
    vectors = model.predict(TEXTS, return_vectors=True)
    assert_type(vectors, tuple[NDArray[Any], NDArray[np.float32]])
    assert vectors[1].dtype == np.float32
    reported = model.predict(TEXTS, lengths="summary")
    assert_type(reported, tuple[NDArray[Any], fse.LengthReport])
    assert isinstance(reported[1], fse.LengthReport)
    all_parts = model.predict(TEXTS, return_vectors=True, lengths="per_row")
    assert_type(all_parts, tuple[NDArray[Any], NDArray[np.float32], fse.LengthReport])
    assert isinstance(all_parts[2], fse.LengthReport)
    assert_type(model.classes_, NDArray[Any])
    assert_type(model.coef_, NDArray[np.float32])
    assert_type(model.intercept_, NDArray[np.float32])
    assert model.coef_.dtype == model.intercept_.dtype == np.float32
    proba = model.predict_proba(TEXTS)
    decision = model.decision_function(TEXTS)
    assert_type(proba, NDArray[np.float32])
    assert_type(decision, NDArray[np.float32])
    assert proba.dtype == decision.dtype == np.float32
    proba_report = model.predict_proba(TEXTS, lengths="summary")
    decision_report = model.decision_function(TEXTS, lengths="per_row")
    assert_type(proba_report, tuple[NDArray[np.float32], fse.LengthReport])
    assert_type(decision_report, tuple[NDArray[np.float32], fse.LengthReport])
    assert proba_report[0].dtype == decision_report[0].dtype == np.float32
    no_proba = model.predict_proba(TEXTS, sink=fse.MemorySink(3))
    no_decision = model.decision_function(TEXTS, sink=fse.MemorySink(3))
    assert_type(no_proba, None)
    assert_type(no_decision, None)
    assert no_proba is no_decision is None
    sink_proba = model.predict_proba(TEXTS, sink=fse.MemorySink(3), lengths="summary")
    sink_decision = model.decision_function(TEXTS, sink=fse.MemorySink(3), lengths="per_row")
    assert_type(sink_proba, tuple[None, fse.LengthReport])
    assert_type(sink_decision, tuple[None, fse.LengthReport])
    assert sink_proba[0] is sink_decision[0] is None
    if TYPE_CHECKING:
        foreign_sink: Any = object()
        assert_type(model.predict_proba(TEXTS, sink=foreign_sink), Any)
        assert_type(model.decision_function(TEXTS, sink=foreign_sink), Any)


def test_regressor_predictions() -> None:
    model = fse.Regressor.from_arrays(_encoder(), np.ones(3, np.float32))
    values = model.predict(TEXTS)
    assert_type(values, NDArray[np.float32])
    assert values.dtype == np.float32
    with_vectors = model.predict(TEXTS, return_vectors=True)
    assert_type(with_vectors, tuple[NDArray[np.float32], NDArray[np.float32]])
    assert with_vectors[0].dtype == with_vectors[1].dtype == np.float32
    report = model.predict(TEXTS, lengths="summary")
    assert_type(report, tuple[NDArray[np.float32], fse.LengthReport])
    assert report[0].dtype == np.float32
    all_parts = model.predict(TEXTS, return_vectors=True, lengths="per_row")
    assert_type(all_parts, tuple[NDArray[np.float32], NDArray[np.float32], fse.LengthReport])
    assert isinstance(all_parts[2], fse.LengthReport)
    taken = model.predict(TEXTS, sink=fse.MemorySink(1))
    assert_type(taken, None)
    assert taken is None
    taken_report = model.predict(TEXTS, sink=fse.MemorySink(1), lengths="summary")
    assert_type(taken_report, tuple[None, fse.LengthReport])
    assert taken_report[0] is None


def test_heads_mapping_and_predictions() -> None:
    encoder = _encoder()
    c = fse.Classifier.from_arrays(encoder, np.eye(3, dtype=np.float32))
    r = fse.Regressor.from_arrays(encoder, np.ones(3, np.float32))
    heads = fse.Heads(c=c, r=r)
    assert_type(heads["c"], fse.Classifier | fse.Regressor)
    found = heads.get("r")
    assert_type(found, fse.Classifier | fse.Regressor | None)
    assert isinstance(found, fse.Regressor)
    for model in heads.values():
        assert_type(model, fse.Classifier | fse.Regressor)
        assert isinstance(model, fse.Classifier | fse.Regressor)
    result = heads.predict(TEXTS)
    assert_type(result, Mapping[str, NDArray[Any]])
    assert list(result) == ["c", "r"]
    vectors = heads.predict(TEXTS, return_vectors=True)
    assert_type(vectors, tuple[Mapping[str, NDArray[Any]], NDArray[np.float32]])
    assert vectors[1].dtype == np.float32
    report = heads.predict(TEXTS, output={"c": "proba"}, lengths="summary")
    assert_type(report, tuple[Mapping[str, NDArray[Any]], fse.LengthReport])
    assert report[0]["c"].dtype == np.float32
    all_parts = heads.predict(TEXTS, return_vectors=True, lengths="per_row")
    assert_type(all_parts, tuple[Mapping[str, NDArray[Any]], NDArray[np.float32], fse.LengthReport])
    assert isinstance(all_parts[2], fse.LengthReport)


def test_classifier_resume_returns(tmp_path: Path) -> None:
    model = fse.Classifier.from_arrays(_encoder(), np.eye(3, dtype=np.float32))
    with fse.ScoreSink(tmp_path / "scores", model, name="c", resume=True) as score:
        result = model.predict(TEXTS, score_sink=score)
        assert_type(result, None)
        assert result is None
    with fse.ScoreSink(tmp_path / "report", model, name="c", resume=True) as score:
        report = model.predict(TEXTS, score_sink=score, lengths="summary")
        assert_type(report, tuple[None, fse.LengthReport])
        assert report[0] is None
    with fse.ShardSink(tmp_path / "vectors", model.encoder, resume=True) as vector:
        result = model.predict(TEXTS, vector_sink=vector, return_vectors=True)
        assert_type(result, None)
        assert result is None
    with (
        fse.ShardSink(tmp_path / "combined-vectors", model.encoder, resume=True) as vector,
        fse.ScoreSink(tmp_path / "combined-scores", model, name="c", resume=True) as score,
    ):
        report = model.predict(
            TEXTS, vector_sink=vector, score_sink=score, return_vectors=True, lengths="per_row"
        )
        assert_type(report, tuple[None, fse.LengthReport])
        assert report[0] is None


def test_regressor_resume_returns(tmp_path: Path) -> None:
    model = fse.Regressor.from_arrays(_encoder(), np.ones(3, np.float32))
    with fse.ScoreSink(tmp_path / "scores", model, name="r", resume=True) as score:
        result = model.predict(TEXTS, score_sink=score)
        assert_type(result, None)
        assert result is None
    with fse.ScoreSink(tmp_path / "report", model, name="r", resume=True) as score:
        report = model.predict(TEXTS, score_sink=score, lengths="summary")
        assert_type(report, tuple[None, fse.LengthReport])
        assert report[0] is None
    with fse.ShardSink(tmp_path / "vectors", model.encoder, resume=True) as vector:
        result = model.predict(TEXTS, vector_sink=vector, return_vectors=True)
        assert_type(result, None)
        assert result is None
    with (
        fse.ShardSink(tmp_path / "combined-vectors", model.encoder, resume=True) as vector,
        fse.ScoreSink(tmp_path / "combined-scores", model, name="r", resume=True) as score,
    ):
        report = model.predict(
            TEXTS, vector_sink=vector, score_sink=score, return_vectors=True, lengths="per_row"
        )
        assert_type(report, tuple[None, fse.LengthReport])
        assert report[0] is None


def test_heads_resume_returns(tmp_path: Path) -> None:
    encoder = _encoder()
    heads = fse.Heads(c=fse.Classifier.from_arrays(encoder, np.eye(3, dtype=np.float32)))
    with fse.ScoreSink(tmp_path / "scores", heads, resume=True) as score:
        result = heads.predict(TEXTS, score_sink=score)
        assert_type(result, None)
        assert result is None
    with fse.ScoreSink(tmp_path / "report", heads, resume=True) as score:
        report = heads.predict(TEXTS, score_sink=score, lengths="summary")
        assert_type(report, tuple[None, fse.LengthReport])
        assert report[0] is None
    with fse.ShardSink(tmp_path / "vectors", encoder, resume=True) as vector:
        result = heads.predict(TEXTS, vector_sink=vector, return_vectors=True)
        assert_type(result, None)
        assert result is None
    with (
        fse.ShardSink(tmp_path / "combined-vectors", encoder, resume=True) as vector,
        fse.ScoreSink(tmp_path / "combined-scores", heads, resume=True) as score,
    ):
        report = heads.predict(
            TEXTS, vector_sink=vector, score_sink=score, return_vectors=True, lengths="per_row"
        )
        assert_type(report, tuple[None, fse.LengthReport])
        assert report[0] is None


def test_predicted_batches_are_generators_with_typed_fields() -> None:
    encoder = _encoder()
    c = fse.Classifier.from_arrays(encoder, np.eye(3, dtype=np.float32))
    r = fse.Regressor.from_arrays(encoder, np.ones(3, np.float32))
    classifier_batches = c.predict_batches([TEXTS], return_vectors=True)
    assert_type(classifier_batches, Generator[fse.PredictedBatch, None, None])
    item = next(classifier_batches)
    assert_type(item.outputs, NDArray[Any] | None)
    assert_type(item.vectors, NDArray[np.float32] | None)
    assert item.outputs is not None
    assert item.vectors is not None
    assert item.vectors.dtype == np.float32
    classifier_batches.close()
    reported = c.predict_batches([TEXTS], lengths="summary")
    assert_type(reported, Generator[tuple[fse.PredictedBatch, fse.LengthReport], None, None])
    assert isinstance(next(reported)[1], fse.LengthReport)
    reported.close()
    regressor_batches = r.predict_batches([TEXTS])
    assert_type(regressor_batches, Generator[fse.PredictedBatch, None, None])
    assert next(regressor_batches).outputs is not None
    regressor_batches.close()
    rreported = r.predict_batches([TEXTS], lengths="per_row")
    assert_type(rreported, Generator[tuple[fse.PredictedBatch, fse.LengthReport], None, None])
    assert isinstance(next(rreported)[1], fse.LengthReport)
    rreported.close()
    taken = r.predict_batches([TEXTS], sink=fse.MemorySink(1))
    assert next(taken).outputs is None
    taken.close()
    heads = fse.Heads(c=c, r=r)
    grouped = heads.predict_batches([TEXTS], output={"c": "proba"}, lengths="per_row")
    assert_type(grouped, Generator[fse.HeadsBatch, None, None])
    assert isinstance(next(grouped).lengths, fse.LengthReport)
    grouped.close()


def _optional_classifier_sinks(
    model: fse.Classifier, score: fse.Sink | None, vector: fse.Sink | None, flag: bool
) -> None:
    result = model.predict(TEXTS, score_sink=score, vector_sink=vector, return_vectors=flag)
    assert_type(result, NDArray[Any] | tuple[NDArray[Any], NDArray[np.float32]] | None)
    assert result is None or isinstance(result, np.ndarray | tuple)
    reported = model.predict(
        TEXTS, score_sink=score, vector_sink=vector, return_vectors=flag, lengths="summary"
    )
    assert_type(
        reported,
        tuple[NDArray[Any], fse.LengthReport]
        | tuple[NDArray[Any], NDArray[np.float32], fse.LengthReport]
        | tuple[None, fse.LengthReport],
    )
    assert isinstance(reported[-1], fse.LengthReport)


def _optional_regressor_sinks(
    model: fse.Regressor, score: fse.Sink | None, vector: fse.Sink | None, flag: bool
) -> None:
    result = model.predict(TEXTS, score_sink=score, vector_sink=vector, return_vectors=flag)
    assert_type(
        result, NDArray[np.float32] | tuple[NDArray[np.float32], NDArray[np.float32]] | None
    )
    assert result is None or isinstance(result, np.ndarray | tuple)
    reported = model.predict(
        TEXTS, score_sink=score, vector_sink=vector, return_vectors=flag, lengths="summary"
    )
    assert_type(
        reported,
        tuple[NDArray[np.float32], fse.LengthReport]
        | tuple[NDArray[np.float32], NDArray[np.float32], fse.LengthReport]
        | tuple[None, fse.LengthReport],
    )
    assert isinstance(reported[-1], fse.LengthReport)


def _optional_group_sinks(
    heads: fse.Heads, score: fse.Sink | None, vector: fse.Sink | None, flag: bool
) -> None:
    result = heads.predict(TEXTS, score_sink=score, vector_sink=vector, return_vectors=flag)
    assert_type(
        result,
        Mapping[str, NDArray[Any]] | tuple[Mapping[str, NDArray[Any]], NDArray[np.float32]] | None,
    )
    assert result is None or isinstance(result, Mapping | tuple)
    reported = heads.predict(
        TEXTS, score_sink=score, vector_sink=vector, return_vectors=flag, lengths="summary"
    )
    assert_type(
        reported,
        tuple[Mapping[str, NDArray[Any]], fse.LengthReport]
        | tuple[Mapping[str, NDArray[Any]], NDArray[np.float32], fse.LengthReport]
        | tuple[None, fse.LengthReport],
    )
    assert isinstance(reported[-1], fse.LengthReport)


def test_optional_sinks_are_arrays_when_none() -> None:
    encoder = _encoder()
    c = fse.Classifier.from_arrays(encoder, np.eye(3, dtype=np.float32))
    r = fse.Regressor.from_arrays(encoder, np.ones(3, np.float32))
    heads = fse.Heads(c=c, r=r)
    for flag in (False, True):
        _optional_classifier_sinks(c, None, None, flag)
        _optional_regressor_sinks(r, None, None, flag)
        _optional_group_sinks(heads, None, None, flag)
