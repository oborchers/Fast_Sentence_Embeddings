"""Caller-facing store and resumed-stream types, without network or optional packages."""

import inspect
from collections.abc import Generator, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any, assert_type

import numpy as np
from numpy.typing import NDArray

import fse
from fse.typing import ScoreHead

TEXTS = ["a b", "c", ""]


def _encoder() -> fse.Encoder:
    return fse.Encoder.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"])


def test_vector_store_types_and_closing(tmp_path: Path) -> None:
    encoder = _encoder()
    memory: fse.Sink = fse.MemorySink(3)
    assert encoder.encode(TEXTS, sink=memory) is None
    with fse.ShardSink(tmp_path / "vectors", encoder, doc_ids=True) as sink:
        assert_type(sink, fse.ShardSink)
        protocol: fse.Sink = sink
        result = encoder.encode(TEXTS, sink=protocol, doc_ids=np.array([4, 5, 6], np.int64))
        assert_type(result, None)
        assert result is None
    with fse.VectorStore.open(tmp_path / "vectors", encoder=encoder) as store:
        batches = store.iter_batches()
        assert_type(batches, Generator[fse.VectorBatch, None, None])
        assert inspect.isgenerator(batches)
        batch = next(batches)
        assert_type(batch.vectors, NDArray[Any] | fse.QuantizedRows)
        assert_type(batch.doc_ids, NDArray[np.int64] | None)
        assert_type(batch.inv_norms, NDArray[np.float32] | None)
        normalized = batch.normalized()
        assert_type(normalized, NDArray[np.float32])
        assert normalized.dtype == np.float32
        batches.close()
        with closing(store.iter_batches()) as opened:
            assert_type(opened, Generator[fse.VectorBatch, None, None])
            assert next(opened).n_rows == 3
        vectors = store.get_vectors()
        ids = store.get_doc_ids()
        assert_type(vectors, NDArray[np.float32])
        assert_type(ids, NDArray[np.int64])
        assert vectors.dtype == np.float32
        assert ids.dtype == np.int64
        result = store.search(vectors[0], k=2)
        assert_type(result.scores, NDArray[np.float32])
        assert_type(result.ids, NDArray[np.int64])
        assert result.scores.dtype == np.float32
        assert result.ids.dtype == np.int64
        taken = store.take_vectors([0, 1])
        taken_ids = store.take_doc_ids([0, 1])
        assert_type(taken, NDArray[np.float32])
        assert_type(taken_ids, NDArray[np.int64])
        assert taken.dtype == np.float32
        assert taken_ids.dtype == np.int64
        model = fse.Regressor.from_arrays(encoder, np.ones(3, np.float32))
        predicted = model.predict_batches(store)
        assert_type(predicted, Generator[fse.PredictedBatch, None, None])
        assert inspect.isgenerator(predicted)
        assert next(predicted).outputs is not None
        predicted.close()


def test_score_store_types(tmp_path: Path) -> None:
    encoder = _encoder()
    classifier = fse.Classifier.from_arrays(encoder, np.eye(3, dtype=np.float32))
    with fse.ScoreSink(tmp_path / "scores", classifier, name="c", doc_ids=True) as sink:
        assert_type(sink, fse.ScoreSink)
        protocol: fse.Sink = sink
        assert protocol.dim == 3
        batches = classifier.predict_batches(
            [(TEXTS, np.array([4, 5, 6], np.int64))], score_sink=sink, doc_ids=True
        )
        assert inspect.isgenerator(batches)
        list(batches)
    with fse.ScoreStore.open(tmp_path / "scores") as store:
        heads = store.heads
        assert_type(heads, Mapping[str, ScoreHead])
        assert isinstance(heads["c"], ScoreHead)
        ids = store.get_doc_ids()
        assert_type(ids, NDArray[np.int64])
        assert ids.dtype == np.int64
        scores = store.get_scores("c", "decision")
        assert_type(scores, NDArray[np.float32])
        assert scores.dtype == np.float32
        stored = store.iter_batches()
        assert_type(stored, Generator[fse.ScoreBatch, None, None])
        assert inspect.isgenerator(stored)
        assert_type(next(stored).outputs, Mapping[str, NDArray[Any]])
        stored.close()
        default = store.predict_batches()
        assert_type(default, Generator[fse.ScoreBatch, None, None])
        assert inspect.isgenerator(default)
        assert next(default).outputs["c"].dtype == np.int64
        default.close()
        probabilities = store.predict_batches(output={"c": "proba"})
        assert_type(probabilities, Generator[fse.ScoreBatch, None, None])
        assert next(probabilities).outputs["c"].dtype == np.float32
        probabilities.close()
        with closing(store.predict_batches(output="decision")) as closing_scores:
            assert next(closing_scores).outputs["c"].dtype == np.float32


def test_resumed_encode_and_position_streams(tmp_path: Path) -> None:
    encoder = _encoder()
    with fse.ShardSink(tmp_path / "resume", encoder, resume=True) as sink:
        result = encoder.encode(TEXTS, sink=sink, shard=(0, 1))
        assert_type(result, None)
        assert result is None
    with fse.ShardSink(tmp_path / "resume-report", encoder, resume=True) as sink:
        report = encoder.encode(TEXTS, sink=sink, shard=(0, 1), lengths="summary")
        assert_type(report, tuple[None, fse.LengthReport])
        assert report[0] is None
    with fse.ShardSink(tmp_path / "positions", encoder, resume=True) as sink:
        positioned = encoder.encode_batches([(TEXTS, "first")], sink=sink, positions=True)
        assert_type(positioned, Generator[fse.EncodedBatch, None, None])
        assert inspect.isgenerator(positioned)
        assert next(positioned).rows is None
        positioned.close()


def test_head_stream_generators_with_both_sinks_and_positions(tmp_path: Path) -> None:
    encoder = _encoder()
    classifier = fse.Classifier.from_arrays(encoder, np.eye(3, dtype=np.float32))
    regressor = fse.Regressor.from_arrays(encoder, np.ones(3, np.float32))
    heads = fse.Heads(c=classifier, r=regressor)
    with (
        fse.ShardSink(tmp_path / "vectors", encoder) as vector,
        fse.ScoreSink(tmp_path / "scores", heads) as score,
    ):
        combined = heads.predict_batches(
            [TEXTS], return_vectors=True, vector_sink=vector, score_sink=score
        )
        assert_type(combined, Generator[fse.HeadsBatch, None, None])
        assert inspect.isgenerator(combined)
        item = next(combined)
        assert item.vectors is None
        combined.close()
    with (
        fse.ShardSink(tmp_path / "position-vectors", encoder, resume=True) as vector,
        fse.ScoreSink(tmp_path / "position-scores", heads, resume=True) as score,
    ):
        positions = heads.predict_batches(
            [(TEXTS, "first")],
            return_vectors=True,
            vector_sink=vector,
            score_sink=score,
            positions=True,
            lengths="per_row",
        )
        assert_type(positions, Generator[fse.HeadsBatch, None, None])
        assert inspect.isgenerator(positions)
        assert isinstance(next(positions).lengths, fse.LengthReport)
        positions.close()
    with fse.ScoreSink(tmp_path / "lone-positions", classifier, name="c", resume=True) as sink:
        lone = classifier.predict_batches(
            [(TEXTS, "first")], score_sink=sink, positions=True, lengths="summary"
        )
        assert_type(lone, Generator[tuple[fse.PredictedBatch, fse.LengthReport], None, None])
        assert inspect.isgenerator(lone)
        assert isinstance(next(lone)[1], fse.LengthReport)
        lone.close()
