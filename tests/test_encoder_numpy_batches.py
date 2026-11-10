"""NumPy text batches follow the same paths and resume rules as list batches."""

from collections.abc import Mapping

import numpy as np
import pytest

import fse
from support import positions_heads as ph
from support import score_heads as sh
from support.encoder import finished
from support.positions import anchors, stream_ids
from support.resume_heads import drop_tail


def _dtype(kind):
    if kind == "unicode":
        return np.str_
    if kind == "object":
        return object
    string_dtype = getattr(np.dtypes, "StringDType", None)
    if string_dtype is None:
        pytest.skip("StringDType requires NumPy 2.0")
    return string_dtype()


def _position_batches(texts, ids, size, arrays):
    for rows, doc_ids, position in stream_ids(texts, ids, size):
        yield np.array(rows, dtype=object) if arrays else rows, doc_ids, position


@pytest.mark.parametrize("kind", ["unicode", "object", "string"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_encoding_array_batches_matches_lists_bit_for_bit(kind, dtype):
    encoder = finished()
    polls = [["w1 w2", "w3"], [], ["", "w4"]]
    expected = list(encoder.encode_batches(polls, dtype=dtype))
    actual = list(
        encoder.encode_batches([np.array(poll, dtype=_dtype(kind)) for poll in polls], dtype=dtype)
    )
    for want, got in zip(expected, actual, strict=True):
        assert (got.first_row, got.n_rows, got.durable_rows) == (
            want.first_row,
            want.n_rows,
            want.durable_rows,
        )
        if dtype == "int8":
            assert got.rows.codes.tobytes() == want.rows.codes.tobytes()
            assert got.rows.scales.tobytes() == want.rows.scales.tobytes()
        else:
            assert got.rows.tobytes() == want.rows.tobytes()


@pytest.mark.parametrize("value", [np.array("w1"), np.array([["w1"]])])
def test_array_dimensions_are_checked_before_reading_a_batch_length(value):
    with pytest.raises(TypeError, match=r"batch 1:.*one-dimensional"):
        list(finished().encode_batches([["w1"], value]))


@pytest.mark.parametrize("value", [np.array([b"w1"]), np.array(["w1", None], dtype=object)])
def test_array_element_errors_name_the_batch_and_row(value):
    with pytest.raises(TypeError, match=r"batch 1.*(?:string|sentence) [01]"):
        list(finished().encode_batches([["w1"], value]))


def test_string_dtype_missing_values_remain_errors():
    _dtype("string")  # skip with the same minimum-version rule
    dtype = np.dtypes.StringDType(na_object=None)
    with pytest.raises(TypeError, match=r"batch 0.*(?:string|sentence) 1"):
        list(finished().encode_batches([np.array(["w1", None], dtype=dtype)]))


@pytest.mark.parametrize("kind", ["unicode", "object", "string"])
def test_array_batches_with_document_ids_write_the_same_store(tmp_path, kind):
    encoder = finished()
    polls = [["w1", "w2"], ["w3"]]
    ids = [[10, 10], [20]]
    for directory, batches in (
        (tmp_path / "lists", polls),
        (tmp_path / "arrays", [np.array(poll, dtype=_dtype(kind)) for poll in polls]),
    ):
        with fse.ShardSink(directory, encoder, doc_ids=True, shard_rows=2) as sink:
            list(encoder.encode_batches(zip(batches, ids, strict=True), sink=sink, doc_ids=True))
    with (
        fse.VectorStore.open(tmp_path / "lists") as expected,
        fse.VectorStore.open(tmp_path / "arrays") as actual,
    ):
        assert actual.get_vectors().tobytes() == expected.get_vectors().tobytes()
        assert actual.get_doc_ids().tobytes() == expected.get_doc_ids().tobytes()


@pytest.mark.parametrize("kind", ["unicode", "object", "string"])
@pytest.mark.parametrize("output", ["labels", "proba", "decision", "regressor", "heads"])
def test_prediction_array_batches_match_lists(kind, output):
    encoder = finished()
    classifier = sh.classifier(encoder)
    regressor = sh.regressor(encoder)
    model = (
        fse.Heads(c=classifier, r=regressor)
        if output == "heads"
        else regressor
        if output == "regressor"
        else classifier
    )
    options = {} if output in ("regressor", "heads") else {"output": output}
    polls = [["w1 w2", "w3"], [], ["", "w4"]]
    expected = list(model.predict_batches(polls, **options))
    actual = list(
        model.predict_batches([np.array(poll, dtype=_dtype(kind)) for poll in polls], **options)
    )
    for want, got in zip(expected, actual, strict=True):
        assert got.n_rows == want.n_rows
        if isinstance(want.outputs, Mapping):
            for name, rows in want.outputs.items():
                assert got.outputs[name].tobytes() == rows.tobytes()
        else:
            assert got.outputs.tobytes() == want.outputs.tobytes()


def test_numpy_batches_resume_two_stores_at_different_lengths_with_ids(tmp_path):
    texts = ph.TEXTS[:23]
    ids = np.arange(len(texts), dtype=np.int64) // 2
    stores, positions = [], []
    for arrays in (False, True):
        directory = tmp_path / ("arrays" if arrays else "lists")
        ph.run(directory, _position_batches(texts, ids, 5, arrays), rows=4, stop=3, doc_ids=True)
        drop_tail(directory / "v", 1, rows=4)
        drop_tail(directory / "s", 2, rows=4)
        ph.run(directory, _position_batches(texts, ids, 10, arrays), rows=4, doc_ids=True)
        stores.append(ph.stores(directory))
        positions.append((anchors(directory / "v"), anchors(directory / "s")))
        with fse.VectorStore.open(directory / "v") as store:
            assert store.get_doc_ids().tobytes() == ids.tobytes()
    assert positions[0] == positions[1]
    for want, got in zip(stores[0], stores[1], strict=True):
        assert got.tobytes() == want.tobytes()
