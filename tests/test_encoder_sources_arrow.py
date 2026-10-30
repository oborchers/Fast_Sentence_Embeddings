"""Arrow string columns passed to encode, encode_batches and the heads: pyarrow, pandas,
Polars and Hugging Face datasets as producers (each skipped where it is not installed)."""

import logging

import numpy as np
import pytest

import fse
from fse._encoder._sources import opened
from fse._text import ArrowSource
from support.encoder import finished
from support.runtime import texts

# ImportError too: pyarrow 26 refuses the numpy 1.26 of the minimum job.
pa = pytest.importorskip("pyarrow", exc_type=ImportError)

ENCODER = finished()
SENTENCES = texts(700)
WANT = ENCODER.encode(SENTENCES)


def same(got, want=WANT):
    assert got.shape == want.shape
    assert got.view(np.uint32).tobytes() == want.view(np.uint32).tobytes()


@pytest.mark.parametrize("kind", [pa.string(), pa.large_string(), pa.string_view()])
def test_pyarrow_arrays_give_the_rows_of_the_list(kind):
    array = pa.array(SENTENCES, type=kind)
    same(ENCODER.encode(array, workers=3))
    same(ENCODER.encode(array.slice(100, 250)), WANT[100:350])
    chunked = pa.chunked_array([array.slice(0, 3), array.slice(3, 400), array.slice(403)])
    same(ENCODER.encode(chunked, workers=2))
    same(ENCODER.encode(pa.table({"id": range(700), "text": array})["text"]))


@pytest.mark.parametrize("kind", [pa.string(), pa.large_string()])
def test_the_text_is_read_where_pyarrow_holds_it(kind):
    array = pa.array(SENTENCES, type=kind).slice(3)
    data, offsets = (
        array.buffers()[2],
        np.frombuffer(array.buffers()[1], np.int32 if kind == pa.string() else np.int64),
    )
    with opened(array, "fse_words@1") as source:
        assert isinstance(source, ArrowSource)
        assert source.copied_bytes == 0
        for row in (0, 1, 500):
            assert source._column.row_address(row) == data.address + int(offsets[3 + row])


def test_long_views_are_read_in_their_buffers():
    array = pa.array(["a short one", "a string of more than twelve bytes"], type=pa.string_view())
    with opened(array, "fse_words@1") as source:
        address = source._column.row_address(1)
    buffers = [b for b in array.buffers()[2:] if b is not None]
    assert any(b.address <= address < b.address + b.size for b in buffers)


def test_a_column_is_released_after_the_call():
    array = pa.array(["w1 w2"])
    with opened(array, "fse_words@1") as source:
        pass
    with pytest.raises(ValueError, match="closed"):
        source.scan()


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (
            pa.table({"text": ["a"], "n": [1]}),
            TypeError,
            r"select the column, such as table\['text'\]",
        ),
        (pa.array(["a", "b"]).dictionary_encode(), TypeError, "dictionary-encoded"),
        (pa.array([b"a"], type=pa.binary()), TypeError, "binary values"),
        (pa.array([b"a"], type=pa.large_binary()), TypeError, "binary values"),
        (pa.array([1, 2]), TypeError, "not a string format"),
        (pa.array(["w1", None, "w2"]), ValueError, "row 1 of the Arrow column is null"),
        (pa.chunked_array([["w1"], ["w2", None]]), ValueError, "row 2 of the Arrow column is null"),
    ],
)
def test_other_columns_are_refused(value, error, message):
    with pytest.raises(error, match=message):
        ENCODER.encode(value)


def test_an_extension_type_is_refused():
    json = pa.array(['{"a": 1}'], type=pa.json_()) if hasattr(pa, "json_") else None
    if json is None:
        pytest.skip("this pyarrow has no json extension type")
    with pytest.raises(TypeError, match=r"extension type 'arrow\.json'"):
        ENCODER.encode(json)


def test_invalid_utf8_from_raw_buffers_raises():
    offsets = pa.py_buffer(np.array([0, 5, 7], np.int32).tobytes())
    bad = pa.Array.from_buffers(pa.string(), 2, [None, offsets, pa.py_buffer(b"w1 w2\xff\xfe")])
    with pytest.raises(UnicodeDecodeError, match=r"in sentence 1 \(byte 0 of the row\)"):
        ENCODER.encode(bad)


def test_batches_take_arrow_columns_and_count_their_rows():
    reader = pa.RecordBatchReader.from_batches(
        pa.schema([("text", pa.string())]),
        [
            pa.record_batch([pa.array(SENTENCES[i : i + 300])], names=["text"])
            for i in range(0, 700, 300)
        ],
    )
    got = list(ENCODER.encode_batches(batch.column("text") for batch in reader))
    assert [(b.first_row, b.n_rows) for b in got] == [(0, 300), (300, 300), (600, 100)]
    same(np.concatenate([b.rows for b in got]))
    with pytest.raises(
        ValueError, match=r"batch 1 \(from row 2 of the run\).*row 1 of the Arrow column is null"
    ):
        list(ENCODER.encode_batches([pa.array(["w1", "w2"]), pa.array(["w3", None])]))
    with pytest.raises(TypeError, match="or an Arrow string column"):
        list(ENCODER.encode_batches([{"w1"}]))


def test_a_sink_takes_the_rows_of_an_arrow_column(tmp_path):
    with fse.ShardSink(tmp_path / "store", ENCODER) as sink:
        ENCODER.encode(pa.array(SENTENCES, type=pa.string_view()), sink=sink, workers=2)
    same(fse.VectorStore.open(tmp_path / "store").get_vectors())


def test_heads_take_arrow_columns():
    rng = np.random.default_rng(0)
    model = fse.Classifier.from_arrays(
        ENCODER,
        rng.standard_normal((3, ENCODER.output_dim)).astype(np.float32),
        np.zeros(3, np.float32),
        labels=["a", "b", "c"],
    )
    want = model.predict_proba(SENTENCES)
    got = model.predict_proba(pa.chunked_array([SENTENCES[:10], SENTENCES[10:]]))
    assert got.tobytes() == want.tobytes()
    batches = list(model.predict_batches([pa.array(SENTENCES[:5]), SENTENCES[5:9]]))
    assert [(b.first_row, b.n_rows) for b in batches] == [(0, 5), (5, 4)]


def test_pandas_series():
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    series = pd.Series(SENTENCES)
    assert hasattr(type(series), "__arrow_c_stream__")
    same(ENCODER.encode(series, workers=2))
    with pytest.raises(ValueError, match="row 2 of the Arrow column is null"):
        ENCODER.encode(pd.Series(["w1", "w2", None]))
    with pytest.raises(TypeError, match="select the column"):
        ENCODER.encode(pd.DataFrame({"text": SENTENCES[:3]}))


def test_a_producer_that_cannot_export_is_read_as_a_stream(caplog):
    class NoArrow(list):
        def __arrow_c_stream__(self, requested_schema=None):
            raise ImportError("pyarrow is not installed")

    with caplog.at_level(logging.WARNING, logger="fse._encoder._sources"):
        got = ENCODER.encode(NoArrow(SENTENCES[:50]))
    assert "read it as a stream" in caplog.text or "reads it as a stream" in caplog.text
    same(got, WANT[:50])
    batches = list(ENCODER.encode_batches([NoArrow(SENTENCES[:4])]))
    assert batches[0].n_rows == 4


def test_polars_series():
    pl = pytest.importorskip("polars", exc_type=ImportError)
    series = pl.concat([pl.Series(SENTENCES[:200]), pl.Series(SENTENCES[200:])], rechunk=False)
    assert series.n_chunks() == 2
    same(ENCODER.encode(series, workers=3))
    with pytest.raises(ValueError, match="row 1 of the Arrow column is null"):
        ENCODER.encode(pl.Series(["w1", None]))
    with pytest.raises(TypeError, match="select the column"):
        ENCODER.encode(pl.DataFrame({"text": SENTENCES[:3]}))


def test_hugging_face_datasets(tmp_path):
    datasets = pytest.importorskip("datasets", exc_type=ImportError)
    datasets.disable_progress_bars()
    datasets.Dataset.from_dict({"text": SENTENCES}).save_to_disk(str(tmp_path / "ds"))
    ds = datasets.load_from_disk(str(tmp_path / "ds"))
    same(ENCODER.encode(ds.with_format("arrow")["text"], workers=2))
    picked = ds.select([5, 3, 9])
    same(ENCODER.encode(picked.with_format("arrow")["text"]), WANT[[5, 3, 9]])
    with pytest.raises(TypeError, match=r"with_format\('arrow'\)\['text'\]"):
        ENCODER.encode(ds["text"])
