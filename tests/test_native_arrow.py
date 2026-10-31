"""Tests of _native_arrow.pxi: Arrow string columns read in place, with a producer in
ctypes (tests/support/arrow_c.py), so no pyarrow is needed and every invalid array can be
built."""

import numpy as np
import pytest

from fse import _native
from fse._text import StringSource, make_lookup
from support.arrow_c import Producer
from support.text import TEXTS, vectors

FORMATS = [b"u", b"U", b"vu"]
STRINGS = [t.encode() for t in TEXTS] + [b"a sentence of more than twelve bytes", b"x" * 13]


def column(producer, fmt, strings, **kw):
    schema = producer.schema(fmt)
    return _native.ArrowColumn.from_array(
        *producer.array_capsules(schema, producer.strings(fmt, strings, **kw))
    )


def rows_of(col):
    return [col.row_bytes(i) for i in range(col.rows)]


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_layout_gives_its_rows(fmt):
    p = Producer()
    col = column(p, fmt, STRINGS)
    assert rows_of(col) == STRINGS
    assert col.rows == len(STRINGS)
    assert col.nbytes == sum(map(len, STRINGS))
    assert col.copied_bytes == 0
    assert list(col.chunk_starts) == [0, len(STRINGS)]
    assert list(col.byte_starts) == [0, col.nbytes]
    assert col.span_bytes(1, 3) == len(STRINGS[1]) + len(STRINGS[2])


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(("offset", "length"), [(0, 0), (2, 3), (1, 8), (8, 1)])
def test_a_slice_reads_from_its_offset(fmt, offset, length):
    col = column(Producer(), fmt, STRINGS, slice_=(offset, length))
    assert rows_of(col) == STRINGS[offset : offset + length]


@pytest.mark.parametrize("fmt", FORMATS)
def test_scan_and_encode_match_a_string_source(fmt):
    lookup = make_lookup(vectors(), "fse_words@1")
    col = column(Producer(), fmt, STRINGS)
    source = StringSource([s.decode() for s in STRINGS])
    data = np.frombuffer(source._data, dtype=np.uint8)
    for chunk_bytes in (1, 7, 1 << 20):
        want = _native.scan_spans(data, source._offsets, 0, len(STRINGS), chunk_bytes)
        assert col.scan(0, col.rows, chunk_bytes) == want
    ws = _native.Workspace()
    got = col.encode(lookup, 2, 7, 102, ws)
    want = lookup.encode(data, source._offsets, 2, 7, 102, ws)
    assert got.first_row == 102
    np.testing.assert_array_equal(got.tokens, want.tokens)
    np.testing.assert_array_equal(got.offsets, want.offsets)


def test_fasttext_batches_carry_their_ngrams():
    lookup = make_lookup(vectors(fasttext=True), "fse_words@1", max_ngrams=4)
    col = column(Producer(), b"vu", STRINGS)
    source = StringSource([s.decode() for s in STRINGS])
    ws = _native.Workspace()
    got = col.encode(lookup, 0, col.rows, 0, ws)
    want = lookup.encode(
        np.frombuffer(source._data, np.uint8), source._offsets, 0, len(STRINGS), 0, ws
    )
    for name in ("tokens", "offsets", "ngram_offsets", "ngrams"):
        np.testing.assert_array_equal(getattr(got, name), getattr(want, name))


def test_views_in_several_buffers_and_out_of_order():
    p = Producer()
    long = [b"the first long string", b"a second long string", b"third long string!"]
    buffers = [np.frombuffer(long[1] + long[0], np.uint8), np.frombuffer(b"--" + long[2], np.uint8)]
    views = np.zeros((4, 16), np.uint8)

    def view(i, s, buffer, offset):
        views[i, :4] = np.array([len(s)], np.int32).view(np.uint8)
        views[i, 4:8] = np.frombuffer(s[:4], np.uint8)
        views[i, 8:12] = np.array([buffer], np.int32).view(np.uint8)
        views[i, 12:] = np.array([offset], np.int32).view(np.uint8)

    view(0, long[0], 0, len(long[1]))
    view(1, long[2], 1, 2)
    view(2, long[1], 0, 0)
    view(3, long[0], 0, len(long[1]))  # a reused view
    sizes = np.array([len(b) for b in buffers], np.int64)
    a = p.array([None, views, *buffers, sizes], 4)
    col = _native.ArrowColumn.from_array(*p.array_capsules(p.schema(b"vu"), a))
    assert rows_of(col) == [long[0], long[2], long[1], long[0]]


@pytest.mark.parametrize("fmt", FORMATS)
def test_empty_arrays_and_empty_strings(fmt):
    assert column(Producer(), fmt, []).rows == 0
    col = column(Producer(), fmt, [b"", b"", b""])
    assert rows_of(col) == [b"", b"", b""]
    assert col.nbytes == 0
    assert col.scan(0, 3, 1) == (3, [(0, 3, 0)], -1, -1)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(
    ("offset", "nulls", "row"), [(0, [3], 3), (5, [7], 2), (3, [8], 5), (0, [0, 4], 0)]
)
def test_a_null_ends_the_column_and_names_its_row(fmt, offset, nulls, row):
    col = column(Producer(), fmt, STRINGS, nulls=nulls, slice_=(offset, len(STRINGS) - offset))
    assert (col.null_row, col.rows) == (row, row)  # the rows before it, nothing after
    assert rows_of(col) == STRINGS[offset : offset + row]
    error = _native.arrow_null_error(col.null_row)
    assert f"row {row} of the Arrow column is null: fill or drop" in str(error)


@pytest.mark.parametrize("fmt", FORMATS)
def test_nulls_outside_the_slice_are_not_rows(fmt):
    col = column(Producer(), fmt, STRINGS, nulls=[0, 8], slice_=(1, 7))
    assert rows_of(col) == STRINGS[1:8]


@pytest.mark.parametrize("fmt", FORMATS)
def test_the_bitmap_is_read_whatever_the_null_count_says(fmt):
    assert column(Producer(), fmt, STRINGS, nulls=[2], null_count=0).null_row == 2
    assert column(Producer(), fmt, STRINGS, nulls=[2], null_count=-1).null_row == 2


@pytest.mark.parametrize("null_count", [-2, len(STRINGS) + 1])
def test_a_null_count_out_of_range_raises(null_count):
    with pytest.raises(ValueError, match="null count"):
        column(Producer(), b"u", STRINGS, null_count=null_count)


@pytest.mark.parametrize("null_count", [-1, 1])
def test_no_bitmap_needs_a_null_count_of_zero(null_count):
    with pytest.raises(ValueError, match="without a validity bitmap"):
        column(Producer(), b"u", STRINGS, null_count=null_count)
