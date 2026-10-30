"""Tests of _native_arrow.pxi: every field of an Arrow array is checked before a pointer is
formed from it; invalid offsets, views, counts, pointers and formats raise."""

import numpy as np
import pytest

from fse import _native
from fse._text import make_lookup
from support.arrow_c import Producer, layout
from support.text import vectors

INT64_MAX = 2**63 - 1
STRINGS = [b"one", b"a string of more than twelve bytes", b"", b"three"]


def from_array(p, fmt, array, **schema):
    return _native.ArrowColumn.from_array(*p.array_capsules(p.schema(fmt, **schema), array))


def all_rows(col):
    lookup = make_lookup(vectors(), "fse_words@1")
    col.scan(0, col.rows, 1 << 20)
    return col.encode(lookup, 0, col.rows, 0, _native.Workspace())


@pytest.mark.parametrize("fmt", [b"u", b"U"])
@pytest.mark.parametrize(
    "offsets", [[0, 3, 2, 5, 8], [0, 3, 40, 5, 8], [-1, 3, 5, 5, 8], [0, 3, 5, 5, -2]]
)
def test_invalid_offsets_raise_with_their_row(fmt, offsets):
    p = Producer()
    data = np.frombuffer(b"x" * 8, np.uint8)
    index = np.array(offsets, np.int32 if fmt == b"u" else np.int64)
    with pytest.raises(ValueError, match="invalid Arrow offsets"):
        all_rows(from_array(p, fmt, p.array([None, index, data], 4)))


def test_the_last_offset_bounds_every_row():
    p = Producer()
    index = np.array([0, 3, 9, 4], np.int32)  # row 1 ends beyond the last offset
    col = from_array(p, b"u", p.array([None, index, np.frombuffer(b"abcdefghi", np.uint8)], 3))
    with pytest.raises(ValueError, match="invalid Arrow offsets or view at row 1"):
        col.scan(0, 3, 1 << 20)
    with pytest.raises(ValueError, match="invalid Arrow offsets or view at row 1"):
        col.encode(make_lookup(vectors(), "fse_words@1"), 0, 3, 0, _native.Workspace())
    with pytest.raises(ValueError, match="at row 1"):
        col.row_bytes(1)


def _view(length, buffer=0, offset=0):
    v = np.zeros(16, np.uint8)
    v[:4] = np.array([length], np.int32).view(np.uint8)
    v[8:12] = np.array([buffer], np.int32).view(np.uint8)
    v[12:] = np.array([offset], np.int32).view(np.uint8)
    return v


@pytest.mark.parametrize(
    "view",
    [
        _view(-1),
        _view(20, buffer=1),
        _view(20, buffer=-1),
        _view(20, offset=-1),
        _view(20, offset=5),
        _view(2**31 - 1, offset=2**31 - 1),
    ],
)
def test_a_view_outside_its_buffer_raises(view):
    p = Producer()
    data = np.frombuffer(b"y" * 24, np.uint8)
    views = np.stack([_view(3), view])
    with pytest.raises(
        ValueError, match="the view of row 1 of the Arrow column points outside its data"
    ):
        from_array(p, b"vu", p.array([None, views, data, np.array([24], np.int64)], 2))


def test_the_sizes_of_view_buffers_are_checked():
    p = Producer()
    views = np.stack([_view(20)])
    data = np.frombuffer(b"z" * 24, np.uint8)
    with pytest.raises(ValueError, match="has the size -1"):
        from_array(p, b"vu", p.array([None, views, data, np.array([-1], np.int64)], 1))
    with pytest.raises(ValueError, match="data buffer 0 of an Arrow view array is missing"):
        from_array(p, b"vu", p.array([None, views, None, np.array([24], np.int64)], 1))
    with pytest.raises(ValueError, match="no buffer of data sizes"):
        from_array(p, b"vu", p.array([None, views, data, None], 1))


def test_view_sizes_need_not_be_aligned():
    p = Producer()
    validity, views, data, sizes = layout(b"vu", STRINGS)
    raw = np.zeros(9, np.uint8)
    raw[1:] = sizes.view(np.uint8)
    col = from_array(p, b"vu", p.array([validity, views, data, raw.ctypes.data + 1], len(STRINGS)))
    p.keep(raw)
    assert [col.row_bytes(i) for i in range(col.rows)] == STRINGS
    assert col.copied_bytes == 0


@pytest.mark.parametrize(("fmt", "width"), [(b"u", 4), (b"U", 8), (b"vu", 16)])
def test_misaligned_offsets_or_views_are_copied_and_counted(fmt, width):
    p = Producer()
    buffers = layout(fmt, STRINGS)
    index = buffers[1]
    raw = np.zeros(index.nbytes + 1, np.uint8)
    raw[1:] = index.view(np.uint8).ravel()
    p.keep(raw)
    buffers[1] = raw.ctypes.data + 1
    col = from_array(p, fmt, p.array(buffers, len(STRINGS) - 1, offset=1))
    assert [col.row_bytes(i) for i in range(col.rows)] == STRINGS[1:]
    entries = len(STRINGS) if fmt != b"vu" else len(STRINGS) - 1
    assert col.copied_bytes == entries * width


def test_null_pointers_raise():
    p = Producer()
    buffers = layout(b"u", STRINGS)
    with pytest.raises(ValueError, match="no buffers"):
        from_array(p, b"u", p.array(buffers, 4, no_buffers=True))
    with pytest.raises(ValueError, match="no offsets or views"):
        from_array(p, b"u", p.array([None, None, buffers[2]], 4))
    with pytest.raises(ValueError, match="no data buffer"):
        from_array(p, b"u", p.array([None, buffers[1], None], 4))
    empty = from_array(p, b"u", p.array([None, np.zeros(3, np.int32), None], 2))
    assert [empty.row_bytes(0), empty.row_bytes(1)] == [b"", b""]
    assert from_array(p, b"u", p.array([None, None, None], 0)).rows == 0


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"offset": -1}, "length 4 and offset -1"),
        ({"offset": INT64_MAX - 3}, "offset and length of an Arrow array overflow"),
        ({"n_buffers": 2}, "has not 2 buffers"),
        ({"n_buffers": 4}, "has not 4 buffers"),
        ({"children": 1}, "no children"),
    ],
)
def test_invalid_array_fields_raise(kw, message):
    p = Producer()
    with pytest.raises(ValueError, match=message):
        from_array(p, b"u", p.array(layout(b"u", STRINGS), 4, **kw))


def test_offsets_beyond_the_address_space_raise():
    p = Producer()
    with pytest.raises(ValueError, match="overflow"):
        from_array(p, b"u", p.array(layout(b"u", STRINGS), 4, offset=2**62))


@pytest.mark.parametrize(
    ("schema", "error", "message"),
    [
        ({"fmt": b"z"}, TypeError, "binary values, not strings"),
        ({"fmt": b"Z"}, TypeError, "binary values, not strings"),
        ({"fmt": b"vz"}, TypeError, "binary values, not strings"),
        ({"fmt": b"i"}, TypeError, "format 'i', not a string format"),
        ({"fmt": b"u", "dictionary": True}, TypeError, "dictionary-encoded"),
    ],
)
def test_other_types_are_refused_with_a_hint(schema, error, message):
    p = Producer()
    fmt = schema.pop("fmt")
    with pytest.raises(error, match=message):
        from_array(p, fmt, p.array(layout(b"u", STRINGS), 4), **schema)


def _metadata(pairs):
    out = np.array([len(pairs)], np.int32).tobytes()
    for k, v in pairs:
        out += (
            np.array([len(k)], np.int32).tobytes() + k + np.array([len(v)], np.int32).tobytes() + v
        )
    return out


def test_extension_types_are_refused_and_other_metadata_is_not():
    p = Producer()
    meta = _metadata([(b"other", b"x"), (b"ARROW:extension:name", b"arrow.json")])
    with pytest.raises(TypeError, match=r"extension type 'arrow\.json'"):
        from_array(p, b"u", p.array(layout(b"u", STRINGS), 4), metadata=meta)
    col = from_array(p, b"u", p.array(layout(b"u", STRINGS), 4), metadata=_metadata([(b"k", b"v")]))
    assert col.rows == 4


def test_a_table_names_its_string_columns():
    p = Producer()
    kids = [
        p.schema(b"u", name=b"text"),
        p.schema(b"l", name=b"id"),
        p.schema(b"vu", name=b"title"),
    ]
    with pytest.raises(TypeError, match=r"select the column.*\['text', 'title'\]"):
        from_array(p, b"+s", p.array([None], 4, children=3), children=kids)
    with pytest.raises(TypeError, match="it has no string column"):
        from_array(p, b"+s", p.array([None], 4, children=1), children=[p.schema(b"l")])


def test_capsules_of_another_name_are_refused():
    p = Producer()
    schema, array = p.array_capsules(p.schema(b"u"), p.array(layout(b"u", STRINGS), 4))
    with pytest.raises(TypeError, match="expected a PyCapsule named 'arrow_array'"):
        _native.ArrowColumn.from_array(schema, schema)
    with pytest.raises(TypeError, match="arrow_array_stream"):
        _native.ArrowColumn.from_stream(array)
    with pytest.raises(TypeError, match="arrow_schema"):
        _native.ArrowColumn.from_array(object(), array)


@pytest.mark.parametrize("fmt", [b"u", b"U", b"vu"])
@pytest.mark.parametrize(
    "bad", [b"\xff", b"ok \xed\xa0\x80", b"\xe2\x82", b"long enough text \xc3"]
)
def test_invalid_utf8_is_reported_with_its_row_and_byte(fmt, bad):
    p = Producer()
    col = from_array(p, fmt, p.strings(fmt, [b"fine", bad]))
    count, _, _, chunks, byte, sentence = col.scan(0, 2, 1 << 20)
    assert (count, chunks, sentence) == (1, [], 1)
    with pytest.raises(UnicodeDecodeError):
        bad.decode()
    assert byte >= 0
    assert bad[byte:].decode("utf-8", "replace").startswith("�")
    with pytest.raises(UnicodeDecodeError, match="of row 1"):
        col.encode(make_lookup(vectors(), "fse_words@1"), 0, 2, 0, _native.Workspace())


def test_ranges_and_sizes_are_checked():
    p = Producer()
    col = from_array(p, b"u", p.strings(b"u", STRINGS))
    for first, last in [(-1, 2), (2, 1), (0, 5)]:
        with pytest.raises(ValueError, match="outside the 4 rows"):
            col.scan(first, last, 1)
    with pytest.raises(ValueError, match="chunk_bytes must be positive"):
        col.scan(0, 4, 0)
    with pytest.raises(ValueError, match="chunk_rows must be between"):
        col.scan(0, 4, 1, 0)
