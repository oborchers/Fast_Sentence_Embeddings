"""Tests of _native_arrow.pxi: Arrow C streams, the take-over of arrays from their capsules
and the release of every struct exactly once (Arrow C Data and C Stream interfaces)."""

import ctypes
import gc
import threading

import numpy as np
import pytest

from fse import _native
from support.arrow_c import Producer, layout

PARTS = [[b"a b", b"c"], [], [b"a longer string of text", b""], [b"d"]]


def stream_of(p, fmt=b"u", parts=PARTS, **kw):
    arrays = [p.strings(fmt, part) for part in parts]
    return p.stream(p.schema(fmt), arrays, **kw), arrays


@pytest.mark.parametrize("fmt", [b"u", b"U", b"vu"])
def test_a_stream_gives_its_chunks_as_one_column(fmt):
    p = Producer()
    capsule, _ = stream_of(p, fmt)
    col = _native.ArrowColumn.from_stream(capsule)
    assert [col.row_bytes(i) for i in range(col.rows)] == [s for part in PARTS for s in part]
    assert list(col.chunk_starts) == [0, 2, 4, 5]
    assert col.span_bytes(1, 4) == len(b"c") + len(b"a longer string of text")
    assert col.scan(0, 5, 1)[3] == [
        (0, 1, 0),
        (1, 2, 1),
        (2, 3, 2),
        (3, 5, 3),
    ]  # the empty row 3 joins row 4


def test_the_stream_and_schema_are_released_at_once_and_the_arrays_with_the_column():
    p = Producer()
    capsule, _ = stream_of(p)
    col = _native.ArrowColumn.from_stream(capsule)
    del capsule
    gc.collect()
    assert (p.releases.count("stream"), p.releases.count("schema"), p.releases.count("array")) == (
        1,
        1,
        0,
    )
    assert col.row_bytes(0) == b"a b"  # the arrays outlive the stream and its capsule
    col.release()
    assert p.releases.count("array") == len(PARTS)
    col.release()
    del col
    gc.collect()
    assert p.releases.count("array") == len(PARTS)  # once each


def test_arrays_are_moved_with_their_private_data():
    p = Producer()
    array = p.strings(b"u", [b"x", b"y"], private_data=0xBEEF)
    schema, capsule = p.array_capsules(p.schema(b"u"), array)
    col = _native.ArrowColumn.from_array(schema, capsule)
    assert not array.release  # the producer's struct was marked as moved
    del schema, capsule
    gc.collect()
    assert [col.row_bytes(0), col.row_bytes(1)] == [b"x", b"y"]
    col.release()
    ((kind, address, private),) = p.releases.calls
    assert (kind, private) == ("array", 0xBEEF)
    assert address != ctypes.addressof(array)  # released where fse moved it
    assert p.releases.count("schema") == 0  # the schema of an array stays with its capsule


def test_a_column_is_released_when_it_is_freed():
    p = Producer()
    capsule, _ = stream_of(p)
    col = _native.ArrowColumn.from_stream(capsule)
    del col
    gc.collect()
    assert p.releases.count("array") == len(PARTS)


@pytest.mark.parametrize(
    ("message", "text"), [(b"disk on fire", "disk on fire"), (None, "error code 5")]
)
def test_an_error_of_get_next_raises_and_releases_what_was_taken(message, text):
    p = Producer()
    capsule, _ = stream_of(p, fail_at=2, message=message)
    with pytest.raises(OSError, match=f"get_next: {text}") as error:
        _native.ArrowColumn.from_stream(capsule)
    assert error.value.errno == 5
    gc.collect()
    assert p.releases.count("array") == 2  # the two arrays before the error, once each
    assert (p.releases.count("stream"), p.releases.count("schema")) == (1, 1)


def test_an_error_at_the_first_get_next_touches_no_array():
    p = Producer()
    capsule, _ = stream_of(p, fail_at=0)
    with pytest.raises(OSError, match="get_next"):
        _native.ArrowColumn.from_stream(capsule)
    gc.collect()
    assert p.releases.count("array") == 0


def test_an_error_of_get_schema_raises():
    p = Producer()
    capsule, _ = stream_of(p, schema_code=22)
    with pytest.raises(OSError, match="get_schema"):
        _native.ArrowColumn.from_stream(capsule)
    gc.collect()
    assert (p.releases.count("stream"), p.releases.count("schema"), p.releases.count("array")) == (
        1,
        0,
        0,
    )


def test_an_invalid_array_of_a_stream_releases_every_array_taken():
    p = Producer()
    arrays = [
        p.strings(b"u", [b"a"]),
        p.strings(b"u", [b"b", b"c"], nulls=[1]),
        p.strings(b"u", [b"d"]),
    ]
    capsule = p.stream(p.schema(b"u"), arrays)
    with pytest.raises(ValueError, match="row 2 of the Arrow column is null"):
        _native.ArrowColumn.from_stream(capsule)
    gc.collect()
    assert p.releases.count("array") == 2


def test_a_stream_of_a_table_is_refused_and_released():
    p = Producer()
    schema = p.schema(b"+s", children=[p.schema(b"u", name=b"text")])
    capsule = p.stream(schema, [])
    with pytest.raises(TypeError, match=r"select the column, such as table\['text'\]"):
        _native.ArrowColumn.from_stream(capsule)
    assert (p.releases.count("stream"), p.releases.count("schema")) == (1, 1)


def test_released_structs_are_refused():
    p = Producer()
    capsule, _ = stream_of(p)
    _native.ArrowColumn.from_stream(capsule)
    with pytest.raises(ValueError, match="stream was released already"):
        _native.ArrowColumn.from_stream(capsule)
    schema, array = p.array_capsules(p.schema(b"u"), p.strings(b"u", [b"a"]))
    _native.ArrowColumn.from_array(schema, array)
    with pytest.raises(ValueError, match="array was released already"):
        _native.ArrowColumn.from_array(schema, array)


def test_a_released_column_reads_nothing():
    p = Producer()
    col = _native.ArrowColumn.from_stream(stream_of(p)[0])
    col.release()
    assert col.released
    assert (col.rows, col.nbytes) == (0, 0)
    with pytest.raises(ValueError, match="was released"):
        col.scan(0, 0, 1)
    with pytest.raises(ValueError, match="was released"):
        col.row_bytes(0)


def test_a_column_being_read_is_not_released():
    p = Producer()
    strings = [b"some words of text"] * 2_000_000
    buffers = layout(b"u", strings)
    col = _native.ArrowColumn.from_array(
        *p.array_capsules(p.schema(b"u"), p.array(buffers, len(strings)))
    )
    refused = []
    worker = threading.Thread(target=col.scan, args=(0, col.rows, 1 << 30))
    worker.start()
    while worker.is_alive():
        try:
            col.release()
        except RuntimeError as error:
            refused.append(str(error))
            continue
        break
    worker.join()
    assert refused
    assert "being read" in refused[0]
    col.release()
    assert p.releases.count("array") == 1
    assert np.all(buffers[1] >= 0)  # the producer's buffers were never written
