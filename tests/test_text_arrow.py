"""Tests of fse._text._arrow: ArrowSource, an Arrow string column as a text source, with
the producer in ctypes (no pyarrow needed)."""

import logging

import numpy as np
import pytest

from fse import _native
from fse._runtime._tasks import plan
from fse._text import ArrowSource, StringSource, is_arrow, make_lookup
from support.arrow_c import Producer, layout
from support.runtime import texts
from support.text import TEXTS, concat, encode_all, vectors

FORMATS = [b"u", b"U", b"vu"]


class Exporter:
    """An object that exports one stream of chunks and has no __len__."""

    def __init__(self, fmt, parts, producer=None):
        self.producer = producer or Producer()
        self.fmt, self.parts = fmt, parts

    def __arrow_c_stream__(self, requested_schema=None):
        assert requested_schema is None  # fse requests no schema
        p = self.producer
        return p.stream(p.schema(self.fmt), [p.strings(self.fmt, part) for part in self.parts])


class ArrayExporter:
    """An object that exports one array, with a __len__ that lies."""

    def __init__(self, fmt, parts, producer=None):
        self.producer = producer or Producer()
        self.fmt, self.parts = fmt, parts

    def __len__(self):
        return 999

    def __arrow_c_array__(self, requested_schema=None):
        p = self.producer
        strings = [s for part in self.parts for s in part]
        return p.array_capsules(p.schema(self.fmt), p.strings(self.fmt, strings))


def parts_of(sentences, size):
    encoded = [s.encode() for s in sentences]
    return [encoded[i : i + size] for i in range(0, len(encoded), size)]


def test_is_arrow_looks_at_the_type():
    assert is_arrow(Exporter(b"u", []))
    assert is_arrow(ArrayExporter(b"u", []))
    assert not is_arrow(["a"])
    assert not is_arrow(np.array(["a"]))
    with pytest.raises(TypeError, match="exports no Arrow data"):
        ArrowSource(["a"])


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("size", [1, 3, 1000])
@pytest.mark.parametrize("chunk_bytes", [1, 7, None])
def test_batches_match_a_string_source(fmt, size, chunk_bytes):
    lookup = make_lookup(vectors(), "fse_words@1")
    source = ArrowSource(Exporter(fmt, parts_of(TEXTS, size)))
    assert len(source) == len(TEXTS)
    assert source.nbytes == sum(len(t.encode()) for t in TEXTS)
    scan, batches = encode_all(source, lookup, chunk_bytes=chunk_bytes)
    _, want = encode_all(StringSource(TEXTS), lookup, chunk_bytes=chunk_bytes)
    rows = [r for c in scan.chunks for r in range(c.first_row, c.first_row + c.n_sentences)]
    assert rows == list(range(len(TEXTS)))
    for got, expected in zip(concat(batches), concat(want), strict=True):
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("workers", [1, 3])
def test_chunks_of_the_runtime_span_arrow_chunks(workers):
    sentences = texts(5000)
    source = ArrowSource(Exporter(b"vu", parts_of(sentences, 10)))
    assert len(source._column.chunk_starts) == 501
    scan = source.scan(workers=workers, chunk_bytes=1 << 14)
    assert scan.sentences == 5000
    assert len(scan.chunks) < 100  # tasks of 16 KiB, not one per Arrow chunk of 10 rows
    lookup = make_lookup(vectors(), "fse_words@1")
    _, got = encode_all(source, lookup, workers=workers, chunk_bytes=1 << 14)
    _, want = encode_all(StringSource(sentences), lookup, chunk_bytes=1 << 14)
    for a, b in zip(concat(got), concat(want), strict=True):
        np.testing.assert_array_equal(a, b)


def test_very_unequal_chunks_and_empty_columns():
    sentences = texts(3000)
    parts = [[s.encode() for s in sentences[:2900]], *parts_of(sentences[2900:], 7), []]
    source = ArrowSource(Exporter(b"U", parts))
    scan = source.scan(workers=4, chunk_bytes=1 << 12)
    assert sum(c.n_sentences for c in scan.chunks) == 3000
    assert ArrowSource(Exporter(b"u", [])).scan(workers=2).chunks == ()
    only_empty = ArrowSource(Exporter(b"vu", [[b""] * 5, [b""]]))
    assert only_empty.scan().sentences == 6
    assert only_empty.nbytes == 0


def test_the_runtime_plans_an_arrow_source_like_a_string_source():
    small = plan(
        ArrowSource(Exporter(b"u", parts_of(TEXTS, 2))),
        workers=4,
        chunk_bytes=None,
        chunk_rows=_native.MAX_CHUNK_ROWS,
        tokenizer="fse_words@1",
    )
    assert small.inline
    assert small.rows == len(TEXTS)
    big = texts(20000)
    planned = plan(
        ArrowSource(Exporter(b"u", parts_of(big, 1000))),
        workers=4,
        chunk_bytes=None,
        chunk_rows=_native.MAX_CHUNK_ROWS,
        tokenizer="fse_words@1",
    )
    assert not planned.inline
    assert planned.rows == 20000


def test_rows_come_from_the_source_not_from_len():
    exporter = ArrayExporter(b"u", parts_of(TEXTS, 3))
    source = ArrowSource(exporter)
    assert len(source) == len(TEXTS)
    assert len(source._column.chunk_starts) == 2  # one array, not a stream


@pytest.mark.parametrize("fmt", FORMATS)
def test_invalid_utf8_names_the_row_and_byte(fmt):
    p = Producer()
    exporter = Exporter(fmt, [[b"fine"], [b"also fine", b"ok \xff here"]], producer=p)
    source = ArrowSource(exporter)
    with pytest.raises(UnicodeDecodeError, match=r"in sentence 2 \(byte 3 of the row\)"):
        source.scan()


def test_a_closed_source_reads_nothing_and_releases_its_arrays():
    p = Producer()
    source = ArrowSource(Exporter(b"u", [[b"a"], [b"b"]], producer=p))
    with source:
        assert source.scan().sentences == 2
    assert p.releases.count("array") == 2
    with pytest.raises(ValueError, match="closed"):
        source.scan()
    with pytest.raises(ValueError, match="closed"):
        source.encode(None, None, None)


def test_misaligned_offsets_are_copied_and_logged(caplog):
    p = Producer()
    buffers = layout(b"U", [b"one", b"two"])
    raw = np.zeros(buffers[1].nbytes + 1, np.uint8)
    raw[1:] = buffers[1].view(np.uint8)
    p.keep(raw)
    buffers[1] = raw.ctypes.data + 1

    class Misaligned:
        def __arrow_c_array__(self, requested_schema=None):
            return p.array_capsules(p.schema(b"U"), p.array(buffers, 2))

    with caplog.at_level(logging.WARNING, logger="fse._text._arrow"):
        source = ArrowSource(Misaligned())
    assert source.copied_bytes == 24
    assert "copied 24 bytes of them (no text)" in caplog.text
    assert ArrowSource(Exporter(b"U", [[b"x"]])).copied_bytes == 0
