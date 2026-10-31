"""Tests of _native_scan.pxi and ArrowColumn.scan: the pre-scan against CPython itself.

An oracle splits the lines (or items, or rows) with CPython, takes the first invalid
sentence and its byte from ``bytes.decode`` and cuts the chunks by the rule of the scan,
for the three ways in: the lines of a buffer, the strings of a buffer with offsets, and an
Arrow column in u, U and vu. It reads results through ``core`` and so does not care how
many counters the tuple carries beside sentences, chunks, error byte and error sentence.
"""

import random

import numpy as np
import pytest

from fse import _native
from support.arrow_c import Producer

MAX = _native.MAX_CHUNK_ROWS
MODES = ["lines", "items", "u", "U", "vu"]
CHUNK_BYTES = [1, 3, 7, 64, 1 << 20]
CHUNK_ROWS = [1, 2, MAX]

GOOD = [
    b"",
    b"plain",
    b"two words",
    b"  lead and trail  ",
    "ä ö ü ß".encode(),  # 2 bytes
    "€ euro".encode(),  # 3 bytes
    "😀 grin".encode(),  # 4 bytes
    b"a\xe2\x80\xa8b\xe2\x80\xa9c",  # U+2028 and U+2029 are no line ends
    b"\xe3\x80\x80\xc2\xa0\xe1\x9a\x80",  # U+3000, NBSP, U+1680: only spaces
    b"a sentence of more than twelve bytes",
    b"\x00\x01 control",
]
BAD = [
    b"\xff",
    b"\xc0\x80",  # overlong
    b"\xed\xa0\x80",  # surrogate
    b"\xf4\x90\x80\x80",  # above U+10FFFF
    b"\xf5",
    b"\x80",  # lone continuation
    b"ok \xc3",  # cut 2-byte sequence
    b"abc\xe2\x82",  # cut 3-byte sequence
    b"\xf0\x9f\x98",  # cut 4-byte sequence
    b"\xe2\x41\x41",  # bad continuation
    b"a long enough text before \xe2\x82\xff",
]


def core(result):
    """The sentences, chunks, error byte and error sentence of a 6 or a 4 value result."""
    return (result[0], *result[-3:])


def cut(chunks, start, pos, first, done):
    chunks.append((start, pos, first))
    return pos, done


def oracle(units, first, last, chunk_bytes, chunk_rows):
    """The scan of units, a list of (position after the unit, bytes, text), in the units
    of positions; returns (sentences, chunks, error index or None, ...) for the caller."""
    sentences, chunks, start, start_sentence, size = 0, [], first, 0, 0
    for pos, length, text in units:
        try:
            text.decode()
        except UnicodeDecodeError as err:
            return sentences, [], err.start, pos, sentences
        sentences += 1
        size += length
        if (size >= chunk_bytes or sentences - start_sentence >= chunk_rows) and pos < last:
            start, start_sentence = cut(chunks, start, pos, start_sentence, sentences)
            size = 0
    if start < last:
        chunks.append((start, last, start_sentence))
    return sentences, chunks, -1, 0, -1


def lines_of(data, first, last):
    """The lines that start in [first, last) as (end position, length, bytes)."""
    pos, units = first, []
    while pos < last:
        end = data.find(b"\n", pos)
        stop = len(data) if end < 0 else end
        units.append((stop + 1 if end >= 0 else len(data), stop - pos, data[pos:stop], pos))
        pos = units[-1][0]
    return units


def want_lines(data, first, last, chunk_bytes, chunk_rows):
    """Lines: every unit is a line; positions are bytes; the lengths count a line end
    for the cut, as the scan measures from line start to line start."""
    lines = lines_of(data, first, last)
    start, start_sentence, sentences, chunks = first, 0, 0, []
    for pos, _, text, base in lines:
        try:
            text.decode()
        except UnicodeDecodeError as err:
            return sentences, [], base + err.start, sentences
        sentences += 1
        if (pos - start >= chunk_bytes or sentences - start_sentence >= chunk_rows) and pos < last:
            start, start_sentence = cut(chunks, start, pos, start_sentence, sentences)
    if start < last:
        chunks.append((start, last, start_sentence))
    return sentences, chunks, -1, -1


def want_items(items, first, last, chunk_bytes, chunk_rows, *, absolute):
    """Items or Arrow rows: positions are indices; an item's error byte is absolute in
    the buffer, a row's relative to the row."""
    bases = np.concatenate([[0], np.cumsum([len(i) for i in items])]).tolist()
    units = [(i + 1, len(items[i]), items[i]) for i in range(first, last)]
    sentences, chunks, byte, pos, error = oracle(units, first, last, chunk_bytes, chunk_rows)
    if byte >= 0:
        return sentences, [], byte + (bases[pos - 1] if absolute else 0), error
    return sentences, chunks, -1, -1


def spans(firsts, lasts):
    """A few (first, last) pairs of the given positions: the whole, and cut at each end."""
    firsts_ = {firsts[0], firsts[len(firsts) // 2], firsts[-1], firsts[min(1, len(firsts) - 1)]}
    lasts_ = {lasts[-1], lasts[0], lasts[max(len(lasts) - 2, 0)]}
    return sorted((a, b) for a in firsts_ for b in lasts_ if a <= b)


def check_lines(sentences, trailing, chunk_bytes, chunk_rows):
    data = b"\n".join(sentences) + (b"\n" if trailing and sentences else b"")
    array = np.frombuffer(data, dtype=np.uint8)
    starts = [0] + [i + 1 for i, b in enumerate(data) if b == 10]
    for first, last in spans(starts, sorted({*starts, len(data)})):
        got = core(_native.scan_spans(array, None, first, last, chunk_bytes, chunk_rows))
        assert got == want_lines(data, first, last, chunk_bytes, chunk_rows), (data, first, last)


def check_items(sentences, chunk_bytes, chunk_rows):
    array = np.frombuffer(b"".join(sentences), dtype=np.uint8)
    offsets = np.cumsum([0] + [len(s) for s in sentences]).astype(np.int64)
    positions = list(range(len(sentences) + 1))
    for first, last in spans(positions, positions):
        got = core(_native.scan_spans(array, offsets, first, last, chunk_bytes, chunk_rows))
        want = want_items(sentences, first, last, chunk_bytes, chunk_rows, absolute=True)
        assert got == want, (sentences, first, last)


def check_arrow(fmt, sentences, chunk_bytes, chunk_rows):
    p = Producer()
    schema = p.schema(fmt.encode())
    col = _native.ArrowColumn.from_array(
        *p.array_capsules(schema, p.strings(fmt.encode(), sentences))
    )
    positions = list(range(len(sentences) + 1))
    for first, last in spans(positions, positions):
        got = core(col.scan(first, last, chunk_bytes, chunk_rows))
        want = want_items(sentences, first, last, chunk_bytes, chunk_rows, absolute=False)
        assert got == want, (fmt, sentences, first, last)


def check(mode, sentences, trailing, chunk_bytes, chunk_rows):
    if mode == "lines":
        check_lines(sentences, trailing, chunk_bytes, chunk_rows)
    elif mode == "items":
        check_items(sentences, chunk_bytes, chunk_rows)
    else:
        check_arrow(mode, sentences, chunk_bytes, chunk_rows)


def all_params(mode, sentences, trailing=True):
    for chunk_bytes in CHUNK_BYTES:
        for chunk_rows in CHUNK_ROWS:
            check(mode, sentences, trailing, chunk_bytes, chunk_rows)


GOOD_LISTS = [[], [b""], [b"", b"", b""], [b"a"], [b"a b", b""], [b"", b"x y"], GOOD, GOOD[::-1]]


def with_bad():
    for bad in BAD:
        yield [bad]
        yield [b"fine", bad]
        yield [b"fine", b"also fine words", bad]
        yield [b"", bad, b"last"]
        yield [bad, b"first", bad]
        yield [b"x", b"", bad, b"", b""]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("trailing", [True, False])
def test_valid_sentences(mode, trailing):
    for sentences in GOOD_LISTS:
        all_params(mode, sentences, trailing)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("trailing", [True, False])
def test_invalid_utf8_at_the_start_in_the_middle_and_at_the_end(mode, trailing):
    for sentences in with_bad():
        all_params(mode, sentences, trailing)


def test_empty_lines_are_sentences_in_line_mode():
    for data, lines in [(b"\n\n\n", 3), (b"", 0), (b"\n", 1), (b"a", 1), (b"a\n\nb", 3)]:
        array = np.frombuffer(data, dtype=np.uint8)
        got = core(_native.scan_spans(array, None, 0, len(data), 1 << 20))
        assert got == (lines, [(0, len(data), 0)] if lines else [], -1, -1)


def test_a_cut_sequence_at_a_line_end_is_invalid_in_its_line():
    data = b"ok\n\xe2\x82\nnext\n"
    got = core(_native.scan_spans(np.frombuffer(data, dtype=np.uint8), None, 0, len(data), 64))
    assert got == (1, [], 3, 1)


def test_the_error_byte_of_an_arrow_row_is_relative_to_its_row():
    p = Producer()
    rows = [b"fine", b"ab\xffcd"]
    col = _native.ArrowColumn.from_array(*p.array_capsules(p.schema(b"u"), p.strings(b"u", rows)))
    assert core(col.scan(0, 2, 1 << 20)) == (1, [], 2, 1)


@pytest.mark.parametrize("mode", MODES)
def test_seeded_fuzz(mode):
    rng = random.Random(177)
    pieces = GOOD + BAD
    for _ in range(400):
        sentences = [
            b"".join(rng.choices(pieces, k=rng.randrange(0, 4))) for _ in range(rng.randrange(0, 7))
        ]
        check(mode, sentences, rng.random() < 0.5, rng.choice(CHUNK_BYTES), rng.choice(CHUNK_ROWS))
