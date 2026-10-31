"""Tests of fse._text._scan: scans, chunks and their sizes."""

import numpy as np
import pytest

from fse import _native, _text
from fse._text import (
    StringSource,
)
from fse._text_file import FileSource

F32 = np.float32


def test_small_sources_and_one_worker_scan_without_threads(monkeypatch):

    def no_threads(**_kwargs):
        raise AssertionError("a thread pool was created")

    monkeypatch.setattr(_text._scan, "ThreadPoolExecutor", no_threads)
    assert StringSource(["a b", "c"] * 10).scan(workers=8).sentences == 20
    big = StringSource(["word " * 40] * 6000)  # about 1.2 MB
    assert big.scan(workers=1).sentences == 6000


@pytest.mark.parametrize("workers", [2, 4])
def test_inline_and_threaded_scans_give_the_same_chunks(monkeypatch, tmp_path, workers):

    lines = [f"sentence {i} ".encode() * (i % 7) for i in range(5000)]
    good = tmp_path / "good.txt"
    good.write_bytes(b"\n".join(lines) + b"\n")
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\n".join([*lines[:3333], b"bad \xff text", *lines[3333:]]) + b"\n")
    runs, errors = {}, {}
    for name, limit in (("inline", 1 << 40), ("threads", 0)):
        monkeypatch.setattr(_text._scan, "_SCAN_INLINE_BYTES", limit)
        with FileSource(good) as source:
            runs[name] = source.scan(workers=workers, chunk_bytes=4096)
        with FileSource(bad) as source, pytest.raises(UnicodeDecodeError) as error:
            source.scan(workers=workers, chunk_bytes=4096)
        errors[name] = str(error.value)
    assert runs["inline"] == runs["threads"]
    assert len(runs["inline"].chunks) > workers
    assert errors["inline"] == errors["threads"]
    assert "3333" in errors["inline"]


@pytest.mark.parametrize("rows", [1, 3, 100])
def test_chunk_rows_bound_the_rows_of_a_chunk(rows):
    texts = [""] * 250 + ["a b"] * 50  # empty sentences take no bytes
    scan = StringSource(texts).scan(chunk_bytes=1 << 20, chunk_rows=rows)
    assert scan.sentences == 300
    assert max(c.n_sentences for c in scan.chunks) <= rows
    assert sum(c.n_sentences for c in scan.chunks) == 300


@pytest.mark.parametrize("rows", [0, -1, _native.MAX_CHUNK_ROWS + 1, True, 2.0])
def test_chunk_rows_are_checked(rows):
    with pytest.raises(ValueError, match="chunk_rows"):
        StringSource(["a"]).scan(chunk_rows=rows)


def test_numpy_integers_are_sizes():
    scan = StringSource(["a b"] * 10).scan(
        workers=np.int64(2), chunk_bytes=np.int32(4), chunk_rows=np.int64(3)
    )
    assert max(c.n_sentences for c in scan.chunks) <= 3


def test_scan_ranges_are_not_smaller_than_a_chunk(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_bytes(b"a\n" * 65_537)  # just over two chunks of 65,536 bytes
    with FileSource(path) as source:
        scan = source.scan(workers=8, chunk_bytes=65_536)
    assert len(scan.chunks) <= 2


def test_one_long_line_with_one_row_per_chunk(tmp_path):
    path = tmp_path / "long.txt"
    path.write_bytes(b"word " * (1 << 20) + b"\n")
    with FileSource(path) as source:
        scan = source.scan(chunk_bytes=1 << 30, chunk_rows=1)
    assert [c.n_sentences for c in scan.chunks] == [1]


def test_full_chunks_of_neighbour_ranges_stay_apart(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_bytes((b"x" * 65_535 + b"\n") * 32)  # 32 lines of exactly one chunk each
    with FileSource(path) as source:
        scan = source.scan(workers=8, chunk_bytes=65_536)
    assert len(scan.chunks) == 32


def test_strings_are_cut_from_their_offsets():
    from itertools import pairwise

    from fse._text._plan import offset_cuts

    texts = [f"sentence {i} " * (i % 9) for i in range(5000)]
    for source in (StringSource(texts), StringSource.encoded([t.encode() for t in texts])):
        for workers in (1, 4):
            scan = source.scan(workers=workers, chunk_bytes=4096, chunk_rows=300)
            cuts = offset_cuts(source._offsets, 4096, 300).tolist()
            assert [(c.start, c.end, c.first_row) for c in scan.chunks] == [
                (a, b, a) for a, b in pairwise(cuts)
            ]


@pytest.mark.parametrize("workers", [1, 2, 8, 10, 16])
@pytest.mark.parametrize("total", [100 << 10, 2_430_000, 8_350_000, 50_000_000])
def test_the_default_rule_gives_about_two_tasks_per_worker(workers, total):
    from fse._text._scan import TASKS_PER_WORKER, default_chunk_bytes

    texts = ["x" * 99] * (total // 99)
    source = StringSource(texts)
    size = default_chunk_bytes(source.nbytes, workers)
    assert size == max(1 << 16, min(1 << 20, -(-source.nbytes // (TASKS_PER_WORKER * workers))))
    scan = source.scan(workers=workers)
    assert len(scan.chunks) == max(1, -(-len(texts) * 99 // size))
    assert all((c.end - c.start) * 99 <= size + 99 for c in scan.chunks)


def test_a_stream_keeps_eight_tasks_per_collection():
    from fse._runtime._tasks import STREAM_TASKS_PER_COLLECTION, plan
    from fse._stream import StreamSource

    texts = ["x" * 99] * 40_000  # one collection of about 4 MB
    planned = plan(
        StreamSource(texts),
        workers=4,
        chunk_bytes=None,
        chunk_rows=1 << 16,
        tokenizer="fse_words@1",
    )
    assert len(list(planned.tasks)) == STREAM_TASKS_PER_COLLECTION


def _lines_across_blocks(block):
    """Lines of 0 to 3 blocks, newlines just before, at and just after the block edges,
    and a last line without a newline."""
    lengths = [block - 1, block, block + 1, 0, 3 * block, 1, block - 2, 2 * block + 1, 0, 5]
    return [bytes([97 + i % 26]) * n for i, n in enumerate(lengths)]


def _starts(data):
    """The byte where every line starts, then the length of the data (naively)."""
    starts = [0] + [i + 1 for i, b in enumerate(data) if b == 10]
    return starts if starts[-1] != len(data) else starts[:-1]


@pytest.mark.parametrize("block", [1, 4, 7, 16])
def test_prefixes_of_file_chunks_across_line_blocks(monkeypatch, tmp_path, block):
    monkeypatch.setattr(_text._scan, "_LINE_BLOCK", block)
    data = b"\n".join(_lines_across_blocks(block))  # the last line ends without a newline
    path = tmp_path / "lines.txt"
    path.write_bytes(data)
    starts = [*_starts(data), len(data)]
    with FileSource(path) as source:
        for chunk in source.scan(chunk_bytes=2 * block + 1).chunks:
            first = starts.index(chunk.start)
            for rows in range(1, chunk.n_sentences + 1):
                got = source.prefix(chunk, rows)
                assert (got.start, got.end, got.n_sentences) == (
                    chunk.start,
                    starts[first + rows],
                    rows,
                ), (chunk, rows)


@pytest.mark.parametrize("block", [1, 4, 7, 16])
@pytest.mark.parametrize("bad", [2, 4, 7, 9])  # lines longer than a block, and the last
def test_an_invalid_line_across_line_blocks(monkeypatch, tmp_path, block, bad):
    monkeypatch.setattr(_text._scan, "_LINE_BLOCK", block)
    lines = _lines_across_blocks(block)
    lines[bad] = lines[bad] + b"\xff" + lines[bad]
    data = b"\n".join(lines)
    path = tmp_path / "lines.txt"
    path.write_bytes(data)
    starts = _starts(data)
    with FileSource(path) as source:
        scan, error = source.scan_until_error(workers=2, chunk_bytes=block, first_row=500)
    assert isinstance(error, UnicodeDecodeError)
    assert f"in sentence {500 + bad} " in str(error)
    assert scan.sentences == bad
    assert sum(c.n_sentences for c in scan.chunks) == bad
    assert [c.first_row for c in scan.chunks] == [
        sum(c.n_sentences for c in scan.chunks[:i]) for i in range(len(scan.chunks))
    ]
    if scan.chunks:
        assert scan.chunks[-1].end == starts[bad]  # the chunks end where the bad line starts


def _sentences(n=2000):
    return [f"sentence {i} " * (1 + i % 7) for i in range(n)]


@pytest.mark.parametrize("kind", ["strings", "file", "arrow"])
def test_split_tail_cuts_the_last_chunks_in_halves(monkeypatch, tmp_path, kind):
    from support.runtime import assert_chunks_tile, source

    monkeypatch.setattr(_text._scan, "_CHUNK_MIN_BYTES", 1)
    src = source(kind, _sentences(), tmp_path)
    scan = src.scan(chunk_bytes=8000)
    lines = None if kind != "file" else bytes(src._data)
    assert len(scan.chunks) > 8
    for workers in (1, 4, len(scan.chunks), 100):
        chunks = src.split_tail(scan.chunks, workers)
        assert_chunks_tile(chunks, scan.sentences, lines=lines)
        whole = len(scan.chunks) - min(workers, len(scan.chunks))
        assert chunks[:whole] == scan.chunks[:whole]  # only the tail changes
        assert len(chunks) == len(scan.chunks) + min(workers, len(scan.chunks))
        # Each of the last chunks gives two parts of about equal rows.
        for old, a, b in zip(
            scan.chunks[whole:], chunks[whole::2], chunks[whole + 1 :: 2], strict=True
        ):
            assert a.n_sentences + b.n_sentences == old.n_sentences
            assert abs(a.n_sentences - b.n_sentences) <= 1
    # The result depends only on the arguments.
    assert src.split_tail(scan.chunks, 4) == src.split_tail(scan.chunks, 4)


def test_split_tail_keeps_small_chunks_and_single_sentences_whole(monkeypatch):
    src = StringSource(_sentences(400))
    scan = src.scan(chunk_bytes=8000)
    sizes = sorted(src._span_bytes(c) for c in scan.chunks)
    monkeypatch.setattr(_text._scan, "_CHUNK_MIN_BYTES", sizes[-1])  # no part reaches it
    assert src.split_tail(scan.chunks, 4) == scan.chunks
    monkeypatch.setattr(_text._scan, "_CHUNK_MIN_BYTES", sizes[0] // 2)  # every chunk splits
    assert len(src.split_tail(scan.chunks, 4)) == 2 * len(scan.chunks)
    monkeypatch.setattr(_text._scan, "_CHUNK_MIN_BYTES", 1)
    one = StringSource(["a long sentence " * 200])
    assert one.split_tail(one.scan().chunks, 8) == one.scan().chunks


def test_split_tail_of_fewer_chunks_than_workers_splits_all_of_them(monkeypatch):
    monkeypatch.setattr(_text._scan, "_CHUNK_MIN_BYTES", 1)
    src = StringSource(_sentences(300))
    scan = src.scan(chunk_bytes=1 << 20)
    assert len(scan.chunks) == 1
    assert len(src.split_tail(scan.chunks, 8)) == 2
