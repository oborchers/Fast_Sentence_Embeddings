import os

import numpy as np
import pytest

from fse import _native
from fse import _reference_text as ref
from fse._text import Chunk, FileSource, StringSource, TokenListSource, make_lookup, tokenize
from fse._wordvectors import FastTextParams, WordVectors

F32 = np.float32
TEXTS = [
    "I don't know.",
    "",
    "The cat sat; the dog didn't.",
    "caf\u00e9 \u4e2d\u56fd \u0130stanbul",
    "   ",
    "well-known U.S. facts, 1,000 of them",
    "x",
]
WORDS = [
    "i",
    "do",
    "not",
    "know",
    "the",
    "cat",
    "sat",
    "dog",
    "did",
    "caf\u00e9",
    "istanbul",
    "well",
    "known",
    "u.s",
    "1,000",
    "of",
    "x",
]


def _vectors(fasttext=False):
    if not fasttext:
        return WordVectors.from_arrays(np.ones((len(WORDS), 3), F32), words=WORDS)
    return WordVectors.from_arrays(
        np.ones((len(WORDS), 3), F32),
        words=WORDS,
        fasttext=FastTextParams(3, 5, 50),
        ngrams=np.ones((50, 3), F32),
    )


def _ids():
    return {w.encode(): i for i, w in enumerate(WORDS)}


def _encode_all(source, lookup, workers=1, chunk_bytes=None):
    scan = source.scan(workers=workers, chunk_bytes=chunk_bytes)
    ws = _native.Workspace()
    return scan, [source.encode(chunk, lookup, ws) for chunk in scan.chunks]


def _concat(batches):
    tokens = np.concatenate([b.tokens for b in batches]) if batches else np.zeros(0, np.uint32)
    offsets = [0]
    for b in batches:
        base = offsets[-1]
        offsets.extend(base + int(o) for o in b.offsets[1:])
    return tokens, np.array(offsets, np.int64)


@pytest.mark.parametrize("chunk_bytes", [1, 7, None])
def test_string_source_gives_the_reference_batches(chunk_bytes):
    source = StringSource(TEXTS)
    assert len(source) == len(TEXTS)
    scan, batches = _encode_all(
        source, make_lookup(_vectors(), "fse_words@1"), chunk_bytes=chunk_bytes
    )
    assert scan.sentences == len(TEXTS)
    rows = [row for c in scan.chunks for row in range(c.first_row, c.first_row + c.n_sentences)]
    assert rows == list(range(len(TEXTS)))
    want = ref.encode_texts(TEXTS, _ids(), tokenizer="fse_words@1")
    tokens, offsets = _concat(batches)
    np.testing.assert_array_equal(tokens, want.tokens)
    np.testing.assert_array_equal(offsets, want.offsets)
    for chunk, batch in zip(scan.chunks, batches, strict=True):
        assert batch.first_row == chunk.first_row


def test_file_source_reads_lines(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_bytes("\n".join(TEXTS).encode() + b"\n")
    with FileSource(path) as source:
        assert len(source) == len(TEXTS)
        scan, batches = _encode_all(
            source,
            make_lookup(_vectors(True), "fse_words@1", max_ngrams=4),
            workers=3,
            chunk_bytes=5,
        )
    want = ref.encode_texts(
        TEXTS, _ids(), tokenizer="fse_words@1", fasttext=(3, 5, 50), max_ngrams=4
    )
    assert scan.sentences == len(TEXTS)
    tokens, offsets = _concat(batches)
    np.testing.assert_array_equal(tokens, want.tokens)
    np.testing.assert_array_equal(offsets, want.offsets)
    assert sum(len(b.ngrams) for b in batches) == len(want.ngrams)


@pytest.mark.parametrize(
    ("data", "lines"),
    [
        (b"", []),
        (b"\n", [""]),
        (b"a", ["a"]),
        (b"a\n", ["a"]),
        (b"a\nb", ["a", "b"]),
        (b"a\r\n\n\nb\n", ["a\r", "", "", "b"]),
    ],
)
def test_file_lines(tmp_path, data, lines):
    path = tmp_path / "f.txt"
    path.write_bytes(data)
    with FileSource(path) as source:
        assert len(source) == len(lines) == len(ref.split_lines(data))
        scan, batches = _encode_all(source, make_lookup(_vectors(), "external"))
    assert scan.sentences == len(lines)
    assert sum(b.n_sentences for b in batches) == len(lines)


def test_file_source_validates(tmp_path):
    with pytest.raises(ValueError, match="not a regular file"):
        FileSource(tmp_path)
    gz = tmp_path / "a.txt.gz"
    gz.write_bytes(b"x")
    with pytest.raises(ValueError, match="looks compressed"):
        FileSource(gz)
    bom = tmp_path / "bom.txt"
    bom.write_bytes(b"\xef\xbb\xbfa\n")
    with pytest.raises(ValueError, match="byte order mark"):
        FileSource(bom)
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"ok\nalso ok\nx \xff y\n")
    with (
        FileSource(bad) as source,
        pytest.raises(UnicodeDecodeError, match=r"in sentence 2 \(byte 13 of the source\)"),
    ):
        source.scan()


def test_file_source_notices_changes_and_closing(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"a b\nc\n")
    lookup = make_lookup(_vectors(), "external")
    source = FileSource(path)
    scan = source.scan()
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    with pytest.raises(RuntimeError, match="changed while it was read"):
        source.encode(scan.chunks[0], lookup, _native.Workspace())
    with pytest.raises(RuntimeError, match="changed while it was read"):
        source.scan()
    with pytest.raises(RuntimeError, match="changed while it was read"):
        len(source)
    source.close()
    with pytest.raises(ValueError, match="is closed"):
        source.scan()
    with pytest.raises(ValueError, match="is closed"):
        len(source)


def test_string_source_validates():
    with pytest.raises(TypeError, match="not a set"):
        StringSource({"a"})
    with pytest.raises(TypeError, match="list, tuple or numpy array"):
        StringSource("abc")
    with pytest.raises(TypeError, match="string 1 is a bytes"):
        StringSource(["a", b"b"])
    with pytest.raises(UnicodeEncodeError, match="in string 1"):
        StringSource(["a", "b\ud800"])
    # Strings are valid UTF-8 by construction; only a corrupted buffer fails.
    corrupted = StringSource(["a", "b"])
    corrupted._data = b"a\xff"
    with pytest.raises(UnicodeDecodeError, match="in sentence 1"):
        corrupted.scan()
    source = StringSource(["a"])
    with pytest.raises(ValueError, match="workers must be"):
        source.scan(workers=0)
    with pytest.raises(ValueError, match="chunk_bytes must be"):
        source.scan(chunk_bytes=0)
    with pytest.raises(TypeError, match="must be a Chunk"):
        source.encode((0, 1, 0, 1), make_lookup(_vectors(), "external"), _native.Workspace())


def test_a_stale_chunk_raises():
    source = StringSource(["a", "b"])
    with pytest.raises(RuntimeError, match="changed since it was scanned"):
        source.encode(Chunk(0, 1, 0, 2), make_lookup(_vectors(), "external"), _native.Workspace())


def test_empty_sources():
    for source in (StringSource([]), TokenListSource([])):
        assert len(source) == 0
        assert source.scan().sentences == 0
        assert source.scan().chunks == ()


def test_token_list_source():
    sentences = [["I", "don't"], [], ["caf\u00e9", "x"]]
    source = TokenListSource(sentences)
    lookup = make_lookup(_vectors(), "external")
    _, batches = _encode_all(source, lookup)
    want = ref.encode_texts([" ".join(s) for s in sentences], _ids(), tokenizer="external")
    tokens, offsets = _concat(batches)
    np.testing.assert_array_equal(tokens, want.tokens)
    np.testing.assert_array_equal(offsets, want.offsets)
    with pytest.raises(ValueError, match="for the external tokenizer"):
        source.encode(
            source.scan().chunks[0], make_lookup(_vectors(), "fse_words@1"), _native.Workspace()
        )
    for bad, error in [
        ([["a b"]], ValueError),
        ([[""]], ValueError),
        ([["a", 1]], TypeError),
        (["ab"], TypeError),
    ]:
        with pytest.raises(error):
            TokenListSource(bad)


def test_make_lookup_validates():
    with pytest.raises(TypeError, match="WordVectors"):
        make_lookup({"a": 0}, "external")
    with pytest.raises(ValueError, match="only to FastText"):
        make_lookup(_vectors(), "external", max_ngrams=3)
    with pytest.raises(TypeError, match="need max_ngrams"):
        make_lookup(_vectors(True), "external")
    with pytest.raises(ValueError, match="between 1 and"):
        make_lookup(_vectors(True), "external", max_ngrams=0)
    with pytest.raises(ValueError, match="not supported"):
        make_lookup(_vectors(), "nltk@1")


def test_tokenize():
    assert tokenize("Don't stop") == ["do", "not", "stop"]
    assert tokenize("Don't stop", "external") == ["Don't", "stop"]


def test_small_sources_and_one_worker_scan_without_threads(monkeypatch):
    from fse import _text

    def no_threads(**_kwargs):
        raise AssertionError("a thread pool was created")

    monkeypatch.setattr(_text, "ThreadPoolExecutor", no_threads)
    assert StringSource(["a b", "c"] * 10).scan(workers=8).sentences == 20
    big = StringSource(["word " * 40] * 6000)  # about 1.2 MB
    assert big.scan(workers=1).sentences == 6000


@pytest.mark.parametrize("workers", [2, 4])
def test_inline_and_threaded_scans_give_the_same_chunks(monkeypatch, tmp_path, workers):
    from fse import _text

    lines = [f"sentence {i} ".encode() * (i % 7) for i in range(5000)]
    good = tmp_path / "good.txt"
    good.write_bytes(b"\n".join(lines) + b"\n")
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\n".join([*lines[:3333], b"bad \xff text", *lines[3333:]]) + b"\n")
    runs, errors = {}, {}
    for name, limit in (("inline", 1 << 40), ("threads", 0)):
        monkeypatch.setattr(_text, "_SCAN_INLINE_BYTES", limit)
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
