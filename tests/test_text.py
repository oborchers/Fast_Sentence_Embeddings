import numpy as np
import pytest

from fse import _native, _text
from fse import _reference_text as ref
from fse._text import (
    Chunk,
    StringSource,
    TokenListSource,
    make_lookup,
    make_vocabulary,
    tokenize,
)
from fse._text_file import FileSource
from fse._wordvectors import WordVectors
from support.text import TEXTS, WORDS, concat, encode_all, ids, vectors

F32 = np.float32


@pytest.mark.parametrize("chunk_bytes", [1, 7, None])
def test_string_source_gives_the_reference_batches(chunk_bytes):
    source = StringSource(TEXTS)
    assert len(source) == len(TEXTS)
    scan, batches = encode_all(
        source, make_lookup(vectors(), "fse_words@1"), chunk_bytes=chunk_bytes
    )
    assert scan.sentences == len(TEXTS)
    rows = [row for c in scan.chunks for row in range(c.first_row, c.first_row + c.n_sentences)]
    assert rows == list(range(len(TEXTS)))
    want = ref.encode_texts(TEXTS, ids(), tokenizer="fse_words@1")
    tokens, offsets = concat(batches)
    np.testing.assert_array_equal(tokens, want.tokens)
    np.testing.assert_array_equal(offsets, want.offsets)
    for chunk, batch in zip(scan.chunks, batches, strict=True):
        assert batch.first_row == chunk.first_row


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
        source.encode((0, 1, 0, 1), make_lookup(vectors(), "external"), _native.Workspace())


def test_a_stale_chunk_raises():
    source = StringSource(["a", "b"])
    with pytest.raises(RuntimeError, match="changed since it was scanned"):
        source.encode(Chunk(0, 1, 0, 2), make_lookup(vectors(), "external"), _native.Workspace())


def test_empty_sources():
    for source in (StringSource([]), TokenListSource([])):
        assert len(source) == 0
        assert source.scan().sentences == 0
        assert source.scan().chunks == ()


def test_token_list_source():
    sentences = [["I", "don't"], [], ["caf\u00e9", "x"]]
    source = TokenListSource(sentences)
    lookup = make_lookup(vectors(), "external")
    _, batches = encode_all(source, lookup)
    want = ref.encode_texts([" ".join(s) for s in sentences], ids(), tokenizer="external")
    tokens, offsets = concat(batches)
    np.testing.assert_array_equal(tokens, want.tokens)
    np.testing.assert_array_equal(offsets, want.offsets)
    with pytest.raises(ValueError, match="for the external tokenizer"):
        source.encode(
            source.scan().chunks[0], make_lookup(vectors(), "fse_words@1"), _native.Workspace()
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
        make_lookup(vectors(), "external", max_ngrams=3)
    with pytest.raises(TypeError, match="need max_ngrams"):
        make_lookup(vectors(True), "external")
    with pytest.raises(ValueError, match="between 1 and"):
        make_lookup(vectors(True), "external", max_ngrams=0)
    with pytest.raises(ValueError, match="not supported"):
        make_lookup(vectors(), "nltk@1")


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


def test_full_chunks_of_neighbour_ranges_stay_apart(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_bytes((b"x" * 65_535 + b"\n") * 32)  # 32 lines of exactly one chunk each
    with FileSource(path) as source:
        scan = source.scan(workers=8, chunk_bytes=65_536)
    assert len(scan.chunks) == 32


def test_a_vocabulary_is_built_once_and_reused(monkeypatch):
    built = []
    real = _native.VocabTable

    def counted(*args):
        built.append(1)
        return real(*args)

    wv = vectors()
    vocabulary = make_vocabulary(wv)
    monkeypatch.setattr(_text._native, "VocabTable", counted)
    first = make_lookup(wv, "fse_words@1", vocabulary=vocabulary)
    second = make_lookup(wv.quantize("int8"), "external", vocabulary=vocabulary)
    assert built == []
    source = StringSource(TEXTS)
    want = encode_all(source, make_lookup(wv, "fse_words@1"))[1]
    got = encode_all(source, first)[1]
    assert [b.tokens.tolist() for b in got] == [b.tokens.tolist() for b in want]
    assert second.tokenizer != first.tokenizer
    assert built == [1]  # only the lookup made without the vocabulary


def test_a_vocabulary_of_other_words_or_order_is_refused():
    wv = vectors()
    swapped = WordVectors.from_arrays(np.ones((len(WORDS), 3), F32), words=WORDS[::-1])
    assert swapped.vocabulary_fingerprint != wv.vocabulary_fingerprint
    with pytest.raises(ValueError, match="is not the one of the word vectors"):
        make_lookup(swapped, "fse_words@1", vocabulary=make_vocabulary(wv))
    with pytest.raises(TypeError, match="vocabulary must be a Vocabulary"):
        make_lookup(wv, "fse_words@1", vocabulary=object())
    with pytest.raises(TypeError, match="word_vectors must be WordVectors"):
        make_vocabulary(object())


def test_a_vocabulary_is_built_from_checked_words(tmp_path):
    """The table and its fingerprint come from one copy of the words, checked against the
    checksums they were loaded with."""
    from fse._container import CorruptArtifactError

    path = tmp_path / "wv.safetensors"
    vectors().save(path)
    data = bytearray(path.read_bytes())
    at = data.rindex(b"istanbul")  # in vocab.bytes, the last tensor of the file
    data[at] = ord("j")
    path.write_bytes(bytes(data))
    mapped = WordVectors.load(path, mmap=True)
    with pytest.raises(CorruptArtifactError, match=r"vocab\.bytes .* does not match its checksum"):
        make_vocabulary(mapped)
