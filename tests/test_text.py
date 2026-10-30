import numpy as np
import pytest

from fse import _native
from fse import _reference_text as ref
from fse._text import (
    Chunk,
    StringSource,
    TokenListSource,
    make_lookup,
)
from support.text import TEXTS, concat, encode_all, ids, vectors

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
    # Strings are valid UTF-8 by construction; only a corrupted buffer fails, which is
    # no longer the buffer of the join, so it is scanned like any other.
    corrupted = StringSource(["a", "b"])
    corrupted._data, corrupted._verified = b"a\xff", False
    with pytest.raises(UnicodeDecodeError, match="in sentence 1"):
        corrupted.scan()
    # Without the scan the tokenizer still finds it.
    corrupted._verified = True
    chunk = corrupted.scan().chunks[0]
    with pytest.raises(UnicodeDecodeError, match="in sentence 1"):
        corrupted.encode(chunk, make_lookup(vectors(), "external"), _native.Workspace())
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
