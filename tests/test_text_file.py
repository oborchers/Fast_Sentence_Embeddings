import os

import numpy as np
import pytest

from fse import _native
from fse import _reference_text as ref
from fse._text import make_lookup
from fse._text_file import FileSource
from support.text import TEXTS, concat, encode_all, ids, vectors


def test_file_source_reads_lines(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_bytes("\n".join(TEXTS).encode() + b"\n")
    with FileSource(path) as source:
        assert len(source) == len(TEXTS)
        scan, batches = encode_all(
            source,
            make_lookup(vectors(True), "fse_words@1", max_ngrams=4),
            workers=3,
            chunk_bytes=5,
        )
    want = ref.encode_texts(
        TEXTS, ids(), tokenizer="fse_words@1", fasttext=(3, 5, 50), max_ngrams=4
    )
    assert scan.sentences == len(TEXTS)
    tokens, offsets = concat(batches)
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
        scan, batches = encode_all(source, make_lookup(vectors(), "external"))
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
    lookup = make_lookup(vectors(), "external")
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
