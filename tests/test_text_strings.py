"""Tests of fse._text._strings: str and token lists joined into one buffer."""

import numpy as np
import pytest

from fse import _native, _text
from fse import _reference_text as ref
from fse._text import (
    StringSource,
    TokenListSource,
    make_lookup,
)
from support.text import concat, encode_all, ids, vectors

F32 = np.float32


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


def test_token_lists_are_joined_natively(monkeypatch):
    def python_path(_sentences):
        raise AssertionError("valid lists of str took the Python path")

    monkeypatch.setattr(_text._strings, "_token_lists", python_path)
    source = TokenListSource([["a", "b\u00e9"], ("c",)])
    assert source._data == "a b\u00e9c".encode()
    np.testing.assert_array_equal(source._offsets, [0, 5, 6])


def test_strings_are_joined_natively(monkeypatch):
    def python_path(_texts, _what):
        raise AssertionError("valid lists of str took the Python path")

    monkeypatch.setattr(_text._strings, "_encode_strings", python_path)
    for texts in (["a b", "", "caf\u00e9"], ("a b", "", "caf\u00e9")):
        source = StringSource(texts)
        assert source._data == "a bcaf\u00e9".encode()
        np.testing.assert_array_equal(source._offsets, [0, 3, 3, 8])


def test_strings_the_native_pass_leaves_take_the_python_path():
    for texts in (np.array(["a b", "c"]), ["a b", type("Text", (str,), {})("c")]):
        assert _native.join_strings(texts) is None
        source = StringSource(texts)
        assert source._data == b"a bc"
        np.testing.assert_array_equal(source._offsets, [0, 3, 4])


def test_the_python_path_decides_an_invalid_verdict():
    sentence = []

    class Repair(str):
        def split(self):  # called by the Python path, after the native one stopped
            sentence[:] = ["ok"]
            return ["ok"]

    sentence[:] = ["\ud800", Repair("ok")]
    assert _native.join_token_lists([sentence]) is False
    source = TokenListSource([sentence])  # the Python path reads the repaired sentence
    assert source._data == b"ok"
    np.testing.assert_array_equal(source._offsets, [0, 2])


def test_bytes_fse_did_not_write_are_scanned():
    class Raw(str):
        def encode(self, *_args):  # the Python path takes these bytes as they are
            return b"ok \xff"

    source = StringSource(["fine", Raw("x")])
    with pytest.raises(UnicodeDecodeError, match=r"in sentence 1 \(byte 7 of the source\)"):
        source.scan()
    encoded = StringSource.encoded([b"fine", b"ok \xff"])
    with pytest.raises(UnicodeDecodeError, match=r"in sentence 1 \(byte 7 of the source\)"):
        encoded.scan()


def test_only_buffers_of_the_native_join_are_verified():
    class Own(str):
        pass

    assert StringSource(["a b", "café"])._verified
    assert TokenListSource([["a", "b"], ["c"]])._verified
    assert not StringSource(["a b", Own("c")])._verified  # a subclass: the Python path
    assert not StringSource(np.array(["a b", "c"]))._verified
    assert not StringSource.encoded([b"a b", b"c"])._verified
    assert not TokenListSource([["a", Own("b")]])._verified
    with pytest.raises(UnicodeEncodeError, match="in string 1"):
        StringSource(["a", "\ud800"])


def test_verified_buffers_are_not_scanned_again(monkeypatch):
    texts = [f"sentence {i} " * (i % 7) for i in range(3000)]
    want = {}
    for kind in ("verified", "forced"):
        source = StringSource(texts)
        if kind == "forced":
            source._verified = False  # the scan reads the read-only offsets as they are
        want[kind] = source.scan(workers=4, chunk_bytes=4096)
    assert want["verified"] == want["forced"]

    def fails(*_args):
        raise AssertionError("a verified buffer was scanned")

    monkeypatch.setattr(_text._scan._native, "scan_spans", fails)
    assert StringSource(texts).scan(workers=4, chunk_bytes=4096) == want["verified"]
    assert TokenListSource([t.split() for t in texts]).scan().sentences == 3000
    with pytest.raises(AssertionError, match="was scanned"):
        StringSource.encoded([t.encode() for t in texts]).scan()


def test_the_offsets_of_a_verified_buffer_are_read_only():
    source = StringSource(["a b", "c"])
    with pytest.raises(ValueError, match="read-only"):
        source._offsets[1] = 0
