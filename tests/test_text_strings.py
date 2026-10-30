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
