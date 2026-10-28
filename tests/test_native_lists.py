"""Token lists joined natively (_native_lists.pxi) against the Python path of fse._text."""

import sys

import numpy as np
import pytest

from fse import _native, _text
from fse._text import TokenListSource

# Code points of every UTF-8 length and string kind: ASCII, Latin-1, BMP, astral.
ALPHABET = ["a", "Z", "0", "_", "\x00", "\xe9", "\xff", "Ā", "日", "￿", "\U0001d11e"]


def _python(sentences):
    return _text._token_lists(sentences)


def _same(sentences):
    data, offsets = _native.join_token_lists(sentences)
    want_data, want_offsets = _python(sentences)
    assert data == want_data
    np.testing.assert_array_equal(offsets, want_offsets)
    assert offsets.dtype == np.int64


def test_every_code_point_is_split_like_str_split():
    for c in range(sys.maxunicode + 1):
        if 0xD800 <= c <= 0xDFFF:
            continue
        for token in (chr(c), f"a{chr(c)}b"):
            joined = _native.join_token_lists([[token]])
            if token.split() == [token]:
                assert joined is not False, hex(c)
                assert joined[0] == token.encode(), hex(c)
            else:
                assert joined is False, hex(c)


@pytest.mark.parametrize("seed", range(5))
def test_random_corpora_equal_the_python_path(seed):
    rng = np.random.default_rng(seed)
    sentences = [
        [
            # By index: numpy strings would drop a trailing NUL.
            "".join(ALPHABET[i] for i in rng.integers(0, len(ALPHABET), rng.integers(1, 6)))
            for _ in range(rng.integers(0, 8))
        ]
        for _ in range(300)
    ]
    _same(sentences)
    _same(tuple(tuple(s) for s in sentences))


def test_empty_inputs():
    _same([])
    _same([[], [], ["a"], []])


@pytest.mark.parametrize(
    "bad",
    [
        [["a"], "ab"],  # a sentence that is not a list
        [["a", 1]],  # a token that is not a str
        [["a"], [""]],  # an empty token
        [["a\x85b"]],  # U+0085 is whitespace for str.split()
        [["a　b"]],
        [["\x1c"]],
        [["a\ud800"]],  # a lone surrogate, which UTF-8 cannot hold
        [["\udfff"], ["a b"]],  # a later whitespace error wins over an earlier surrogate
        [["\ud800"], [1]],
    ],
)
def test_errors_are_those_of_the_python_path(bad):
    with pytest.raises((TypeError, ValueError)) as want:  # UnicodeEncodeError is a ValueError
        _python(bad)
    with pytest.raises(want.type) as got:
        TokenListSource(bad)
    assert str(got.value) == str(want.value)


class Word(str):
    pass


class Sentence(list):
    pass


def test_other_types_take_the_python_path():
    sentences = [["a", Word("b")], Sentence(["c"]), ("d",)]
    assert _native.join_token_lists(sentences) is None
    assert _native.join_token_lists(np.array([["a"], ["b", "c"]], dtype=object)) is None
    source = TokenListSource(sentences)
    data, offsets = _python([["a", "b"], ["c"], ["d"]])
    assert source._data == data
    np.testing.assert_array_equal(source._offsets, offsets)


def test_a_str_subclass_is_never_read_natively():
    class Spaced(str):
        def split(self):  # what the Python path calls
            return [str(self)]

    # The native pass would reject the space; the subclass decides through its own split.
    assert _native.join_token_lists([[Spaced("a b")]]) is None
    assert TokenListSource([[Spaced("a b")]])._data == b"a b"
