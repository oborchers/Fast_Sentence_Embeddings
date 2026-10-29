"""Strings and token lists joined natively (_native_lists.pxi) against the Python path."""

import sys

import numpy as np
import pytest

from fse import _native, _text
from fse._text import StringSource, TokenListSource

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


def _many(n):
    return [[f"w{i}", "\u65e5" * (i % 3 + 1)] if i % 7 else [] for i in range(n)]


@pytest.mark.parametrize("threads", [2, 3, 8])
def test_threads_join_the_same_bytes(threads):
    sentences = _many(8 * 4096 + 17)  # 8 parts of at least 4,096 sentences, and a rest
    data, offsets = _native.join_token_lists(sentences, threads)
    want_data, want_offsets = _python(sentences)
    assert data == want_data
    np.testing.assert_array_equal(offsets, want_offsets)


@pytest.mark.parametrize(("bad", "verdict"), [("a b", False), (Word("a"), None)])
def test_any_part_decides_the_verdict(bad, verdict):
    sentences = _many(4 * 4096)
    sentences[-3] = ["x", bad]  # in the last part
    assert _native.join_token_lists(sentences, 4) is verdict
    sentences[-3] = ["\ud800"]
    sentences[5] = [Word("a")]  # an unsupported type in one part wins over an invalid token
    assert _native.join_token_lists(sentences, 4) is None


def test_the_verdict_of_mixed_problems_depends_on_the_split():
    sentences = _many(2 * 4096)
    sentences[0] = [""]  # invalid in the first part
    sentences[4096] = [Word("a")]  # unsupported in the second
    assert _native.join_token_lists(sentences, 1) is False
    assert _native.join_token_lists(sentences, 2) is None
    # Either way the Python path raises the error of sentence 0.
    with pytest.raises(ValueError, match="sentence 0 holds the token ''"):
        TokenListSource(sentences)


def test_threads_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        _native.join_token_lists([["a"]], 0)


@pytest.fixture
def faults():
    yield _native._test_join_faults
    _native._test_join_faults()


def test_parts_whose_threads_do_not_start_run_in_the_caller(faults):
    sentences = _many(8 * 4096)
    want = _native.join_token_lists(sentences, 8)
    for mask in (0b10, 0b11111110, 0b1010_1010):
        faults(failed_starts=mask)
        data, offsets = _native.join_token_lists(sentences, 8)
        assert data == want[0]
        np.testing.assert_array_equal(offsets, want[1])


GROW_TOKEN_LISTS = "^could not grow the buffer of the token lists$"


@pytest.mark.parametrize("threads", [1, 4])
def test_a_failed_growth_raises_and_frees(faults, threads):
    sentences = _many(4 * 4096)
    faults(growth_limit=0)
    with pytest.raises(MemoryError, match=GROW_TOKEN_LISTS):
        _native.join_token_lists(sentences, threads)
    assert _native._test_join_freed() == 0  # nothing was allocated
    faults()
    _same(sentences)  # the next call works


# With a limit of 16 KiB a buffer grows to 4, 8 and 16 KiB, then fails: the text of a part
# of long sentences first, the ends (8 bytes per sentence) of short ones after 2,048.
@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("buffer", ["text", "ends"])
def test_a_growth_that_fails_midway_raises_and_frees_every_buffer(faults, threads, buffer):
    sentence = ["w" * 60, "\u65e5" * 40] if buffer == "text" else ["a"]  # 181 or 1 bytes
    sentences = [sentence] * 4096 * threads  # one part per thread
    faults(growth_limit=16384)
    with pytest.raises(MemoryError, match=GROW_TOKEN_LISTS):
        _native.join_token_lists(sentences, threads)
    assert _native._test_join_freed() == 2 * threads  # text and ends of every part
    faults()
    _same(sentences)


# Strings, one sentence each (join_strings), against encode_string of the Python path.

TEXT_ALPHABET = [*ALPHABET, " ", "\t", "\n", "\x85", "\xa0", "\u3000", "\u2028"]


def _same_strings(texts, threads=1):
    data, offsets = _native.join_strings(texts, threads)
    want_data, want_offsets = _text._encode_strings(texts, "string")
    assert data == want_data
    np.testing.assert_array_equal(offsets, want_offsets)
    assert offsets.dtype == np.int64


def _random_texts(rng, n):
    # By index: numpy strings would drop a trailing NUL. A fifth of them are empty.
    return [
        "".join(TEXT_ALPHABET[i] for i in rng.integers(0, len(TEXT_ALPHABET), rng.integers(0, 12)))
        for _ in range(n)
    ]


@pytest.mark.parametrize("seed", range(5))
def test_random_strings_equal_the_python_path(seed):
    texts = _random_texts(np.random.default_rng(seed), 300)
    _same_strings(texts)
    _same_strings(tuple(texts))


def test_every_code_point_is_written_as_its_utf8():
    texts = [chr(c) for c in range(sys.maxunicode + 1) if not 0xD800 <= c <= 0xDFFF]
    _same_strings(texts, 8)
    _same_strings(["".join(texts)])


def test_empty_strings():
    _same_strings([])
    _same_strings([""])
    _same_strings(["", "a", "", " ", ""])


@pytest.mark.parametrize("n", [4095, 4096, 4097, 2 * 4096 - 1, 2 * 4096, 3 * 4096 + 5])
@pytest.mark.parametrize("threads", [1, 2, 3, 8])
def test_threads_join_the_same_strings(n, threads):
    _same_strings(_random_texts(np.random.default_rng(n), n), threads)


@pytest.mark.parametrize("at", [0, 1, 4095, 4096, 3 * 4096 - 1, 4 * 4096 - 1])
@pytest.mark.parametrize("surrogate", ["\ud800", "a\udfffb", "\U0001d11e\udbff"])
def test_a_lone_surrogate_in_any_part_leaves_the_strings_to_python(at, surrogate):
    texts = [f"t{i} 日" for i in range(4 * 4096)]
    texts[at] = surrogate
    for threads in (1, 4):
        assert _native.join_strings(texts, threads) is None


class Text(str):
    pass


@pytest.mark.parametrize(
    "texts",
    [
        ["a", Text("b")],  # a subclass, which may override encode
        ["a", 1],
        ["a", b"b"],
        ["a", None],
        np.array(["a", "b"]),
        np.array(["a", "b"], dtype=object),
        iter(["a"]),
        {"a": 1},
        "ab",
    ],
)
def test_other_types_leave_the_strings_to_python(texts):
    assert _native.join_strings(texts) is None


@pytest.mark.parametrize("threads", [1, 4])
def test_a_wrong_type_in_any_part_leaves_the_strings_to_python(threads):
    texts = [f"t{i}" for i in range(4 * 4096)]
    texts[-1] = Text("x")
    assert _native.join_strings(texts, threads) is None


def test_a_str_subclass_is_encoded_by_its_own_encode():
    class Upper(str):
        def encode(self, *args):  # what the Python path calls
            return str.encode(self.upper(), *args)

    assert StringSource(["a", Upper("b")])._data == b"aB"


@pytest.mark.parametrize(
    "bad",
    [
        ["a", "b\ud800"],
        ["\udfff", 1],  # the surrogate of row 0 wins over the type of row 1
        ["a", 1, "\ud800"],
        ["a", b"b"],
        ["a", Text("\ud800")],
        np.array(["a", 1], dtype=object),
    ],
)
def test_string_errors_are_those_of_the_python_path(bad):
    with pytest.raises((TypeError, ValueError)) as want:  # UnicodeEncodeError is a ValueError
        _text._encode_strings(bad, "string")
    with pytest.raises(want.type) as got:
        StringSource(bad)
    assert str(got.value) == str(want.value)
    assert got.value.args == want.value.args


def test_threads_must_be_positive_for_strings():
    with pytest.raises(ValueError, match="at least 1"):
        _native.join_strings(["a"], 0)


def _many_texts(n):
    return [f"w{i} " + "日" * (i % 3) if i % 7 else "" for i in range(n)]


def test_string_parts_whose_threads_do_not_start_run_in_the_caller(faults):
    texts = _many_texts(8 * 4096)
    want = _native.join_strings(texts, 8)
    for mask in (0b10, 0b11111110, 0b1010_1010):
        faults(failed_starts=mask)
        data, offsets = _native.join_strings(texts, 8)
        assert data == want[0]
        np.testing.assert_array_equal(offsets, want[1])


GROW_STRINGS = "^could not grow the buffer of the strings$"


@pytest.mark.parametrize("threads", [1, 4])
def test_a_failed_growth_of_strings_raises_and_frees(faults, threads):
    texts = _many_texts(4 * 4096)
    faults(growth_limit=0)
    with pytest.raises(MemoryError, match=GROW_STRINGS):
        _native.join_strings(texts, threads)
    assert _native._test_join_freed() == 0
    faults()
    _same_strings(texts, threads)  # the next call works


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("buffer", ["text", "ends"])
def test_a_growth_of_strings_that_fails_midway_raises_and_frees(faults, threads, buffer):
    # Empty strings grow only the ends; long ones outgrow the text first.
    text = "w" * 60 + "\u65e5" * 40 if buffer == "text" else ""
    texts = [text] * 4096 * threads  # one part per thread
    faults(growth_limit=16384)
    with pytest.raises(MemoryError, match=GROW_STRINGS):
        _native.join_strings(texts, threads)
    assert _native._test_join_freed() == (2 if buffer == "text" else 1) * threads
    faults()
    _same_strings(texts, threads)
