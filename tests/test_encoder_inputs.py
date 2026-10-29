"""The rules of in-memory input: texts or token lists, the first row decides."""

import numpy as np
import pytest

from fse._encoder._sources import opened
from fse._text import StringSource, TokenListSource


def _source(texts, tokenizer="external"):
    with opened(texts, tokenizer) as source:
        return source


def test_the_first_row_decides_the_kind():
    assert isinstance(_source(["the cat", "a dog"]), StringSource)
    assert isinstance(_source([["the", "cat"], ("a", "dog")]), TokenListSource)
    assert isinstance(_source(np.array(["a", "b"])), StringSource)
    assert len(_source(np.array(["a", "b"]))) == 2


def test_empty_inputs():
    assert len(_source([], "fse_words@1")) == 0  # no kind: every tokenizer
    assert len(_source([[]])) == 1  # one text without tokens
    with pytest.raises(TypeError, match="for the external tokenizer"):
        _source([[]], "fse_words@1")
    assert len(_source(["a b"], "fse_words@1")) == 1  # texts take every tokenizer


@pytest.mark.parametrize(
    ("texts", "match"),
    [
        (["a", ["b"]], "row 1 is a list, but row 0 is a str"),
        ([["a"], "b"], "row 1 is a str, but row 0 is a list"),
        ([("a",), ["b"], "c"], "row 2 is a str, but row 0 is a tuple"),
    ],
)
def test_mixed_rows_raise_naming_both(texts, match):
    with pytest.raises(TypeError, match=match):
        _source(texts)


@pytest.mark.parametrize(
    ("texts", "error", "match"),
    [
        (["\ud800", ["a"]], UnicodeEncodeError, "string 0"),  # the UTF-8 of row 0 first
        ([[1], "a"], TypeError, "sentence 0 holds a int"),  # the bad token of row 0 first
        ([["a b"], "c"], ValueError, "sentence 0 holds the token 'a b'"),
        (["a", 2], TypeError, "string 1 is a int"),  # a wrong type that is no other kind
    ],
)
def test_the_first_error_of_a_row_wins_over_mixed_rows(texts, error, match):
    with pytest.raises(error, match=match):
        _source(texts)


@pytest.mark.parametrize(
    ("texts", "match"),
    [
        ("one text", "not one str"),
        (b"bytes", "not bytes"),
        ({"a"}, "ordered"),
        (np.array([["a", "b"]]), "one-dimensional, not 2-D: pass arr.tolist()"),
        (np.array("a"), "one-dimensional, not 0-D$"),
    ],
)
def test_encode_refuses_what_is_no_collection_of_texts(texts, match):
    with pytest.raises(TypeError, match=match):
        _source(texts)


def test_paths_and_streams_go_to_encode_directly(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("a b\nc\n")
    with opened(path, "fse_words@1") as source:
        assert len(source) == 2
    with opened(iter(["a"]), "fse_words@1") as source:
        assert type(source).__name__ == "StreamSource"
    with pytest.raises(TypeError, match="not 2-D"), opened(np.array([["a"]]), "external"):
        pass


class BrokenStr(str):
    def encode(self, encoding="utf-8", errors="strict"):
        raise TypeError(f"broken encode ({encoding}, {errors})")


def test_a_type_error_of_a_str_subclass_is_not_a_mixed_row():
    with pytest.raises(TypeError, match="broken encode"):
        _source([BrokenStr("a"), ["b"]])


class BrokenSplit(str):
    def split(self):  # what the Python path of token lists calls
        raise TypeError("broken split")


class ExplodingList(list):
    def __iter__(self):
        raise RuntimeError("a later row was read")


def test_no_later_row_is_read_after_a_type_error():
    texts = [[BrokenSplit("a")], ExplodingList(["b"])]
    with pytest.raises(TypeError, match="broken split"):
        _source(texts)


class GuardedRows(list):
    def __getitem__(self, i):
        if i == 1:
            raise RuntimeError("row 1 was read by index")
        return super().__getitem__(i)


def test_the_mixed_rows_error_reads_nothing_again():
    with pytest.raises(TypeError, match="row 1 is a list, but row 0 is a str"):
        _source(GuardedRows(["a", ["b"]]))


def test_other_row_type_errors_stay_plain_type_errors():
    with pytest.raises(TypeError) as raised:
        _source(["a", 1])
    assert type(raised.value) is TypeError
    assert str(raised.value) == "string 1 is a int, not a str"
