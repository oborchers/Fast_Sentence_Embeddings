"""fse.Corpus: texts or token lists in memory, read once, and the rules of in-memory input."""

from pathlib import Path

import numpy as np
import pytest

import fse
from fse._encoder._sources import opened
from fse._text import StringSource, TokenListSource


def test_the_first_row_decides_the_kind():
    texts = fse.Corpus(["the cat", "a dog"])
    tokens = fse.Corpus([["the", "cat"], ("a", "dog")])
    assert isinstance(texts._source, StringSource)
    assert isinstance(tokens._source, TokenListSource)
    assert repr(texts) == "Corpus(2 texts, 12 bytes)"
    assert repr(tokens) == "Corpus(2 token lists, 12 bytes)"
    assert len(texts) == len(tokens) == 2
    assert len(fse.Corpus(np.array(["a", "b"]))) == 2


def test_empty_corpora():
    with opened(fse.Corpus([]), "fse_words@1") as source:  # no kind: every tokenizer
        assert len(source) == 0
    one_empty = fse.Corpus([[]])  # one text without tokens
    assert len(one_empty) == 1
    with (
        pytest.raises(TypeError, match="for the external tokenizer"),
        opened(one_empty, "fse_words@1"),
    ):
        pass


@pytest.mark.parametrize(
    ("texts", "match"),
    [
        (["a", ["b"]], "row 1 is a list, but row 0 is a str"),
        ([["a"], "b"], "row 1 is a str, but row 0 is a list"),
        ([("a",), ["b"], "c"], "row 2 is a str, but row 0 is a tuple"),
    ],
)
def test_mixed_rows_raise_naming_both(texts, match):
    for make in (fse.Corpus, lambda t: opened(t, "external").__enter__()):
        with pytest.raises(TypeError, match=match):
            make(texts)


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
    for make in (fse.Corpus, lambda t: opened(t, "external").__enter__()):
        with pytest.raises(error, match=match):
            make(texts)


@pytest.mark.parametrize(
    ("texts", "match"),
    [
        ("one text", "not one str"),
        (b"bytes", "not bytes"),
        ({"a"}, "ordered"),
        (Path("lines.txt"), "pass a path to fit or encode"),
        (iter(["a"]), "pass a stream to fit or encode directly"),
        (np.array([["a", "b"]]), "one-dimensional, not 2-D: pass arr.tolist()"),
        (np.array("a"), "one-dimensional, not 0-D$"),
    ],
)
def test_a_corpus_refuses_what_is_no_collection_of_texts(texts, match):
    with pytest.raises(TypeError, match=match):
        fse.Corpus(texts)


def test_paths_and_streams_still_go_to_fit_and_encode(tmp_path):
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
    for make in (fse.Corpus, lambda t: opened(t, "external").__enter__()):
        with pytest.raises(TypeError, match="broken encode"):
            make([BrokenStr("a"), ["b"]])


def test_a_corpus_does_not_change():
    texts = ["a", "b"]
    corpus = fse.Corpus(texts)
    texts.append("c")
    assert len(corpus) == 2
    with pytest.raises(AttributeError, match="does not change"):
        corpus._source = None
    with pytest.raises(AttributeError, match="does not change"):
        del corpus._source
    with pytest.raises(AttributeError, match="does not change"):
        corpus.__init__(["other"])
    assert len(corpus) == 2
    again = fse.Corpus(corpus)
    assert again._source is corpus._source  # shares the built source


class BrokenSplit(str):
    def split(self):  # what the Python path of token lists calls
        raise TypeError("broken split")


class ExplodingList(list):
    def __iter__(self):
        raise RuntimeError("a later row was read")


def test_no_later_row_is_read_after_a_type_error():
    texts = [[BrokenSplit("a")], ExplodingList(["b"])]
    for make in (fse.Corpus, lambda t: opened(t, "external").__enter__()):
        with pytest.raises(TypeError, match="broken split"):
            make(texts)


class GuardedRows(list):
    def __getitem__(self, i):
        if i == 1:
            raise RuntimeError("row 1 was read by index")
        return super().__getitem__(i)


def test_the_mixed_rows_error_reads_nothing_again():
    for make in (fse.Corpus, lambda t: opened(t, "external").__enter__()):
        with pytest.raises(TypeError, match="row 1 is a list, but row 0 is a str"):
            make(GuardedRows(["a", ["b"]]))


def test_other_row_type_errors_stay_plain_type_errors():
    with pytest.raises(TypeError) as raised:
        fse.Corpus(["a", 1])
    assert type(raised.value) is TypeError
    assert str(raised.value) == "string 1 is a int, not a str"
