from pathlib import Path

import numpy as np
import pytest

from fse._encoder._sources import is_stream, opened
from fse._stream import StreamSource
from fse._text import StringSource, TokenLists
from fse._text_file import FileSource


def test_each_input_becomes_its_source(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("a b\nc\n")
    cases = [
        (["a b", "c"], "fse_words@1", StringSource),
        (("a b",), "fse_words@1", StringSource),
        (np.array(["a b", "c"]), "fse_words@1", StringSource),
        ([], "fse_words@1", StringSource),
        ([["a", "b"], ["c"]], "external", TokenLists),
        ((s for s in ["a"]), "fse_words@1", StreamSource),
        (iter(["a"]), "fse_words@1", StreamSource),
    ]
    for texts, tokenizer, kind in cases:
        with opened(texts, tokenizer) as source:
            assert type(source) is kind
    given = StringSource(["x"])
    with opened(given, "fse_words@1") as source:
        assert source is given
    assert is_stream(StreamSource([]))
    assert not is_stream(given)


def test_a_path_is_opened_and_closed_by_the_call(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("a b\nc\n")
    with opened(Path(path), "fse_words@1") as source:
        assert isinstance(source, FileSource)
        assert len(source) == 2
    with pytest.raises(ValueError, match="is closed"):
        source.scan()


@pytest.mark.parametrize(
    ("texts", "match"),
    [
        ("one sentence", "not one str"),
        (b"bytes", "not bytes"),
        (memoryview(b"x"), "not memoryview"),
        ({"a", "b"}, "ordered"),
        ({"a": 1}, "ordered"),
        (5, "iterable"),
    ],
)
def test_inputs_that_are_no_sentences_raise_before_reading(texts, match):
    with pytest.raises(TypeError, match=match), opened(texts, "fse_words@1"):
        pass


def test_token_lists_need_the_external_tokenizer():
    with (
        pytest.raises(TypeError, match="for the external tokenizer"),
        opened([["a"]], "fse_words@1"),
    ):
        pass


def test_given_token_lists_need_the_external_tokenizer():
    given = TokenLists([["a"]])
    with opened(given, "external") as source:
        assert source is given
    with (
        pytest.raises(TypeError, match="for the external tokenizer"),
        opened(given, "fse_words@1"),
    ):
        pass


def test_token_lists_are_read_once_for_fit_and_encode(monkeypatch):
    import fse
    from fse import _native
    from support.runtime import texts, word_vectors

    sentences = [s.split() for s in texts(600)]
    wv = word_vectors()
    counts = np.arange(1, len(wv) + 1)
    recipe = fse.SIF(tokenizer="external", components=2)
    want_encoder = recipe.fit(wv, sentences, counts=counts)
    want = want_encoder.encode(sentences, workers=2)
    joins = []
    real = _native.join_token_lists
    monkeypatch.setattr(_native, "join_token_lists", lambda *a: joins.append(1) or real(*a))
    given = fse.TokenLists(sentences)
    encoder = recipe.fit(wv, given, counts=counts, workers=2)
    rows = encoder.encode(given, workers=2)
    assert len(joins) == 1
    assert encoder.fingerprint == want_encoder.fingerprint
    np.testing.assert_array_equal(rows.view(np.uint32), want.view(np.uint32))
    _, fitted = recipe.fit_transform(wv, given, counts=counts)
    np.testing.assert_array_equal(fitted.view(np.uint32), want.view(np.uint32))
    with pytest.raises(TypeError, match="for the external tokenizer"):
        fse.SIF().fit(wv, given, counts=counts)
