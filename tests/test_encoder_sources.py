from pathlib import Path

import numpy as np
import pytest

from fse._encoder._sources import is_stream, opened
from fse._stream import StreamSource
from fse._text import StringSource, TokenListSource
from fse._text_file import FileSource


def test_each_input_becomes_its_source(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("a b\nc\n")
    cases = [
        (["a b", "c"], "fse_words@1", StringSource),
        (("a b",), "fse_words@1", StringSource),
        (np.array(["a b", "c"]), "fse_words@1", StringSource),
        ([], "fse_words@1", StringSource),
        ([["a", "b"], ["c"]], "external", TokenListSource),
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


def test_a_corpus_of_token_lists_needs_the_external_tokenizer():
    import fse

    corpus = fse.Corpus([["a"]])
    with opened(corpus, "external") as source:
        assert source is corpus._source
    with (
        pytest.raises(TypeError, match="for the external tokenizer"),
        opened(corpus, "fse_words@1"),
    ):
        pass
    with opened(fse.Corpus(["a b"]), "fse_words@1") as source:  # texts take every tokenizer
        assert isinstance(source, StringSource)


def test_the_tokenizer_is_checked_before_the_rows_of_a_list():
    with (
        pytest.raises(TypeError, match="for the external tokenizer"),
        opened([[""]], "fse_words@1"),
    ):
        pass


def _bits(rows):
    if isinstance(rows, tuple):  # int8: codes and scales
        return rows[0].tobytes() + rows[1].tobytes()
    return rows.tobytes()


@pytest.mark.parametrize("dtype", ["float32", "int8"])
@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("kind", ["tokens", "texts"])
def test_a_corpus_is_read_once_for_fit_and_encode(monkeypatch, dtype, fold, kind):
    import fse
    from fse import _text
    from support.runtime import texts, word_vectors

    sentences = texts(600)
    if kind == "tokens":
        sentences = [s.split() for s in sentences]
    wv = word_vectors()
    counts = np.arange(1, len(wv) + 1)
    recipe = fse.SIF(tokenizer="external", components=2, fold=fold)
    want_encoder = recipe.fit(wv, sentences, counts=counts)
    want = want_encoder.encode(sentences, dtype=dtype, workers=2)
    builds = []
    source = _text.TokenListSource if kind == "tokens" else _text.StringSource
    real = source.__init__
    monkeypatch.setattr(source, "__init__", lambda self, *a: builds.append(1) or real(self, *a))
    corpus = fse.Corpus(sentences)
    encoder = recipe.fit(wv, corpus, counts=counts, workers=2)
    rows = encoder.encode(corpus, dtype=dtype, workers=2)
    assert len(builds) == 1
    assert encoder.fingerprint == want_encoder.fingerprint
    assert _bits(rows) == _bits(want)
