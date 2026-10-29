from pathlib import Path

import numpy as np
import pytest

from fse._encoder._sources import opened
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
@pytest.mark.parametrize("kind", ["tokens", "texts"])
def test_the_native_join_encodes_like_the_python_path(monkeypatch, dtype, kind):
    from fse import _native
    from support.encoder import finished
    from support.runtime import texts

    sentences = [*texts(3 * 4096), "", " ", "caf\u00e9 \U0001d11e\x00"]
    if kind == "tokens":
        sentences = [s.split() for s in sentences]
    encoder = finished(tokenizer="external")
    native = encoder.encode(sentences, dtype=dtype, workers=4)
    name = "join_token_lists" if kind == "tokens" else "join_strings"
    monkeypatch.setattr(_native, name, lambda *_: None)  # every list to the Python path
    assert _bits(native) == _bits(encoder.encode(sentences, dtype=dtype, workers=4))
