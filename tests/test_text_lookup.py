"""Tests of fse._text._lookup: the lookup of stage 2, tokenize and the Vocabulary."""

import numpy as np
import pytest

from fse import _native
from fse._text import (
    StringSource,
    make_lookup,
    make_vocabulary,
    tokenize,
)
from fse._wordvectors import WordVectors
from support.text import TEXTS, WORDS, encode_all, vectors

F32 = np.float32


def test_make_lookup_validates():
    with pytest.raises(TypeError, match="WordVectors"):
        make_lookup({"a": 0}, "external")
    with pytest.raises(ValueError, match="only to FastText"):
        make_lookup(vectors(), "external", max_ngrams=3)
    with pytest.raises(TypeError, match="need max_ngrams"):
        make_lookup(vectors(True), "external")
    with pytest.raises(ValueError, match="between 1 and"):
        make_lookup(vectors(True), "external", max_ngrams=0)
    with pytest.raises(ValueError, match="not supported"):
        make_lookup(vectors(), "nltk@1")


def test_tokenize():
    assert tokenize("Don't stop") == ["do", "not", "stop"]
    assert tokenize("Don't stop", "external") == ["Don't", "stop"]


def test_a_vocabulary_is_built_once_and_reused(monkeypatch):
    built = []
    real = _native.VocabTable

    def counted(*args):
        built.append(1)
        return real(*args)

    wv = vectors()
    vocabulary = make_vocabulary(wv)
    monkeypatch.setattr(_native, "VocabTable", counted)
    first = make_lookup(wv, "fse_words@1", vocabulary=vocabulary)
    second = make_lookup(wv.quantize("int8"), "external", vocabulary=vocabulary)
    assert built == []
    source = StringSource(TEXTS)
    want = encode_all(source, make_lookup(wv, "fse_words@1"))[1]
    got = encode_all(source, first)[1]
    assert [b.tokens.tolist() for b in got] == [b.tokens.tolist() for b in want]
    assert second.tokenizer != first.tokenizer
    assert built == [1]  # only the lookup made without the vocabulary


def test_a_vocabulary_of_other_words_or_order_is_refused():
    wv = vectors()
    swapped = WordVectors.from_arrays(np.ones((len(WORDS), 3), F32), words=WORDS[::-1])
    assert swapped.vocabulary_fingerprint != wv.vocabulary_fingerprint
    with pytest.raises(ValueError, match="is not the one of the word vectors"):
        make_lookup(swapped, "fse_words@1", vocabulary=make_vocabulary(wv))
    with pytest.raises(TypeError, match="vocabulary must be a Vocabulary"):
        make_lookup(wv, "fse_words@1", vocabulary=object())
    with pytest.raises(TypeError, match="word_vectors must be WordVectors"):
        make_vocabulary(object())


def test_a_vocabulary_is_built_from_checked_words(tmp_path):
    """The table and its fingerprint come from one copy of the words, checked against the
    checksums they were loaded with."""
    from fse._container import CorruptArtifactError

    path = tmp_path / "wv.safetensors"
    vectors().save(path)
    data = bytearray(path.read_bytes())
    at = data.rindex(b"istanbul")  # in vocab.bytes, the last tensor of the file
    data[at] = ord("j")
    path.write_bytes(bytes(data))
    mapped = WordVectors.load(path, mmap=True)
    with pytest.raises(CorruptArtifactError, match=r"vocab\.bytes .* does not match its checksum"):
        make_vocabulary(mapped)
