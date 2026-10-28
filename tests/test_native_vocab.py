"""Tests of the vocabulary table of fse._native, VocabTable."""

import numpy as np
import pytest

from fse import _native

F32 = np.float32


def _vocab(words):
    from fse._wordvectors import WordVectors

    wv = WordVectors.from_arrays(np.zeros((len(words), 2), F32), words=words)
    return wv.vocab_bytes, wv.vocab_offsets


def test_vocab_table_finds_every_word():
    words = ["a", "ab", "café", "x" * 16, "y" * 17, "中国" * 5, "\U0001f600"]
    table = _native.VocabTable(*_vocab(words))
    assert len(table) == len(words)
    for i, word in enumerate(words):
        assert table.get(word) == i
        assert table.get(word.encode()) == i
    for missing in ["", "A", "abc", "x" * 15, "y" * 18, "caf"]:
        assert table.get(missing) is None


def test_vocab_table_on_many_words():
    words = [f"w{i}" for i in range(50_000)]
    table = _native.VocabTable(*_vocab(words))
    rng = np.random.default_rng(0)
    for i in rng.integers(0, len(words), 1000):
        assert table.get(words[i]) == i
    assert table.get("w50000") is None


def test_vocab_table_rejects_duplicates():
    data = np.frombuffer(b"abcab", np.uint8)
    with pytest.raises(ValueError, match="word 2 is also word 0"):
        _native.VocabTable(data, np.array([0, 2, 3, 5], np.int64))
    long = np.frombuffer(b"z" * 40, np.uint8)
    with pytest.raises(ValueError, match="word 1 is also word 0"):
        _native.VocabTable(long, np.array([0, 20, 40], np.int64))


@pytest.mark.parametrize(
    ("data", "offsets", "match"),
    [
        (b"ab", [0, 1], "run from 0 to 2"),
        (b"ab", [1, 2], "run from 0 to 2"),
        (b"ab", [], "run from 0 to 2"),
        (b"abc", [0, 2, 1, 3], "must not decrease"),
    ],
)
def test_vocab_table_rejects_bad_offsets(data, offsets, match):
    with pytest.raises(ValueError, match=match):
        _native.VocabTable(np.frombuffer(data, np.uint8), np.array(offsets, np.int64))


def test_vocab_table_checks_its_arrays():
    with pytest.raises(ValueError, match="uint8"):
        _native.VocabTable(np.zeros(2, np.int8), np.array([0, 2], np.int64))
    with pytest.raises(ValueError, match="int64"):
        _native.VocabTable(np.zeros(2, np.uint8), np.array([0, 2], np.int32))
    empty = _native.VocabTable(np.zeros(0, np.uint8), np.zeros(1, np.int64))
    assert len(empty) == 0
    assert empty.get("a") is None


def test_vocab_table_reads_mapped_vectors(tmp_path):
    from fse._wordvectors import WordVectors

    words = ["short", "a much longer word than sixteen bytes"]
    WordVectors.from_arrays(np.ones((2, 3), F32), words=words).save(tmp_path / "wv.safetensors")
    wv = WordVectors.load(tmp_path / "wv.safetensors", mmap=True)
    table = _native.VocabTable(wv.vocab_bytes, wv.vocab_offsets)
    assert [table.get(w) for w in words] == [0, 1]
