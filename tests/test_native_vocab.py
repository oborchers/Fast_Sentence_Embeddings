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


def _table(words: list[bytes]):
    offsets = np.zeros(len(words) + 1, np.int64)
    np.cumsum([len(w) for w in words], out=offsets[1:])
    data = np.frombuffer(b"".join(words), np.uint8) if words else np.zeros(0, np.uint8)
    return _native.VocabTable(data, offsets)


def test_vocab_table_is_exact_for_every_length_and_byte():
    rng = np.random.default_rng(0)
    words = list(
        dict.fromkeys(
            rng.integers(0, 256, n, dtype=np.uint8).tobytes() for n in range(41) for _ in range(30)
        )
    )
    table = _table(words)
    index = {w: i for i, w in enumerate(words)}
    for word in words:
        assert table.get(word) == index[word]
        for changed in (
            word + b"\0",
            word[:-1],
            word[1:],
            b"\0" + word,
            word[:-1] + bytes([(word[-1:] or b"\0")[0] ^ 1]),
        ):
            assert table.get(changed) == index.get(changed)


def test_the_padded_and_the_unpadded_lookup_agree():
    # fse_words@1 tokens are read 16 bytes wide from the candidate buffer, external ones
    # only within the input: both must find the same words at every length.
    words = ["abcdefghijklmnopqrstuvwxyzabcdefghijklmn"[:n] for n in range(1, 41)]
    table = _table([w.encode() for w in words])
    data = np.frombuffer(" ".join(words).encode(), np.uint8)
    ids = []
    for tokenizer in ("fse_words@1", "external"):
        lookup = _native.TextLookup(table, tokenizer)
        batch = lookup.encode(
            data, np.array([0, len(data)], np.int64), 0, 1, 0, _native.Workspace()
        )
        ids.append(batch.tokens.tolist())
    assert ids[0] == ids[1] == list(range(len(words)))


def test_vocab_table_nbytes():
    words = [f"w{i}".encode() for i in range(1000)]
    table = _table(words)
    assert table.nbytes == 2048 * 32 + sum(map(len, words))


def test_vocab_table_is_built_once():
    table = _table([b"a", b"b"])
    with pytest.raises(RuntimeError, match="built once"):
        table.__init__(np.zeros(0, np.uint8), np.zeros(1, np.int64))
    assert table.get("b") == 1


def test_an_unbuilt_or_failed_vocab_table_is_never_read():
    unbuilt = _native.VocabTable.__new__(_native.VocabTable)
    failed = _native.VocabTable.__new__(_native.VocabTable)
    with pytest.raises(ValueError, match="word 1 is also word 0"):
        failed.__init__(np.frombuffer(b"aa", np.uint8), np.array([0, 1, 2], np.int64))
    for table in (unbuilt, failed):
        for use in (lambda t: t.get("a"), len, lambda t: t.nbytes):
            with pytest.raises(RuntimeError, match="was not built"):
                use(table)
        with pytest.raises(RuntimeError, match="was not built"):
            _native.TextLookup(table, "external")
    with pytest.raises(RuntimeError, match="built once"):
        failed.__init__(np.zeros(0, np.uint8), np.zeros(1, np.int64))
    unbuilt.__init__(np.frombuffer(b"a", np.uint8), np.array([0, 1], np.int64))
    assert unbuilt.get("a") == 0


def test_a_text_lookup_is_built_once_and_only_a_built_one_encodes():
    table = _table([b"a"])
    lookup = _native.TextLookup(table, "external")
    with pytest.raises(RuntimeError, match="built once"):
        lookup.__init__(table, "fse_words@1")
    assert lookup.tokenizer == _native.TOKENIZER_EXTERNAL
    data = np.frombuffer(b"a", np.uint8)
    unbuilt = _native.TextLookup.__new__(_native.TextLookup)
    with pytest.raises(RuntimeError, match="was not built"):
        unbuilt.encode(data, None, 0, 1, 0, _native.Workspace())
    failed = _native.TextLookup.__new__(_native.TextLookup)
    with pytest.raises(ValueError, match="max_ngrams needs fasttext"):
        failed.__init__(table, "external", max_ngrams=3)
    with pytest.raises(RuntimeError, match="was not built"):
        failed.encode(data, None, 0, 1, 0, _native.Workspace())


def test_a_text_lookup_refuses_a_second_init_while_it_is_built(monkeypatch):
    # The tokenizer check runs Python code; a second __init__ from there (as another
    # thread could start one) must not overwrite what the first one sets.
    from fse import _contract

    table = _table([b"a"])
    lookup = _native.TextLookup.__new__(_native.TextLookup)
    real = _contract.check_tokenizer
    refused = []

    def reentrant(tokenizer):
        with pytest.raises(RuntimeError, match="built once"):
            lookup.__init__(table, "fse_words@1")
        refused.append(True)
        return real(tokenizer)

    monkeypatch.setattr(_contract, "check_tokenizer", reentrant)
    lookup.__init__(table, "external")
    assert refused == [True]
    assert lookup.tokenizer == _native.TOKENIZER_EXTERNAL
