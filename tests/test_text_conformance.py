"""The native tokenizers give the bytes of fse._reference_text.

The golden tokens pin the reference; random text over all of Unicode compares the native
code with it. Meta-tests at the end check that the corpus catches typical mistakes, so
the comparison cannot pass by accident.
"""

import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from fse import _native
from fse import _reference_text as ref
from fse._contract import OOV_TOKEN, Batch
from fse._unicode import SIMPLE_LOWERCASE

GOLDEN = json.loads(
    (Path(__file__).parent / "test_data" / "v2" / "fse_words.json").read_text(encoding="utf-8")
)
TOKENIZERS = ["fse_words@1", "external"]
REFERENCE = {"fse_words@1": ref.tokenize_fse_words, "external": ref.tokenize_external}

# A weighted alphabet: the characters of every rule, every whitespace and separator,
# length-changing and special lowercase mappings, combining marks, non-ASCII digits,
# control characters and 4-byte characters.
_ALPHABET = [
    *"abcXYZ019_.-,:'`=/+",
    *"....----''''::,,  ",
    *sorted(ref.WHITESPACE),
    *sorted(ref.SEPARATORS),
    *"\u0130\u03a3\u03c2\u212a\u023a\u2c65\u00e9\u0301\u4e2d\U0001f600\u1e9e\u00df\x00\x7f\x1b",
    *"\u0663\u00b2\uff11",  # digits that are not 0 to 9
    *"tnsmdlrev",  # the letters of the contractions and clitics
]


def _random_text(rng: random.Random) -> str:
    chars = []
    for _ in range(rng.randint(0, 40)):
        if rng.random() < 0.85:
            chars.append(rng.choice(_ALPHABET))
        else:
            lo, hi = rng.choice(
                [(0x80, 0x7FF), (0x800, 0xD7FF), (0xE000, 0xFFFF), (0x10000, 0x10FFFF)]
            )
            chars.append(chr(rng.randint(lo, hi)))
    return "".join(chars)


def _corpus() -> list[str]:
    rng = random.Random(0)
    texts = [text for text, _ in GOLDEN["cases"]]
    texts += [_random_text(rng) for _ in range(20_000)]
    words = ["cannot", "gonna", "'tis", "'twas", "more'n", "d'ye", "don't", "n't", "it's", "we'll"]
    texts += [
        " ".join(rng.choice([*words, "x", "a-b", "..."]) for _ in range(6)) for _ in range(500)
    ]
    return texts


CORPUS = _corpus()


def assert_tokens_equal(got: list[str], want: list[str], text: str) -> None:
    """The comparison of every token test: the same tokens in the same order."""
    if got != want:
        raise AssertionError(f"{text!r}: {got!r} != {want!r}")


def test_the_token_comparison_catches_wrong_tokens():
    assert_tokens_equal(["a", "b"], ["a", "b"], "a b")
    for wrong in (["b", "a"], ["a"], ["a", "b", "c"], ["a", "B"], []):
        with pytest.raises(AssertionError):
            assert_tokens_equal(wrong, ["a", "b"], "a b")


def test_native_matches_the_golden_tokens():
    for text, tokens in GOLDEN["cases"]:
        assert_tokens_equal(_native.tokenize(text), tokens, text)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_native_matches_the_reference_on_random_unicode(tokenizer):
    want = REFERENCE[tokenizer]
    for text in CORPUS:
        assert_tokens_equal(_native.tokenize(text, tokenizer), want(text), text)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_bytes_and_str_give_the_same_tokens(tokenizer):
    for text in CORPUS[:2000]:
        assert _native.tokenize(text.encode(), tokenizer) == _native.tokenize(text, tokenizer)


def test_native_lowercase_table_matches_the_python_table():
    cps = np.arange(sys.maxunicode + 1, dtype=np.uint32)
    want = cps.copy()
    for cp, lower in SIMPLE_LOWERCASE.items():
        want[cp] = lower
    np.testing.assert_array_equal(_native._simple_lower(cps), want)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_invalid_utf8_raises_where_python_does(tokenizer):
    rng = random.Random(1)
    pieces = [b"a", b" ", b".", b"\xc3\xa9", b"\xe4\xb8\xad", b"\xf0\x9f\x98\x80", b"'"]
    bad = [
        b"\xff",
        b"\xc3",
        b"\xe4\xb8",
        b"\xed\xa0\x80",
        b"\xc0\xaf",
        b"\xf4\x90\x80\x80",
        b"\x80",
    ]
    for _ in range(3000):
        data = b"".join(rng.choice(pieces) for _ in range(rng.randint(0, 8)))
        data += rng.choice(bad) + b"".join(rng.choice(pieces) for _ in range(rng.randint(0, 4)))
        with pytest.raises(UnicodeDecodeError) as python:
            data.decode("utf-8")
        with pytest.raises(UnicodeDecodeError) as native:
            _native.tokenize(data, tokenizer)
        assert native.value.start == python.value.start, data


def test_tokenize_validates_its_arguments():
    with pytest.raises(TypeError, match="str or bytes"):
        _native.tokenize(["a"])
    with pytest.raises(ValueError, match="not supported"):
        _native.tokenize("a", "nltk@1")
    with pytest.raises(UnicodeEncodeError):
        _native.tokenize("a\ud800b")


# --- Meta-tests: the corpus catches typical mistakes ------------------------------------------


@pytest.mark.parametrize(
    ("mistake", "attribute", "value"),
    [
        ("full lowercase as str.lower", "simple_lower", str.lower),
        (
            "str.isdigit for , and :",
            "_DIGITS",
            frozenset(ch for ch in map(chr, range(0x10000)) if ch.isdigit()),
        ),
        ("no contraction table", "CONTRACTIONS", {}),
        ("no clitics", "CLITICS", ()),
        ("every character a word character", "_is_word", lambda ch: True),
        ("the separators of the 1.x regex only", "SEPARATORS", frozenset('!"#$%&()*;<>?@[]`{}')),
    ],
)
def test_the_corpus_catches(monkeypatch, mistake, attribute, value):
    want = [ref.tokenize_fse_words(text) for text in CORPUS]
    monkeypatch.setattr(ref, attribute, value)
    caught = 0
    for text, tokens in zip(CORPUS, want, strict=True):
        try:
            assert_tokens_equal(ref.tokenize_fse_words(text), tokens, text)
        except AssertionError:
            caught += 1
    assert caught, f"no text catches {mistake}"


# --- Stage 2: batches -------------------------------------------------------------------------


def _lookup_setup(texts, share=0.5):
    from fse._wordvectors import WordVectors

    vocab = sorted(
        {t for s in texts for t in ref.tokenize_fse_words(s)}
        | {t for s in texts for t in s.split()}
    )
    keep = vocab[: int(len(vocab) * share)]
    wv = WordVectors.from_arrays(np.zeros((len(keep), 2), np.float32), words=keep)
    table = _native.VocabTable(wv.vocab_bytes, wv.vocab_offsets)
    return table, {w.encode(): i for i, w in enumerate(keep)}


def _items(texts):
    encoded = [s.encode() for s in texts]
    offsets = np.zeros(len(texts) + 1, np.int64)
    np.cumsum([len(b) for b in encoded], out=offsets[1:])
    return np.frombuffer(b"".join(encoded), np.uint8), offsets


def assert_batches_equal(got, want):
    assert got.first_row == want.first_row
    for name in ("offsets", "tokens", "ngram_offsets", "ngrams"):
        a, b = getattr(got, name), getattr(want, name)
        assert (a is None) == (b is None), name
        if a is not None:
            assert a.dtype == b.dtype, name
            np.testing.assert_array_equal(a, b, err_msg=name)


ENCODE_TEXTS = CORPUS[:4000]
FASTTEXT = [(None, 0), ((3, 5, 1000), 0), ((3, 5, 1000), 2), ((1, 3, 97), 1000), ((2, 2, 0), 5)]


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
@pytest.mark.parametrize(("fasttext", "max_ngrams"), FASTTEXT)
def test_native_batches_match_the_reference(tokenizer, fasttext, max_ngrams):
    table, words = _lookup_setup(ENCODE_TEXTS)
    lookup = _native.TextLookup(table, tokenizer, fasttext, max_ngrams)
    data, offsets = _items(ENCODE_TEXTS)
    got = lookup.encode(data, offsets, 0, len(ENCODE_TEXTS), 7, _native.Workspace())
    want = ref.encode_texts(
        ENCODE_TEXTS,
        words,
        tokenizer=tokenizer,
        fasttext=fasttext,
        max_ngrams=max_ngrams,
        first_row=7,
    )
    assert_batches_equal(got, want)


def test_lines_give_the_batches_of_their_strings():
    texts = [t.replace("\n", " ").replace("\r", " ") for t in ENCODE_TEXTS]
    table, words = _lookup_setup(texts)
    lookup = _native.TextLookup(table, "fse_words@1", (3, 5, 1000), 3)
    data = np.frombuffer(("\n".join(texts) + "\n").encode(), np.uint8)
    got = lookup.encode(data, None, 0, len(data), 0, _native.Workspace())
    want = ref.encode_texts(
        texts, words, tokenizer="fse_words@1", fasttext=(3, 5, 1000), max_ngrams=3
    )
    assert_batches_equal(got, want)


def test_ranges_split_a_source_without_changing_its_batches():
    table, words = _lookup_setup(ENCODE_TEXTS)
    lookup = _native.TextLookup(table, "fse_words@1", (3, 5, 1000), 4)
    data, offsets = _items(ENCODE_TEXTS)
    ws = _native.Workspace()
    cuts = [0, 1, 1, 17, 500, 2048, len(ENCODE_TEXTS)]
    for a, b in itertools.pairwise(cuts):
        got = lookup.encode(data, offsets, a, b, a, ws)
        want = ref.encode_texts(
            ENCODE_TEXTS[a:b],
            words,
            tokenizer="fse_words@1",
            fasttext=(3, 5, 1000),
            max_ngrams=4,
            first_row=a,
        )
        assert_batches_equal(got, want)


def test_encoding_runs_in_threads():
    from concurrent.futures import ThreadPoolExecutor

    table, _ = _lookup_setup(ENCODE_TEXTS)
    lookup = _native.TextLookup(table, "fse_words@1", (3, 5, 1000), 3)
    data, offsets = _items(ENCODE_TEXTS)
    ranges = [(i, min(i + 250, len(ENCODE_TEXTS))) for i in range(0, len(ENCODE_TEXTS), 250)]
    alone = [lookup.encode(data, offsets, a, b, a, _native.Workspace()) for a, b in ranges]

    def work(chunk):
        ws = _native.Workspace()
        return [lookup.encode(data, offsets, a, b, a, ws) for a, b in chunk]

    with ThreadPoolExecutor(4) as pool:
        parts = list(pool.map(work, [ranges[k::4] for k in range(4)]))
    together = {b.first_row: b for part in parts for b in part}
    for batch in alone:
        assert_batches_equal(together[batch.first_row], batch)


def test_encode_reports_invalid_utf8_with_its_sentence():
    table, _ = _lookup_setup(["a b"], share=1)
    lookup = _native.TextLookup(table, "fse_words@1")
    data = np.frombuffer(b"a b\nok\nx \xff y\n", np.uint8)
    with pytest.raises(
        UnicodeDecodeError, match=r"in sentence 2 of the range \(byte 9 of the source\)"
    ):
        lookup.encode(data, None, 0, len(data), 0, _native.Workspace())


def test_text_lookup_validates():
    table, _ = _lookup_setup(["a"], share=1)
    with pytest.raises(ValueError, match="not supported"):
        _native.TextLookup(table, "nltk@1")
    with pytest.raises(ValueError, match="max_ngrams needs fasttext"):
        _native.TextLookup(table, "external", None, 3)
    with pytest.raises(ValueError, match="min_n <= max_n"):
        _native.TextLookup(table, "external", (4, 3, 10))
    with pytest.raises(ValueError, match="max_ngrams must be between"):
        _native.TextLookup(table, "external", (3, 4, 10), -1)
    lookup = _native.TextLookup(table, "external")
    data, offsets = _items(["a", "b"])
    with pytest.raises(ValueError, match="outside"):
        lookup.encode(data, offsets, 0, 3, 0, _native.Workspace())
    with pytest.raises(ValueError, match="start of a line"):
        lookup.encode(np.frombuffer(b"ab\ncd\n", np.uint8), None, 1, 6, 0, _native.Workspace())


def _batch(tokens=(0, OOV_TOKEN, 1), offsets=(0, 2, 3), ngrams=(5, 6)):
    return Batch(
        0,
        np.array(offsets, np.int64),
        np.array(tokens, np.uint32),
        np.array([0, len(ngrams)], np.int64),
        np.array(ngrams, np.uint32),
    )


def test_the_batch_comparison_catches_wrong_batches():
    """assert_batches_equal fails for wrong ids, offsets, ngram order and ngram values."""
    assert_batches_equal(_batch(), _batch())
    wrong = [
        _batch(tokens=(1, OOV_TOKEN, 0)),
        _batch(offsets=(0, 1, 3)),
        _batch(ngrams=(6, 5)),
        _batch(ngrams=(5, 7)),
    ]
    for batch in wrong:
        with pytest.raises(AssertionError):
            assert_batches_equal(batch, _batch())


def test_text_lookup_checks_its_types():
    table, _ = _lookup_setup(["a"], share=1)
    with pytest.raises(TypeError, match="three ints"):
        _native.TextLookup(table, "external", (True, 3, 100))
    with pytest.raises(TypeError, match="three ints"):
        _native.TextLookup(table, "external", [3, 4, 100], 2)
    with pytest.raises(TypeError, match="max_ngrams must be an int"):
        _native.TextLookup(table, "external", (3, 4, 100), True)


def test_vocab_table_keeps_its_own_copy():
    long = b"a word longer than sixteen bytes"
    base = np.frombuffer(b"ab" + long, np.uint8).copy()
    view = base.view()
    view.flags.writeable = False  # read-only, but its base stays writable
    offsets = np.array([0, 2, 2 + len(long)], np.int64)
    table = _native.VocabTable(view, offsets)
    base[:] = ord("z")
    offsets[1] = 1
    assert table.get(b"ab") == 0
    assert table.get(long) == 1


def test_encode_keeps_its_own_offsets():
    table, _ = _lookup_setup(["a b"], share=1)
    lookup = _native.TextLookup(table, "external")
    data, offsets = _items(["a", "b"])
    batch = lookup.encode(data, offsets, 1, 2, 1, _native.Workspace())
    offsets[:] = 10**9  # changing the caller's array afterwards changes nothing
    assert batch.tokens.tolist() == [table.get("b")]
