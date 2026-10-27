"""The native tokenizers give the bytes of fse._reference_text.

The golden tokens pin the reference; random text over all of Unicode compares the native
code with it. Meta-tests at the end check that the corpus catches typical mistakes, so
the comparison cannot pass by accident.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from fse import _native
from fse import _reference_text as ref
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


def test_native_matches_the_golden_tokens():
    for text, tokens in GOLDEN["cases"]:
        assert _native.tokenize(text) == tokens, text


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_native_matches_the_reference_on_random_unicode(tokenizer):
    want = REFERENCE[tokenizer]
    for text in CORPUS:
        assert _native.tokenize(text, tokenizer) == want(text), repr(text)


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
    wrong = sum(
        ref.tokenize_fse_words(text) != tokens for text, tokens in zip(CORPUS, want, strict=True)
    )
    assert wrong, f"no text catches {mistake}"
