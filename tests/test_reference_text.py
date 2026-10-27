import json
from pathlib import Path

import numpy as np
import pytest

from fse._contract import OOV_TOKEN, UnsupportedArtifactError
from fse._reference_text import (
    SEPARATORS,
    WHITESPACE,
    encode_texts,
    ft_ngram_hashes,
    simple_lower,
    split_lines,
    tokenize,
    tokenize_external,
    tokenize_fse_words,
)

DATA = Path(__file__).parent / "test_data" / "v2"

# The examples of ARCHITECTURE.md, "Tokenizers".
EXAMPLES = [
    ("I don't know.", ["i", "do", "not", "know"]),
    ("Can't, won't", ["ca", "not", "wo", "not"]),
    ("'Tis", ["t", "is"]),
    ("O'Reilly's", ["o'reilly", "s"]),
    ("James' book", ["james", "book"]),
    ("rock'n'roll", ["rock'n'roll"]),
    ("rock\u2019n\u2019roll", ["rock", "n", "roll"]),
    ("x='abc'", ["x=", "abc"]),
    ("U.S. e.g.", ["u.s", "e.g"]),
    ("Wait... what?", ["wait", "what"]),
    ("1,000 and 10:30", ["1,000", "and", "10:30"]),
    ("a,b c:d", ["a", "b", "c", "d"]),
    ("http://example.com/a-b?x=1", ["http", "//example.com/a", "b", "x=1"]),
    ("well-known --- a--b -x", ["well", "known", "a", "b", "x"]),
    ("caf\u00e9 \u4e2d\u56fd Paris", ["caf\u00e9", "paris"]),
    ("\u0130stanbul", ["istanbul"]),
    ("\u03a3\u039f\u03a6\u039f\u03a3 2\u03a3", ["2\u03c3"]),
    ("5\u212a", ["5k"]),
    ("snake_case foo/bar a+b", ["snake_case", "foo/bar", "a+b"]),
    ("...", []),
    ("", []),
    (" \t\u3000 ", []),
]

# Further cases, one per rule.
RULES = [
    (
        "cannot gonna gimme lemme wanna gotta",
        ["can", "not", "gon", "na", "gim", "me", "lem", "me", "wan", "na", "got", "ta"],
    ),
    ("more'n d'ye 'twas", ["more", "n", "d", "ye", "t", "was"]),
    ("n't isn't o''n't", ["not", "is", "not", "o", "not"]),
    (
        "I'm he'd we'll they're I've it's",
        ["i", "m", "he", "d", "we", "ll", "they", "re", "i", "ve", "it", "s"],
    ),
    ("''quoted'' x''s", ["quoted", "x", "s"]),
    ("3.14 a.b. .5 5.", ["3.14", "a.b", "5", "5"]),
    ("12:30: 1,2, ,3 :4", ["12:30", "1,2", ",3", "4"]),
    ("-- - a-", ["a"]),
    (
        "\u00bfqu\u00e9? \u00abs\u00ed\u00bb \u201cso\u201d \u2018it\u2019 \u201elow\u201c a\u2013b a\u2014b",
        ["\u00bfqu\u00e9", "s\u00ed", "so", "it", "low", "a", "b", "a", "b"],
    ),
    (
        "\u00dcn\u00efc\u00f6d\u00e9 \u00c0\u00c9\u00ce",
        ["\u00fcn\u00efc\u00f6d\u00e9"],  # no a-z in \u00e0\u00e9\u00ee,
    ),
    ("\u00e9\u0301 e\u0301", ["e\u0301"]),
    ("tab\there\nnew\u2028line\u0085x", ["tab", "here", "new", "line", "x"]),
    ("a\x00b \x01", ["a\x00b"]),
    ("\U0001f600 x\U0001f600", ["x\U0001f600"]),
    ("\u023a", []),
    ("X\u023aY", ["x\u2c65y"]),
    ("=' ='a /'s", ["a", "s"]),
]


@pytest.mark.parametrize(("text", "tokens"), EXAMPLES + RULES)
def test_fse_words_examples(text, tokens):
    assert tokenize_fse_words(text) == tokens


def test_simple_lower():
    assert simple_lower("ABC\u0130\u03a3\u212a\u1e9e") == "abci\u03c3k\u00df"
    assert simple_lower("\u00df\u03c2") == "\u00df\u03c2"  # no uppercase, no final sigma rule


def test_classes_are_pinned():
    assert len(WHITESPACE) == 29
    assert {ch for ch in map(chr, range(0x110000)) if ch.isspace()} == WHITESPACE
    assert not SEPARATORS & {"-", "'", ".", ",", ":"}


def test_external_splits_at_whitespace_only():
    text = "  A-b,\u00a0C'd\u3000  e\n"
    assert tokenize_external(text) == text.split() == ["A-b,", "C'd", "e"]
    assert tokenize(text, "external") == text.split()


def test_tokenize_checks_the_tokenizer():
    assert tokenize("Don't", "fse_words@1") == ["do", "not"]
    with pytest.raises(UnsupportedArtifactError):
        tokenize("x", "fse_words@2")


def test_split_lines():
    assert split_lines(b"") == []
    assert split_lines(b"\n") == [""]
    assert split_lines(b"a\nb") == ["a", "b"]
    assert split_lines(b"a\nb\n") == ["a", "b"]
    assert split_lines(b"a\r\n\nb") == ["a\r", "", "b"]
    with pytest.raises(UnicodeDecodeError):
        split_lines(b"a\n\xff")
    with pytest.raises(ValueError, match="byte order mark"):
        split_lines(b"\xef\xbb\xbfa")


GOLDEN_NGRAMS = json.loads((DATA / "ft_ngram_hashes.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", GOLDEN_NGRAMS["cases"], ids=lambda c: repr(c["word"][:8]))
def test_ft_ngram_hashes_match_gensim(case):
    got = ft_ngram_hashes(case["word"], case["min_n"], case["max_n"], case["bucket"], 10**6)
    assert got == case["hashes"]


def test_ft_ngram_hashes_are_capped():
    full = ft_ngram_hashes("example", 3, 6, 1000, 10**6)
    assert ft_ngram_hashes("example", 3, 6, 1000, 4) == full[:4]
    assert ft_ngram_hashes("example", 3, 6, 0, 10) == []


def test_encode_texts():
    words = {b"i": 0, b"do": 1, b"not": 2, b"know": 3}
    batch = encode_texts(
        ["I don't know.", "", "know what"], words, tokenizer="fse_words@1", first_row=4
    )
    assert batch.first_row == 4
    assert batch.offsets.tolist() == [0, 4, 4, 5]
    assert batch.tokens.tolist() == [0, 1, 2, 3, 3]
    assert batch.ngrams is None


def test_encode_texts_with_fasttext():
    words = {b"know": 0}
    batch = encode_texts(
        ["know what x"], words, tokenizer="fse_words@1", fasttext=(3, 4, 100), max_ngrams=3
    )
    assert batch.tokens.tolist() == [0, OOV_TOKEN, OOV_TOKEN]
    assert batch.ngram_offsets.tolist() == [0, 3, 4]
    assert batch.ngrams.tolist() == [
        *ft_ngram_hashes("what", 3, 4, 100, 3),
        *ft_ngram_hashes("x", 3, 4, 100, 3),
    ]
    none = encode_texts(["what"], words, tokenizer="external", fasttext=(3, 4, 100), max_ngrams=0)
    assert none.tokens.tolist() == [OOV_TOKEN]
    assert none.ngram_offsets.tolist() == [0, 0]
    assert none.ngrams.dtype == np.uint32


GOLDEN = json.loads((DATA / "fse_words.json").read_text(encoding="utf-8"))


def test_reference_matches_the_golden_tokens():
    assert GOLDEN["tokenizer"] == "fse_words@1"
    assert GOLDEN["unicode"] == "15.1.0"
    assert len(GOLDEN["cases"]) > 2700
    for text, tokens in GOLDEN["cases"]:
        assert tokenize_fse_words(text) == tokens, text
