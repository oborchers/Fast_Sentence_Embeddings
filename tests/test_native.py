"""Tests of the native routines of fse._native that do not average."""

import json
import random
import sys

import pytest

from fse._native import MAX_NGRAMS_LIMIT, ft_ngram_hashes

GOLDEN = json.loads(
    (
        __import__("pathlib").Path(__file__).parent / "test_data" / "v2" / "ft_ngram_hashes.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    "case",
    GOLDEN["cases"],
    ids=lambda c: f"{c['word'][:8]!r}-{c['min_n']}-{c['max_n']}-{c['bucket']}",
)
def test_ft_ngram_hashes_match_the_pinned_gensim_values(case):
    expected = case["hashes"]
    got = ft_ngram_hashes(case["word"], case["min_n"], case["max_n"], case["bucket"], 10**6)
    assert got == expected
    assert (
        ft_ngram_hashes(case["word"], case["min_n"], case["max_n"], case["bucket"], 2)
        == expected[:2]
    )


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="gensim does not build on Python 3.14")
def test_ft_ngram_hashes_match_gensim_on_random_unicode():
    """Random code points across all planes, including control and private use characters."""
    from gensim.models.fasttext import ft_ngram_hashes as gensim_ft_ngram_hashes

    rng = random.Random(0)
    for _ in range(2_000):
        word = "".join(
            chr(
                rng.choice(
                    [
                        rng.randrange(0x20, 0x7F),
                        rng.randrange(0xA0, 0xD800),
                        rng.randrange(0xE000, 0x110000),
                    ]
                )
            )
            for _ in range(rng.randrange(0, 12))
        )
        expected = gensim_ft_ngram_hashes(word, 3, 6, 2_000_000)
        assert ft_ngram_hashes(word, 3, 6, 2_000_000, 10**6) == expected, repr(word)


def test_ft_ngram_hashes_without_buckets():
    assert ft_ngram_hashes("dog", 3, 6, 0, 40) == []


def test_ft_ngram_hashes_reject_unencodable_words():
    with pytest.raises(UnicodeEncodeError):
        ft_ngram_hashes("\ud800", 3, 6, 100, 40)


def test_ngram_limit():
    assert MAX_NGRAMS_LIMIT == 1 << 24
