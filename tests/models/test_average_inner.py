"""Tests for the helpers of the compiled extension fse.models.average_inner."""

import random

import pytest
from gensim.models.fasttext import ft_ngram_hashes as gensim_ft_ngram_hashes

from fse.models.average_inner import MAX_NGRAMS_IN_BATCH, ft_ngram_hashes

WORDS = [
    "",
    "a",
    "dog",
    "Straße",
    "Ünïcödé",
    "東京大学",
    "naïve",
    "été",  # combining accents
    "\U0001f600\U0001f44d\U0001f3fd",  # emoji with a skin tone modifier
    "x" * 500,
]


@pytest.mark.parametrize("word", WORDS)
@pytest.mark.parametrize(("min_n", "max_n"), [(3, 6), (1, 1), (2, 3), (5, 5), (3, 0)])
@pytest.mark.parametrize("bucket", [2_000_000, 7, 1])
def test_ft_ngram_hashes_match_gensim(word, min_n, max_n, bucket):
    expected = gensim_ft_ngram_hashes(word, min_n, max_n, bucket)
    assert ft_ngram_hashes(word, min_n, max_n, bucket, 10**6) == expected
    assert (
        ft_ngram_hashes(word, min_n, max_n, bucket, MAX_NGRAMS_IN_BATCH)
        == (expected[:MAX_NGRAMS_IN_BATCH])
    )
    assert ft_ngram_hashes(word, min_n, max_n, bucket, 2) == expected[:2]


def test_ft_ngram_hashes_match_gensim_on_random_unicode():
    """Random code points across all planes, including control and private use characters."""
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
    """A model without ngram buckets has no ngram vectors."""
    assert ft_ngram_hashes("dog", 3, 6, 0, 40) == []


def test_ft_ngram_hashes_reject_unencodable_words():
    with pytest.raises(UnicodeEncodeError):
        ft_ngram_hashes("\ud800", 3, 6, 100, 40)
