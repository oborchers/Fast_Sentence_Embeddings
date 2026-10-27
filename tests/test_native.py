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


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="gensim does not build on Python 3.14 and newer")
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


# --- The weighted mean kernel: arguments (its arithmetic is in test_conformance.py) ---

import numpy as np  # noqa: E402

from fse import _native  # noqa: E402
from fse._contract import OOV_TOKEN, Batch  # noqa: E402

F32 = np.float32


def _batch(tokens, offsets=None, ngram_offsets=None, ngrams=None):
    tokens = np.array(tokens, dtype=np.uint32)
    offsets = np.array(offsets if offsets is not None else [0, len(tokens)], dtype=np.int64)
    if ngram_offsets is None:
        return Batch(0, offsets, tokens)
    return Batch(0, offsets, tokens, np.array(ngram_offsets, np.int64), np.array(ngrams, np.uint32))


def test_isas():
    isas = _native.available_isas()
    assert isas[0] == "portable"
    assert _native.default_isa() == isas[-1]
    assert set(isas) <= {"portable", "avx2", "neon"}
    with pytest.raises(ValueError, match="isa must be one of"):
        _native.WeightedMean(np.ones((2, 2), F32), np.ones(2, F32), isa="sse")
    missing = {"avx2", "neon"} - set(isas)
    for isa in missing:
        with pytest.raises(ValueError, match="cannot run the"):
            _native.WeightedMean(np.ones((2, 2), F32), np.ones(2, F32), isa=isa)
    wm = _native.WeightedMean(np.ones((2, 2), F32), np.ones(2, F32))
    assert wm.isa_name == isas[-1]
    assert wm.dim == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"vectors": np.ones((2, 2))}, "float32, float16 or int8"),
        ({"vectors": np.ones(2, F32)}, "matrix"),
        ({"vectors": np.ones((2, 4), F32)[:, ::2]}, "C-contiguous"),
        ({"vectors": np.ones((2, 0), F32)}, "at least one dimension"),
        ({"weights": np.ones(3, F32)}, "weights must be a float32 vector of 2"),
        ({"weights": np.ones(2)}, "weights must be a float32 vector of 2"),
        ({"scales": np.ones(2, F32)}, "scales are given exactly for int8"),
        ({"vectors": np.ones((2, 2), np.int8)}, "scales are given exactly for int8"),
        ({"ngram_vectors": np.ones((3, 3), F32)}, "dtype and dimension of vectors"),
        ({"ngram_scales": np.ones(3, F32)}, "ngram_scales without ngram_vectors"),
        (
            {
                "vectors": np.ones((2, 2), np.int8),
                "scales": np.ones(2, F32),
                "ngram_vectors": np.ones((3, 2), np.int8),
            },
            "ngram_scales are given exactly",
        ),
    ],
)
def test_weighted_mean_rejects(kwargs, match):
    args = {"vectors": np.ones((2, 2), F32), "weights": np.ones(2, F32), **kwargs}
    with pytest.raises(ValueError, match=match):
        _native.WeightedMean(**args)


def test_batches_are_checked():
    wm = _native.WeightedMean(np.ones((2, 2), F32), np.ones(2, F32))
    with pytest.raises(TypeError, match="must be a Batch"):
        wm.pool([0, 1])
    with pytest.raises(ValueError, match="token ids must be below 2"):
        wm.pool(_batch([0, 2]))
    with pytest.raises(ValueError, match="needs ngram vectors"):
        wm.pool(_batch([OOV_TOKEN], ngram_offsets=[0, 1], ngrams=[0]))
    ft = _native.WeightedMean(
        np.ones((2, 2), F32), np.ones(2, F32), ngram_vectors=np.ones((3, 2), F32)
    )
    with pytest.raises(ValueError, match="needs oov_weight"):
        ft.pool(_batch([OOV_TOKEN], ngram_offsets=[0, 1], ngrams=[0]))
    ft = _native.WeightedMean(
        np.ones((2, 2), F32), np.ones(2, F32), ngram_vectors=np.ones((3, 2), F32), oov_weight=1.0
    )
    with pytest.raises(ValueError, match="ngram buckets must be below 3"):
        ft.pool(_batch([OOV_TOKEN], ngram_offsets=[0, 1], ngrams=[3]))
    np.testing.assert_array_equal(
        ft.pool(_batch([OOV_TOKEN, 1], ngram_offsets=[0, 1], ngrams=[2])), [[1, 1]]
    )


def test_out():
    vectors = np.arange(4, dtype=F32).reshape(2, 2)
    wm = _native.WeightedMean(vectors, np.ones(2, F32))
    out = np.full((1, 2), 9, F32)
    assert wm.pool(_batch([0, 1]), out) is out
    np.testing.assert_array_equal(out, [[1, 2]])
    for bad in (
        np.empty((2, 2), F32),
        np.empty((1, 2)),
        np.empty((1, 4), F32)[:, ::2],
        vectors[:1],
    ):
        with pytest.raises(ValueError, match="out must"):
            wm.pool(_batch([0, 1]), bad)
    readonly = np.empty((1, 2), F32)
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        wm.pool(_batch([0, 1]), readonly)
    assert wm.pool(Batch(0, np.zeros(1, np.int64), np.zeros(0, np.uint32))).shape == (0, 2)


def test_pooling_runs_in_threads():
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((100, 64)).astype(F32)
    wm = _native.WeightedMean(vectors, rng.random(100).astype(F32))
    batch = _batch(rng.integers(0, 100, 5000), offsets=np.arange(0, 5001, 50))
    want = wm.pool(batch)
    with ThreadPoolExecutor(4) as pool:
        for got in pool.map(lambda _: wm.pool(batch), range(8)):
            np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
