"""Tests of the native routines of fse._native that do not average."""

import json
import os
import platform
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


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="gensim does not build on Python 3.14 and newer"
)
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


def test_out_must_not_share_memory_with_the_batch():
    tokens = np.zeros(2, dtype=np.uint32)
    batch = Batch(0, np.array([0, 2], np.int64), tokens)
    kernel = _native.WeightedMean(np.ones((1, 2), F32), np.ones(1, F32))
    with pytest.raises(ValueError, match="share memory with the batch"):
        kernel.pool(batch, tokens.view(F32).reshape(1, 2))


def test_batch_fields_are_read_only():
    batch = Batch(0, np.array([0, 1], np.int64), np.zeros(1, np.uint32))
    with pytest.raises(ValueError, match="read-only"):
        batch.offsets[-1] = 5


def test_offsets_are_checked_again_at_the_kernel():
    offsets = np.array([0, 1], np.int64)
    batch = Batch(0, offsets, np.zeros(1, np.uint32))
    offsets[-1] = 1000  # the caller still holds a writable array
    kernel = _native.WeightedMean(np.ones((1, 2), F32), np.ones(1, F32))
    with pytest.raises(ValueError, match="offsets must end with 1"):
        kernel.pool(batch)


def test_ngram_offsets_are_checked_again_at_the_kernel():
    ngram_offsets = np.array([0, 1], np.int64)
    batch = Batch(
        0,
        np.array([0, 1], np.int64),
        np.array([OOV_TOKEN], np.uint32),
        ngram_offsets,
        np.zeros(1, np.uint32),
    )
    ngram_offsets[-1] = 1000
    kernel = _native.WeightedMean(
        np.ones((1, 2), F32), np.ones(1, F32), ngram_vectors=np.ones((1, 2), F32), oov_weight=1.0
    )
    with pytest.raises(ValueError, match="ngram_offsets must end with 1"):
        kernel.pool(batch)


def test_the_kernel_limits_the_dimension():
    assert _native._check_dim(_native.MAX_DIM) == _native.MAX_DIM
    with pytest.raises(ValueError, match="at most"):
        _native._check_dim(_native.MAX_DIM + 1)


@pytest.mark.skipif(not os.environ.get("CI"), reason="CI runners have the expected ISA")
def test_ci_runs_the_vector_kernel():
    """A build that lost its NEON or AVX2 kernel would otherwise test only the portable one."""
    machine = platform.machine().lower()
    want = "neon" if machine in ("arm64", "aarch64") else "avx2"
    assert want in _native.available_isas()


def test_store_rows_checks_its_arguments_and_edges():
    rows = np.ones((3, 2), np.float32)
    with pytest.raises(ValueError, match="float32 matrix"):
        _native.store_rows(
            rows.astype(np.float64), np.empty((3, 2), np.int8), np.empty(3, np.float32)
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        _native.store_rows(
            np.asfortranarray(np.ones((3, 2), np.float32)),
            np.empty((3, 2), np.int8),
            np.empty(3, np.float32),
        )
    with pytest.raises(ValueError, match="codes must be"):
        _native.store_rows(rows, np.empty((2, 2), np.float16))
    with pytest.raises(ValueError, match="scales must be"):
        _native.store_rows(rows, np.empty((3, 2), np.int8), None)
    with pytest.raises(ValueError, match="have no scales"):
        _native.store_rows(rows, np.empty((3, 2), np.float16), np.empty(3, np.float32))
    buffer = np.empty(9, np.float32)
    with pytest.raises(ValueError, match="share memory"):
        _native.store_rows(buffer[:6].reshape(3, 2), np.empty((3, 2), np.int8), buffer[4:7])
    scales = np.full(3, 7, np.float32)
    _native.store_rows(np.zeros((3, 0), np.float32), np.empty((3, 0), np.int8), scales)
    assert scales.tolist() == [0, 0, 0]
    _native.store_rows(np.zeros((0, 4), np.float32), np.empty((0, 4), np.float16))


def test_store_rows_names_the_first_bad_row_of_the_call():
    rows = np.ones((6, 2), np.float32)
    rows[1, 0] = 70_000  # beyond float16, before the infinite row
    rows[4, 1] = np.inf
    with pytest.raises(ValueError, match=r"^row 14 contains NaN or infinite values$"):
        _native.store_rows(rows, np.empty((6, 2), np.float16), None, 10)
    rows[4, 1] = 1
    with pytest.raises(ValueError, match=r"^row 11 has values beyond the float16 range"):
        _native.store_rows(rows, np.empty((6, 2), np.float16), None, 10)


def test_the_workspace_counts_the_tokens_of_the_last_encode():
    from fse._text import StringSource, make_lookup
    from fse._wordvectors import WordVectors

    wv = WordVectors.from_arrays(np.ones((2, 3), np.float32), words=["a", "b"])
    lookup = make_lookup(wv, "fse_words@1")
    workspace = _native.Workspace()
    assert workspace.tokens_seen == 0
    for sentences, seen in ((["a b zz", "", "q a"], 5), (["a"], 1)):
        source = StringSource(sentences)
        batch = source.encode(source.scan().chunks[0], lookup, workspace)
        assert workspace.tokens_seen == seen  # unknown words included, reset per encode
        assert len(batch.tokens) <= seen


def test_postprocess_pooled_gives_the_bits_of_postprocess_in_place():
    from fse._contract import Normalize

    rng = np.random.default_rng(7)
    rows = rng.standard_normal((37, 12)).astype(np.float32)
    rows[3] = 0  # a zero row stays zero
    want = _native.postprocess(rows, (Normalize(),), 5)
    pooled = rows.copy()
    got = _native.postprocess_pooled(pooled, (Normalize(),), 5)
    assert got.tobytes() == want.tobytes()
    assert got is pooled
    assert _native.postprocess_pooled(rows, (), 5) is rows  # no step changes nothing


def test_postprocess_pooled_takes_only_writable_float32_matrices():
    from fse._contract import Normalize

    frozen = np.ones((2, 3), np.float32)
    frozen.setflags(write=False)
    for bad in (
        np.ones((2, 3), np.float64),
        np.ones(3, np.float32),
        frozen,
        np.ones((3, 2), np.float32).T,
    ):
        with pytest.raises(ValueError, match="writable C-contiguous float32 matrix"):
            _native.postprocess_pooled(bad, (Normalize(),))
