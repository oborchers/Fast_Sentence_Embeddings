from fractions import Fraction

import numpy as np
import pytest

from fse._codes import quantize_into
from fse._contract import OOV_TOKEN, Batch, Normalize, Project, RemoveComponents
from fse._reference import (
    accumulate,
    cosine,
    dequantize,
    dot16,
    fma32,
    inverse_norms,
    pool_weighted_mean,
    postprocess,
    quantize_rows,
    score,
    top_k,
)

F32 = np.float32


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def round_f32(value: Fraction) -> np.float32:
    """The float32 nearest to an exact rational, ties to even.

    For finite results only, and blind to the sign of zero: the tests that use it stay
    in that range, and signed zeros and overflow have explicit cases.
    """
    guess = F32(float(value))
    candidates = {guess, np.nextafter(guess, F32(np.inf)), np.nextafter(guess, F32(-np.inf))}
    candidates = [c for c in candidates if np.isfinite(c)]

    def key(c):
        return (abs(Fraction(float(c)) - value), int(bits(c)) & 1)

    return min(candidates, key=key)


def exact_fma(a, b, c):
    return round_f32(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)))


def test_fma32_matches_exact_rounding_on_random_triples():
    rng = np.random.default_rng(0)
    n = 20_000
    a = (rng.standard_normal(n) * 2.0 ** rng.integers(-30, 30, n)).astype(F32)
    b = (rng.standard_normal(n) * 2.0 ** rng.integers(-30, 30, n)).astype(F32)
    # c close to -a*b gives cancellation, the hard case for a double rounding.
    c = np.where(rng.random(n) < 0.5, -(a * b), rng.standard_normal(n)).astype(F32)
    c = (c * (1 + rng.standard_normal(n).astype(F32) * F32(1e-6))).astype(F32)
    got = fma32(a, b, c)
    expected = np.array([exact_fma(x, y, z) for x, y, z in zip(a, b, c, strict=True)])
    np.testing.assert_array_equal(bits(got), bits(expected))


def test_fma32_edge_cases():
    tiny = F32(np.finfo(F32).smallest_subnormal)
    big = np.finfo(F32).max
    cases = [
        (F32(1 + 2**-12), F32(1 + 2**-12), F32(-1)),  # cancellation
        (F32(1 + 2**-23), F32(1 - 2**-23), F32(-1)),
        (tiny, F32(0.5), F32(0)),  # subnormal result, tie
        (tiny, F32(1.5), F32(0)),
        (F32(3), tiny, -tiny),
        (F32(1), F32(1), F32(2**-24)),  # tie at the last bit
        (F32(1), F32(1), F32(3 * 2**-25)),
        (F32(2**60), F32(2**60), F32(-1)),  # large exponent gaps
        (F32(1e-30), F32(1e-30), F32(1)),
        (F32(-0.0), F32(1), F32(0.0)),
    ]
    for a, b, c in cases:
        assert bits(fma32(a, b, c)) == bits(exact_fma(a, b, c)), (a, b, c)
    assert np.isposinf(fma32(big, F32(2), F32(0)))
    # Not the same as rounding the product first.
    a, b, c = cases[0]
    assert bits(fma32(a, b, c)) != bits(F32(a * b) + c)
    # A zero result has a positive sign unless both terms are negative zeros.
    assert bits(fma32(F32(-0.0), F32(1), F32(0))) == 0
    assert bits(fma32(F32(-0.0), F32(1), F32(-0.0))) == bits(F32(-0.0))


@pytest.mark.parametrize("sign", [1, -1])
def test_fma32_needs_round_to_odd(sign):
    # a * b lies exactly halfway between two float32 values; the tiny c decides the
    # rounding but vanishes in a plain float64 addition, which would then round to even.
    a = F32(sign * (1 + 2**-12))
    b = F32(1 + 2**-12)
    c = F32(sign * 2**-80)
    expected = F32(sign * (1 + 2**-11 + 2**-23))
    assert bits(fma32(a, b, c)) == bits(expected)
    assert bits(F32(np.float64(a) * np.float64(b) + np.float64(c))) != bits(expected)


def dot16_loop(row, vector):
    partial = [F32(0)] * 16
    for i, (x, y) in enumerate(zip(row, vector, strict=True)):
        partial[i % 16] = F32(fma32(x, y, partial[i % 16]))
    for width in (8, 4, 2, 1):
        for lane in range(width):
            partial[lane] = F32(partial[lane] + partial[lane + width])
    return partial[0]


@pytest.mark.parametrize("dim", [0, 1, 15, 16, 17, 300])
def test_dot16_follows_its_order(dim):
    rng = np.random.default_rng(dim)
    rows = rng.standard_normal((3, dim)).astype(F32)
    vector = rng.standard_normal(dim).astype(F32)
    got = dot16(rows, vector)
    assert got.dtype == F32
    for row, value in zip(rows, got, strict=True):
        assert bits(value) == bits(dot16_loop(row, vector))


def test_dot16_keeps_negative_zero_lanes():
    # Products that round to -0.0 leave a lane at -0.0, so zero padding is not neutral:
    # an implementation must not pad the reduction with zeros.
    rows = np.full((1, 16), -(2.0**-149), dtype=F32)
    vector = np.full(16, 0.5, dtype=F32)
    assert bits(dot16(rows, vector)[0]) == bits(F32(-0.0))
    padded = np.concatenate([rows, np.zeros((1, 16), F32)], axis=1)
    assert bits(dot16(padded, np.concatenate([vector, np.zeros(16, F32)]))[0]) == 0


def _batch(sentences, first_row=0, ngrams=None):
    offsets = np.cumsum([0, *map(len, sentences)]).astype(np.int64)
    tokens = np.array([t for s in sentences for t in s], dtype=np.uint32)
    if ngrams is None:
        return Batch(first_row, offsets, tokens)
    ngram_offsets = np.cumsum([0, *map(len, ngrams)]).astype(np.int64)
    flat = np.array([b for n in ngrams for b in n], dtype=np.uint32)
    return Batch(first_row, offsets, tokens, ngram_offsets, flat)


def pool_loop(sentences, vectors, weights, scales=None):
    rows = []
    for sentence in sentences:
        acc = np.zeros(vectors.shape[1], dtype=F32)
        for t in sentence:
            factor = F32(weights[t]) if scales is None else F32(F32(weights[t]) * F32(scales[t]))
            for i in range(len(acc)):
                acc[i] = fma32(factor, F32(vectors[t, i]), acc[i])
        if sentence:
            acc = F32(F32(1) / F32(len(sentence))) * acc
        rows.append(acc)
    return np.array(rows, dtype=F32)


def test_pool_by_hand():
    vectors = np.array([[1, 2], [3, -4], [0.5, 0.25]], dtype=F32)
    weights = np.array([1, 0.5, 2], dtype=F32)
    rows = pool_weighted_mean(_batch([[0, 1], [], [2, 2, 0]]), vectors, weights)
    np.testing.assert_array_equal(rows[0], [1.25, 0.0])
    np.testing.assert_array_equal(bits(rows[1]), [0, 0])  # +0.0, no EPS
    np.testing.assert_array_equal(rows[2], F32(1 / 3) * np.array([3, 3], dtype=F32))


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_pool_matches_a_loop(dtype):
    rng = np.random.default_rng(7)
    rows = rng.standard_normal((50, 19)).astype(F32)
    vectors, scales = quantize_rows(rows, dtype)
    weights = rng.random(50).astype(F32)
    sentences = [list(rng.integers(0, 50, rng.integers(0, 12))) for _ in range(30)]
    got = pool_weighted_mean(_batch(sentences), vectors, weights, scales)
    np.testing.assert_array_equal(bits(got), bits(pool_loop(sentences, vectors, weights, scales)))


def test_pool_is_independent_of_batch_splits():
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((40, 9)).astype(F32)
    weights = rng.random(40).astype(F32)
    sentences = [list(rng.integers(0, 40, rng.integers(0, 9))) for _ in range(17)]
    whole = pool_weighted_mean(_batch(sentences), vectors, weights)
    for cut in range(len(sentences) + 1):
        parts = [
            pool_weighted_mean(_batch(sentences[:cut]), vectors, weights),
            pool_weighted_mean(_batch(sentences[cut:], first_row=cut), vectors, weights),
        ]
        np.testing.assert_array_equal(bits(np.concatenate(parts)), bits(whole))


def test_pool_finishes_with_the_inverse_length():
    # (1/3) * acc and acc / 3 differ for this value; the contract takes (1/3) * acc.
    value = F32(1 + 2**-23)
    vectors = np.array([[value]], dtype=F32)
    weights = np.array([1], dtype=F32)
    rows = pool_weighted_mean(_batch([[0], [0, 0, 0]]), vectors, weights)
    acc = fma32(value, F32(1), fma32(value, F32(1), value))
    assert F32(F32(1) / F32(3)) * acc != acc / F32(3)
    assert bits(rows[1, 0]) == bits(F32(F32(1) / F32(3)) * acc)


def test_pool_fasttext_words():
    vectors = np.array([[1, 1]], dtype=F32)
    ngram_vectors = np.array([[2, 0], [0, 4], [8, 8]], dtype=F32)
    weights = np.array([1], dtype=F32)
    batch = _batch([[0, OOV_TOKEN], [OOV_TOKEN, 0]], ngrams=[[0, 1], []])
    rows = pool_weighted_mean(batch, vectors, weights, None, ngram_vectors, None, oov_weight=0.5)
    # Unknown word: (1/2) * 0.5 per ngram; one without ngrams counts for the length only.
    np.testing.assert_array_equal(rows[0], [0.5 * (1 + 0.5), 0.5 * (1 + 1)])
    np.testing.assert_array_equal(rows[1], [0.5, 0.5])
    codes, scales = quantize_rows(ngram_vectors, "int8")
    int8_rows = pool_weighted_mean(batch, vectors, weights, None, codes, scales, oov_weight=0.5)
    shared = F32(F32(F32(1) / F32(2)) * F32(0.5))
    acc = fma32(F32(1), vectors[0], np.zeros(2, F32))
    for b in (0, 1):
        acc = fma32(F32(shared * scales[b]), codes[b].astype(F32), acc)
    np.testing.assert_array_equal(bits(int8_rows[0]), bits(F32(0.5) * acc))
    with pytest.raises(ValueError, match="needs ngram vectors"):
        pool_weighted_mean(batch, vectors, weights)
    with pytest.raises(ValueError, match="needs oov_weight"):
        pool_weighted_mean(batch, vectors, weights, None, ngram_vectors)


def test_remove_components_matches_a_loop():
    rng = np.random.default_rng(5)
    rows = rng.standard_normal((6, 20)).astype(F32)
    q, _ = np.linalg.qr(rng.standard_normal((20, 3)))
    components = np.ascontiguousarray(q.T, dtype=F32)
    weights = np.array([1, 0.5, 0.25], dtype=F32)
    for w in (None, weights):
        got = postprocess(rows, [RemoveComponents(components, w)])
        for row, out in zip(rows, got, strict=True):
            projection = np.zeros(20, dtype=F32)
            for j, u in enumerate(components):
                c = dot16_loop(row, u)
                c = c if w is None else F32(c * w[j])
                projection = fma32(c, u, projection)
            np.testing.assert_array_equal(bits(out), bits(row - projection))
    removed = postprocess(rows, [RemoveComponents(components)])
    np.testing.assert_allclose(removed @ components.T, 0, atol=1e-5)


def test_project():
    rng = np.random.default_rng(9)
    rows = rng.standard_normal((5, 20)).astype(F32)
    q, _ = np.linalg.qr(rng.standard_normal((20, 4)))
    basis = np.ascontiguousarray(q.T, dtype=F32)
    out = postprocess(rows, [Project(basis), Normalize()])
    assert out.shape == (5, 4)
    projected = postprocess(rows, [Project(basis)])
    for row, got in zip(rows, projected, strict=True):
        np.testing.assert_array_equal(bits(got), bits([dot16_loop(row, axis) for axis in basis]))
    np.testing.assert_allclose(projected, rows @ basis.T, rtol=1e-5, atol=1e-6)
    assert postprocess(np.zeros((0, 20), dtype=F32), [Project(basis)]).shape == (0, 4)


@pytest.mark.parametrize("which", ["remove", "project", "both"])
def test_folding_linear_steps_matches_within_rounding(which):
    """Removing components and projecting commute with the weighted mean.

    An encoder may therefore fold both steps into its word and ngram vectors; the rows
    then agree with the unfolded steps within float32 rounding, not bit for bit. The
    projection basis is independent of the components, so both steps show in the result.
    """
    rng = np.random.default_rng(10)
    vectors = rng.standard_normal((30, 16)).astype(F32)
    ngram_vectors = rng.standard_normal((50, 16)).astype(F32)
    weights = rng.random(30).astype(F32)
    removal = RemoveComponents(
        np.ascontiguousarray(np.linalg.qr(rng.standard_normal((16, 2)))[0].T, dtype=F32),
        np.array([1, 0.5], F32),
    )
    projection = Project(
        np.ascontiguousarray(np.linalg.qr(rng.standard_normal((16, 6)))[0].T, dtype=F32)
    )
    steps = {"remove": [removal], "project": [projection], "both": [removal, projection]}[which]
    sentences = [list(rng.integers(0, 30, rng.integers(1, 10))) for _ in range(20)]
    sentences[3].append(OOV_TOKEN)
    ngrams = [list(rng.integers(0, 50, 4))]
    batch = _batch(sentences, ngrams=ngrams)
    kwargs = {"oov_weight": 0.7}
    unfolded = postprocess(
        pool_weighted_mean(batch, vectors, weights, None, ngram_vectors, **kwargs), steps
    )
    folded = pool_weighted_mean(
        batch,
        postprocess(vectors, steps),
        weights,
        None,
        postprocess(ngram_vectors, steps),
        **kwargs,
    )
    np.testing.assert_allclose(folded, unfolded, rtol=1e-5, atol=1e-5)
    # Each step changes the rows, so a fold that skipped one would fail.
    plain = pool_weighted_mean(batch, vectors, weights, None, ngram_vectors, **kwargs)
    for step in steps:
        assert not np.allclose(postprocess(plain, [step])[:, :6], plain[:, :6], atol=1e-3)


def test_pool_reports_overflow_with_the_row():
    vectors = np.array([[127, 1]], dtype=np.int8)
    weights = np.array([np.finfo(F32).max], dtype=F32)
    scales = np.array([2], dtype=F32)
    with pytest.raises(ValueError, match="row 5 contains NaN or infinite"):
        pool_weighted_mean(_batch([[0]], first_row=5), vectors, weights, scales)


def test_scores_and_cosine():
    stored = np.array([[127, 0], [0, 64], [0, 0]], dtype=np.int8)
    scales = np.array([2, 0.5, 0], dtype=F32)
    np.testing.assert_array_equal(score(stored, scales, np.array([1, 1], F32), 1), [255, 33, 1])
    inverse = inverse_norms(stored)
    np.testing.assert_array_equal(inverse, [F32(1 / 127), F32(1 / 64), 0])
    # The cosine ignores the scale: [127, 0] with scale 2 points along the query.
    np.testing.assert_allclose(cosine(stored, inverse, np.array([3, 0], F32)), [1, 0, 0])
    assert top_k(np.array([0.5, 0.9, 0.9, 0.1], F32), 3).tolist() == [1, 2, 0]


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_reads_follow_their_formulas_bit_for_bit(dtype):
    rng = np.random.default_rng(12)
    rows = rng.standard_normal((9, 37)).astype(F32)
    stored, scales = quantize_rows(rows, dtype)
    weights = rng.standard_normal(37).astype(F32)
    query = rng.standard_normal(37).astype(F32)
    got = score(stored, scales, weights, 0.25)
    inverse = inverse_norms(stored)
    unit = postprocess(query[None, :], [Normalize()])[0]
    cos = cosine(stored, inverse, query)
    for r in range(len(rows)):
        dot = dot16_loop(stored[r].astype(F32), weights)
        dot = dot if scales is None else F32(dot * scales[r])
        assert bits(got[r]) == bits(F32(dot + F32(0.25)))
        squares = sum(float(v) * float(v) for v in stored[r].astype(np.float64))
        assert bits(inverse[r]) == bits(F32(1 / np.sqrt(squares)))
        assert bits(cos[r]) == bits(F32(dot16_loop(stored[r].astype(F32), unit) * inverse[r]))


def test_reads_raise_on_overflow():
    big = np.array([[np.finfo(F32).max, 0]], dtype=F32)
    with pytest.raises(ValueError, match="score of row 0 overflows"):
        score(big, None, np.array([2, 0], F32))
    with pytest.raises(ValueError, match="inverse norm of row 0 is beyond float32"):
        inverse_norms(np.array([[2.0**-149, 0]], dtype=F32))
    stored = np.array([[1, 0]], dtype=F32)
    with pytest.raises(ValueError, match="zero vector"):
        cosine(stored, inverse_norms(stored), np.zeros(2, F32))
    for query in ([3e38, 3e38], [1e-30, 0]):
        with pytest.raises(ValueError, match="cannot be normalized"):
            cosine(stored, inverse_norms(stored), np.array(query, F32))


def test_normalize():
    rng = np.random.default_rng(6)
    rows = np.concatenate([rng.standard_normal((4, 30)), np.zeros((1, 30))]).astype(F32)
    out = postprocess(rows, [Normalize()], first_row=10)
    np.testing.assert_allclose(np.linalg.norm(out[:4], axis=1), 1, rtol=1e-6)
    np.testing.assert_array_equal(bits(out[4]), 0)
    np.testing.assert_array_equal(out[0], rows[0] * (F32(1) / np.sqrt(dot16(rows[:1], rows[0]))))
    with pytest.raises(ValueError, match="row 11 cannot be normalized"):
        postprocess(np.array([[1, 1], [1e-30, 0]], dtype=F32), [Normalize()], first_row=10)
    with pytest.raises(ValueError, match="row 0 cannot be normalized"):
        postprocess(np.array([[3e38, 3e38]], dtype=F32), [Normalize()])


def test_postprocess_reports_overflow_with_the_row():
    rows = np.array([[1, 0], [np.finfo(F32).max, 0]], dtype=F32)
    step = RemoveComponents(np.array([[1, 0]], dtype=F32), np.array([-2], dtype=F32))
    with pytest.raises(ValueError, match="row 8 contains NaN or infinite"):
        postprocess(rows, [step], first_row=7)


def test_postprocess_rejects_non_finite_rows():
    with pytest.raises(ValueError, match="row 4 contains NaN"):
        postprocess(np.array([[1, 0], [np.inf, 0]], dtype=F32), [], first_row=3)
    rows = np.array([[1, 2]], dtype=F32)
    out = postprocess(rows, [])
    assert out is not rows
    np.testing.assert_array_equal(out, rows)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_quantize_rows_matches_1x(dtype):
    rng = np.random.default_rng(8)
    big = np.finfo(F32).max
    rows = np.concatenate(
        [
            rng.standard_normal((40, 7)) * 10.0 ** rng.integers(-40, 4, (40, 1)),
            np.zeros((1, 7)),
            np.full((1, 7), 65504.0),
            np.array([[big, -big, 0, 1, 2, 3, 4]]) if dtype == "int8" else np.ones((1, 7)),
        ]
    ).astype(F32)
    codes, scales = quantize_rows(rows, dtype)
    expected = np.empty_like(
        rows, dtype={"float32": F32, "float16": np.float16, "int8": np.int8}[dtype]
    )
    expected_scales = np.empty(len(rows), dtype=F32) if dtype == "int8" else None
    quantize_into(rows, expected, expected_scales)
    np.testing.assert_array_equal(codes.view(np.uint8), expected.view(np.uint8))
    if dtype == "int8":
        np.testing.assert_array_equal(bits(scales), bits(expected_scales))
        np.testing.assert_array_equal(
            bits(dequantize(codes, scales)), bits(codes.astype(F32) * scales[:, None])
        )
    else:
        assert scales is None
        np.testing.assert_array_equal(dequantize(codes, None), codes.astype(F32))


def test_quantize_rows_with_a_scale_that_rounds_to_zero():
    # absmax / 127 underflows to 0 for the smallest subnormal: the codes are 0, as for a
    # zero row, and as in 1.x.
    rows = np.array([[2.0**-149, 0], [0, 0]], dtype=F32)
    codes, scales = quantize_rows(rows, "int8")
    np.testing.assert_array_equal(codes, 0)
    np.testing.assert_array_equal(bits(scales), 0)
    expected = np.empty((2, 2), dtype=np.int8)
    quantize_into(rows, expected, np.empty(2, dtype=F32))
    np.testing.assert_array_equal(codes, expected)


def test_quantize_rows_rejects():
    with pytest.raises(ValueError, match="finite"):
        quantize_rows(np.array([[np.nan]], dtype=F32), "int8")
    with pytest.raises(ValueError, match="float16 range"):
        quantize_rows(np.array([[65520]], dtype=F32), "float16")
    with pytest.raises(ValueError, match="unknown storage dtype"):
        quantize_rows(np.ones((1, 1), dtype=F32), "int4")
    codes, scales = quantize_rows(np.zeros((2, 0), dtype=F32), "int8")
    assert codes.shape == (2, 0)
    np.testing.assert_array_equal(scales, 0)
