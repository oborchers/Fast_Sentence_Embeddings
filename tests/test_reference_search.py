"""The reference of the search: cosine and score of many queries or weight vectors."""

import numpy as np
import pytest

from fse._reference import cosine, cosines, inverse_norms, quantize_rows, score, scores

F32 = np.float32


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_every_column_is_the_rule_of_one_vector(dtype):
    rng = np.random.default_rng(3)
    stored, scales = quantize_rows(rng.standard_normal((11, 19)).astype(F32), dtype)
    inverse = inverse_norms(stored)
    queries = rng.standard_normal((4, 19)).astype(F32)
    got = cosines(stored, inverse, queries)
    assert got.shape == (11, 4)
    for j, query in enumerate(queries):
        np.testing.assert_array_equal(bits(got[:, j]), bits(cosine(stored, inverse, query)))
    biases = rng.standard_normal(4).astype(F32)
    linear = scores(stored, scales, queries, biases)
    for j, weights in enumerate(queries):
        np.testing.assert_array_equal(
            bits(linear[:, j]), bits(score(stored, scales, weights, biases[j]))
        )
    one_bias = scores(stored, scales, queries, F32(0.5))
    np.testing.assert_array_equal(
        bits(one_bias[:, 1]), bits(score(stored, scales, queries[1], 0.5))
    )


def test_every_query_is_checked_before_any_is_used():
    stored = np.ones((3, 2), F32)
    queries = np.array([[1, 0], [0, 0], [0, 0]], F32)
    with pytest.raises(ValueError, match="query 1 is the zero vector"):
        cosines(stored, inverse_norms(stored), queries)


def test_errors_name_the_query_and_the_row():
    big = F32(3e38)
    stored = np.array([[1, 0], [big, big]], F32)
    inverse = np.array([1, 1], F32)  # not the true norms: the products overflow
    with pytest.raises(ValueError, match=r"query 1: the cosine of row 1 is not finite"):
        cosines(stored, inverse, np.array([[1, 0], [1, 1]], F32))
    with pytest.raises(ValueError, match=r"weights 0: the score of row 1 overflows"):
        scores(stored, None, np.array([[big, big]], F32))


@pytest.mark.parametrize("bad", [np.ones(3, F32), np.ones((1, 1, 3), F32)])
def test_queries_and_weights_are_matrices(bad):
    stored = np.ones((2, 3), F32)
    with pytest.raises(ValueError, match="queries must be a matrix"):
        cosines(stored, inverse_norms(stored), bad)
    with pytest.raises(ValueError, match="weights must be a matrix"):
        scores(stored, None, bad)
