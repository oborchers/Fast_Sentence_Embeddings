"""Tests for the principal components of fit (moved from the 1.x utilities)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fse._contract import RemoveComponents
from fse._gram import compute_principal_components
from fse._reference import postprocess


def test_compute_components():
    m = np.random.uniform(size=(500, 10)).astype(np.float32)
    out = compute_principal_components(vectors=m)
    assert len(out) == 2
    assert len(out[1]) == 1
    assert out[1].dtype == np.float32

    m = np.random.uniform(size=(500, 10))
    out = compute_principal_components(vectors=m, components=5)
    assert len(out) == 2
    assert len(out[1]) == 5


def _reference(vectors, components):
    """Singular values and the subspace of the right singular vectors, in float64."""
    _, s, vt = np.linalg.svd(vectors.astype(np.float64), full_matrices=False)
    return s[:components], vt[:components]


def _subspace_cosines(a, b):
    """Cosines of the principal angles between the row spaces of a and b."""
    qa, _ = np.linalg.qr(np.asarray(a, dtype=np.float64).T)
    qb, _ = np.linalg.qr(np.asarray(b, dtype=np.float64).T)
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_components_match_an_exact_svd(dtype):
    rng = np.random.default_rng(0)
    # A common offset and decaying scales, like sentence vectors with a dominant direction.
    scales = np.linspace(3.0, 0.1, 20)
    vectors = (rng.standard_normal((2_000, 20)) * scales + rng.standard_normal(20)).astype(dtype)
    singular_values, components = compute_principal_components(vectors, components=5)
    expected_values, expected_components = _reference(vectors, 5)
    assert singular_values.dtype == components.dtype == np.float32
    assert_allclose(singular_values, expected_values, rtol=1e-5)
    assert_allclose(_subspace_cosines(components, expected_components), 1.0, atol=1e-5)
    assert_allclose(np.linalg.norm(components, axis=1), 1.0, rtol=1e-6)


def _with_singular_values(values, rows, dim, seed):
    """A float32 matrix U diag(values) V^T with orthonormal U and V, and V^T."""
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.standard_normal((rows, len(values))))
    right, _ = np.linalg.qr(rng.standard_normal((dim, len(values))))
    return (left * values @ right.T).astype(np.float32), right.T


def test_compute_components_pair_values_and_vectors():
    """Each singular value belongs to its own vector: X^T X v_i = s_i^2 v_i."""
    vectors, _ = _with_singular_values([9.0, 7.0, 5.0, 3.0, 1.0], 400, 10, seed=3)
    singular_values, components = compute_principal_components(vectors, components=4)
    gram = vectors.astype(np.float64).T @ vectors.astype(np.float64)
    assert_allclose(singular_values, [9.0, 7.0, 5.0, 3.0], rtol=1e-5)
    for value, vector in zip(singular_values, components.astype(np.float64), strict=True):
        assert_allclose(gram @ vector, value**2 * vector, atol=1e-3)


def test_weighted_removal_matches_an_independent_svd():
    """uSIF weights reach the right directions, compared with numpy's own SVD."""
    vectors, _ = _with_singular_values([8.0, 4.0, 2.0, 1.0], 300, 6, seed=4)
    svd_res = compute_principal_components(vectors, components=3)
    _, s, vt = np.linalg.svd(vectors.astype(np.float64), full_matrices=False)
    weights = s[:3] ** 2 / (s[:3] ** 2).sum()
    v64 = vectors.astype(np.float64)
    expected = v64 - (v64 @ vt[:3].T * weights) @ vt[:3]
    weights = (svd_res[0] ** 2 / (svd_res[0] ** 2).sum()).astype(np.float32)
    out = postprocess(vectors.astype(np.float32), [RemoveComponents(svd_res[1], weights)])
    assert_allclose(out, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("values", "components", "selected"),
    [
        # Two equal values inside the selection: the subspace is determined.
        ([10.0, 6.0, 6.0, 2.0, 1.0], 3, 3),
        # Nearly equal values: the subspace of the first two is determined.
        ([10.0, 10.0 - 1e-4, 3.0, 1.0], 2, 2),
    ],
)
def test_compute_components_with_equal_singular_values(values, components, selected):
    """Equal values leave the basis open, but not the subspace or the removal."""
    vectors, right = _with_singular_values(values, 500, 12, seed=1)
    svd_res = compute_principal_components(vectors, components=components)
    expected_components = right[:selected]
    assert_allclose(_subspace_cosines(svd_res[1], expected_components), 1.0, atol=1e-4)
    removed = postprocess(vectors.astype(np.float32), [RemoveComponents(svd_res[1])])
    v64 = vectors.astype(np.float64)
    expected = v64 - v64 @ expected_components.T @ expected_components
    assert_allclose(removed, expected, atol=1e-4)


def test_compute_components_have_a_fixed_sign():
    """The largest entry of every component is positive, so flipped inputs agree."""
    vectors = np.random.default_rng(2).random((300, 6), dtype=np.float32)
    _, components = compute_principal_components(vectors, components=3)
    largest = np.abs(components).argmax(axis=1)
    assert (components[np.arange(3), largest] > 0).all()
    _, flipped = compute_principal_components(-vectors, components=3)
    assert_allclose(flipped, components, atol=1e-6)


def test_compute_components_over_blocks_equal_one_block(monkeypatch, tmp_path):
    """Block boundaries, a short last block and a memmap give the same components."""
    from fse import _gram as utils

    rows, dim = 1_000, 8
    vectors = np.memmap(tmp_path / "sv", dtype=np.float32, mode="w+", shape=(rows, dim))
    # Well separated singular values, so float32 rounding cannot rotate the vectors.
    vectors[:], _ = _with_singular_values([9.0, 7.0, 5.0, 3.0, 1.0], rows, dim, seed=5)
    whole = compute_principal_components(np.array(vectors), components=3)
    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 7)
    blocked = compute_principal_components(vectors, components=3)
    # Each block product is a float32 product: other blocks round differently.
    assert_allclose(blocked[0], whole[0], rtol=1e-6)
    assert_allclose(blocked[1], whole[1], atol=1e-5)


def test_compute_components_are_reproducible():
    m = np.random.default_rng(1).random((1000, 10), dtype=np.float32)
    first = compute_principal_components(vectors=m, components=3)
    second = compute_principal_components(vectors=m, components=3)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_compute_components_of_a_rank_deficient_matrix():
    """More components than the rank give zero singular values, never NaN."""
    vectors = np.ones((50, 4), dtype=np.float32)
    singular_values, components = compute_principal_components(vectors, components=3)
    assert_allclose(singular_values, [np.sqrt(200), 0, 0], atol=1e-3)
    assert np.isfinite(components).all()


def test_compute_components_memory_is_bounded_by_a_block(tmp_path):
    """A memmap is streamed: about one block in float64 is allocated, not the matrix."""
    import tracemalloc

    from fse import _gram as utils

    rows, dim = 500_000, 50
    vectors = np.memmap(tmp_path / "sv", dtype=np.float32, mode="w+", shape=(rows, dim))
    vectors[:] = np.random.default_rng(0).random((rows, dim), dtype=np.float32)
    tracemalloc.start()
    compute_principal_components(vectors, components=2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 3 * utils.REMOVAL_BLOCK_ROWS * dim * 8
    assert peak < rows * dim * 4 / 4


@pytest.mark.parametrize(
    ("vectors", "components", "message"),
    [
        (np.ones((10, 4), dtype=np.float32), 5, r"dimension \(4\), got 5"),
        (np.ones((10, 4), dtype=np.float32), 0, r"got 0"),
        (np.ones((2, 4), dtype=np.float32), 3, r"number of vectors \(2\)"),
        (np.ones((0, 4), dtype=np.float32), 1, r"number of vectors \(0\)"),
        (np.ones(4, dtype=np.float32), 1, "2-dimensional"),
        (np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float32), 1, "NaN or infinite"),
        (np.array([[1.0, np.inf], [0.0, 1.0]], dtype=np.float32), 1, "NaN or infinite"),
        (np.array([[np.inf, -np.inf], [0.0, 1.0]]), 1, "NaN or infinite"),
        # Finite, but the sum of a row and the products overflow.
        (np.array([[1e308, 1e308], [0.0, 1.0]]), 1, "overflows"),
        (np.array([[1e200, 0.0], [0.0, 1.0]]), 1, "overflows"),
        (np.array([[1e200, -1e200], [1e200, 1e200]]), 1, "overflows"),
    ],
)
def test_compute_components_reject_invalid_input(vectors, components, message):
    with pytest.raises(ValueError, match=message):
        compute_principal_components(vectors, components=components)


def test_invalid_values_behind_a_block_boundary_are_found(monkeypatch):
    from fse import _gram as utils

    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 4)
    vectors = np.ones((10, 3), dtype=np.float32)
    vectors[9, 1] = np.nan
    with pytest.raises(ValueError, match="in rows from 8"):
        compute_principal_components(vectors, components=1)


@pytest.mark.parametrize(
    ("vectors", "components"),
    [
        (np.array([["1", "2"], ["3", "4"]]), 1),
        (np.ones((3, 2), dtype=np.complex64), 1),
        (np.ones((3, 2), dtype=bool), 1),
        (np.ones((3, 2), dtype=np.float32), 1.0),
        (np.ones((3, 2), dtype=np.float32), True),
    ],
)
def test_compute_components_reject_wrong_types(vectors, components):
    with pytest.raises(TypeError):
        compute_principal_components(vectors, components=components)


def test_compute_components_accept_integers_exactly():
    vectors = np.array([[3, 0], [0, 1], [3, 0]], dtype=np.int64)
    singular_values, components = compute_principal_components(vectors, components=2)
    assert_allclose(singular_values, [np.sqrt(18), 1.0], rtol=1e-6)
    assert_allclose(np.abs(components), np.eye(2), atol=1e-6)


def test_undetermined_components_are_logged(caplog):
    vectors = np.ones((50, 4), dtype=np.float32)
    with caplog.at_level("WARNING", logger="fse._gram"):
        compute_principal_components(vectors, components=3)
    assert "2 of 3 principal components have a variance below a heuristic bound" in caplog.text
