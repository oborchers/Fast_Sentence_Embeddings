"""Automated tests for the principal component helpers."""

import sys
import threading
from typing import ClassVar

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_raises

from fse.models.utils import (
    check_components_fit,
    compute_principal_components,
    remove_principal_components,
    set_madvise_for_mmap,
)


@pytest.fixture
def ones():
    return np.ones((500, 10), dtype=np.float32)


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
    out = remove_principal_components(
        vectors, svd_res, weights=(svd_res[0] ** 2 / (svd_res[0] ** 2).sum()), inplace=False
    )
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
    removed = remove_principal_components(vectors, svd_res, inplace=False)
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
    from fse.models import utils

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

    from fse.models import utils

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
    from fse.models import utils

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
    with caplog.at_level("WARNING", logger="fse.models.utils"):
        compute_principal_components(vectors, components=3)
    assert "2 of 3 principal components have a variance below a heuristic bound" in caplog.text


def test_cache_size_gb_is_deprecated():
    """The warning names the caller's line, so it is shown outside of fse."""
    vectors = np.random.default_rng(0).random((20, 4), dtype=np.float32)
    with pytest.warns(DeprecationWarning, match="cache_size_gb has no effect") as record:
        compute_principal_components(vectors, cache_size_gb=1.0)
    assert record[0].filename == __file__


def test_remove_components_inplace(ones):
    c = np.copy(ones)
    out = compute_principal_components(vectors=ones)
    remove_principal_components(ones, svd_res=out)
    assert_allclose(ones, 0.0, atol=1e-5)
    with assert_raises(AssertionError):
        assert_allclose(ones, c)


def test_remove_components(ones):
    c = np.copy(ones)
    out = compute_principal_components(vectors=ones)
    res = remove_principal_components(ones, svd_res=out, inplace=False)
    assert_allclose(res, 0.0, atol=1e-5)
    assert_allclose(ones, c)


def test_remove_weighted_components_inplace(ones):
    c = np.copy(ones)
    out = compute_principal_components(vectors=ones)
    remove_principal_components(ones, svd_res=out, weights=np.array([0.5]))
    assert_allclose(ones, 0.5, atol=1e-5)
    with assert_raises(AssertionError):
        assert_allclose(ones, c)


def test_remove_weighted_components(ones):
    c = np.copy(ones)
    out = compute_principal_components(vectors=ones)
    res = remove_principal_components(ones, svd_res=out, weights=np.array([0.5]), inplace=False)
    assert_allclose(res, 0.5, atol=1e-5)
    assert_allclose(ones, c)


@pytest.mark.parametrize("inplace", [True, False])
def test_remove_weighted_components_matches_usif_reference(inplace):
    """Each component is removed with weight lambda_i, as in the reference uSIF code.

    Reference: https://github.com/kawine/usif/blob/master/usif.py, ``embed``:
    ``v_s - lambda_i * proj(v_s, pc)`` for every principal component ``pc``.
    """
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(200, 8)).astype(np.float32)
    singular_values, components = compute_principal_components(vectors=vectors, components=3)
    weights = (singular_values**2) / (singular_values**2).sum()

    expected = vectors.astype(np.float64)
    for lambda_i, pc in zip(weights, components.astype(np.float64), strict=True):
        expected = expected - lambda_i * np.outer(expected @ pc, pc)

    out = remove_principal_components(
        vectors, svd_res=(singular_values, components), weights=weights, inplace=inplace
    )
    result = vectors if inplace else out
    assert_allclose(result, expected, atol=1e-5)


@pytest.mark.skipif(
    sys.platform not in ("linux", "linux2", "darwin", "aix"),
    reason="madvise is only available on POSIX platforms",
)
def test_madvise(tmp_path):
    p = tmp_path / "test_vectors"
    madvise = set_madvise_for_mmap(True)
    shape = (500, 10)
    mat = np.random.normal(size=shape)
    memvecs = np.memmap(p, dtype=np.float32, mode="w+", shape=shape)
    memvecs[:] = mat[:]
    del memvecs

    mat = np.memmap(p, dtype=np.float32, mode="r", shape=shape)

    assert madvise(mat.ctypes.data, mat.size * mat.dtype.itemsize, 1) == 0


def _remove_full_matrix(vectors, svd_res, weights=None):
    """The formula on the whole matrix at once, as before the blockwise removal."""
    components = svd_res[1].astype(np.float32)
    coefficients = vectors.dot(components.T)
    if weights is not None:
        coefficients *= weights.astype(np.float32)[None, :]
    return vectors - coefficients.dot(components)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_remove_components_blockwise_matches_full_matrix(monkeypatch, inplace, weighted):
    """Block boundaries, including a short last block, give the same result."""
    from fse.models import utils

    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 7)
    rng = np.random.default_rng(0)
    vectors = rng.random((50, 10), dtype=np.float32)
    svd_res = compute_principal_components(vectors, components=3)
    weights = np.array([0.5, 0.3, 0.2], dtype=np.float32) if weighted else None
    expected = _remove_full_matrix(vectors, svd_res, weights)

    out = remove_principal_components(vectors, svd_res, weights=weights, inplace=inplace)
    result = vectors if inplace else out
    # BLAS may sum a block in another order than the whole matrix: float32 rounding only.
    assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_remove_components_memory_is_bounded_by_a_block(tmp_path):
    """Removing components from a memmap allocates about one block, not the whole matrix."""
    import tracemalloc

    from fse.models import utils

    rows, dim = 200_000, 50
    vectors = np.memmap(tmp_path / "sv", dtype=np.float32, mode="w+", shape=(rows, dim))
    vectors[:] = np.random.default_rng(0).random((rows, dim), dtype=np.float32)
    svd_res = compute_principal_components(vectors, components=2)

    tracemalloc.start()
    remove_principal_components(vectors, svd_res, inplace=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    matrix = rows * dim * 4
    block = utils.REMOVAL_BLOCK_ROWS * dim * 4
    assert peak < 3 * block
    assert peak < matrix / 4


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("memmap", [False, True])
def test_remove_components_in_parallel_equals_serial(monkeypatch, tmp_path, weighted, memmap):
    """Blocks removed by several threads give exactly the serial result."""
    from fse.models import utils

    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 7)
    rng = np.random.default_rng(1)
    data = rng.random((64, 10), dtype=np.float32)
    svd_res = compute_principal_components(data, components=3)
    weights = np.array([0.5, 0.3, 0.2], dtype=np.float32) if weighted else None
    serial = data.copy()
    remove_principal_components(serial, svd_res, weights=weights, inplace=True)
    if memmap:
        vectors = np.memmap(tmp_path / "sv", dtype=np.float32, mode="w+", shape=data.shape)
        vectors[:] = data
    else:
        vectors = data.copy()
    remove_principal_components(vectors, svd_res, weights=weights, inplace=True, workers=4)
    np.testing.assert_array_equal(np.asarray(vectors), serial)
    out = remove_principal_components(data, svd_res, weights=weights, inplace=False, workers=4)
    np.testing.assert_array_equal(out, serial)


def test_remove_components_in_parallel_raises_the_error_of_a_block(monkeypatch):
    """An error in one block reaches the caller."""
    from fse.models import utils

    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 4)
    vectors = np.arange(40, dtype=np.int64).reshape(10, 4)
    svd_res = (np.ones(1, dtype=np.float32), np.full((1, 4), 0.5, dtype=np.float32))
    with pytest.raises(TypeError):
        remove_principal_components(vectors, svd_res, inplace=True, workers=3)


def _blas_limits():
    from threadpoolctl import threadpool_info

    return [(i["filepath"], i["num_threads"]) for i in threadpool_info() if i["user_api"] == "blas"]


class _FakeLimits:
    """Records threadpool_limits calls; restore_original_limits marks the end."""

    calls: ClassVar[list] = []

    def __init__(self, limits, user_api):
        _FakeLimits.calls.append(limits)

    def restore_original_limits(self):
        _FakeLimits.calls.append("restored")


@pytest.fixture
def fake_limits(monkeypatch):
    from fse.models import utils

    _FakeLimits.calls = []
    monkeypatch.setattr(utils, "threadpool_limits", _FakeLimits)
    monkeypatch.setattr(utils, "usable_cpus", lambda: 16)
    return _FakeLimits.calls


def test_blas_threads_shares_the_cores_and_restores_the_limits(fake_limits):
    """With several workers BLAS gets its share of the cores, also after an error."""
    from fse.models import utils

    with utils.blas_threads(1):
        pass
    assert fake_limits == []
    with utils.blas_threads(4):
        pass
    with pytest.raises(RuntimeError), utils.blas_threads(32):
        raise RuntimeError("worker failed")
    assert fake_limits == [4, "restored", 1, "restored"]


def test_overlapping_blas_threads_restore_once_in_any_order(fake_limits):
    """Blocks in two threads that end in the other order keep the smallest active limit
    and restore the original limits only when the last one ends."""
    from fse.models import utils

    entered, release_first = threading.Event(), threading.Event()

    def first():
        with utils.blas_threads(4):  # 4 threads
            entered.set()
            release_first.wait()

    thread = threading.Thread(target=first)
    thread.start()
    entered.wait()
    second = utils.blas_threads(8)  # 2 threads
    second.__enter__()
    release_first.set()
    thread.join()
    assert fake_limits == [4, 2, 2]  # the first ended: the second's limit stays
    second.__exit__(None, None, None)
    assert fake_limits == [4, 2, 2, "restored"]


def test_blas_threads_leaves_the_real_limits_as_they_were():
    """The real threadpoolctl limits are back after the block."""
    from fse.models import utils

    before = _blas_limits()
    with utils.blas_threads(64):
        pass
    assert _blas_limits() == before


def test_usable_cpus_is_positive():
    from fse.models import utils

    assert utils.usable_cpus() >= 1


def test_remove_components_keeps_numpy_result_types():
    """Integers give a float result, and in-place removal refuses to truncate."""
    vectors = np.arange(40, dtype=np.int64).reshape(10, 4)
    svd_res = (np.ones(1, dtype=np.float32), np.full((1, 4), 0.5, dtype=np.float32))
    out = remove_principal_components(vectors, svd_res, inplace=False)
    expected = vectors - vectors.dot(svd_res[1].T).dot(svd_res[1])
    assert out.dtype == expected.dtype
    assert_allclose(out, expected)
    with pytest.raises(TypeError):
        remove_principal_components(vectors, svd_res, inplace=True)


@pytest.mark.parametrize(
    ("components", "update", "rows_before", "new_rows", "dim", "message"),
    [
        (3, False, 0, 2, 4, "training would give 2 vectors of dimension 4"),
        (5, False, 10, 10, 4, "training would give 10 vectors of dimension 4"),
        (3, True, 1, 1, 4, "training would give 2 vectors"),
    ],
)
def test_check_components_fit_rejects(components, update, rows_before, new_rows, dim, message):
    with pytest.raises(ValueError, match=message):
        check_components_fit(components, None, update, rows_before, new_rows, dim)


def test_check_components_fit_accepts_reuse_and_zero():
    learned = (np.ones(3, dtype=np.float32), np.eye(3, 4, dtype=np.float32))
    # Two rows could not give three components: only the reuse lets this pass.
    check_components_fit(3, learned, True, 1, 1, 4)
    check_components_fit(0, None, False, 0, 1, 4)
    check_components_fit(2, None, True, 1, 1, 4)
