"""Automated tests for the principal component helpers."""

import sys
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_raises
from sklearn.decomposition import TruncatedSVD

from fse.models.utils import (
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


def test_compute_large_components():
    m = np.random.uniform(size=(int(2e6), 10)).astype(np.float32)
    out = compute_principal_components(vectors=m, cache_size_gb=0.02)
    assert len(out) == 2
    assert len(out[1]) == 1
    assert out[1].dtype == np.float32


def test_compute_components_sample_size():
    """The sample follows cache_size_gb, also below one million vectors (#58)."""
    m = np.random.uniform(size=(1000, 10)).astype(np.float32)
    cache_size_gb = 500 * 10 * 4 / 1024**3
    fitted = []
    original_fit = TruncatedSVD.fit

    def recording_fit(self, x, *args, **kwargs):
        fitted.append(x.shape)
        return original_fit(self, x, *args, **kwargs)

    with patch.object(TruncatedSVD, "fit", recording_fit):
        out = compute_principal_components(vectors=m, cache_size_gb=cache_size_gb)
    assert fitted == [(500, 10)]
    assert len(out[1]) == 1


def test_compute_components_sampling_is_reproducible():
    m = np.random.default_rng(1).random((1000, 10), dtype=np.float32)
    cache_size_gb = 500 * 10 * 4 / 1024**3
    first = compute_principal_components(vectors=m, cache_size_gb=cache_size_gb)
    second = compute_principal_components(vectors=m, cache_size_gb=cache_size_gb)
    assert np.array_equal(first[1], second[1])


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
    assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


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


def test_sample_rows_in_file_order_equal_fancy_indexing(monkeypatch, tmp_path):
    """Reading the sample in file order gives the same matrix, so the same components."""
    from fse.models import utils

    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 7)
    rows, dim = 1_000, 8
    vectors = np.memmap(tmp_path / "sv", dtype=np.float32, mode="w+", shape=(rows, dim))
    vectors[:] = np.random.default_rng(0).random((rows, dim), dtype=np.float32)
    indices = np.random.default_rng(42).choice(rows, size=100, replace=False)
    assert np.array_equal(utils._take_rows_in_file_order(vectors, indices), vectors[indices])
