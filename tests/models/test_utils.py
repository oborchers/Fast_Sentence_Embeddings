"""Automated tests for the principal component helpers."""

import sys
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_raises

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
    with patch("fse.models.utils.choice", wraps=np.random.choice) as choice:
        out = compute_principal_components(vectors=m, cache_size_gb=cache_size_gb)
    assert choice.call_args.kwargs["size"] == 500
    assert len(out[1]) == 1


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
