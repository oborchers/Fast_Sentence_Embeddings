import logging
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_raises

from fse.models.utils import compute_principal_components, remove_principal_components

logger = logging.getLogger(__name__)

TEST_DATA = Path(__file__).parent / "test_data"


class TestUtils(unittest.TestCase):
    def test_compute_components(self):
        m = np.random.uniform(size=(500, 10)).astype(np.float32)
        out = compute_principal_components(vectors=m)
        self.assertEqual(2, len(out))
        self.assertEqual(1, len(out[1]))
        self.assertEqual(np.float32, out[1].dtype)

        m = np.random.uniform(size=(500, 10))
        out = compute_principal_components(vectors=m, components=5)
        self.assertEqual(2, len(out))
        self.assertEqual(5, len(out[1]))

    def test_compute_large_components(self):
        m = np.random.uniform(size=(int(2e6), 10)).astype(np.float32)
        out = compute_principal_components(vectors=m, cache_size_gb=0.02)
        self.assertEqual(2, len(out))
        self.assertEqual(1, len(out[1]))
        self.assertEqual(np.float32, out[1].dtype)

    def test_compute_components_sample_size(self):
        """The sample follows cache_size_gb, also below one million vectors (#58)."""
        m = np.random.uniform(size=(1000, 10)).astype(np.float32)
        cache_size_gb = 500 * 10 * 4 / 1024**3
        with patch("fse.models.utils.choice", wraps=np.random.choice) as choice:
            out = compute_principal_components(vectors=m, cache_size_gb=cache_size_gb)
        self.assertEqual(500, choice.call_args.kwargs["size"])
        self.assertEqual(1, len(out[1]))

    def test_remove_components_inplace(self):
        m = np.ones((500, 10), dtype=np.float32)
        c = np.copy(m)
        out = compute_principal_components(vectors=m)
        remove_principal_components(m, svd_res=out)
        assert_allclose(m, 0.0, atol=1e-5)
        with assert_raises(AssertionError):
            assert_allclose(m, c)

    def test_remove_components(self):
        m = np.ones((500, 10), dtype=np.float32)
        c = np.copy(m)
        out = compute_principal_components(vectors=m)
        res = remove_principal_components(m, svd_res=out, inplace=False)
        assert_allclose(res, 0.0, atol=1e-5)
        assert_allclose(m, c)

    def test_remove_weighted_components_inplace(self):
        m = np.ones((500, 10), dtype=np.float32)
        c = np.copy(m)
        out = compute_principal_components(vectors=m)
        remove_principal_components(m, svd_res=out, weights=np.array([0.5]))
        assert_allclose(m, 0.75, atol=1e-5)
        with assert_raises(AssertionError):
            assert_allclose(m, c)

    def test_remove_weighted_components(self):
        m = np.ones((500, 10), dtype=np.float32)
        c = np.copy(m)
        out = compute_principal_components(vectors=m)
        res = remove_principal_components(
            m, svd_res=out, weights=np.array([0.5]), inplace=False
        )
        assert_allclose(res, 0.75, atol=1e-5)
        assert_allclose(m, c)

    def test_madvise(self):
        from sys import platform

        from fse.models.utils import set_madvise_for_mmap

        if platform in ["linux", "linux2", "darwin", "aix"]:
            p = TEST_DATA / "test_vectors"
            madvise = set_madvise_for_mmap(True)
            shape = (500, 10)
            mat = np.random.normal(size=shape)
            memvecs = np.memmap(p, dtype=np.float32, mode="w+", shape=shape)
            memvecs[:] = mat[:]
            del memvecs

            mat = np.memmap(p, dtype=np.float32, mode="r", shape=shape)

            self.assertEqual(
                madvise(mat.ctypes.data, mat.size * mat.dtype.itemsize, 1), 0
            )
            p.unlink()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
    )
    unittest.main()
