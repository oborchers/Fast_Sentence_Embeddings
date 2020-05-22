#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2020 Oliver Borchers

from shared_imports import *

from fse.models.utils import (
    compute_principal_components,
    remove_principal_components,
    get_ft_word_vector,
)


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
        m = np.random.uniform(size=(int(2e6), 100)).astype(np.float32)
        out = compute_principal_components(vectors=m, cache_size_gb=0.2)
        self.assertEqual(2, len(out))
        self.assertEqual(1, len(out[1]))
        self.assertEqual(np.float32, out[1].dtype)

    def test_remove_components_inplace(self):
        m = np.ones((500, 10), dtype=np.float32)
        out = compute_principal_components(vectors=m)
        remove_principal_components(m, svd_res=out)
        self.assertTrue(np.allclose(0.0, m, atol=1e-5))

    def test_remove_components(self):
        m = np.ones((500, 10), dtype=np.float32)
        out = compute_principal_components(vectors=m)
        res = remove_principal_components(m, svd_res=out, inplace=False)
        self.assertTrue(np.allclose(1.0, res, atol=1e-5))

    def test_remove_weighted_components(self):
        m = np.ones((500, 10), dtype=np.float32)
        out = compute_principal_components(vectors=m)
        remove_principal_components(m, svd_res=out, weights=np.array([0.5]))
        self.assertTrue(np.allclose(0.75, m))

    def test_madvise(self):
        from pathlib import Path
        from sys import platform
        from fse.models.utils import set_madvise_for_mmap

        if platform in ["linux", "linux2", "darwin", "aix"]:
            p = Path("fse/test/test_data/test_vectors")
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

    def test_ft_word_vector(self):
        from fse.models.average import Average

        model = Average(FT_DET)

        word = "123456789"
        vec = get_ft_word_vector(word, model)

        self.assertIsInstance(vec, np.ndarray)
        self.assertTrue(np.allclose(max(vec), 979980.94))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
    )
    unittest.main()
