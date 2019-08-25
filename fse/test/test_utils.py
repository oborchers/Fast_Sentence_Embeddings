import logging
import unittest

import numpy as np

from fse.models.utils import compute_principal_components, remove_principal_components

logger = logging.getLogger(__name__)


class TestUtils(unittest.TestCase):

    def test_compute_components(self):
        m = np.random.uniform(size=(500, 10)).astype(np.float32)
        out = compute_principal_components(vectors = m)
        self.assertEqual(2, len(out))
        self.assertEqual(1, len(out[1]))
        self.assertEqual(np.float32, out[1].dtype)

        m = np.random.uniform(size=(500, 10))
        out = compute_principal_components(vectors = m, components=5)
        self.assertEqual(2, len(out))
        self.assertEqual(5, len(out[1]))
    
    def test_remove_components_inplace(self):
        m = np.ones((500,10), dtype=np.float32)
        out = compute_principal_components(vectors = m)
        remove_principal_components(m, svd_res=out)
        self.assertTrue(np.allclose(0., m, atol=1e-5))
    
    def test_remove_components(self):
        m = np.ones((500,10), dtype=np.float32)
        out = compute_principal_components(vectors = m)
        res = remove_principal_components(m, svd_res=out, inplace=False)
        self.assertTrue(np.allclose(1., res, atol=1e-5))

    def test_remove_weighted_components(self):
        m = np.ones((500,10), dtype=np.float32)
        out = compute_principal_components(vectors = m)
        remove_principal_components(m, svd_res=out, weights=np.array([0.5]))
        self.assertTrue(np.allclose(0.75, m))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()