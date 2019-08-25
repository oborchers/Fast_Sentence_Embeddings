import logging
import unittest

from pathlib import Path

import numpy as np

from fse.models.sif import SIF, compute_principal_components, remove_principal_components
from fse.inputs import IndexedLineDocument

from gensim.models import Word2Vec

logger = logging.getLogger(__name__)

CORPUS = Path("fse/test/test_data/test_sentences.txt")
DIM = 50
W2V = Word2Vec(min_count=1, size=DIM)
SENTENCES = [l.split() for l in open(CORPUS, "r")]
W2V.build_vocab(SENTENCES)


class TestSIFFunctions(unittest.TestCase):
    def setUp(self):
        self.sentences = IndexedLineDocument(CORPUS)
        self.model = SIF(W2V, lang_freq="en")
    
    def test_parameter_sanity(self):
        with self.assertRaises(ValueError):
            m = SIF(W2V, alpha= -1)
            m._check_parameter_sanity()
        with self.assertRaises(ValueError):
            m = SIF(W2V, components=-1)
            m._check_parameter_sanity()
        with self.assertRaises(ValueError):
            m = SIF(W2V)
            m.word_weights = np.ones_like(m.word_weights) + 2
            m._check_parameter_sanity()

    def test_pre_train_calls(self):
        self.model._pre_train_calls()

    def test_post_train_calls(self):
        self.model.sv.vectors = np.ones((200, 10), dtype=np.float32)
        self.model._post_train_calls()
        self.assertTrue(np.allclose(self.model.sv.vectors, 0, atol=1e-5))
    
    def test_post_train_calls_no_removal(self):
        self.model.components = 0
        self.model.sv.vectors = np.ones((200, 10), dtype=np.float32)
        self.model._post_train_calls()
        self.assertTrue(np.allclose(self.model.sv.vectors, 1, atol=1e-5))
    
    def test_post_inference_calls(self):
        self.model.sv.vectors = np.ones((200, 10), dtype=np.float32)
        self.model._post_train_calls()

        output = np.ones((200, 10), dtype=np.float32)
        self.model._post_inference_calls(output=output)
        self.assertTrue(np.allclose(output, 0, atol=1e-5))

    def test_post_inference_calls_no_svd(self):
        self.model.sv.vectors = np.ones((200, 10), dtype=np.float32)
        self.model.svd_res = None
        with self.assertRaises(RuntimeError):
            self.model._post_inference_calls(output=None)
    
    def test_post_inference_calls_no_removal(self):
        self.model.components = 0
        self.model.sv.vectors = np.ones((200, 10), dtype=np.float32)
        self.model._post_train_calls()
        self.model._post_inference_calls(output=None)
        self.assertTrue(np.allclose(self.model.sv.vectors, 1, atol=1e-5))

    def test_dtype_sanity_word_weights(self):
        self.model.word_weights = np.ones_like(self.model.word_weights, dtype=int)
        with self.assertRaises(TypeError):
            self.model._check_dtype_santiy()
    
    def test_dtype_sanity_svd_vals(self):
        self.model.svd_res = (np.ones_like(self.model.word_weights, dtype=int), np.array(0, dtype=np.float32))
        with self.assertRaises(TypeError):
            self.model._check_dtype_santiy()

    def test_dtype_sanity_svd_vecs(self):
        self.model.svd_res = (np.array(0, dtype=np.float32), np.ones_like(self.model.word_weights, dtype=int))
        with self.assertRaises(TypeError):
            self.model._check_dtype_santiy()
    
    def test_compute_sif_weights(self):
        cs = 1095661426
        w = "Good"
        pw = 1.916650481770269e-08
        alpha = self.model.alpha
        sif = alpha / (alpha + pw)

        idx = self.model.wv.vocab[w].index
        self.model._compute_sif_weights()
        self.assertTrue(np.allclose(self.model.word_weights[idx], sif))

    def test_train(self):
        output = self.model.train(self.sentences)
        self.assertEqual((100,1450), output)
        self.assertTrue(np.isfinite(self.model.sv.vectors).all())

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()