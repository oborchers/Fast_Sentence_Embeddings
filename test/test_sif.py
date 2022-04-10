import logging
import unittest
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from fse.inputs import IndexedLineDocument
from fse.models.sif import SIF

logger = logging.getLogger(__name__)

TEST_DATA = Path(__file__).parent / "test_data"
CORPUS = TEST_DATA / "test_sentences.txt"
DIM = 50
W2V = Word2Vec(min_count=1, vector_size=DIM)
with open(CORPUS, "r") as file:
    SENTENCES = [l.split() for _, l in enumerate(file)]
W2V.build_vocab(SENTENCES)


class TestSIFFunctions(unittest.TestCase):
    def setUp(self):
        self.sentences = IndexedLineDocument(CORPUS)
        self.model = SIF(W2V, lang_freq="en")

    def test_parameter_sanity(self):
        with self.assertRaises(ValueError):
            m = SIF(W2V, alpha=-1)
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
        self.model.svd_res = (
            np.ones_like(self.model.word_weights, dtype=int),
            np.array(0, dtype=np.float32),
        )
        with self.assertRaises(TypeError):
            self.model._check_dtype_santiy()

    def test_dtype_sanity_svd_vecs(self):
        self.model.svd_res = (
            np.array(0, dtype=np.float32),
            np.ones_like(self.model.word_weights, dtype=int),
        )
        with self.assertRaises(TypeError):
            self.model._check_dtype_santiy()

    def test_compute_sif_weights(self):
        w = "Good"
        pw = 1.916650481770269e-08
        alpha = self.model.alpha
        sif = alpha / (alpha + pw)

        idx = self.model.wv.key_to_index[w]
        self.model._compute_sif_weights()
        self.assertTrue(np.allclose(self.model.word_weights[idx], sif))

    def test_train(self):
        output = self.model.train(self.sentences)
        self.assertEqual((100, 1450), output)
        self.assertTrue(np.isfinite(self.model.sv.vectors).all())
        self.assertEqual(2, len(self.model.svd_res))

    def test_save_issue(self):
        model = SIF(W2V)
        model.train(self.sentences)

        p = TEST_DATA / "test_emb.model"
        model.save(str(p))
        model = SIF.load(str(p))
        p.unlink()

        self.assertEqual(2, len(model.svd_res))
        model.sv.similar_by_sentence("test sentence".split(), model=model)

    def test_broken_vocab(self):
        w2v = Word2Vec(min_count=1, vector_size=DIM)
        with open(CORPUS, "r") as file:
            w2v.build_vocab([l.split() for l in file])

        for k in w2v.wv.key_to_index:
            w2v.wv.set_vecattr(k, "count", -1)

        model = SIF(w2v)
        with self.assertRaises(ValueError):
            model.train(self.sentences)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
    )
    unittest.main()
