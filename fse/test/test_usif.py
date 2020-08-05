#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <oliver-borchers@outlook.de>
# Copyright (C) 2020 Oliver Borchers

from fse.models.usif import uSIF

from fse.test.model_shared_imports import *


class TestuSIFFunctions(unittest.TestCase):
    def setUp(self):
        self.sentences = IndexedLineDocument(CORPUS)
        self.model = uSIF(W2V_RNG_LRG, lang_freq="en")

    def test_parameter_sanity(self):
        with self.assertRaises(ValueError):
            m = uSIF(W2V_RNG_LRG, length=0)
            m._check_parameter_sanity()
        with self.assertRaises(ValueError):
            m = uSIF(W2V_RNG_LRG, components=-1, length=11)
            m._check_parameter_sanity()

    def test_pre_train_calls(self):
        kwargs = {"average_length": 10}
        self.model._pre_train_calls(**kwargs)
        self.assertEqual(10, self.model.length)

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

    def test_compute_usif_weights(self):
        w = "Good"
        pw = 1.916650481770269e-08
        idx = self.model.wv.vocab[w].index
        self.model.length = 11
        a = 0.17831555484795414
        usif = a / ((a / 2) + pw)
        self.model._compute_usif_weights()
        self.assertTrue(np.allclose(self.model.word_weights[idx], usif))

    def test_train(self):
        output = self.model.train(self.sentences)
        self.assertEqual((100, 1450), output)
        self.assertTrue(np.isfinite(self.model.sv.vectors).all())

    def test_broken_vocab(self):
        w2v = Word2Vec(min_count=1, size=50)
        w2v.build_vocab([l.split() for l in open(CORPUS, "r")])
        for k in w2v.wv.vocab:
            w2v.wv.vocab[k].count = np.nan

        model = uSIF(w2v)

        with self.assertRaises(RuntimeError):
            model.train(self.sentences)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
    )
    unittest.main()
