#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers


"""Automated tests for checking the sentence vectors."""

import logging
import unittest
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from fse.inputs import IndexedLineDocument, IndexedList
from fse.models.average import Average
from fse.models.sentencevectors import SentenceVectors, _l2_norm

logger = logging.getLogger(__name__)

TEST_DATA = Path(__file__).parent / "test_data"
CORPUS = TEST_DATA / "test_sentences.txt"
DIM = 5
W2V = Word2Vec(min_count=1, vector_size=DIM, seed=42)
with open(CORPUS, "r") as file:
    SENTENCES = [l.split() for _, l in enumerate(file)]
W2V.build_vocab(SENTENCES)

rng = np.random.default_rng(12345)
W2V.wv.vectors = rng.uniform(size=W2V.wv.vectors.shape).astype(np.float32)


class TestSentenceVectorsFunctions(unittest.TestCase):
    def setUp(self):
        self.sv = SentenceVectors(2)
        self.sv.vectors = np.arange(10).reshape(5, 2)

    def test_getitem(self):
        self.assertTrue(([0, 1] == self.sv[0]).all())
        self.assertTrue(([[0, 1], [4, 5]] == self.sv[[0, 2]]).all())

    def test_isin(self):
        self.assertIn(0, self.sv)
        self.assertNotIn(5, self.sv)

    def test_init_sims_wo_replace(self):
        self.sv.init_sims()
        self.assertIsNotNone(self.sv.vectors_norm)
        self.assertFalse((self.sv.vectors == self.sv.vectors_norm).all())

        v1 = self.sv.vectors[0]
        v1 = v1 / np.sqrt(np.sum(v1 ** 2))

        v2 = self.sv.vectors[1]
        v2 = v2 / np.sqrt(np.sum(v2 ** 2))

        self.assertTrue(np.allclose(v1, self.sv.vectors_norm[0]))
        self.assertTrue(np.allclose(v2, self.sv.vectors_norm[1]))
        self.assertTrue(np.allclose(v2, self.sv.get_vector(1, True)))

    def test_get_vector(self):
        self.assertTrue(([0, 1] == self.sv.get_vector(0)).all())
        self.assertTrue(([2, 3] == self.sv.get_vector(1)).all())

    def test_init_sims_w_replace(self):
        self.sv.init_sims(True)
        self.assertTrue(np.allclose(self.sv.vectors[0], self.sv.vectors_norm[0]))

    def test_init_sims_w_mapfile(self):
        p = TEST_DATA / "test_vectors"
        self.sv.mapfile_path = str(p.absolute())
        self.sv.init_sims()
        p = TEST_DATA / "test_vectors.vectors_norm"
        self.assertTrue(p.exists())
        p.unlink()

    def test_save_load(self):
        p = TEST_DATA / "test_vectors.vectors"
        self.sv.save(str(p.absolute()))
        self.assertTrue(p.exists())
        sv2 = SentenceVectors.load(str(p.absolute()))
        self.assertTrue((self.sv.vectors == sv2.vectors).all())
        p.unlink()

    def test_save_load_with_memmap(self):
        p = TEST_DATA / "test_vectors"
        p_target = TEST_DATA / "test_vectors.vectors"
        p_not_exists = TEST_DATA / "test_vectors.vectors.npy"

        sv = SentenceVectors(2, mapfile_path=str(p))

        shape = (1000, 1000)
        sv.vectors = np.ones(shape, dtype=np.float32)

        memvecs = np.memmap(p_target, dtype=np.float32, mode="w+", shape=shape)
        memvecs[:] = sv.vectors[:]
        del memvecs

        self.assertTrue(p_target.exists())
        sv.save(str(p.absolute()))
        self.assertTrue(p.exists())
        self.assertFalse(p_not_exists.exists())

        sv = SentenceVectors.load(str(p.absolute()))
        self.assertEqual(shape, sv.vectors.shape)

        for t in [p, p_target]:
            t.unlink()

    def test_len(self):
        self.assertEqual(5, len(self.sv))

    def test_similarity(self):
        v1 = self.sv.vectors[0]
        v1 = v1 / np.sqrt(np.sum(v1 ** 2))

        v2 = self.sv.vectors[1]
        v2 = v2 / np.sqrt(np.sum(v2 ** 2))

        self.assertTrue(np.allclose(v1.dot(v2), self.sv.similarity(0, 1)))
        self.assertTrue(np.allclose(1 - v1.dot(v2), self.sv.distance(0, 1)))

    def test_most_similar(self):
        sent_ind = IndexedList(SENTENCES)
        sentences = IndexedLineDocument(CORPUS)
        m = Average(W2V)
        m.train(sentences)
        o = m.sv.most_similar(positive=0)
        self.assertEqual(50, o[0][0])
        self.assertEqual(58, o[1][0])
        o = m.sv.most_similar(positive=0, indexable=sentences)
        self.assertEqual(
            "A basic phone which does what it is suppose to do , the thing i like most is the distinctive ring",
            o[0][0],
        )

        o = m.sv.most_similar(positive=0, indexable=sent_ind)
        self.assertEqual(
            "A basic phone which does what it is suppose to do , the thing i like most is the distinctive ring".split(),
            o[0][0][0],
        )

    def test_most_similar_vec(self):
        sentences = IndexedLineDocument(CORPUS)
        m = Average(W2V)
        m.train(sentences)
        m.sv.init_sims()
        v = m.sv.get_vector(0, use_norm=True)
        o = m.sv.most_similar(positive=v)
        self.assertEqual(0, o[0][0])
        self.assertEqual(50, o[1][0])
        self.assertEqual(58, o[2][0])

    def test_most_similar_vectors(self):
        sentences = IndexedLineDocument(CORPUS)
        m = Average(W2V)
        m.train(sentences)
        m.sv.init_sims()
        v = m.sv[[0, 1]]
        o = m.sv.most_similar(positive=v)
        self.assertEqual(10, o[0][0])
        self.assertEqual(11, o[1][0])

    def test_most_similar_wrong_indexable(self):
        def indexable(self):
            pass

        sentences = IndexedLineDocument(CORPUS)
        m = Average(W2V)
        m.train(sentences)
        with self.assertRaises(RuntimeError):
            m.sv.most_similar(positive=0, indexable=indexable)

    def test_most_similar_topn(self):
        sentences = IndexedLineDocument(CORPUS)
        m = Average(W2V)
        m.train(sentences)
        o = m.sv.most_similar(positive=0, topn=20)
        self.assertEqual(20, len(o))

    def test_most_similar_restrict_size(self):
        sentences = IndexedLineDocument(CORPUS)
        m = Average(W2V)
        m.train(sentences)
        o = m.sv.most_similar(positive=20, topn=20, restrict_size=5)
        self.assertEqual(5, len(o))

    def test_most_similar_restrict_size_tuple(self):
        sentences = IndexedLineDocument(CORPUS)
        m = Average(W2V)
        m.train(sentences)
        o = m.sv.most_similar(positive=20, topn=20, restrict_size=(5, 25))
        self.assertEqual(19, len(o))
        self.assertEqual(22, o[0][0])

        o = m.sv.most_similar(positive=1, topn=20, restrict_size=(5, 25))
        self.assertEqual(20, len(o))
        self.assertEqual(11, o[0][0])

        o = m.sv.most_similar(
            positive=1, topn=20, restrict_size=(5, 25), indexable=sentences
        )
        self.assertEqual(20, len(o))
        self.assertEqual(11, o[0][1])

    def test_similar_by_word(self):
        sentences = IndexedLineDocument(CORPUS)
        m = Average(W2V)
        m.train(sentences)
        o = m.sv.similar_by_word(word="the", wv=m.wv)
        self.assertEqual(5, o[0][0])
        o = m.sv.similar_by_word(word="the", wv=m.wv, indexable=sentences)
        self.assertEqual(5, o[0][1])

    def test_similar_by_vector(self):
        sentences = IndexedLineDocument(CORPUS)
        m = Average(W2V)
        m.train(sentences)
        o = m.sv.similar_by_vector(m.wv["the"])
        self.assertEqual(5, o[0][0])

    def test_similar_by_sentence(self):
        sentences = IndexedLineDocument(CORPUS)
        m = Average(W2V)
        m.train(sentences)
        o = m.sv.similar_by_sentence(sentence=["the", "product", "is", "good"], model=m)
        self.assertEqual(26, o[0][0])

    def test_similar_by_sentence_wrong_model(self):
        sentences = IndexedLineDocument(CORPUS)
        m = Average(W2V)
        m.train(sentences)
        with self.assertRaises(RuntimeError):
            m.sv.similar_by_sentence(
                sentence=["the", "product", "is", "good"], model=W2V
            )

    def test_l2_norm(self):
        out = np.random.normal(size=(200, 50)).astype(np.float32)
        result = _l2_norm(out, False)
        lens = np.sqrt(np.sum((result ** 2), axis=-1))
        self.assertTrue(np.allclose(1, lens, atol=1e-6))

        out = np.random.normal(size=(200, 50)).astype(np.float32)
        out = _l2_norm(out, True)
        lens = np.sqrt(np.sum((out ** 2), axis=-1))
        self.assertTrue(np.allclose(1, lens, atol=1e-6))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
    )
    unittest.main()
