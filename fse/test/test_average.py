#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers


"""
Automated tests for checking the average model.
"""


import logging
import unittest

from pathlib import Path

import numpy as np

from fse.models.average import Average
from fse.models.average import train_average_np
from fse.models.average_inner import train_average_cy
from fse.models.average_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
from fse.models.inputs import IndexedSentence

from gensim.models import Word2Vec

logger = logging.getLogger(__name__)

CORPUS = Path("fse/test/test_data/test_sentences.txt")
DIM = 5
W2V = Word2Vec(min_count=1, size=DIM)
SENTENCES = [l.split() for i, l in enumerate(open(CORPUS, "r"))]
W2V.build_vocab(SENTENCES)
W2V.wv.vectors[:,] = np.arange(len(W2V.wv.vectors), dtype=np.float32)[:, None]

class TestAverageFunctions(unittest.TestCase):
    def setUp(self):
        self.sentences = [["They", "admit"], ["So", "Apple", "bought", "buds"], ["go", "12345"]]
        self.sentences = [IndexedSentence(s, i) for i,s in enumerate(self.sentences)]
        self.model = Average(W2V)
        self.model.prep.prepare_vectors(sv=self.model.sv, total_sentences=len(self.sentences), update=False)
        self.model._pre_train_calls()

    def test_average_train_np(self):
        self.model.sv.vectors = np.zeros_like(self.model.sv.vectors, dtype=np.float32)
        output = train_average_np(self.model, self.sentences)
        self.assertEqual((3, 7), output)
        self.assertTrue((183 == self.model.sv[0]).all())
        self.assertTrue((164.5 == self.model.sv[1]).all())
        self.assertTrue((self.model.wv.vocab["go"].index == self.model.sv[2]).all())

    def test_cython(self):
        self.assertTrue(FAST_VERSION)
        self.assertEqual(10000,MAX_WORDS_IN_BATCH)

    def test_average_train_cy(self):
        self.model.sv.vectors = np.zeros_like(self.model.sv.vectors, dtype=np.float32)
        output = train_average_cy(self.model, self.sentences)
        self.assertEqual((3, 7), output)
        self.assertTrue((183 == self.model.sv[0]).all())
        self.assertTrue((164.5 == self.model.sv[1]).all())
        self.assertTrue((self.model.wv.vocab["go"].index == self.model.sv[2]).all())

    def test_cy_equal_np(self):
        m1 = Average(W2V)
        m1.prep.prepare_vectors(sv=m1.sv, total_sentences=len(self.sentences), update=False)
        m1._pre_train_calls()
        o1 = train_average_np(m1, self.sentences)

        m2 = Average(W2V)
        m2.prep.prepare_vectors(sv=m2.sv, total_sentences=len(self.sentences), update=False)
        m2._pre_train_calls()
        o2 = train_average_cy(m2, self.sentences)

        self.assertEqual(o1, o2)
        self.assertTrue((m1.sv.vectors == m2.sv.vectors).all())

    def test_cy_single_equal_multi(self):
        m1 = Average(W2V, workers=1)
        m1.train_average = train_average_cy
        o1 = m1.train([IndexedSentence(s, i) for i,s in enumerate(SENTENCES)])
        
        m2 = Average(W2V, workers=2)
        m2.train_average = train_average_cy
        o2 = m2.train([IndexedSentence(s, i) for i,s in enumerate(SENTENCES)])
        
        self.assertEqual(o1, o2)
        self.assertTrue((m1.sv.vectors == m2.sv.vectors).all())

    def test_do_train_job(self):
        self.model.prep.prepare_vectors(sv=self.model.sv, total_sentences=len(SENTENCES), update=True)
        self.assertEqual((100,1450), self.model._do_train_job([IndexedSentence(s, i) for i,s in enumerate(SENTENCES)]))
        self.assertEqual((103,DIM), self.model.sv.vectors.shape)

    def test_train(self):
        self.assertEqual((100,1450), self.model.train([IndexedSentence(s, i) for i,s in enumerate(SENTENCES)]))
    
    def test_train_single_from_disk(self):
        p = Path("fse/test/test_data/test_vecs")
        p_res = Path("fse/test/test_data/test_vecs.vectors")
        p_target = Path("fse/test/test_data/test_vecs_wv.vectors")

        se1 = Average(W2V)
        se2 = Average(W2V, mapfile_path=str(p.absolute()), wv_from_disk=True)
        se1.train([IndexedSentence(s, i) for i,s in enumerate(SENTENCES)])
        se2.train([IndexedSentence(s, i) for i,s in enumerate(SENTENCES)])

        self.assertTrue(p_target.exists())
        self.assertTrue((se1.wv.vectors == se2.wv.vectors).all())
        self.assertFalse(se2.wv.vectors.flags.writeable)

        self.assertTrue((se1.sv.vectors == se2.sv.vectors).all())
        p_res.unlink()
        p_target.unlink()

    def test_train_multi_from_disk(self):
        p = Path("fse/test/test_data/test_vecs")
        p_res = Path("fse/test/test_data/test_vecs.vectors")
        p_target = Path("fse/test/test_data/test_vecs_wv.vectors")

        se1 = Average(W2V, workers=2)
        se2 = Average(W2V, workers=2,mapfile_path=str(p.absolute()), wv_from_disk=True)
        se1.train([IndexedSentence(s, i) for i,s in enumerate(SENTENCES)])
        se2.train([IndexedSentence(s, i) for i,s in enumerate(SENTENCES)])

        self.assertTrue(p_target.exists())
        self.assertTrue((se1.wv.vectors == se2.wv.vectors).all())
        self.assertFalse(se2.wv.vectors.flags.writeable)

        self.assertTrue((se1.sv.vectors == se2.sv.vectors).all())
        p_res.unlink()
        p_target.unlink()

    def test_check_parameter_sanity(self):
        se = Average(W2V)
        se.word_weights = np.full(20, 2., dtype=np.float32)
        with self.assertRaises(ValueError):
            se._check_parameter_sanity()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()