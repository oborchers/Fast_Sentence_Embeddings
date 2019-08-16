#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers


"""
Automated tests for checking the base_s2v class.
"""

import logging
import unittest

from pathlib import Path

import numpy as np

from fse.models.base_s2v import BaseSentence2VecModel, BaseSentence2VecPreparer
from fse.models.inputs import IndexedSentence

from gensim.models import Word2Vec, FastText
from gensim.models.keyedvectors import BaseKeyedVectors

from wordfreq import get_frequency_dict

logger = logging.getLogger(__name__)


CORPUS = Path("fse/test/test_data/test_sentences.txt")
DIM = 5
W2V = Word2Vec(min_count=1, size=DIM)
SENTENCES = [l.split() for i, l in enumerate(open(CORPUS, "r"))]
W2V.build_vocab(SENTENCES)


class TestBaseSentence2VecModelFunctions(unittest.TestCase):
    def test_init_wo_model(self):
        with self.assertRaises(TypeError):
            BaseSentence2VecModel()

    def test_init_w_wrong_model(self):
        with self.assertRaises(RuntimeError):
            BaseSentence2VecModel(int)

    def test_init_w_empty_w2v_model(self):
        with self.assertRaises(RuntimeError):
            w2v = Word2Vec()
            del w2v.wv.vectors
            BaseSentence2VecModel(w2v)

    def test_init_w_ft_model(self):
        with self.assertRaises(NotImplementedError):
            BaseSentence2VecModel(FastText(min_count=1, size=DIM))

    def test_include_model(self):
        se = BaseSentence2VecModel(W2V)
        self.assertTrue(isinstance(se.wv, BaseKeyedVectors))
    
    def test_model_w_language(self):
        se = BaseSentence2VecModel(W2V, lang_freq="en")
        freq = int((2**31 - 1) * get_frequency_dict("en", wordlist="best")["help"])
        self.assertEqual(freq, se.wv.vocab["help"].count)
        self.assertEqual(21, se.wv.vocab["79"].count)

    def test_model_w_wrong_language(self):
        with self.assertRaises(ValueError):
            BaseSentence2VecModel(W2V, lang_freq="test")

    def test_save_load(self):
        se = BaseSentence2VecModel(W2V, lang_freq="en")
        p = Path("fse/test/test_data/test_emb.model")
        se.save(str(p.absolute()))
        self.assertTrue(p.exists())
        se2 = BaseSentence2VecModel.load(str(p.absolute()))
        self.assertTrue((se.wv.vectors == se2.wv.vectors).all())
        self.assertTrue(se.wv.index2word == se2.wv.index2word)
        self.assertEqual(se.workers, se2.workers)
        self.assertEqual(se.min_count, se2.min_count)
        p.unlink()

    def test_input_check(self):
        se = BaseSentence2VecModel(W2V)

        class BadIterator():
            def __init__(self):
                pass
        def Generator():
            for i in range(10):
                yield i

        with self.assertRaises(TypeError):
            se._check_input_data_sanity()
        with self.assertRaises(TypeError):
            se._check_input_data_sanity(data_iterable = "Hello there!")
        with self.assertRaises(TypeError):
            se._check_input_data_sanity(data_iterable = BadIterator())
        with self.assertRaises(TypeError):
            se._check_input_data_sanity(data_iterable = Generator())

    def test_scan_w_a_list(self):
        se = BaseSentence2VecModel(W2V)
        self.assertTrue((100, 1450, 14, 0) == se.scan_sentences(SENTENCES, progress_per=0))

    def test_scan_w_a_IndexedSentence(self):
        se = BaseSentence2VecModel(W2V)
        id_sent = [IndexedSentence(s, i) for i,s in enumerate(SENTENCES)]
        self.assertTrue(
            (100, 1450, 14, 0) == se.scan_sentences(id_sent)
            )

    def test_scan_w_empty(self):
        se = BaseSentence2VecModel(W2V)
        for i in [5, 10, 15]:
            SENTENCES[i] = []
        self.assertEqual(3, se.scan_sentences(SENTENCES)[-1])

    def test_scan_w_wrong_input(self):
        se = BaseSentence2VecModel(W2V)
        sentences = ["the dog hit the car", "he was very fast"]
        
        with self.assertRaises(TypeError):
            se.scan_sentences(sentences)
        with self.assertRaises(TypeError):
            se.scan_sentences([IndexedSentence(s, i) for i,s in enumerate(sentences)])
        with self.assertRaises(TypeError):
            se.scan_sentences([list(range(10) for _ in range(2))])

        with self.assertRaises(RuntimeError):
            se.scan_sentences([IndexedSentence(s, i+1) for i,s in enumerate(SENTENCES)])

    def test_estimate_memory(self):
        se = BaseSentence2VecModel(W2V)
        self.assertEqual(2000013704, se.estimate_memory(int(1e8))["Total"])

    def test_train(self):
        se = BaseSentence2VecModel(W2V)

        with self.assertRaises(NotImplementedError):
            se.train(SENTENCES)
        

class TestBaseSentence2VecPreparerFunctions(unittest.TestCase):

    def test_reset_vectors(self):
        se = BaseSentence2VecModel(W2V)
        trainables = BaseSentence2VecPreparer()
        trainables.reset_vectors(se.sv, 20)
        self.assertEqual((20,DIM), se.sv.vectors.shape)
        self.assertEqual(np.float32, se.sv.vectors.dtype)
        self.assertTrue((np.zeros((20, DIM)) == se.sv.vectors).all())
        self.assertTrue(se.sv.vectors_norm is None)

    def test_reset_vectors_memmap(self):
        p = Path("fse/test/test_data/test_vectors")
        p_target = Path("fse/test/test_data/test_vectors.vectors")
        se = BaseSentence2VecModel(W2V, mapfile_path=str(p.absolute()))
        trainables = BaseSentence2VecPreparer()
        trainables.reset_vectors(se.sv, 20)
        self.assertTrue(p_target.exists())
        self.assertEqual((20,DIM), se.sv.vectors.shape)
        self.assertEqual(np.float32, se.sv.vectors.dtype)
        self.assertTrue((np.zeros((20, DIM)) == se.sv.vectors).all())
        self.assertTrue(se.sv.vectors_norm is None)
        p_target.unlink()

    def test_update_vectors(self):
        se = BaseSentence2VecModel(W2V)
        trainables = BaseSentence2VecPreparer()
        trainables.reset_vectors(se.sv, 20)
        se.sv.vectors[:] = 1.
        trainables.update_vectors(se.sv, 10)
        self.assertEqual((30,DIM), se.sv.vectors.shape)
        self.assertEqual(np.float32, se.sv.vectors.dtype)
        self.assertTrue((np.ones((20, DIM)) == se.sv.vectors[:20]).all())
        self.assertTrue((np.zeros((10, DIM)) == se.sv.vectors[20:]).all())
        self.assertTrue(se.sv.vectors_norm is None)

    def test_update_vectors_memmap(self):
        p = Path("fse/test/test_data/test_vectors")
        p_target = Path("fse/test/test_data/test_vectors.vectors")
        se = BaseSentence2VecModel(W2V, mapfile_path=str(p.absolute()))
        trainables = BaseSentence2VecPreparer()
        trainables.reset_vectors(se.sv, 20)
        se.sv.vectors[:] = 1.
        trainables.update_vectors(se.sv, 10)
        self.assertTrue(p_target.exists())
        self.assertEqual((30,DIM), se.sv.vectors.shape)
        self.assertEqual(np.float32, se.sv.vectors.dtype)
        self.assertTrue((np.ones((20, DIM)) == se.sv.vectors[:20]).all())
        self.assertTrue((np.zeros((10, DIM)) == se.sv.vectors[20:]).all())
        self.assertTrue(se.sv.vectors_norm is None)
        p_target.unlink()

    def test_prepare_vectors(self):
        se = BaseSentence2VecModel(W2V)
        trainables = BaseSentence2VecPreparer()
        trainables.prepare_vectors(se.sv, 20, update=False)
        self.assertEqual((20,DIM), se.sv.vectors.shape)
        trainables.prepare_vectors(se.sv, 40, update=True)
        self.assertEqual((60,DIM), se.sv.vectors.shape)
        


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
