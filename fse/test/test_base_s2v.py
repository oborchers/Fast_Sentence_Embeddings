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

    def test_init_w_empty_ft_model(self):
        ft = FastText(min_count=1, size=DIM)
        ft.wv.vectors = np.zeros(10)
        ft.wv.vectors_ngrams = None
        with self.assertRaises(RuntimeError):
            BaseSentence2VecModel(ft)

    def test_init_w_incompatible_ft_model(self):
        ft = FastText(min_count=1, size=DIM, compatible_hash=False)
        with self.assertRaises(RuntimeError):
            BaseSentence2VecModel(ft)

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

    def test_scan_w_list(self):
        se = BaseSentence2VecModel(W2V)
        with self.assertRaises(TypeError):
            se.scan_sentences(SENTENCES)

    def test_scan_w_IndexedSentence(self):
        se = BaseSentence2VecModel(W2V)
        id_sent = [IndexedSentence(s, i) for i,s in enumerate(SENTENCES)]
        self.assertTrue(
            (100, 1450, 14, 0) == se.scan_sentences(id_sent, progress_per=0)
            )

    def test_scan_w_wrong_IndexedSentence(self):
        se = BaseSentence2VecModel(W2V)
        id_sent = [IndexedSentence(s, str(i)) for i,s in enumerate(SENTENCES)]
        with self.assertRaises(TypeError):
            se.scan_sentences(id_sent)

    def test_scan_w_empty(self):
        se = BaseSentence2VecModel(W2V)
        for i in [5, 10, 15]:
            SENTENCES[i] = []
        self.assertEqual(3, se.scan_sentences([IndexedSentence(s, i) for i,s in enumerate(SENTENCES)])[-1])

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
            se.train([IndexedSentence(s, i) for i,s in enumerate(SENTENCES)])

    def test_log_end(self):
        se = BaseSentence2VecModel(W2V)
        se._log_train_end(eff_sentences=2000, eff_words=4000, overall_time=10)

    def test_child_requirements(self):
        se = BaseSentence2VecModel(W2V)

        with self.assertRaises(NotImplementedError):
            se._do_train_job(None)
        with self.assertRaises(NotImplementedError):
            se._pre_train_calls()
        with self.assertRaises(NotImplementedError):
            se._post_train_calls()
        with self.assertRaises(NotImplementedError):
            se._check_parameter_sanity()
        with self.assertRaises(NotImplementedError):
            se._check_dtype_santiy()  

    def test_pre_training_sanity(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)
        for w in ft.wv.vocab.keys():
            ft.wv.vocab[w].count = 1

        se = BaseSentence2VecModel(ft)

        # Just throws multiple warnings warning
        se._check_pre_training_sanity(1,1,1)
        
        with self.assertRaises(ValueError):
            se._check_pre_training_sanity(0,1,1)
        with self.assertRaises(ValueError):
            se._check_pre_training_sanity(1,0,1)
        with self.assertRaises(ValueError):
            se._check_pre_training_sanity(1,1,0)
        
        se.word_weights = np.ones(20, dtype=bool)
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)
        se.sv.vectors = np.zeros((20,20), dtype=int)
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)
        se.wv.vectors = np.zeros((20,20), dtype=np.float64)
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)
        se.vectors_ngrams = np.ones(30, dtype=np.float16)
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)

        se.word_weights = None
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)
        se.sv.vectors = None
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)
        del se.sv.vectors
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)

        se.wv.vectors_ngrams = []
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)
        se.wv.vectors = []
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)
        se.wv = None
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)

    def test_post_training_sanity(self):
        w2v = Word2Vec()
        w2v.build_vocab(SENTENCES)
        se = BaseSentence2VecModel(w2v)
        se.prep.prepare_vectors(se.sv, 20)
        with self.assertRaises(ValueError):
            se._check_post_training_sanity(0,1)
        with self.assertRaises(ValueError):
            se._check_post_training_sanity(1,0)
        
        se.sv.vectors[5,3] = np.inf
        with self.assertRaises(RuntimeError):
            se._check_post_training_sanity(1,1)
        
        se.wv.vectors[50,3] = np.nan
        with self.assertRaises(RuntimeError):
            se._check_post_training_sanity(1,1)

    def test_move_vectors_to_disk_w2v(self):
        se = BaseSentence2VecModel(W2V)
        p = Path("fse/test/test_data/test_vecs")
        p_target = Path("fse/test/test_data/test_vecs_wv.vectors")
        se.wv.vectors[0,1] = 10
        vecs = se.wv.vectors.copy()
        output = se._move_vectors_to_disk(se.wv.vectors, name="wv", mapfile_path=str(p.absolute()))
        self.assertTrue(p_target.exists())
        self.assertFalse(output.flags.writeable)
        self.assertTrue((vecs == output).all())
        p_target.unlink()

    def test_move_vectors_to_disk_wo_file(self):
        se = BaseSentence2VecModel(W2V)
        with self.assertRaises(TypeError):
            output = se._move_vectors_to_disk(se.wv.vectors)

    def test_move_w2v_vectors_to_disk_from_init(self):
        p = Path("fse/test/test_data/test_vecs")
        se = BaseSentence2VecModel(W2V, mapfile_path=str(p.absolute()), wv_from_disk=True)
        p_target = Path("fse/test/test_data/test_vecs_wv.vectors")
        self.assertTrue(p_target.exists())
        self.assertFalse(se.wv.vectors.flags.writeable)
        p_target.unlink()

    def test_move_ft_vectors_to_disk_from_init(self):
        ft = FastText(min_count=1, size=DIM)
        ft.build_vocab(SENTENCES)

        p = Path("fse/test/test_data/test_vecs")
        p_target_wv = Path("fse/test/test_data/test_vecs_wv.vectors")
        p_target_ngram = Path("fse/test/test_data/test_vecs_ngrams.vectors")

        se = BaseSentence2VecModel(ft, mapfile_path=str(p.absolute()), wv_from_disk=True)

        self.assertTrue(p_target_wv.exists())
        self.assertFalse(se.wv.vectors.flags.writeable)

        self.assertTrue(p_target_ngram.exists())
        self.assertFalse(se.wv.vectors_ngrams.flags.writeable)

        p_target_wv.unlink()
        p_target_ngram.unlink()

    def test_train_manager(self):
        se = BaseSentence2VecModel(W2V, workers=2)
        def temp_train_job(data_iterable):
            v1 = v2 = sum(1 for _ in data_iterable)
            return v1*2, v2*3
        se._do_train_job = temp_train_job
        job_output = se._train_manager(data_iterable=[IndexedSentence(s, i) for i,s in enumerate(SENTENCES)], total_sentences=len(SENTENCES),report_delay=0.01)
        self.assertEqual((100,200,300), job_output)

    def test_remaining_thread_funcs(self):
        # TODO
        pass


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
