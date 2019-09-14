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

from fse.models.base_s2v import BaseSentence2VecModel, BaseSentence2VecPreparer, EPS

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

    def test_init_w_empty_vocab_model(self):
        with self.assertRaises(RuntimeError):
            w2v = Word2Vec()
            del w2v.wv.vocab
            BaseSentence2VecModel(w2v)

    def test_init_w_ft_model_wo_vecs(self):
        ft = FastText(SENTENCES, size=5)
        with self.assertRaises(RuntimeError):
            ft.wv.vectors_vocab = None
            BaseSentence2VecModel(ft)
        with self.assertRaises(RuntimeError):
            ft.wv.vectors_ngrams = None
            BaseSentence2VecModel(ft)

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
        se = BaseSentence2VecModel(W2V)
        p = Path("fse/test/test_data/test_emb.model")
        se.save(str(p.absolute()))
        self.assertTrue(p.exists())
        se2 = BaseSentence2VecModel.load(str(p.absolute()))
        self.assertTrue((se.wv.vectors == se2.wv.vectors).all())
        self.assertTrue(se.wv.index2word == se2.wv.index2word)
        self.assertEqual(se.workers, se2.workers)
        p.unlink()

    def test_save_load_with_memmap(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)
        shape = (1000, 1000)
        ft.wv.vectors = np.zeros(shape, np.float32)

        p = Path("fse/test/test_data/test_emb")
        p_vecs = Path("fse/test/test_data/test_emb_wv.vectors")
        p_ngrams = Path("fse/test/test_data/test_emb_ngrams.vectors")
        p_vocab = Path("fse/test/test_data/test_emb_vocab.vectors")

        p_not_exists = Path("fse/test/test_data/test_emb.wv.vectors.npy")

        se = BaseSentence2VecModel(ft, wv_mapfile_path=str(p))
        self.assertTrue(p_vecs.exists())
        self.assertTrue(p_ngrams.exists())
        self.assertTrue(p_vocab.exists())

        se.save(str(p.absolute()))
        self.assertTrue(p.exists())
        self.assertFalse(p_not_exists.exists())

        se = BaseSentence2VecModel.load(str(p.absolute()))
        self.assertFalse(se.wv.vectors_vocab.flags.writeable)
        self.assertEqual(shape, se.wv.vectors.shape)
        self.assertEqual((2000000, 5), se.wv.vectors_ngrams.shape)

        for p in [p, p_vecs, p_ngrams, p_vocab]:
            p.unlink()

    def test_map_all_vectors_to_disk(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)

        p = Path("fse/test/test_data/test_emb")
        p_vecs = Path("fse/test/test_data/test_emb_wv.vectors")
        p_ngrams = Path("fse/test/test_data/test_emb_ngrams.vectors")
        p_vocab = Path("fse/test/test_data/test_emb_vocab.vectors")

        se = BaseSentence2VecModel(ft, wv_mapfile_path=str(p))

        self.assertTrue(p_vecs.exists())
        self.assertTrue(p_ngrams.exists())
        self.assertTrue(p_vocab.exists())

        for p in [p_vecs, p_ngrams, p_vocab]:
            p.unlink()

    def test_input_check(self):
        se = BaseSentence2VecModel(W2V)

        class BadIterator():
            def __init__(self):
                pass

        with self.assertRaises(TypeError):
            se._check_input_data_sanity()
        with self.assertRaises(TypeError):
            se._check_input_data_sanity(data_iterable = None)
        with self.assertRaises(TypeError):
            se._check_input_data_sanity(data_iterable = "Hello there!")
        with self.assertRaises(TypeError):
            se._check_input_data_sanity(data_iterable = BadIterator())

    def test_scan_w_list(self):
        se = BaseSentence2VecModel(W2V)
        with self.assertRaises(TypeError):
            se.scan_sentences(SENTENCES)

    def test_str_rep(self):
        output = str(BaseSentence2VecModel(W2V))
        self.assertEqual("BaseSentence2VecModel based on Word2VecKeyedVectors, size=0", output)

    def test_scan_w_ituple(self):
        se = BaseSentence2VecModel(W2V)
        id_sent = [(s, i) for i,s in enumerate(SENTENCES)]
        stats = se.scan_sentences(id_sent, progress_per=0)

        self.assertEqual(100, stats["total_sentences"])
        self.assertEqual(1417, stats["total_words"])
        self.assertEqual(14, stats["average_length"])
        self.assertEqual(3, stats["empty_sentences"])
        self.assertEqual(100, stats["max_index"])

    def test_scan_w_wrong_tuple(self):
        se = BaseSentence2VecModel(W2V)
        id_sent = [(s, str(i)) for i,s in enumerate(SENTENCES)]
        with self.assertRaises(TypeError):
            se.scan_sentences(id_sent)

    def test_scan_w_empty(self):
        se = BaseSentence2VecModel(W2V)
        for i in [5, 10, 15]:
            SENTENCES[i] = []
        self.assertEqual(3, se.scan_sentences([(s, i) for i,s in enumerate(SENTENCES)])["empty_sentences"])

    def test_scan_w_wrong_input(self):
        se = BaseSentence2VecModel(W2V)
        sentences = ["the dog hit the car", "he was very fast"]
        
        with self.assertRaises(TypeError):
            se.scan_sentences(sentences)
        with self.assertRaises(TypeError):
            se.scan_sentences([(s, i) for i,s in enumerate(sentences)])
        with self.assertRaises(TypeError):
            se.scan_sentences([list(range(10) for _ in range(2))])

        with self.assertRaises(RuntimeError):
            se.scan_sentences([(s, i+1) for i,s in enumerate(SENTENCES)])
        with self.assertRaises(ValueError):
            se.scan_sentences([(s, i-1) for i,s in enumerate(SENTENCES)])
        
    def test_scan_w_many_to_one_input(self):
        se = BaseSentence2VecModel(W2V)
        output = se.scan_sentences([(s, 0) for i,s in enumerate(SENTENCES)])["max_index"]
        self.assertEqual(1, output)

    def test_estimate_memory(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)
        se = BaseSentence2VecModel(ft)
        self.assertEqual(2040025124, se.estimate_memory(int(1e8))["Total"])

    def test_train(self):
        se = BaseSentence2VecModel(W2V)
        with self.assertRaises(NotImplementedError):
            se.train([(s, i) for i,s in enumerate(SENTENCES)])

    def test_log_end(self):
        se = BaseSentence2VecModel(W2V)
        se._log_train_end(eff_sentences=2000, eff_words=4000, overall_time=10)

    def test_child_requirements(self):
        se = BaseSentence2VecModel(W2V)

        with self.assertRaises(NotImplementedError):
            se._do_train_job(None, None, None)
        with self.assertRaises(NotImplementedError):
            se._pre_train_calls()
        with self.assertRaises(NotImplementedError):
            se._post_train_calls()
        with self.assertRaises(NotImplementedError):
            se._check_parameter_sanity()
        with self.assertRaises(NotImplementedError):
            se._check_dtype_santiy()  
        with self.assertRaises(NotImplementedError):
            se._post_inference_calls()  

    def test_check_pre_train_san_no_wv(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)
        se = BaseSentence2VecModel(ft)
        se.wv = None
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)

    def test_check_pre_train_san_no_wv_len(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)
        se = BaseSentence2VecModel(ft)
        se.wv.vectors = []
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)

    def test_check_pre_train_san_no_ngrams_vectors(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)
        se = BaseSentence2VecModel(ft)
        se.wv.vectors_ngrams = []
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)
        se.wv.vectors_ngrams = [1]
        se.wv.vectors_vocab = []
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)

    def test_check_pre_train_san_no_sv_vecs(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)
        se = BaseSentence2VecModel(ft)
        se.sv.vectors = None
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)

    def test_check_pre_train_san_no_word_weights(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)
        se = BaseSentence2VecModel(ft)
        se.word_weights = None
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)

    def test_check_pre_train_san_incos_len(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)
        se = BaseSentence2VecModel(ft)
        se.word_weights = np.ones(20)
        with self.assertRaises(RuntimeError):
            se._check_pre_training_sanity(1,1,1)

    def test_check_pre_train_dtypes(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)
        se = BaseSentence2VecModel(ft)

        se.wv.vectors = np.zeros((len(se.wv.vocab),20), dtype=np.float64)
        with self.assertRaises(TypeError):
            se._check_pre_training_sanity(1,1,1)
        se.wv.vectors = np.zeros((len(se.wv.vocab),20), dtype=np.float32)

        se.wv.vectors_ngrams = np.ones(len(se.wv.vocab), dtype=np.float16)
        with self.assertRaises(TypeError):
            se._check_pre_training_sanity(1,1,1)
        se.wv.vectors_ngrams = np.ones(len(se.wv.vocab), dtype=np.float32)

        se.wv.vectors_vocab = np.ones(len(se.wv.vocab), dtype=np.float16)
        with self.assertRaises(TypeError):
            se._check_pre_training_sanity(1,1,1)
        se.wv.vectors_vocab = np.ones(len(se.wv.vocab), dtype=np.float32)

        se.sv.vectors = np.zeros((len(se.wv.vocab),20), dtype=int)
        with self.assertRaises(TypeError):
            se._check_pre_training_sanity(1,1,1)
        se.sv.vectors = np.zeros((len(se.wv.vocab),20), dtype=np.float32)

        se.word_weights = np.ones(len(se.wv.vocab), dtype=bool)
        with self.assertRaises(TypeError):
            se._check_pre_training_sanity(1,1,1)
        se.word_weights = np.ones(len(se.wv.vocab), dtype=np.float32)

    def test_check_pre_train_statistics(self):
        ft = FastText(min_count=1, size=5)
        ft.build_vocab(SENTENCES)
        se = BaseSentence2VecModel(ft)

        for v in se.wv.vocab:
            se.wv.vocab[v].count = 1

        # Just throws multiple warnings warning
        se._check_pre_training_sanity(1,1,1)
        
        with self.assertRaises(ValueError):
            se._check_pre_training_sanity(0,1,1)
        with self.assertRaises(ValueError):
            se._check_pre_training_sanity(1,0,1)
        with self.assertRaises(ValueError):
            se._check_pre_training_sanity(1,1,0)


    def test_post_training_sanity(self):
        w2v = Word2Vec()
        w2v.build_vocab(SENTENCES)
        se = BaseSentence2VecModel(w2v)
        se.prep.prepare_vectors(se.sv, 20)
        with self.assertRaises(ValueError):
            se._check_post_training_sanity(0,1)
        with self.assertRaises(ValueError):
            se._check_post_training_sanity(1,0)
            
    def test_move_ndarray_to_disk_w2v(self):
        se = BaseSentence2VecModel(W2V)
        p = Path("fse/test/test_data/test_vecs")
        p_target = Path("fse/test/test_data/test_vecs_wv.vectors")
        se.wv.vectors[0,1] = 10
        vecs = se.wv.vectors.copy()
        output = se._move_ndarray_to_disk(se.wv.vectors, name="wv", mapfile_path=str(p.absolute()))
        self.assertTrue(p_target.exists())
        self.assertFalse(output.flags.writeable)
        self.assertTrue((vecs == output).all())
        p_target.unlink()

    def test_move_ndarray_to_disk_wo_file(self):
        se = BaseSentence2VecModel(W2V)
        with self.assertRaises(TypeError):
            output = se._move_ndarray_to_disk(se.wv.vectors)

    def test_move_w2v_vectors_to_disk_from_init(self):
        p = Path("fse/test/test_data/test_vecs")
        se = BaseSentence2VecModel(W2V, wv_mapfile_path=str(p.absolute()))
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
        p_target_vocab = Path("fse/test/test_data/test_vecs_vocab.vectors")

        se = BaseSentence2VecModel(ft, wv_mapfile_path=str(p.absolute()))

        self.assertTrue(p_target_wv.exists())
        self.assertFalse(se.wv.vectors.flags.writeable)

        self.assertTrue(p_target_ngram.exists())
        self.assertFalse(se.wv.vectors_ngrams.flags.writeable)

        p_target_wv.unlink()
        p_target_ngram.unlink()
        p_target_vocab.unlink()

    def test_train_manager(self):
        se = BaseSentence2VecModel(W2V, workers=2)
        def temp_train_job(data_iterable, target, memory):
            v1 = v2 = sum(1 for _ in data_iterable)
            return v1*2, v2*3
        se._do_train_job = temp_train_job
        job_output = se._train_manager(data_iterable=[(s, i) for i,s in enumerate(SENTENCES)], total_sentences=len(SENTENCES),report_delay=0.01)
        self.assertEqual((100,200,300), job_output)

    def test_infer_method(self):
        se = BaseSentence2VecModel(W2V)
        def temp_train_job(data_iterable, target, memory):
            for i in data_iterable:
                target += 1
            return target

        def pass_method(**kwargs): pass
        se._post_inference_calls = pass_method
        se._do_train_job = temp_train_job
        output = se.infer([(s, i) for i,s in enumerate(SENTENCES)])
        self.assertTrue((100 == output).all())

    def test_infer_method_cy_overflow(self):
        se = BaseSentence2VecModel(W2V)
        
        from fse.models.average_inner import MAX_WORDS_IN_BATCH
        from fse.models.average_inner import train_average_cy
        def _do_train_job(data_iterable, target, memory):
            eff_sentences, eff_words = train_average_cy(model=se, indexed_sentences=data_iterable, target=target, memory=memory)
            return eff_sentences, eff_words

        def pass_method(**kwargs): pass
        se._post_inference_calls = pass_method
        se._do_train_job = _do_train_job
        tmp = []
        for i in range(20):
            tmp.extend(SENTENCES)
        bs = 0
        for i, s in enumerate(tmp):
            if bs >= MAX_WORDS_IN_BATCH:
                min_index = i
                break
            bs += len(s)
        sents = [(s, i) for i,s in enumerate(tmp)]
        output = se.infer(sents)
        output = output[i:]
        self.assertTrue((0 != output).all())

    def test_infer_many_to_one(self):
        se = BaseSentence2VecModel(W2V)
        def temp_train_job(data_iterable, target, memory):
            for i in data_iterable:
                target += 1
            return target
        def pass_method(**kwargs): pass
        se._post_inference_calls = pass_method
        se._do_train_job = temp_train_job
        output = se.infer([(s, 0) for i,s in enumerate(SENTENCES)])
        self.assertTrue((100 == output).all())
        self.assertEqual((1, 5), output.shape)

    def test_infer_use_norm(self):
        se = BaseSentence2VecModel(W2V)
        def temp_train_job(data_iterable, target, memory):
            for i in data_iterable:
                target += 1
            return target
        def pass_method(**kwargs): pass
        se._post_inference_calls = pass_method
        se._do_train_job = temp_train_job
        output = se.infer([(s, i) for i,s in enumerate(SENTENCES)], use_norm=True)

        self.assertTrue(np.allclose(1., np.sqrt(np.sum(output[0]**2))))

class TestBaseSentence2VecPreparerFunctions(unittest.TestCase):

    def test_reset_vectors(self):
        se = BaseSentence2VecModel(W2V)
        trainables = BaseSentence2VecPreparer()
        trainables.reset_vectors(se.sv, 20)
        self.assertEqual((20,DIM), se.sv.vectors.shape)
        self.assertEqual(np.float32, se.sv.vectors.dtype)
        self.assertTrue((EPS == se.sv.vectors).all())
        self.assertTrue(se.sv.vectors_norm is None)

    def test_reset_vectors_memmap(self):
        p = Path("fse/test/test_data/test_vectors")
        p_target = Path("fse/test/test_data/test_vectors.vectors")
        se = BaseSentence2VecModel(W2V, sv_mapfile_path=str(p.absolute()))
        trainables = BaseSentence2VecPreparer()
        trainables.reset_vectors(se.sv, 20)
        self.assertTrue(p_target.exists())
        self.assertEqual((20,DIM), se.sv.vectors.shape)
        self.assertEqual(np.float32, se.sv.vectors.dtype)
        self.assertTrue((EPS == se.sv.vectors).all())
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
        self.assertTrue((EPS == se.sv.vectors[20:]).all())
        self.assertTrue(se.sv.vectors_norm is None)

    def test_update_vectors_memmap(self):
        p = Path("fse/test/test_data/test_vectors")
        p_target = Path("fse/test/test_data/test_vectors.vectors")
        se = BaseSentence2VecModel(W2V, sv_mapfile_path=str(p.absolute()))
        trainables = BaseSentence2VecPreparer()
        trainables.reset_vectors(se.sv, 20)
        se.sv.vectors[:] = 1.
        trainables.update_vectors(se.sv, 10)
        self.assertTrue(p_target.exists())
        self.assertEqual((30,DIM), se.sv.vectors.shape)
        self.assertEqual(np.float32, se.sv.vectors.dtype)
        self.assertTrue((np.ones((20, DIM)) == se.sv.vectors[:20]).all())
        self.assertTrue((EPS == se.sv.vectors[20:]).all())
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
