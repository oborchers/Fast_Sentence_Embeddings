#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <oliver-borchers@outlook.de>
# Copyright (C) 2020 Oliver Borchers
# For License information, see corresponding LICENSE file.

"""
Example for computation of convolution length

window=5
stride=3

window=5
stride=1

Consider, that w2v does not contain "12345"

all     w2v     ft
"They", "admit"
2       2       2  
        1       1

"So", "Apple", "bought", "buds"
4       4       4
1       3       3
        2       2
        1       1

"go", "12345"
2       1       2
                1

"pull", "12345678910111213"
2       0       2
                1

"this"	"is"	"a"	"longer"	"test"	"sentence"	"test"	"longer"	"sentences"
0       1       2   3           4
                    0           1       2           3       4
                                                    0       1           2
                                    
"this"          5       5       5    
"is"	                4       5
"a"	                    4       5
"longer"	    5       4       5
"test"	                3       5
"sentence"	            2       4
"test"	        3       2       3
"longer"	            1       2
"sentences"             0       1
"""


from fse.models.pooling import MaxPooling, train_pooling_np

from fse.test.model_shared_imports import *

class TestPoolingFunctions(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            ["They", "admit"],
            ["So", "Apple", "bought", "buds"],
            ["go", "12345"],
            ["pull", "12345678910111213"],
            "this is a longer test sentence test longer sentences".split(),
        ]
        self.sentences = [(s, i) for i, s in enumerate(self.sentences)]
        self.model = MaxPooling(W2V_DET)
        self.model.prep.prepare_vectors(
            sv=self.model.sv, total_sentences=len(self.sentences), update=False
        )
        self.model._pre_train_calls()

    def set_convolution(self):
        self.model.window_size=5
        self.model.window_stride=1
    
    def unset_convolution(self):
        self.model.window_size=1
        self.model.window_stride=1

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_cython(self):
        from fse.models.pooling_inner import (
            FAST_VERSION,
            MAX_WORDS_IN_BATCH,
            MAX_NGRAMS_IN_BATCH,
            train_pooling_cy,
        )

        self.assertTrue(FAST_VERSION)
        self.assertTrue(callable(train_pooling_cy))
        self.assertEqual(10000, MAX_WORDS_IN_BATCH)
        self.assertEqual(40, MAX_NGRAMS_IN_BATCH)

    def test_check_parameter_sanity(self):
        se = MaxPooling(W2V_DET)
        se.word_weights = np.full(20, 2.0, dtype=np.float32)
        with self.assertRaises(ValueError):
            se._check_parameter_sanity()

        se = MaxPooling(W2V_DET, window_size=0)
        with self.assertRaises(ValueError):
            se._check_parameter_sanity()

        se = MaxPooling(W2V_DET, window_size=3, window_stride=0)
        with self.assertRaises(ValueError):
            se._check_parameter_sanity()

        se = MaxPooling(W2V_DET, window_size=3, window_stride=4)
        with self.assertRaises(ValueError):
            se._check_parameter_sanity()

    def test_train(self):
        self.assertEqual(
            (100, 1450), self.model.train([(s, i) for i, s in enumerate(SENTENCES)])
        )

    def test_do_train_job(self):
        self.model.prep.prepare_vectors(
            sv=self.model.sv, total_sentences=len(SENTENCES), update=True
        )
        mem = self.model._get_thread_working_mem()
        self.assertEqual(
            (100, 1450),
            self.model._do_train_job(
                [(s, i) for i, s in enumerate(SENTENCES)],
                target=self.model.sv.vectors,
                memory=mem,
            ),
        )
        self.assertEqual((105, DIM), self.model.sv.vectors.shape)

    ### Basic Pooling Tests start here

    def test_pool_train_np_w2v(self):
        self.model.sv.vectors = np.zeros_like(self.model.sv.vectors, dtype=np.float32)
        mem = self.model._get_thread_working_mem()

        output = train_pooling_np(
            self.model, self.sentences, self.model.sv.vectors, mem
        )

        self.assertEqual((5, 14), output)
        self.assertTrue((241 == self.model.sv[0]).all())
        self.assertTrue((306 == self.model.sv[1]).all())
        self.assertTrue((self.model.wv.vocab["go"].index == self.model.sv[2]).all())

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_pool_train_cy_w2v(self):
        self.model.sv.vectors = np.zeros_like(self.model.sv.vectors, dtype=np.float32)
        mem = self.model._get_thread_working_mem()

        from fse.models.pooling_inner import train_pooling_cy

        output = train_pooling_cy(
            self.model, self.sentences, self.model.sv.vectors, mem
        )

        self.assertEqual((5, 14), output)
        self.assertTrue((241 == self.model.sv[0]).all())
        self.assertTrue((306 == self.model.sv[1]).all())
        self.assertTrue((self.model.wv.vocab["go"].index == self.model.sv[2]).all())


    ##### Worked until here

    def test_pool_train_np_ft(self):
        m = MaxPooling(FT_DET)
        m.prep.prepare_vectors(
            sv=m.sv, total_sentences=len(self.sentences), update=False
        )
        m._pre_train_calls()
        mem = m._get_thread_working_mem()

        output = train_pooling_np(m, self.sentences, m.sv.vectors, mem)

        self.assertEqual((5, 19), output)
        self.assertTrue((1. == m.sv[0]).all())
        self.assertTrue(
            np.allclose(737413.9, m.sv[2])
        )
        self.assertTrue(
            np.allclose(1080970.2, m.sv[3])
        )

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_pool_train_cy_ft(self):
        m = MaxPooling(FT_DET)
        m.prep.prepare_vectors(
            sv=m.sv, total_sentences=len(self.sentences), update=False
        )
        m._pre_train_calls()
        mem = m._get_thread_working_mem()

        from fse.models.pooling_inner import train_pooling_cy

        output = train_pooling_cy(m, self.sentences, m.sv.vectors, mem)

        self.assertEqual((5, 19), output)
        self.assertTrue((1. == m.sv[0]).all())
        self.assertTrue(
            np.allclose(737413.9, m.sv[2])
        )
        self.assertTrue(
            np.allclose(1080970.2, m.sv[3])
        )

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_pool_cy_equal_np_w2v(self):
        m1 = MaxPooling(W2V_DET)
        m1.prep.prepare_vectors(
            sv=m1.sv, total_sentences=len(self.sentences), update=False
        )
        m1._pre_train_calls()
        mem1 = m1._get_thread_working_mem()
        o1 = train_pooling_np(m1, self.sentences, m1.sv.vectors, mem1)

        m2 = MaxPooling(W2V_DET)
        m2.prep.prepare_vectors(
            sv=m2.sv, total_sentences=len(self.sentences), update=False
        )
        m2._pre_train_calls()
        mem2 = m2._get_thread_working_mem()

        from fse.models.pooling_inner import train_pooling_cy

        o2 = train_pooling_cy(m2, self.sentences, m2.sv.vectors, mem2)

        self.assertEqual(o1, o2)
        self.assertTrue((m1.sv.vectors == m2.sv.vectors).all())

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_pool_cy_equal_np_w2v_random(self):
        m1 = MaxPooling(W2V_RNG)
        m1.prep.prepare_vectors(
            sv=m1.sv, total_sentences=len(self.sentences), update=False
        )
        m1._pre_train_calls()
        mem1 = m1._get_thread_working_mem()
        o1 = train_pooling_np(m1, self.sentences, m1.sv.vectors, mem1)

        m2 = MaxPooling(W2V_RNG)
        m2.prep.prepare_vectors(
            sv=m2.sv, total_sentences=len(self.sentences), update=False
        )
        m2._pre_train_calls()
        mem2 = m2._get_thread_working_mem()

        from fse.models.pooling_inner import train_pooling_cy

        o2 = train_pooling_cy(m2, self.sentences, m2.sv.vectors, mem2)

        self.assertTrue(np.allclose(m1.sv.vectors, m2.sv.vectors, atol=1e-6))

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_pool_cy_equal_np_ft_random(self):
        m1 = MaxPooling(FT_RNG)
        m1.prep.prepare_vectors(
            sv=m1.sv, total_sentences=len(self.sentences), update=False
        )
        m1._pre_train_calls()

        from fse.models.pooling_inner import MAX_NGRAMS_IN_BATCH

        m1.batch_ngrams = MAX_NGRAMS_IN_BATCH
        mem1 = m1._get_thread_working_mem()
        o1 = train_pooling_np(m1, self.sentences[:2], m1.sv.vectors, mem1)

        m2 = MaxPooling(FT_RNG)
        m2.prep.prepare_vectors(
            sv=m2.sv, total_sentences=len(self.sentences), update=False
        )
        m2._pre_train_calls()
        mem2 = m2._get_thread_working_mem()

        from fse.models.pooling_inner import train_pooling_cy

        o2 = train_pooling_cy(m2, self.sentences[:2], m2.sv.vectors, mem2)

        self.assertEqual(o1, o2)
        self.assertTrue(np.allclose(m1.sv.vectors, m2.sv.vectors, atol=1e-6))

    def test_pool_np_w2v_non_negative(self):
        mpool = MaxPooling(W2V_RNG)
        mpool.train(self.sentences)
        self.assertTrue((mpool.sv.vectors >= 0).all())

    def test_pool_np_ft_non_negative(self):
        mpool = MaxPooling(FT_RNG)
        mpool.train(self.sentences)
        self.assertTrue((mpool.sv.vectors >= 0).all())

    ## Hierarchical Tests start here

    def test_conv_pool_train_np_w2v(self):
        self.model.sv.vectors = np.zeros_like(self.model.sv.vectors, dtype=np.float32)
        mem = self.model._get_thread_working_mem()

        self.set_convolution()

        output = train_pooling_np(
            self.model, self.sentences, self.model.sv.vectors, mem
        )
        self.unset_convolution()

        # TODO: The count does not match the expectation
        self.assertEqual((5, 39), output)
        self.assertTrue((183 == self.model.sv[0]).all())
        self.assertTrue(np.allclose(self.model.sv[4], 184.8))

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_conv_pool_train_cy_w2v(self):
        self.model.sv.vectors = np.zeros_like(self.model.sv.vectors, dtype=np.float32)
        mem = self.model._get_thread_working_mem()

        self.set_convolution()

        from fse.models.pooling_inner import train_pooling_cy

        output = train_pooling_cy(
            self.model, self.sentences, self.model.sv.vectors, mem
        )

        self.unset_convolution()

        self.assertEqual((5, 39), output)
        self.assertTrue((183 == self.model.sv[0]).all())
        self.assertTrue(np.allclose(self.model.sv[4], 184.8))

    def test_conv_pool_train_np_ft(self):
        m = MaxPooling(FT_DET, window_size=2, window_stride=2)
        m.prep.prepare_vectors(
            sv=m.sv, total_sentences=len(self.sentences), update=False
        )
        m._pre_train_calls()
        mem = m._get_thread_working_mem()

        output = train_pooling_np(m, self.sentences, m.sv.vectors, mem)

        self.assertEqual((5, 19), output)
        self.assertTrue((1 == m.sv[0]).all())
        self.assertTrue(np.allclose(368707.44, m.sv[2]))
        self.assertTrue(np.allclose(961940.2, m.sv[3]))
        """
        Note to future self:
        Due to the size of the ngram vectors,
        an ngram at the last position of the senence
        will always be the highest value.
        TODO: This unittest is thus a bit flawed. Maybe fix?
        """

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_conv_pool_train_cy_ft(self):
        m = MaxPooling(FT_DET, window_size=2, window_stride=2)
        m.prep.prepare_vectors(
            sv=m.sv, total_sentences=len(self.sentences), update=False
        )
        m._pre_train_calls()
        mem = m._get_thread_working_mem()

        from fse.models.pooling_inner import train_pooling_cy

        output = train_pooling_cy(m, self.sentences, m.sv.vectors, mem)

        self.assertEqual((5, 19), output)
        self.assertTrue((1 == m.sv[0]).all())
        self.assertTrue(np.allclose(368707.44, m.sv[2]))
        self.assertTrue(np.allclose(961940.2, m.sv[3]))

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_conv_pool_cy_equal_np_w2v_random(self):
        m1 = MaxPooling(W2V_RNG, window_size=5, window_stride=1)
        m1.prep.prepare_vectors(
            sv=m1.sv, total_sentences=len(self.sentences), update=False
        )
        m1._pre_train_calls()
        mem1 = m1._get_thread_working_mem()
        o1 = train_pooling_np(m1, self.sentences, m1.sv.vectors, mem1)

        m2 = MaxPooling(W2V_RNG, window_size=5, window_stride=1)
        m2.prep.prepare_vectors(
            sv=m2.sv, total_sentences=len(self.sentences), update=False
        )
        m2._pre_train_calls()
        mem2 = m2._get_thread_working_mem()

        from fse.models.pooling_inner import train_pooling_cy

        o2 = train_pooling_cy(m2, self.sentences, m2.sv.vectors, mem2)

        self.assertTrue(np.allclose(m1.sv.vectors, m2.sv.vectors, atol=1e-6))

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_conv_pool_cy_equal_np_ft_random(self):
        m1 = MaxPooling(FT_RNG, window_size=5, window_stride=1)
        m1.prep.prepare_vectors(
            sv=m1.sv, total_sentences=len(self.sentences), update=False
        )
        m1._pre_train_calls()

        from fse.models.pooling_inner import MAX_NGRAMS_IN_BATCH

        m1.batch_ngrams = MAX_NGRAMS_IN_BATCH
        mem1 = m1._get_thread_working_mem()
        o1 = train_pooling_np(m1, self.sentences[:2], m1.sv.vectors, mem1)

        m2 = MaxPooling(FT_RNG, window_size=5, window_stride=1)
        m2.prep.prepare_vectors(
            sv=m2.sv, total_sentences=len(self.sentences), update=False
        )
        m2._pre_train_calls()
        mem2 = m2._get_thread_working_mem()

        from fse.models.pooling_inner import train_pooling_cy

        o2 = train_pooling_cy(m2, self.sentences[:2], m2.sv.vectors, mem2)

        self.assertEqual(o1, o2)
        self.assertTrue(np.allclose(m1.sv.vectors, m2.sv.vectors, atol=1e-6))

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_conv_pool_cy_w2v_non_negative(self):
        mpool = MaxPooling(W2V_RNG, window_size=5, window_stride=1)
        mpool.train(self.sentences)
        self.assertTrue((mpool.sv.vectors >= 0).all())

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_conv_pool_cy_ft_non_negative(self):
        mpool = MaxPooling(FT_RNG, window_size=5, window_stride=1)
        mpool.train(self.sentences)
        self.assertTrue((mpool.sv.vectors >= 0).all())

    ### Hierarchical Test + Stride start here

    def test_conv_pool_stride_train_np_w2v(self):
        self.model.sv.vectors = np.zeros_like(self.model.sv.vectors, dtype=np.float32)
        mem = self.model._get_thread_working_mem()

        self.set_convolution()
        self.model.window_stride = 5

        output = train_pooling_np(
            self.model, self.sentences, self.model.sv.vectors, mem
        )
        self.unset_convolution()

        self.assertEqual((5, 14), output)
        self.assertTrue((183 == self.model.sv[0]).all())
        self.assertTrue((115.5 == self.model.sv[4]).all())

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_conv_pool_stride_train_cy_w2v(self):
        self.model.sv.vectors = np.zeros_like(self.model.sv.vectors, dtype=np.float32)
        mem = self.model._get_thread_working_mem()

        self.set_convolution()
        self.model.window_stride = 5

        from fse.models.pooling_inner import train_pooling_cy

        output = train_pooling_cy(
            self.model, self.sentences, self.model.sv.vectors, mem
        )
        self.unset_convolution()

        self.assertEqual((5, 14), output)
        self.assertTrue((183 == self.model.sv[0]).all())
        self.assertTrue((115.5 == self.model.sv[4]).all())

    def test_conv_pool_stride_train_np_ft(self):
        m = MaxPooling(FT_DET, window_size=5, window_stride=3)
        m.prep.prepare_vectors(
            sv=m.sv, total_sentences=len(self.sentences), update=False
        )
        m._pre_train_calls()
        mem = m._get_thread_working_mem()

        output = train_pooling_np(m, self.sentences, m.sv.vectors, mem)

        self.assertEqual((5, 24), output)
        self.assertTrue((1 == m.sv[0]).all())
        self.assertTrue(np.allclose(368707.44, m.sv[2]))
        self.assertTrue(np.allclose(961940.2, m.sv[3]))

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_conv_pool_stride_train_cy_ft(self):
        m = MaxPooling(FT_DET, window_size=5, window_stride=3)
        m.prep.prepare_vectors(
            sv=m.sv, total_sentences=len(self.sentences), update=False
        )
        m._pre_train_calls()
        mem = m._get_thread_working_mem()

        from fse.models.pooling_inner import train_pooling_cy

        output = train_pooling_cy(m, self.sentences, m.sv.vectors, mem)

        self.assertEqual((5, 24), output)
        self.assertTrue((1 == m.sv[0]).all())
        self.assertTrue(np.allclose(368707.44, m.sv[2]))
        self.assertTrue(np.allclose(961940.2, m.sv[3]))

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_conv_pool_stride_cy_equal_np_w2v_random(self):
        m1 = MaxPooling(W2V_RNG, window_size=5, window_stride=4)
        m1.prep.prepare_vectors(
            sv=m1.sv, total_sentences=len(self.sentences), update=False
        )
        m1._pre_train_calls()
        mem1 = m1._get_thread_working_mem()
        o1 = train_pooling_np(m1, self.sentences, m1.sv.vectors, mem1)

        m2 = MaxPooling(W2V_RNG, window_size=5, window_stride=4)
        m2.prep.prepare_vectors(
            sv=m2.sv, total_sentences=len(self.sentences), update=False
        )
        m2._pre_train_calls()
        mem2 = m2._get_thread_working_mem()

        from fse.models.pooling_inner import train_pooling_cy

        o2 = train_pooling_cy(m2, self.sentences, m2.sv.vectors, mem2)

        self.assertTrue(np.allclose(m1.sv.vectors, m2.sv.vectors, atol=1e-6))

    @unittest.skipIf(IGNORE_CY, "ignoring Cython build")
    def test_conv_pool_stride_cy_equal_np_ft_random(self):
        m1 = MaxPooling(FT_RNG, window_size=5, window_stride=5)
        m1.prep.prepare_vectors(
            sv=m1.sv, total_sentences=len(self.sentences), update=False
        )
        m1._pre_train_calls()

        from fse.models.pooling_inner import MAX_NGRAMS_IN_BATCH

        m1.batch_ngrams = MAX_NGRAMS_IN_BATCH
        mem1 = m1._get_thread_working_mem()
        o1 = train_pooling_np(m1, self.sentences[:2], m1.sv.vectors, mem1)

        m2 = MaxPooling(FT_RNG, window_size=5, window_stride=5)
        m2.prep.prepare_vectors(
            sv=m2.sv, total_sentences=len(self.sentences), update=False
        )
        m2._pre_train_calls()
        mem2 = m2._get_thread_working_mem()

        from fse.models.pooling_inner import train_pooling_cy

        o2 = train_pooling_cy(m2, self.sentences[:2], m2.sv.vectors, mem2)

        self.assertEqual(o1, o2)
        self.assertTrue(np.allclose(m1.sv.vectors, m2.sv.vectors, atol=1e-6))

    def test_conv_pool_stride_np_w2v_non_negative(self):
        mpool = MaxPooling(W2V_RNG, window_size=2, window_stride=2)
        mpool.train(self.sentences)
        self.assertTrue((mpool.sv.vectors >= 0).all())

    def test_conv_pool_stride_np_ft_non_negative(self):
        mpool = MaxPooling(FT_RNG, window_size=2, window_stride=2)
        mpool.train(self.sentences)
        self.assertTrue((mpool.sv.vectors >= 0).all())


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
    )
    unittest.main()
