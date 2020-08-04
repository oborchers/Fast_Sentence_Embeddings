#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2020 Oliver Borchers
# For License information, see corresponding LICENSE file.

"""
Automated tests for checking the average model.
"""

from fse.test.model_shared_imports import *

from fse.models.average import Average, train_average_np

class TestAverageFunctions(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            ["They", "admit"],
            ["So", "Apple", "bought", "buds"],
            ["go", "12345"],
            ["pull", "12345678910111213"],
        ]
        self.sentences = [(s, i) for i, s in enumerate(self.sentences)]
        self.model = Average(W2V_DET)
        self.model.prep.prepare_vectors(
            sv=self.model.sv, total_sentences=len(self.sentences), update=False
        )
        self.model._pre_train_calls()

    def test_cython(self):
        from fse.models.average_inner import (
            FAST_VERSION,
            MAX_WORDS_IN_BATCH,
            MAX_NGRAMS_IN_BATCH,
        )

        self.assertTrue(FAST_VERSION)
        self.assertEqual(10000, MAX_WORDS_IN_BATCH)
        self.assertEqual(40, MAX_NGRAMS_IN_BATCH)

    def test_average_train_np_w2v_det(self):
        self.model.sv.vectors = np.zeros_like(self.model.sv.vectors, dtype=np.float32)
        mem = self.model._get_thread_working_mem()
        output = train_average_np(
            self.model, self.sentences, self.model.sv.vectors, mem
        )
        self.assertEqual((4, 7), output)
        self.assertTrue((183 == self.model.sv[0]).all())
        self.assertTrue((164.5 == self.model.sv[1]).all())
        self.assertTrue((self.model.wv.vocab["go"].index == self.model.sv[2]).all())

    def test_average_train_cy_w2v_det(self):
        self.model.sv.vectors = np.zeros_like(self.model.sv.vectors, dtype=np.float32)
        mem = self.model._get_thread_working_mem()

        from fse.models.average_inner import train_average_cy

        output = train_average_cy(
            self.model, self.sentences, self.model.sv.vectors, mem
        )
        self.assertEqual((4, 7), output)
        self.assertTrue((183 == self.model.sv[0]).all())
        self.assertTrue((164.5 == self.model.sv[1]).all())
        self.assertTrue((self.model.wv.vocab["go"].index == self.model.sv[2]).all())

    def test_average_train_np_ft_det(self):
        m = Average(FT_DET)
        m.prep.prepare_vectors(
            sv=m.sv, total_sentences=len(self.sentences), update=False
        )
        m._pre_train_calls()
        mem = m._get_thread_working_mem()
        output = train_average_np(m, self.sentences, m.sv.vectors, mem)

        self.assertEqual((4, 10), output)
        self.assertTrue((1.0 + EPS == m.sv[0]).all())
        self.assertTrue(np.allclose(368707.44, m.sv[2]))
        self.assertTrue(np.allclose(961940.2, m.sv[3]))
        # "go" -> [1,1...]
        # oov: "12345" -> (14 hashes * 2) / 14 =  2
        # (2 + 1) / 2 = 1.5

    def test_average_train_cy_ft_det(self):
        m = Average(FT_DET)
        m.prep.prepare_vectors(
            sv=m.sv, total_sentences=len(self.sentences), update=False
        )
        m._pre_train_calls()
        mem = m._get_thread_working_mem()

        from fse.models.average_inner import train_average_cy

        output = train_average_cy(m, self.sentences, m.sv.vectors, mem)
        self.assertEqual((4, 10), output)
        self.assertTrue((1.0 + EPS == m.sv[0]).all())
        self.assertTrue(np.allclose(368707.4, m.sv[2]))
        self.assertTrue(np.allclose(961940.0, m.sv[3]))

    def test_cy_equal_np_w2v_det(self):
        m1 = Average(W2V_DET)
        m1.prep.prepare_vectors(
            sv=m1.sv, total_sentences=len(self.sentences), update=False
        )
        m1._pre_train_calls()
        mem1 = m1._get_thread_working_mem()
        o1 = train_average_np(m1, self.sentences, m1.sv.vectors, mem1)

        m2 = Average(W2V_DET)
        m2.prep.prepare_vectors(
            sv=m2.sv, total_sentences=len(self.sentences), update=False
        )
        m2._pre_train_calls()
        mem2 = m2._get_thread_working_mem()

        from fse.models.average_inner import train_average_cy

        o2 = train_average_cy(m2, self.sentences, m2.sv.vectors, mem2)

        self.assertEqual(o1, o2)
        self.assertTrue((m1.sv.vectors == m2.sv.vectors).all())

    def test_cy_equal_np_w2v_rng(self):
        m1 = Average(W2V_RNG)
        m1.prep.prepare_vectors(
            sv=m1.sv, total_sentences=len(self.sentences), update=False
        )
        m1._pre_train_calls()
        mem1 = m1._get_thread_working_mem()
        o1 = train_average_np(m1, self.sentences, m1.sv.vectors, mem1)

        m2 = Average(W2V_RNG)
        m2.prep.prepare_vectors(
            sv=m2.sv, total_sentences=len(self.sentences), update=False
        )
        m2._pre_train_calls()
        mem2 = m2._get_thread_working_mem()

        from fse.models.average_inner import train_average_cy

        o2 = train_average_cy(m2, self.sentences, m2.sv.vectors, mem2)

        self.assertTrue(np.allclose(m1.sv.vectors, m2.sv.vectors, atol=1e-6))

    def test_cy_equal_np_ft_rng(self):
        m1 = Average(FT_RNG)
        m1.prep.prepare_vectors(
            sv=m1.sv, total_sentences=len(self.sentences), update=False
        )
        m1._pre_train_calls()

        from fse.models.average_inner import MAX_NGRAMS_IN_BATCH

        m1.batch_ngrams = MAX_NGRAMS_IN_BATCH
        mem1 = m1._get_thread_working_mem()
        o1 = train_average_np(m1, self.sentences[:2], m1.sv.vectors, mem1)

        m2 = Average(FT_RNG)
        m2.prep.prepare_vectors(
            sv=m2.sv, total_sentences=len(self.sentences), update=False
        )
        m2._pre_train_calls()
        mem2 = m2._get_thread_working_mem()

        from fse.models.average_inner import train_average_cy

        o2 = train_average_cy(m2, self.sentences[:2], m2.sv.vectors, mem2)

        self.assertEqual(o1, o2)
        self.assertTrue(np.allclose(m1.sv.vectors, m2.sv.vectors, atol=1e-6))

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
        self.assertEqual((104, DIM), self.model.sv.vectors.shape)

    def test_train(self):
        self.assertEqual(
            (100, 1450), self.model.train([(s, i) for i, s in enumerate(SENTENCES)])
        )

    def test_train_single_from_disk(self):
        p = Path("fse/test/test_data/test_vecs")
        p_res = Path("fse/test/test_data/test_vecs.vectors")
        p_target = Path("fse/test/test_data/test_vecs_wv.vectors")

        se1 = Average(W2V_DET)
        se2 = Average(
            W2V_DET,
            sv_mapfile_path=str(p.absolute()),
            wv_mapfile_path=str(p.absolute()),
        )
        se1.train([(s, i) for i, s in enumerate(SENTENCES)])
        se2.train([(s, i) for i, s in enumerate(SENTENCES)])

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

        se1 = Average(W2V_DET, workers=2)
        se2 = Average(
            W2V_DET,
            workers=2,
            sv_mapfile_path=str(p.absolute()),
            wv_mapfile_path=str(p.absolute()),
        )
        se1.train([(s, i) for i, s in enumerate(SENTENCES)])
        se2.train([(s, i) for i, s in enumerate(SENTENCES)])

        self.assertTrue(p_target.exists())
        self.assertTrue((se1.wv.vectors == se2.wv.vectors).all())
        self.assertFalse(se2.wv.vectors.flags.writeable)

        self.assertTrue((se1.sv.vectors == se2.sv.vectors).all())
        p_res.unlink()
        p_target.unlink()

    def test_check_parameter_sanity(self):
        se = Average(W2V_DET)
        se.word_weights = np.full(20, 2.0, dtype=np.float32)
        with self.assertRaises(ValueError):
            se._check_parameter_sanity()

        # TODO:
        # se = Average(W2V_DET, window_size=0)
        # with self.assertRaises(ValueError):
        #     se._check_parameter_sanity()

        # se = Average(W2V_DET, window_size=3, window_stride=0)
        # with self.assertRaises(ValueError):
        #     se._check_parameter_sanity()

        # se = Average(W2V_DET, window_size=3, window_stride=4)
        # with self.assertRaises(ValueError):
        #     se._check_parameter_sanity()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
    )
    unittest.main()
