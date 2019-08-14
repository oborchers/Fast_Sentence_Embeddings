#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers


"""
Automated tests for checking the sentence vectors.
"""

import logging
import unittest

from pathlib import Path
import numpy as np
from fse.models import sentencevectors

logger = logging.getLogger(__name__)

class TestSentenceVectorsFunctions(unittest.TestCase):
    def setUp(self):
        self.sv = sentencevectors.SentenceVectors(2)
        self.sv.vectors = np.arange(10).reshape(5,2)

    def test_getitem(self):
        # Done
        self.assertTrue(([0,1] == self.sv[0]).all())
        self.assertTrue(([[0,1],[4,5]] == self.sv[[0,2]]).all())

    def test_isin(self):
        # Done
        self.assertTrue(0 in self.sv)
        self.assertFalse(5 in self.sv)

    def test_init_sims_wo_replace(self):
        # Done
        self.sv.init_sims()
        self.assertIsNotNone(self.sv.vectors_norm)
        self.assertFalse((self.sv.vectors == self.sv.vectors_norm).all())

        v1 = self.sv.vectors[0]
        v1 = v1 / np.sqrt(np.sum(v1**2))

        v2 = self.sv.vectors[1]
        v2 = v2 / np.sqrt(np.sum(v2**2))

        self.assertTrue(np.allclose(v1, self.sv.vectors_norm[0]))
        self.assertTrue(np.allclose(v2, self.sv.vectors_norm[1]))
        self.assertTrue(np.allclose(v2, self.sv.get_vector(1, True)))

    def test_get_vector(self):
        # Done
        self.assertTrue(([0,1] == self.sv.get_vector(0)).all())
        self.assertTrue(([2,3] == self.sv.get_vector(1)).all())

    def test_init_sims_w_replace(self):
        # Done
        self.sv.init_sims(True)
        self.assertTrue((self.sv.vectors[0] == self.sv.vectors_norm[0]).all())

    def test_init_sims_w_mapfile(self):
        # Done
        p = Path("fse/test/test_data/test_vectors")
        self.sv.mapfile_path = str(p.absolute())
        self.sv.init_sims()
        p = Path("fse/test/test_data/test_vectors.vectors_norm")
        self.assertTrue(p.exists())
        p.unlink()

    def test_save_load(self):
        # Done
        p = Path("fse/test/test_data/test_vectors.vectors")
        self.sv.save(str(p.absolute()))
        self.assertTrue(p.exists())
        sv2 = sentencevectors.SentenceVectors.load(str(p.absolute()))
        self.assertTrue((self.sv.vectors == sv2.vectors).all())
        p.unlink()

    def test_len(self):
        # Done
        self.assertEqual(5, len(self.sv))

    def test_similarity(self):
        # Done
        v1 = self.sv.vectors[0]
        v1 = v1 / np.sqrt(np.sum(v1**2))

        v2 = self.sv.vectors[1]
        v2 = v2 / np.sqrt(np.sum(v2**2))

        self.assertEqual(v1.dot(v2), self.sv.similarity(0,1))
        self.assertEqual(1-(v1.dot(v2)), self.sv.distance(0,1))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
