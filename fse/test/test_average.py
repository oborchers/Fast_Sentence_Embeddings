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

from fse.models.average import Average
from fse.models.average import average_train_np

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
        self.sentences = [["They", "admit"], ["So", "Apple", "bought", "buds"]]
        self.word_indices = [[W2V.wv.vocab[w].index for w in s] for s in self.sentences]
        self.model = Average(W2V)
        self.model.prep.prepare_vectors(sv=self.model.sv, total_sentences=len(self.sentences), update=False)

    def test_weight_dtype(self):
        self.assertEqual(np.float32, self.model.word_weights.dtype)

    def test_average_train_np(self):
        output = average_train_np(self.model, self.sentences)
        self.assertEqual((2, 6), output)
        self.assertTrue((183 == self.model.sv[0]).all())
        self.assertTrue((164.5 == self.model.sv[1]).all())

    def test_do_train_job(self):
        self.model.prep.prepare_vectors(sv=self.model.sv, total_sentences=len(SENTENCES), update=True)
        self.assertEqual((100,1450), self.model._do_train_job(SENTENCES))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()