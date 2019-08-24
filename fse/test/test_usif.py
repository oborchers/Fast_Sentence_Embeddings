import logging
import unittest

from pathlib import Path

import numpy as np

from fse.models.usif import uSIF
from fse.inputs import IndexedSentence

from gensim.models import Word2Vec

logger = logging.getLogger(__name__)

CORPUS = Path("fse/test/test_data/test_sentences.txt")
DIM = 50
W2V = Word2Vec(min_count=1, size=DIM)
SENTENCES = [l.split() for i, l in enumerate(open(CORPUS, "r"))]
W2V.build_vocab(SENTENCES)

class TestuSIFFunctions(unittest.TestCase):
    def setUp(self):
        self.sentences = [IndexedSentence(s, i) for i,s in enumerate(SENTENCES)]
        self.model = uSIF(W2V, lang_freq="en")

    def test_length_init(self):
        se = uSIF(W2V, length=11, lang_freq="en")
        self.assertEqual(11, se.length)

    def test_length(self):
        self.model.train(sentences=self.sentences)
        self.assertEqual(14, self.model.length)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()