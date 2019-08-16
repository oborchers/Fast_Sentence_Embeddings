#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers


"""
Automated tests for checking the input models.
"""


import logging
import unittest

from fse.models.inputs import IndexedSentence

logger = logging.getLogger(__name__)

class TestIndexedSentenceFunctions(unittest.TestCase):
    def test__str(self):
        sent_0 = IndexedSentence(["Hello", "there"], 0)
        sent_1 = IndexedSentence(["Hello", "again"], 1)
        self.assertEqual(0, sent_0.index)
        self.assertEqual(1, sent_1.index)
        self.assertEqual(["Hello", "there"], sent_0.words)
        self.assertEqual(["Hello", "again"], sent_1.words)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()