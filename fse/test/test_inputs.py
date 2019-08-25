#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers


"""
Automated tests for checking the input methods.
"""


import logging
import unittest

from fse.inputs import IndexedSentence, IndexedList, IndexedLineDocument

logger = logging.getLogger(__name__)

class TestIndexedSentenceFunctions(unittest.TestCase):
    def test__str(self):
        sent_0 = IndexedSentence(["Hello", "there"], 0)
        sent_1 = IndexedSentence(["Hello", "again"], 1)
        self.assertEqual(0, sent_0.index)
        self.assertEqual(1, sent_1.index)
        self.assertEqual(["Hello", "there"], sent_0.words)
        self.assertEqual(["Hello", "again"], sent_1.words)

class TestIndexedListFuncs(unittest.TestCase):

    def setUp(self):
        self.list_a = ["the dog is good", "it's nice and comfy"]
        self.list_b = ["lorem ipsum dolor", "si amet"]
        self.list_c = [s.split() for s in self.list_a]
        self.set_a = set(["hello there", "its a set"])
        self.il = IndexedList(self.list_a, self.list_b, self.set_a, split=True)

    def test_init_list(self):
        l = IndexedList(self.list_a)
    
    def test_init_multiple_list(self):
        l = IndexedList(self.list_a, self.list_b)
        self.assertEqual(4, len(l))
    
    def test_init_set(self):
        l = IndexedList(self.set_a)
    
    def test_init_dict(self):
        tmp = {0: "hello there"}
        with self.assertRaises(TypeError): 
            IndexedList(tmp)
    
    def test_init_multiple_args(self):
        with self.assertRaises(RuntimeError):
            IndexedList(self.list_a, split=True, split_func=self.list_a)

    def test_init_multiple_splits(self):
        with self.assertRaises(RuntimeError):
            IndexedList(self.list_a, split_func=self.list_a, pre_splitted=True)
    
    def test__len(self):
        l = IndexedList(self.list_a)
        self.assertEqual(2, len(l))

    def test__str(self):
        target = "[\'the dog is good\', \"it's nice and comfy\"]"
        self.assertEqual(target, str(IndexedList(self.list_a)))

    def test_getitem(self):
        self.assertEqual(["the", "dog", "is", "good"], self.il.__getitem__(0).words)
        self.assertEqual(0, self.il.__getitem__(0).index)

    def test_getitem_presplitted(self):
        l = IndexedList(self.list_c, pre_splitted=True)
        self.assertEqual(["the", "dog", "is", "good"], self.il.__getitem__(0).words)

    def test_delitem(self):
        self.il.__delitem__(0)
        self.assertEqual(5, len(self.il))

    def test_setitem(self):
        self.il.__setitem__(0, "is it me?")
        self.assertEqual(["is", "it", "me?"], self.il[0].words)
    
    def test_setitem_wrong_dtype(self):
        with self.assertRaises(TypeError):
            self.il.__setitem__(0, ["is it me?"])

    def test_append(self):
        self.il.append("is it me?")
        self.assertEqual(["is", "it", "me?"], self.il[-1].words)
    
    def test_extend(self):
        self.il.extend(self.list_a, self.list_b)
        self.assertEqual(10, len(self.il))

class TestIndexedLineDocument(unittest.TestCase):

    def setUp(self):
        self.p = "fse/test/test_data/test_sentences.txt"
        self.doc = IndexedLineDocument(self.p)

    def test_getitem(self):
        self.assertEqual("Good stuff i just wish it lasted longer", self.doc[0])
        self.assertEqual("Save yourself money and buy it direct from lg", self.doc[19])
        self.assertEqual("I am not sure if it is a tracfone problem or the battery", self.doc[-1])

    def test_yield(self):
        first = IndexedSentence("Good stuff i just wish it lasted longer".split(), 0)
        last = IndexedSentence("I am not sure if it is a tracfone problem or the battery".split(), 99)
        for i, obj in enumerate(self.doc):
            if i == 0:
                self.assertEqual(first, obj)
            if i == 99:
                self.assertEqual(last, obj)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()