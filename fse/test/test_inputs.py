#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers


"""
Automated tests for checking the input methods.
"""

import logging
import unittest

import numpy as np

from fse.inputs import BaseIndexedList, IndexedList, SplitIndexedList, CSplitIndexedList,  \
    CIndexedList, CSplitCIndexedList, IndexedLineDocument, SplitCIndexedList

logger = logging.getLogger(__name__)

class TestBaseIndexedList(unittest.TestCase):

    def setUp(self):
        self.list_a = ["the dog is good", "it's nice and comfy"]
        self.list_b = ["lorem ipsum dolor", "si amet"]
        self.list_c = [s.split() for s in self.list_a]
        self.set_a = set(["hello there", "its a set"])
        self.arr_a = np.array(self.list_a)
        self.l = BaseIndexedList(self.list_a)
        self.ll = BaseIndexedList(self.list_a, self.list_b, self.list_c)
    
    def test_init(self):
        _ = BaseIndexedList(self.list_a)

    def test_init_mult_arg(self):
        self.assertEqual(6, len(self.ll.items))

    def test_init_ndarray(self):
        _ = BaseIndexedList(self.arr_a)

    def test__check_list_type(self):
        with self.assertRaises(TypeError):
            self.l._check_list_type(1)
        with self.assertRaises(TypeError):
            self.l._check_list_type("Hello")

    def test__check_str_type(self):
        self.assertEqual(1, self.l._check_str_type("Hello"))
        with self.assertRaises(TypeError):
            self.l._check_str_type(1)
        with self.assertRaises(TypeError):
            self.l._check_str_type([])
    
    def test__len(self):
        self.assertEqual(2, len(self.l))

    def test__str(self):
        self.assertEqual("[\'the dog is good\', \"it\'s nice and comfy\"]",
        str(self.l))

    def test__getitem(self):
        with self.assertRaises(NotImplementedError):
            self.l[0]

    def test__delitem(self):
        self.ll.__delitem__(0)
        self.assertEqual(5, len(self.ll))

    def test__setitem(self):
        self.ll.__setitem__(0, "is it me?")
        self.assertEqual("is it me?", self.ll.items[0])
    
    def test_append(self):
        self.ll.append("is it me?")
        self.assertEqual("is it me?", self.ll.items[-1])
    
    def test_extend(self):
        self.ll.extend(self.list_a)
        self.assertEqual(8, len(self.ll))

        self.ll.extend(self.set_a)
        self.assertEqual(10, len(self.ll))

    def test_extend_ndarr(self):
        l = BaseIndexedList(np.array([str(i) for i in [1,2,3,4]]))
        l.extend(np.array([str(i) for i in [1,2,3,4]]))
        self.assertEqual(8, len(l))

class TestIndexedList(unittest.TestCase):
    def setUp(self):
        self.list_a = ["the dog is good", "it's nice and comfy"]
        self.list_b = [s.split() for s in self.list_a]
        self.il = IndexedList(self.list_a, self.list_b)

    def test_init(self):
        _ = IndexedList(self.list_a)

    def test_getitem(self):
        self.assertEqual(("the dog is good", 0), self.il[0])

    def test_split(self):
        l = SplitIndexedList(self.list_a)
        self.assertEqual("the dog is good".split(), l[0][0])

class TestCIndexedList(unittest.TestCase):
    def setUp(self):
        self.list_a = ["The Dog is good", "it's nice and comfy"]
        self.il = CIndexedList(self.list_a, custom_index=[1,1])
    
    def test_cust_index(self):
        self.assertEqual(1, self.il[0][1])

    def test_wrong_len(self):
        with self.assertRaises(RuntimeError):
            il = CIndexedList(self.list_a, custom_index=[1])

    def test_mutable_funcs(self):
        with self.assertRaises(NotImplementedError):
            self.il.__delitem__(0)
        with self.assertRaises(NotImplementedError):
            self.il.__setitem__(0, "the")
    
        with self.assertRaises(NotImplementedError):
            self.il.insert(0, "the")
        with self.assertRaises(NotImplementedError):
            self.il.append("the")
        with self.assertRaises(NotImplementedError):
            self.il.extend(["the", "dog"])

class TestCSplitIndexedList(unittest.TestCase):
    def setUp(self):
        self.list_a = ["The Dog is good", "it's nice and comfy"]
        self.il = CSplitIndexedList(self.list_a, custom_split=self.split_func)

    def split_func(self, input):
        return input.lower().split()
    
    def test_getitem(self):
        self.assertEqual("the dog is good".split(), self.il[0][0])

class TestSplitCIndexedList(unittest.TestCase):
    def setUp(self):
        self.list_a = ["The Dog is good", "it's nice and comfy"]
        self.il = SplitCIndexedList(self.list_a, custom_index=[1,1])
    
    def test_getitem(self):
        self.assertEqual(("The Dog is good".split(), 1), self.il[0])

    def test_mutable_funcs(self):
        with self.assertRaises(NotImplementedError):
            self.il.__delitem__(0)
        with self.assertRaises(NotImplementedError):
            self.il.__setitem__(0, "the")
    
        with self.assertRaises(NotImplementedError):
            self.il.insert(0, "the")
        with self.assertRaises(NotImplementedError):
            self.il.append("the")
        with self.assertRaises(NotImplementedError):
            self.il.extend(["the", "dog"])

class TestCSplitCIndexedList(unittest.TestCase):
    def setUp(self):
        self.list_a = ["The Dog is good", "it's nice and comfy"]
        self.il = CSplitCIndexedList(self.list_a, custom_split=self.split_func, custom_index=[1,1])

    def split_func(self, input):
        return input.lower().split()
    
    def test_getitem(self):
        self.assertEqual(("the dog is good".split(), 1), self.il[0])
    
    def test_mutable_funcs(self):
        with self.assertRaises(NotImplementedError):
            self.il.__delitem__(0)
        with self.assertRaises(NotImplementedError):
            self.il.__setitem__(0, "the")
    
        with self.assertRaises(NotImplementedError):
            self.il.insert(0, "the")
        with self.assertRaises(NotImplementedError):
            self.il.append("the")
        with self.assertRaises(NotImplementedError):
            self.il.extend(["the", "dog"])

class TestIndexedLineDocument(unittest.TestCase):

    def setUp(self):
        self.p = "fse/test/test_data/test_sentences.txt"
        self.doc = IndexedLineDocument(self.p)

    def test_getitem(self):
        self.assertEqual("Good stuff i just wish it lasted longer", self.doc[0])
        self.assertEqual("Save yourself money and buy it direct from lg", self.doc[19])
        self.assertEqual("I am not sure if it is a tracfone problem or the battery", self.doc[-1])

    def test_yield(self):
        first = ("Good stuff i just wish it lasted longer".split(), 0)
        last = ("I am not sure if it is a tracfone problem or the battery".split(), 99)
        for i, obj in enumerate(self.doc):
            if i == 0:
                self.assertEqual(first, obj)
            if i == 99:
                self.assertEqual(last, obj)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()