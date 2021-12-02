#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Automated tests for checking the average model."""

import logging
import unittest


from fse.vectors import Vectors

logger = logging.getLogger(__name__)


class TestVectors(unittest.TestCase):
    def test_from_pretrained(self):
        """Test the pretrained vectors"""
        vectors = Vectors.from_pretrained("glove-wiki-gigaword-50")
        self.assertEqual(vectors.vector_size, 50)
        self.assertEqual(vectors.vectors.shape, (400000, 50))

        vectors = Vectors.from_pretrained("glove-wiki-gigaword-50", mmap="r")
        self.assertEqual(vectors.vector_size, 50)
        self.assertEqual(vectors.vectors.shape, (400000, 50))

    def test_missing_model(self):
        """Tests a missing model"""

        with self.assertRaises(ValueError):
            Vectors.from_pretrained("unittest")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
    )
    unittest.main()
