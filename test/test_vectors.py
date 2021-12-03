#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Automated tests for checking the average model."""

import logging
import unittest
from unittest.mock import patch
from pathlib import Path

from fse.vectors import FTVectors, Vectors

logger = logging.getLogger(__name__)

TEST_DATA = Path(__file__).parent / "test_data"


class TestVectors(unittest.TestCase):
    def test_from_pretrained(self):
        """Test the pretrained vectors."""
        vectors = Vectors.from_pretrained("glove-wiki-gigaword-50")
        self.assertEqual(vectors.vector_size, 50)
        self.assertEqual(vectors.vectors.shape, (400000, 50))

        vectors = Vectors.from_pretrained("glove-wiki-gigaword-50", mmap="r")
        self.assertEqual(vectors.vector_size, 50)
        self.assertEqual(vectors.vectors.shape, (400000, 50))

    def test_missing_model(self):
        """Tests a missing model."""

        with self.assertRaises(ValueError):
            Vectors.from_pretrained("unittest")


class TestFTVectors(unittest.TestCase):
    def test_from_pretrained(self):
        """Test the pretrained vectors."""
        with patch("fse.vectors.snapshot_download") as mock, patch(
            "fse.vectors.FastTextKeyedVectors.load"
        ):
            mock.return_value = TEST_DATA.as_posix()
            FTVectors.from_pretrained("ft")

    def test_missing_model(self):
        """Tests a missing model."""

        with self.assertRaises(ValueError):
            Vectors.from_pretrained("unittest")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
    )
    unittest.main()
