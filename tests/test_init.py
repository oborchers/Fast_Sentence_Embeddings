"""Tests for the package root."""

import importlib
import unittest
from unittest.mock import patch

import fse


class TestInit(unittest.TestCase):
    def test_all_is_importable(self):
        for name in fse.__all__:
            self.assertTrue(hasattr(fse, name), name)

    def test_version(self):
        self.assertIsInstance(fse.__version__, str)
        self.assertNotEqual("0+unknown", fse.__version__)

    def test_requires_gensim_4(self):
        with patch("gensim.__version__", "3.8.3"):
            with self.assertRaises(ImportError):
                importlib.reload(fse)
        importlib.reload(fse)
