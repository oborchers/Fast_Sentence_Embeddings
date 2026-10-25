"""Tests for the package root."""

import importlib
from unittest.mock import patch

import pytest

import fse


@pytest.mark.parametrize("name", fse.__all__)
def test_all_is_importable(name):
    assert hasattr(fse, name), name


def test_version():
    assert isinstance(fse.__version__, str)
    assert fse.__version__ != "0+unknown"


def test_requires_gensim_4():
    with patch("gensim.__version__", "3.8.3"), pytest.raises(ImportError):
        importlib.reload(fse)
    importlib.reload(fse)
