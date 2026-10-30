"""The package fse._static_model: fse.StaticModel loads lazily and imports no library of
another model format."""

import subprocess
import sys

import pytest

import fse
from fse._static_model import StaticModel


def test_fse_exports_the_static_model_lazily():
    assert fse.StaticModel is StaticModel
    assert "StaticModel" in fse.__all__
    assert "StaticModel" in dir(fse)


@pytest.mark.parametrize("module", ["model2vec", "tokenizers", "torch", "sklearn", "safetensors"])
def test_using_a_static_model_imports_none_of_their_libraries(module):
    """fse reads their files itself; a library installed in the environment is not loaded."""
    code = f"import sys, fse\nfse.StaticModel\nassert {module!r} not in sys.modules, {module!r}\n"
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
