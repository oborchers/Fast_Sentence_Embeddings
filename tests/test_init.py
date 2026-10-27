"""Tests for the package root."""

import ast
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import fse

SRC = Path(fse.__file__).parent
# Modules of 2.0: they must import without gensim (and without the 1.x modules).
GENSIM_FREE = ["_codes", "_container", "_contract", "_reference", "_validation", "_wordvectors"]


@pytest.mark.parametrize("name", fse.__all__)
def test_all_is_importable(name):
    assert hasattr(fse, name), name


def test_version():
    assert isinstance(fse.__version__, str)
    assert fse.__version__ != "0+unknown"


def test_unknown_attribute():
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        fse.Nope  # noqa: B018
    assert "Average" in dir(fse)


def test_requires_gensim_4():
    # reload keeps the module dict, so drop the cached class first.
    fse.__dict__.pop("Average", None)
    with patch("gensim.__version__", "3.8.3"), pytest.raises(ImportError, match=r"gensim>=4\.0"):
        fse.Average  # noqa: B018
    assert fse.Average.__name__ == "Average"


def test_import_without_gensim():
    """``import fse`` and the 2.0 classes work in a process where gensim cannot be imported."""
    code = (
        "import builtins\n"
        "real = builtins.__import__\n"
        "def guard(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in ('gensim', 'scipy', 'smart_open'):\n"
        "        raise ImportError(name + ' blocked')\n"
        "    return real(name, *args, **kwargs)\n"
        "builtins.__import__ = guard\n"
        "import numpy as np\n"
        "import fse\n"
        "wv = fse.WordVectors.from_arrays(np.ones((1, 2), np.float32), words=['a'])\n"
        "assert wv.index('a') == 0\n"
        "try:\n"
        "    fse.Average\n"
        "except ImportError as err:\n"
        "    assert \"pip install 'fse[gensim]'\" in str(err), err\n"
        "else:\n"
        "    raise AssertionError('Average imported without gensim')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("module", GENSIM_FREE)
def test_2_0_modules_import_no_gensim(module):
    """A static check of every import statement, also those inside functions."""
    tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            assert name.split(".")[0] not in ("gensim", "scipy", "smart_open"), (module, name)
            assert not name.startswith("fse.models"), (module, name)
            assert not name.startswith(("fse.inputs", "fse.vectors")), (module, name)
