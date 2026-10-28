"""Tests for the package root."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import fse

SRC = Path(fse.__file__).parent
# Every module of fse except the gensim adapter: none may import gensim, scipy or
# smart_open, and none the removed 1.x modules.
MODULES = sorted(
    p.relative_to(SRC).as_posix()
    for p in SRC.rglob("*.py")
    if p.relative_to(SRC).as_posix() not in ("gensim.py", "_version.py")
)
FORBIDDEN = ("gensim", "scipy", "smart_open", "psutil", "threadpoolctl", "wordfreq")


@pytest.mark.parametrize("name", fse.__all__)
def test_all_is_importable(name):
    assert hasattr(fse, name), name


def test_version():
    assert isinstance(fse.__version__, str)
    assert fse.__version__ != "0+unknown"


@pytest.mark.parametrize("name", ["uSIF", "TextList", "Vectors"])
def test_removed_1x_names_point_to_1_1_0(name):
    with pytest.raises(AttributeError, match=r"fse==1\.1\.0"):
        getattr(fse, name)


def test_the_1x_name_usif_points_to_the_2_0_recipe():
    name = "uSIF"
    with pytest.raises(AttributeError, match=r"the 2\.0 recipe is fse\.USIF"):
        getattr(fse, name)


def test_unknown_attribute():
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        fse.Nope  # noqa: B018


@pytest.mark.parametrize("module", ["fse.models", "fse.inputs", "fse.vectors"])
def test_1x_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_import_without_optional_packages():
    """``import fse`` and the 2.0 classes work where gensim and the 1.x dependencies are missing."""
    code = (
        "import builtins\n"
        "real = builtins.__import__\n"
        f"blocked = {FORBIDDEN!r}\n"
        "def guard(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in blocked:\n"
        "        raise ImportError(name + ' blocked')\n"
        "    return real(name, *args, **kwargs)\n"
        "builtins.__import__ = guard\n"
        "import numpy as np\n"
        "import fse, fse._native, fse._reference, fse._gram\n"
        "wv = fse.WordVectors.from_arrays(np.ones((1, 2), np.float32), words=['a'])\n"
        "assert wv.index('a') == 0\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("module", MODULES)
def test_modules_import_no_optional_packages(module):
    """A static check of every import statement, also those inside functions."""
    tree = ast.parse((SRC / module).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            assert name.split(".")[0] not in FORBIDDEN, (module, name)
            assert not name.startswith(("fse.models", "fse.inputs", "fse.vectors")), (module, name)


def test_native_imports_no_optional_packages():
    import re

    # The extension and the .pxi parts it includes.
    parts = sorted(p for p in SRC.glob("_native*") if p.suffix in (".pyx", ".pxi", ".pxd"))
    text = "".join(p.read_text(encoding="utf-8") for p in parts)
    assert "cdef class VocabTable" in text
    for name in FORBIDDEN:
        # Import statements only; comments may name gensim, from which code was adapted.
        assert not re.search(rf"^\s*(c?import|from)\s+{name}\b", text, re.MULTILINE), name


def test_public_api_is_pinned():
    """A new public name needs a deliberate change of this test."""
    names = [
        "SIF",
        "USIF",
        "Average",
        "EncodedBatch",
        "Encoder",
        "MemorySink",
        "QuantizedRows",
        "ShardSink",
        "Sink",
        "WordVectors",
    ]
    assert fse.__all__ == [*names, "__version__"]
    public = {name for name in dir(fse) if not name.startswith("_")}
    assert public - {"gensim"} == set(names)
