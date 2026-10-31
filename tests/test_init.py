"""Tests for the package root."""

import ast
import importlib
import re
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
FORBIDDEN = ("gensim", "scipy", "smart_open", "psutil", "threadpoolctl", "wordfreq", "sklearn")


@pytest.mark.parametrize("name", fse.__all__)
def test_all_is_importable(name):
    assert hasattr(fse, name), name


def test_version():
    assert isinstance(fse.__version__, str)
    assert fse.__version__ != "0+unknown"


@pytest.mark.parametrize("name", ["Average", "SIF", "uSIF", "TextList", "Vectors"])
def test_removed_1x_names_point_to_1_1_0(name):
    with pytest.raises(AttributeError, match=r"part of the 1\.x API.*fse==1\.1\.0"):
        getattr(fse, name)


@pytest.mark.parametrize("name", ["USIF", "Corpus"])
def test_names_of_pre_releases_of_2_0_are_gone(name):
    with pytest.raises(AttributeError, match=f"^module 'fse' has no attribute '{name}'$"):
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
        "import fse, fse._native, fse._reference\n"
        "wv = fse.WordVectors.from_arrays(np.ones((1, 2), np.float32), words=['a'])\n"
        "assert wv.index('a') == 0\n"
        "assert fse.Encoder.from_word_vectors(wv).encode(['a'], workers=1).shape == (1, 2)\n"
        "enc = fse.Encoder.from_word_vectors(wv, normalize=False)\n"
        "clf = fse.Classifier.from_arrays(enc, np.ones(2), labels=[0, 1])\n"
        "assert clf.predict(['a'], workers=1).tolist() == [1]\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_and_local_models_do_not_load_huggingface_hub(tmp_path):
    """huggingface-hub is a dependency, imported only when a model of the Hub is loaded:
    ``import fse``, local files and local folders never load it."""
    code = (
        "import sys\n"
        "import numpy as np\n"
        "import fse\n"
        "assert 'huggingface_hub' not in sys.modules, 'import fse'\n"
        "wv = fse.WordVectors.from_arrays(np.ones((1, 2), np.float32), words=['a'])\n"
        "encoder = fse.Encoder.from_word_vectors(wv)\n"
        f"path = {str(tmp_path / 'encoder.safetensors')!r}\n"
        "encoder.save(path)\n"
        "loaded = fse.Encoder.from_pretrained(path)\n"
        "assert loaded.encode(['a'], workers=1).shape == (1, 2)\n"
        "fse.Encoder.load(path).encode(['a'], workers=1)\n"
        "assert 'huggingface_hub' not in sys.modules, 'a local model'\n"
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
    parts = sorted(p for p in SRC.rglob("_native*") if p.suffix in (".pyx", ".pxi", ".pxd"))
    text = "".join(p.read_text(encoding="utf-8") for p in parts)
    assert "cdef class VocabTable" in text
    for name in FORBIDDEN:
        # Import statements only; comments may name gensim, from which code was adapted.
        assert not re.search(rf"^\s*(c?import|from)\s+{name}\b", text, re.MULTILINE), name


def test_public_api_is_pinned():
    """A new public name needs a deliberate change of this test."""
    names = [
        "Classifier",
        "CorruptArtifactError",
        "EncodedBatch",
        "Encoder",
        "FastTextParams",
        "MemorySink",
        "PredictedBatch",
        "PublishedNotDurableError",
        "QuantizedRows",
        "Regressor",
        "SearchResult",
        "ShardSink",
        "Sink",
        "StaticModel",
        "UnsupportedArtifactError",
        "VectorStore",
        "WordVectors",
    ]
    functions = ["disable_logging", "enable_logging", "repair_store"]
    assert fse.__all__ == [*names, "__version__", *functions]
    public = {name for name in dir(fse) if not name.startswith("_")}
    assert public - {"gensim"} == {*names, *functions}


def test_the_readme_shows_only_names_fse_has():
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    code = "\n".join(re.findall(r"```python\n(.*?)```", readme, re.DOTALL))
    used = set(re.findall(r"\bfse\.([A-Za-z_]\w*)", code)) - {"gensim"}
    used |= set(re.findall(r"`fse\.([A-Za-z_]\w*)", readme)) - {"gensim"}  # inline code
    assert used, "the README shows no fse names"
    assert used <= set(fse.__all__), sorted(used - set(fse.__all__))
    # The removed 1.x names only as code (uSIF stays the name of the method in the text).
    imported = re.findall(r"^from fse import (.+)$", readme, re.MULTILINE)
    names = {n.strip() for line in imported for n in line.split(",")}
    assert not (used | names) & fse._REMOVED, sorted((used | names) & fse._REMOVED)


def test_the_public_errors_and_fasttext_params_are_the_ones_fse_raises_and_takes():
    from fse._container import CorruptArtifactError
    from fse._contract import UnsupportedArtifactError
    from fse._wordvectors_format import FastTextParams

    assert fse.CorruptArtifactError is CorruptArtifactError
    assert fse.UnsupportedArtifactError is UnsupportedArtifactError
    assert fse.FastTextParams is FastTextParams
    assert issubclass(fse.CorruptArtifactError, ValueError)
