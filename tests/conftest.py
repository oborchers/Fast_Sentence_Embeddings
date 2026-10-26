"""Shared fixtures for the fse test suite.

The corpus in ``tests/test_data`` is read-only input. Tests that write files
use pytest's ``tmp_path``.

The word vector models are function scoped on purpose: several tests change
the vectors, the word counts or map the vectors to disk, and the models must
not carry that state into other tests. Building them is cheap.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from gensim.models import FastText, Word2Vec

TEST_DATA = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Directory with the read-only test input data."""
    return TEST_DATA


@pytest.fixture(scope="session")
def corpus_path(test_data_dir: Path) -> Path:
    """Path to the corpus of 100 sentences, one per line."""
    return test_data_dir / "test_sentences.txt"


@pytest.fixture(scope="session")
def corpus(corpus_path: Path) -> tuple[tuple[str, ...], ...]:
    """The tokenized corpus, read once and immutable."""
    with open(corpus_path) as file:
        return tuple(tuple(line.split()) for line in file)


@pytest.fixture
def sentences(corpus: tuple[tuple[str, ...], ...]) -> list[list[str]]:
    """A fresh, mutable copy of the tokenized corpus."""
    return [list(sentence) for sentence in corpus]


@pytest.fixture
def make_w2v(corpus: tuple[tuple[str, ...], ...]) -> Callable[..., Word2Vec]:
    """Build a Word2Vec model with the vocabulary of the corpus."""

    def _make(vector_size: int, **kwargs) -> Word2Vec:
        w2v = Word2Vec(min_count=1, vector_size=vector_size, **kwargs)
        w2v.build_vocab([list(sentence) for sentence in corpus])
        return w2v

    return _make


@pytest.fixture
def make_ft(corpus: tuple[tuple[str, ...], ...]) -> Callable[..., FastText]:
    """Build a FastText model with the vocabulary of the corpus."""

    def _make(vector_size: int, **kwargs) -> FastText:
        ft = FastText(min_count=1, vector_size=vector_size, **kwargs)
        ft.build_vocab([list(sentence) for sentence in corpus])
        return ft

    return _make


@pytest.fixture
def windows_file_maps(monkeypatch):
    """Fail like Windows: a file that a live memmap maps cannot be replaced, cut or deleted.

    macOS and Linux allow all three, so tests use this fixture to catch maps that are
    still referenced when a file is moved.
    """
    import gc
    import os
    import pathlib

    import numpy as np

    def mapped(path) -> bool:
        target = os.path.realpath(path)
        for obj in gc.get_objects():
            if (
                isinstance(obj, np.memmap)
                and getattr(obj, "_mmap", None) is not None
                and obj.filename is not None
                and os.path.realpath(obj.filename) == target
            ):
                return True
        return False

    def guard(original, paths_of):
        def wrapper(*args, **kwargs):
            for path in paths_of(*args):
                if os.path.exists(path) and mapped(path):
                    raise PermissionError(f"[emulated WinError 32] {path} is mapped")
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(os, "replace", guard(os.replace, lambda src, dst, *a: (src, dst)))
    monkeypatch.setattr(os, "truncate", guard(os.truncate, lambda path, *a: (path,)))
    monkeypatch.setattr(
        pathlib.Path, "unlink", guard(pathlib.Path.unlink, lambda self, *a: (str(self),))
    )
