"""Shared fixtures for the fse test suite.

The corpus in ``tests/test_data`` is read-only input. Tests that write files
use pytest's ``tmp_path``.

gensim is only needed by tests/test_gensim.py, which builds its own models.

Every WARNING or higher record of fse that no ``expect_log`` expected fails the test in
the phase it came in (setup, call or teardown; tests/support/logs.py), unless the test is
marked ``expected_log(event, ...)`` for its event.
"""

from pathlib import Path

import pytest

from support.logs import PAUSED, STRICT, TOP, unexpected

TEST_DATA = Path(__file__).parent / "test_data"


def _strict(item):
    STRICT.records.clear()
    if not PAUSED:
        TOP.addHandler(STRICT)
    try:
        result = yield
    finally:
        TOP.removeHandler(STRICT)
    allowed = {e for mark in item.iter_markers("expected_log") for e in mark.args}
    found = unexpected([r for r in STRICT.records if r.fse_event not in allowed])
    STRICT.records.clear()
    if found:
        pytest.fail("unexpected records of fse (use expect_log):\n" + "\n".join(found))
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    return (yield from _strict(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    return (yield from _strict(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item):  # pluggy passes only the arguments a hook names
    return (yield from _strict(item))


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
    with open(corpus_path, encoding="utf-8") as file:
        return tuple(tuple(line.split()) for line in file)


@pytest.fixture
def sentences(corpus: tuple[tuple[str, ...], ...]) -> list[list[str]]:
    """A fresh, mutable copy of the tokenized corpus."""
    return [list(sentence) for sentence in corpus]


@pytest.fixture
def python_sigint():
    """SIGINT raising KeyboardInterrupt, as Python sets it up unless SIGINT was ignored
    when it started (a background job of a shell)."""
    import signal

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


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
            # type(), not isinstance(): isinstance reads __class__, which some objects of
            # third-party modules turn into a deprecation warning (torch.distributed).
            if (
                issubclass(type(obj), np.memmap)
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

    monkeypatch.setattr(os, "replace", guard(os.replace, lambda src, dst, *_args: (src, dst)))
    monkeypatch.setattr(os, "truncate", guard(os.truncate, lambda path, *_args: (path,)))
    monkeypatch.setattr(
        pathlib.Path, "unlink", guard(pathlib.Path.unlink, lambda self, *_args: (str(self),))
    )
