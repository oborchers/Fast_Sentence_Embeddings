"""What the tests of fse.Pool share: the threads of the pool as the process sees them, the
fixture that proves a test leaves none, and a small encoder and head."""

import gc
import threading

import numpy as np
import pytest

from fse import Classifier, Encoder
from support.runtime import word_vectors

F32 = np.float32


def fse_threads():
    return [t for t in threading.enumerate() if t.name.startswith("fse-worker")]


def settled(count=0, timeout=10):
    """Whether the number of fse worker threads is ``count`` (idle threads end a moment
    after their executor does)."""
    wait = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if len(fse_threads()) == count:
            return True
        wait.wait(0.01)
    return len(fse_threads()) == count


@pytest.fixture(autouse=True)
def no_fse_threads_left():
    assert settled(0), "a test before left fse worker threads"
    yield
    gc.collect()
    assert settled(0), "this test left fse worker threads"


def make_encoder():
    return Encoder.from_word_vectors(word_vectors(dim=12), normalize=False)


def make_head(encoder, index=0):
    rng = np.random.default_rng(17 + index)
    return Classifier.from_arrays(
        encoder,
        rng.standard_normal((4, 12)).astype(F32),
        rng.standard_normal(4).astype(F32),
        labels=list("abcd"),
        link="softmax",
    )
