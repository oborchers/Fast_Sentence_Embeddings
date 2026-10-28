from pathlib import Path

import numpy as np
import pytest

from fse._encoder._weights import (
    check_counts,
    sif_weights,
    usif_component_weights,
    usif_weights,
    word_probabilities,
)

FIXTURE = np.load(Path(__file__).parent / "test_data" / "v1_reference" / "outputs.npz")
COUNTS = FIXTURE["kv.counts"]


def test_the_weights_are_the_bits_of_1_1_0():
    for dtype in ("float32", "float16", "int8"):
        sif = sif_weights(COUNTS, 1e-3)
        usif = usif_weights(COUNTS, 11)
        assert sif.dtype == usif.dtype == np.float32
        assert sif.tobytes() == FIXTURE[f"SIF.{dtype}.weights"].tobytes()
        assert usif.tobytes() == FIXTURE[f"uSIF.{dtype}.weights"].tobytes()


def test_the_component_weights_follow_1_1_0():
    weights = FIXTURE["uSIF.float32.component_weights"]
    assert weights.dtype == np.float32
    s = np.array([3.0, 1.0, 0.5], np.float32)
    got = usif_component_weights(s)
    want = (s**2) / (s**2).sum().astype(np.float32)
    assert got.tobytes() == want.astype(np.float32).tobytes()
    with pytest.raises(ValueError, match="no variance"):
        usif_component_weights(np.zeros(2, np.float32))


def test_word_probabilities_round_once():
    counts = np.array([1, 2, 3, 2**40], np.int64)
    want = counts.astype(np.float32)
    want /= counts.sum()  # as 1.1.0
    assert word_probabilities(counts).tobytes() == want.tobytes()


@pytest.mark.parametrize(
    ("counts", "error", "match"),
    [
        (np.array([1.0, 2.0]), TypeError, "vector of integers"),
        (np.ones((2, 1), np.int64), TypeError, "vector of integers"),
        (np.ones(3, np.int64), ValueError, "one count per word"),
        (np.array([1, -1]), ValueError, "must not be negative"),
        (np.zeros(2, np.int64), ValueError, "all counts are zero"),
        (np.array([2**62, 2**62], np.int64), ValueError, "too large"),
    ],
)
def test_counts_are_checked(counts, error, match):
    with pytest.raises(error, match=match):
        check_counts(counts, 2)


def test_counts_of_other_integer_types_are_int64():
    assert check_counts(np.array([1, 2], np.uint16), 2).dtype == np.int64
