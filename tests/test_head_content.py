"""The content of a linear head: weights and biases, labels, links."""

import numpy as np
import pytest

from fse import _native
from fse._contract import Link
from fse._head._content import HeadContent, classes, labels, link, weights

F32 = np.float32


def test_weights_take_a_matrix_or_a_vector():
    w, b = weights(np.ones((2, 3)), [1, -2], 3)
    assert w.dtype == F32
    assert w.shape == (2, 3)
    assert b.tolist() == [1.0, -2.0]
    assert not w.flags.writeable
    assert not b.flags.writeable
    w, b = weights(np.arange(3), None, 3)
    assert w.shape == (1, 3)
    assert b.tolist() == [0.0]
    _, b = weights(np.ones((4, 3)), 0.5, 3)
    assert b.tolist() == [0.5] * 4  # a scalar bias for every output


def test_weights_are_copies():
    given = np.ones((1, 3), F32)
    w, _ = weights(given, None, 3)
    given[0, 0] = 7
    assert w[0, 0] == 1


def test_weights_round_to_nearest_whatever_the_thread_set():
    value = np.array([[1 + 2.0**-25 + 2.0**-40, 1e-40, 0.0]])  # rounds to 1 and a subnormal
    saved = _native._test_fpenv_set(flush=True, round_up=True)
    try:
        w, _ = weights(value, None, 3)
    finally:
        _native._test_fpenv_restore(saved)
    assert w[0, 0] == F32(1 + 2.0**-25 + 2.0**-40)
    assert w[0, 1] == F32(1e-40) != 0


@pytest.mark.parametrize(
    ("weight", "bias", "error", "match"),
    [
        (np.ones((2, 4)), None, ValueError, r"shape \(k, 3\) or \(3,\)"),
        (np.ones((0, 3)), None, ValueError, r"shape \(k, 3\)"),
        (np.ones((1, 1, 3)), None, ValueError, "at most two dimensions"),
        (np.array([["a", "b", "c"]]), None, TypeError, "weight must be numbers"),
        (np.ones((1, 3), bool), None, TypeError, "weight must be numbers"),
        (np.ones((2, 3)), [1, 2, 3], ValueError, "bias must hold 2 values"),
        (np.ones((2, 3)), np.ones((2, 1)), ValueError, "bias must hold 2 values"),
        (np.array([[1, 2, np.nan]]), None, ValueError, r"weight\[0, 2\] is not finite"),
        (np.array([[1, 2, 1e39]]), None, ValueError, r"weight\[0, 2\] is not finite in float32"),
        (np.ones((2, 3)), [0, np.inf], ValueError, r"bias\[1\] is not finite"),
    ],
)
def test_weights_refuse(weight, bias, error, match):
    with pytest.raises(error, match=match):
        weights(weight, bias, 3)


def test_labels():
    assert labels(["no", "yes"], 1) == ("no", "yes")
    assert labels(np.array([3, 1, 2]), 3) == (3, 1, 2)
    assert isinstance(labels(np.array([3, 1], np.int8), 1)[0], int)
    assert classes(("a", "b")).dtype.kind == "U"
    assert classes((1, 2)).dtype == np.int64


@pytest.mark.parametrize(
    ("values", "k", "error", "match"),
    [
        (["a", "b", "c"], 1, ValueError, r"needs 2 labels \(negative first\), not 3"),
        (["a", "b"], 3, ValueError, "needs 3 labels, not 2"),
        ([1, "b"], 1, TypeError, "all int or all str"),
        ([1, 1], 1, ValueError, "unique"),
        ([True, False], 1, TypeError, "not a bool"),
        ([1.5, 2.5], 1, TypeError, "not float"),
        ([2**63, 1], 1, ValueError, "beyond int64"),
        ("ab", 1, TypeError, "sequence of int or str"),
        (3, 1, TypeError, "sequence of int or str"),
    ],
)
def test_labels_refuse(values, k, error, match):
    with pytest.raises(error, match=match):
        labels(values, k)


def test_links():
    assert link("auto", 1) is Link.SIGMOID
    assert link("auto", 3) is Link.SOFTMAX
    assert link("ovr", 2) is Link.OVR
    assert link("none", 1) is None
    with pytest.raises(ValueError, match="sigmoid takes one output, not 2"):
        link("sigmoid", 2)
    with pytest.raises(ValueError, match="softmax takes at least two outputs"):
        link("softmax", 1)
    with pytest.raises(ValueError, match="link must be one of"):
        link("tanh", 2)


def test_head_content_properties():
    content = HeadContent("classification", None, np.zeros(2, F32), False, None, ("a", "b"))
    assert content.k == 2
    assert content.link_id == "none"
    assert not content.standalone
    assert (
        HeadContent("regression", None, np.zeros(1, F32), True, Link.OVR, None).link_id == "ovr@1"
    )
