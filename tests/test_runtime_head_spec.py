"""The immutable multilabel specification and output widths."""

import numpy as np
import pytest

from fse._contract import Link
from fse._runtime._head_spec import HeadSpec


def test_one_label_is_one_column_and_thresholds_are_owned():
    limits = np.array([0.3], np.float32)
    head = HeadSpec(
        np.ones((1, 3), np.float32), np.zeros(1, np.float32), False, Link.MULTILABEL, limits
    )
    limits[0] = 0.9
    assert head.thresholds[0] == np.float32(0.3)
    assert not head.thresholds.flags.writeable
    assert [head.width(kind) for kind in ("labels", "proba", "logits")] == [1, 1, 1]
    default = HeadSpec(None, np.zeros(2, np.float32), False, Link.MULTILABEL)
    assert default.thresholds.tolist() == [0.5, 0.5]


@pytest.mark.parametrize(
    "limits",
    [
        np.zeros(3, np.float32),
        np.array([-0.1, 0.5], np.float32),
        np.array([1, np.nan], np.float32),
        np.zeros(2),
    ],
)
def test_invalid_thresholds_are_refused(limits):
    with pytest.raises(ValueError, match="thresholds"):
        HeadSpec(None, np.zeros(2, np.float32), False, Link.MULTILABEL, limits)


def test_other_links_do_not_accept_thresholds():
    with pytest.raises(ValueError, match="only for multilabel"):
        HeadSpec(None, np.zeros(2, np.float32), False, Link.SOFTMAX, np.zeros(2, np.float32))


def test_the_rule_follows_the_link_unless_given():
    from fse._contract import Head

    def make(link, rule=None, k=2):
        return HeadSpec(
            np.ones((k, 3), np.float32), np.zeros(k, np.float32), False, link, rule=rule
        )

    assert make(Link.SOFTMAX).rule is Head.LINEAR
    assert make(None).rule is Head.LINEAR
    assert make(Link.MULTILABEL).rule is Head.LINEAR2
    assert make(Link.OVR, Head.LINEAR2).rule is Head.LINEAR2
    assert make(Link.SIGMOID, Head.LINEAR2, 1).rule is Head.LINEAR2
    with pytest.raises(ValueError, match="multilabel@1 needs the label rule linear@2"):
        make(Link.MULTILABEL, Head.LINEAR)
    with pytest.raises(ValueError, match="linear@2 needs a link"):
        make(None, Head.LINEAR2)
    with pytest.raises(TypeError, match="rule must be a Head"):
        make(Link.OVR, "linear@2")


def test_a_probability_buffer_has_the_width_of_predict_proba():
    from fse._contract import Head

    one = HeadSpec(np.ones((1, 3), np.float32), np.zeros(1, np.float32), False, Link.SIGMOID)
    assert one.width("proba") == 2
    assert (
        HeadSpec(
            np.ones((1, 3), np.float32),
            np.zeros(1, np.float32),
            False,
            Link.SIGMOID,
            rule=Head.LINEAR2,
        ).width("labels")
        == 1
    )
