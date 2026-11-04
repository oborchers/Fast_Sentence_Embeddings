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
