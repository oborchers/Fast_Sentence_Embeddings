"""The anchor a shard takes: the batch it ends in, or the one before it (docs/architecture/
store.md, "Stream positions")."""

import pytest

from fse._store._anchors import Anchors
from fse._store._position_format import SourcePosition


def test_a_shard_takes_the_batch_it_ends_in():
    anchors = Anchors()
    anchors.begin("a", 0)
    anchors.begin("b", 5)
    assert anchors.take(8) == SourcePosition("b", 5)  # its last row, 7, is in the batch of row 5
    assert anchors.take(5) == SourcePosition("a", 0)  # its last row, 4, is in the one before


def test_a_batch_that_committed_nothing_is_not_the_anchor():
    # A commits 5 rows, B fails without a commit: the shard ends at 5 and takes A (the review).
    anchors = Anchors()
    anchors.begin("a", 0)
    anchors.begin("b", 5)
    anchor = anchors.take(5)
    assert anchor == SourcePosition("a", 0)
    assert anchor.row < 5


def test_the_first_batch_is_the_anchor_of_a_shard_inside_it():
    anchors = Anchors()
    anchors.begin("a", 3)  # the first batch starts before the rows the store holds
    assert anchors.take(10) == SourcePosition("a", 3)


def test_a_shard_before_any_batch_is_a_fault():
    with pytest.raises(RuntimeError, match="before the first batch"):
        Anchors().take(4)
    anchors = Anchors()
    anchors.begin("a", 5)
    with pytest.raises(RuntimeError, match="before the first batch"):
        anchors.take(2)


def test_a_position_is_checked():
    anchors = Anchors()
    with pytest.raises(TypeError, match="a position is a str"):
        anchors.begin(5, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1 to 65536 bytes"):
        anchors.begin("", 0)
