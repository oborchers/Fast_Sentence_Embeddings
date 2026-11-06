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
    other = Anchors()
    other.begin("a", 0)
    other.begin("b", 5)
    assert other.take(5) == SourcePosition("a", 0)  # its last row, 4, is in the one before


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


def test_no_anchor_lies_before_the_last_one_given():
    # the longer of two stores meets a batch that starts before its own last anchor
    anchors = Anchors(SourcePosition("o60", 60))
    anchors.begin("o40", 40)
    assert anchors.take(100) == SourcePosition("o60", 60)
    anchors.begin("o100", 100)
    assert anchors.take(120) == SourcePosition("o100", 100)
    anchors.begin("o90", 90)  # never back again
    assert anchors.take(130) == SourcePosition("o100", 100)


def test_a_shard_that_ends_where_the_partner_has_one_takes_its_anchor():
    anchors = Anchors(SourcePosition("o0", 0))
    anchors.begin("o25", 25)
    shared = {40: SourcePosition("o20", 20)}
    assert anchors.take(40, shared) == SourcePosition("o20", 20)
    assert anchors.take(60, shared) == SourcePosition("o25", 25)  # no partner shard ends here


def test_a_shard_before_any_batch_takes_the_last_anchor_of_the_store():
    assert Anchors(SourcePosition("o3", 3)).take(10) == SourcePosition("o3", 3)
