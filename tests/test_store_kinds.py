"""The closed table of the kinds of a shard store."""

import pytest

from fse._store._kinds import KINDS, kind_named, kind_of, kind_of_root
from fse._store._vectors_format import VECTORS, StoreInfo


def test_the_table_holds_the_vector_kind_alone():
    assert KINDS == {"sentence_vectors": VECTORS}
    assert kind_named("sentence_vectors") is VECTORS


def test_a_name_that_is_not_in_the_table_is_no_kind():
    with pytest.raises(KeyError):
        kind_named("prediction_scores")


def test_every_root_and_store_is_read_as_vectors_for_now():
    assert kind_of_root({"fse.kind": "anything"}) is VECTORS
    assert kind_of_root("not even a dict") is VECTORS  # its parser refuses it, in order
    assert kind_of(StoreInfo("xxh3-128:" + "0" * 32, "float32", 3)) is VECTORS
