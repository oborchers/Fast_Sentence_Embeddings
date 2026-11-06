"""The closed table of the kinds of a shard store."""

import pytest

from fse._contract import UnsupportedArtifactError
from fse._store._kinds import KINDS, kind_named, kind_of, kind_of_root
from fse._store._scores_format import SCORES
from fse._store._vectors_format import VECTORS, StoreInfo


def test_the_table_holds_the_vector_and_the_score_kind():
    assert KINDS == {"sentence_vectors": VECTORS, "prediction_scores": SCORES}
    assert kind_named("sentence_vectors") is VECTORS
    assert kind_named("prediction_scores") is SCORES


def test_a_name_that_is_not_in_the_table_is_no_kind():
    with pytest.raises(KeyError):
        kind_named("something_else")


def test_a_root_is_read_by_the_kind_it_names():
    assert kind_of_root({"fse.kind": "prediction_scores"}) is SCORES
    assert kind_of_root({"fse.kind": "sentence_vectors"}) is VECTORS


def test_a_root_without_a_text_kind_is_read_as_vectors_for_its_old_messages():
    assert kind_of_root({"fse.kind": 3}) is VECTORS
    assert kind_of_root({}) is VECTORS
    assert kind_of_root("not even a dict") is VECTORS  # its parser refuses it, in order


def test_a_kind_that_is_not_in_the_table_is_unsupported():
    with pytest.raises(UnsupportedArtifactError, match="something_else"):
        kind_of_root({"fse.kind": "something_else"})


def test_a_store_is_read_by_the_kind_its_info_names():
    assert kind_of(StoreInfo("xxh3-128:" + "0" * 32, "float32", 3)) is VECTORS
