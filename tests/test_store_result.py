"""Document IDs do not change positional tuple semantics or result serialization."""

import copy
import pickle
import sys

import numpy as np
import pytest

from fse._store._result import DocumentSearchResult, SearchResult


def test_two_fields_and_pickle():
    result = DocumentSearchResult(np.array([1]), np.array([0.5], dtype="float32"), np.array([5]))
    ids, scores = result
    assert ids is result.ids
    assert scores is result.scores
    assert isinstance(result, SearchResult)
    assert list(pickle.loads(pickle.dumps(result)).doc_ids) == [5]
    assert not hasattr(SearchResult(ids, scores), "doc_ids")


def _result():
    return DocumentSearchResult(
        np.array([1, 2]), np.array([0.5, 0.25], "float32"), np.array([7, 8])
    )


def test_replace_keeps_or_replaces_doc_ids():
    result = _result()
    again = result._replace(scores=np.array([1, 1], "float32"))
    assert isinstance(again, DocumentSearchResult)
    assert again.doc_ids is result.doc_ids
    assert again.ids is result.ids
    other = result._replace(doc_ids=np.array([9, 9]))
    assert list(other.doc_ids) == [9, 9]
    assert other.scores is result.scores
    with pytest.raises(TypeError, match="unexpected field names"):
        result._replace(nope=1)


@pytest.mark.skipif(sys.version_info < (3, 13), reason="copy.replace is new in 3.13")
def test_copy_replace_keeps_or_replaces_doc_ids():
    result = _result()
    assert copy.replace(result, scores=np.zeros(2, "float32")).doc_ids is result.doc_ids
    assert list(copy.replace(result, doc_ids=np.array([1, 1])).doc_ids) == [1, 1]


def test_asdict_make_and_repr():
    result = _result()
    assert list(result._asdict()) == ["ids", "scores", "doc_ids"]
    with pytest.raises(TypeError, match="needs doc_ids"):
        DocumentSearchResult._make([result.ids, result.scores])
    assert "doc_ids=array([7, 8])" in repr(result)
    assert len(result) == 2
