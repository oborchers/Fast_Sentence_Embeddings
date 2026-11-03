"""Document IDs do not change positional tuple semantics or result serialization."""

import pickle

import numpy as np

from fse._store._result import DocumentSearchResult, SearchResult


def test_two_fields_and_pickle():
    result = DocumentSearchResult(np.array([1]), np.array([0.5], dtype="float32"), np.array([5]))
    ids, scores = result
    assert ids is result.ids
    assert scores is result.scores
    assert isinstance(result, SearchResult)
    assert list(pickle.loads(pickle.dumps(result)).doc_ids) == [5]
    assert not hasattr(SearchResult(ids, scores), "doc_ids")
