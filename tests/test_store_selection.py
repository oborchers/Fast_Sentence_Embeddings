"""Selection scans IDs against the short filter and admits bounded workspace."""

import numpy as np
import pytest

from fse._store import _selection


def test_membership_does_not_overrun_filter():
    ids = np.array([-(2**63), -1, 0, 2**63 - 1], dtype="int64")
    assert list(_selection.membership(ids, np.array([-1, 0], dtype="int64"))) == [1, 2]
    assert list(_selection.prepare([2, 2, 0], "rows", 3)) == [0, 2]
    with pytest.raises(ValueError, match="positions"):
        _selection.prepare([-1], "rows", 3)


def test_budget_is_checked_before_normalizing(monkeypatch):
    monkeypatch.setattr(_selection, "FILTER_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        _selection.prepare([1], "doc_ids", 3)
