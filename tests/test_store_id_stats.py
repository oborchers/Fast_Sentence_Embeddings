"""ID statistics preserve duplicates and the signed-int64 boundaries."""

import numpy as np
import pytest

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._store._id_stats import IdStats, check_mode, parsed_stats, verify_stats


def test_cross_block_order_and_extremes():
    stats = IdStats()
    stats.update(np.array([-(2**63), 5, 5], dtype="int64"))
    stats.update(np.array([4, 2**63 - 1], dtype="int64"))
    assert not stats.sorted
    expected = parsed_stats(stats.metadata(), "test")
    verify_stats(np.array([-(2**63), 5, 5, 4, 2**63 - 1], dtype="int64"), expected, "test")
    with pytest.raises(CorruptArtifactError, match="statistics"):
        verify_stats(np.array([1, 2], dtype="int64"), expected, "test")


@pytest.mark.parametrize("value", ["-0", "+1", "01", str(2**63), "1.0", ""])
def test_canonical_bounds(value):
    metadata = IdStats(1, 2).metadata()
    metadata["fse.row_ids_min"] = value
    with pytest.raises(CorruptArtifactError):
        parsed_stats(metadata, "test")


def test_unknown_resume_is_unsupported():
    with pytest.raises(UnsupportedArtifactError, match="source_rows@1"):
        check_mode({"fse.resume": "source_rows@1"}, "test")
