"""Caller-facing store types, checked and exercised without network access."""

from pathlib import Path
from typing import Any, assert_type

import numpy as np

import fse


def test_sink_context_type(tmp_path: Path) -> None:
    encoder = fse.Encoder.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"])
    with fse.ShardSink(tmp_path / "vectors", encoder) as sink:
        assert_type(sink, Any)
        encoder.encode(["a b"], sink=sink)
