"""Caller-facing encoder types, checked by mypy and executed by pytest."""

from pathlib import Path
from typing import assert_type

import numpy as np

import fse


def test_encoder_factory_types(tmp_path: Path) -> None:
    vectors = fse.WordVectors.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"])
    assert_type(vectors, fse.WordVectors)
    encoder = fse.Encoder.from_word_vectors(vectors)
    assert_type(encoder, fse.Encoder)
    assert isinstance(encoder, fse.Encoder)
    path = tmp_path / "encoder.safetensors"
    assert_type(encoder.save(path), None)
    loaded = fse.Encoder.load(path)
    assert_type(loaded, fse.Encoder)
    assert isinstance(loaded, fse.Encoder)
