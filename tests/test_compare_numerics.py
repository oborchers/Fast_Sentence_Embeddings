"""Neighborhood comparison includes every eligible query and resolves ties by ID."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def numerics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("compare_numerics")


def test_identical_vectors_have_full_recall(numerics):
    pytest.importorskip("scipy")
    values = np.random.default_rng(0).normal(size=(131, 4))
    got = numerics.compare(values, values, k=5)
    assert got["recall_at_k"] == 1
    assert got["query_rows"] == 131
    assert got["max_absolute_error"] == 0
    assert max(got["principal_angles_degrees"]) < 1e-10


def test_self_is_excluded_and_ties_use_smaller_id(numerics):
    values = np.ones((4, 2))
    assert numerics.neighbors(values, 0, 4, 2).tolist() == [[1, 2], [0, 2], [0, 1], [0, 1]]


def test_rotation_preserves_subspace_and_neighborhoods(numerics):
    pytest.importorskip("scipy")
    values = np.random.default_rng(1).normal(size=(80, 3))
    rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    got = numerics.compare(values, values @ rotation, k=5)
    assert got["recall_at_k"] == 1
    assert max(got["principal_angles_degrees"]) < 1e-10
    assert got["max_absolute_error"] > 0


def test_zero_disagreements_are_explicit(numerics):
    pytest.importorskip("scipy")
    a = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
    b = a.copy()
    b[2] = [1, 0]
    got = numerics.compare(a, b, k=1)
    assert got["zero_disagreements"] == 1
    assert got["query_rows"] == 2


def test_nonfinite_vectors_raise(numerics):
    with pytest.raises(ValueError, match="finite"):
        numerics.normalized(np.array([[np.nan, 0]]))
