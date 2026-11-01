"""Offline SIF folding keeps the recipe's projection and unknown-token divisor."""

import importlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def recipe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality"))
    return importlib.import_module("classic_math")


def test_fold_matches_unfolded_sentence_projection(recipe):
    random = np.random.default_rng(0)
    table = random.normal(size=(100, 12)).astype(np.float32)
    tokens = [["a", "missing", "b"], ["b", "b"], ["missing"]]
    lookup = {"a": 4, "b": 9}
    means = recipe.reference_means(table, lookup, tokens)
    assert np.array_equal(means[0], table[[4, 9]].sum(axis=0, dtype=np.float32) / 3)
    basis = np.linalg.qr(random.normal(size=(12, 12)))[0][:, :10].T.astype(np.float32)
    folded = np.empty_like(table)
    recipe.fold_table(table, basis, folded)
    expected = recipe.remove(means, basis)
    actual = recipe.reference_means(folded, lookup, tokens)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert np.array_equal(actual[-1], np.zeros(12, dtype=np.float32))



def test_components_keep_ten_largest_uncentered_directions(recipe):
    pytest.importorskip("scipy.linalg")
    rows = np.diag(np.sqrt(np.arange(1, 13, dtype=np.float32)))
    gram = recipe.gram_of_batches([rows[:6], rows[6:]])
    basis = recipe.components(gram)
    expected = np.eye(12, dtype=np.float32)[-10:][::-1]
    np.testing.assert_array_equal(basis, expected)


def test_sif_uses_training_frequencies_only_and_requires_overlap(recipe):
    weights = recipe.sif_weights(["seen", "unseen"], Counter({"seen": 10}))
    np.testing.assert_array_equal(weights, np.array([0.001 / 1.001, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="no vocabulary overlap"):
        recipe.sif_weights(["absent"], Counter({"other": 1}))


def test_tokenizer_imports_from_the_comparison_script_path():
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    command = "import sys; sys.path.insert(0, sys.argv[1]); import quality.classic_corpus"
    subprocess.run([sys.executable, "-c", command, str(root)], check=True, capture_output=True)
