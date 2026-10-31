"""Exact numerical comparison over every row, without sampling or ANN.

Principal angles compare centered column spaces over the same ordered texts.
The algorithm is scipy.linalg.subspace_angles (Knyazev and Argentati, 2002).
Recall compares cosine neighborhoods for every nonzero query, excluding itself.
Ties use the smaller document ID. This measures numerical neighborhood stability,
not relevance on a labeled retrieval task. MTEB measures retrieval quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from compare_inputs import identity


def normalized(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("vectors must be a nonempty finite matrix")
    norms = np.linalg.norm(matrix, axis=1)
    return np.divide(
        matrix, norms[:, None], out=np.zeros_like(matrix), where=norms[:, None] != 0
    ), norms != 0


def neighbors(vectors: np.ndarray, begin: int, end: int, k: int) -> np.ndarray:
    scores = vectors[begin:end] @ vectors.T
    scores[np.arange(end - begin), np.arange(begin, end)] = -np.inf
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def recall(a: np.ndarray, b: np.ndarray, keep: np.ndarray, k: int) -> float | None:
    overlap = 0
    queries = 0
    for begin in range(0, len(a), 64):
        end = min(begin + 64, len(a))
        left = neighbors(a, begin, end, k)
        right = neighbors(b, begin, end, k)
        for row, valid in enumerate(keep[begin:end]):
            if valid:
                overlap += len(set(left[row]) & set(right[row]))
                queries += 1
    return overlap / (queries * k) if queries else None


def compare(a: np.ndarray, b: np.ndarray, k: int = 10) -> dict:
    from scipy.linalg import subspace_angles

    if a.shape != b.shape:
        raise ValueError("numerical comparison requires the same matrix shape and row order")
    if not 0 < k < len(a):
        raise ValueError("k must be positive and smaller than the row count")
    left, nonzero_a = normalized(a)
    right, nonzero_b = normalized(b)
    keep = nonzero_a & nonzero_b
    cosines = (left[keep] * right[keep]).sum(axis=1)
    centered_a = np.asarray(a, np.float64) - np.mean(a, axis=0, dtype=np.float64)
    centered_b = np.asarray(b, np.float64) - np.mean(b, axis=0, dtype=np.float64)
    angles = subspace_angles(centered_a, centered_b)
    return {
        "shape": a.shape,
        "k": k,
        "recall_at_k": recall(left, right, keep, k),
        "query_rows": int(keep.sum()),
        "zero_rows_reference": int((~nonzero_a).sum()),
        "zero_rows_candidate": int((~nonzero_b).sum()),
        "zero_disagreements": int(np.count_nonzero(nonzero_a != nonzero_b)),
        "max_absolute_error": float(np.abs(np.asarray(a, np.float64) - b).max()),
        "minimum_cosine": float(cosines.min()) if len(cosines) else None,
        "principal_angles_degrees": np.rad2deg(angles).tolist(),
        "subspace": "centered column spaces, all rows, scipy.linalg.subspace_angles",
    }


def main() -> None:
    from fse._container._hash import checksum

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    a, b = (np.load(path, allow_pickle=False) for path in (args.reference, args.candidate))
    row = {
        **identity(),
        **compare(a, b, args.k),
        "reference": args.reference.name,
        "candidate": args.candidate.name,
        "reference_digest": checksum(a),
        "candidate_digest": checksum(b),
    }
    with args.results.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
