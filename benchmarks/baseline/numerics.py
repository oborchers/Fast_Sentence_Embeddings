"""Numerics of fse 1.1.0 beyond STS: Gram rounding, components, search under quantization.

On the first ``--sentences`` lines of a tokenized corpus (by default the WikiText-103 fit
corpus of benchmarks/quality/run_mteb.py) with paranmt-300:

- **Gram matrix:** the SIF-weighted sentence vectors (``components=0``) give the Gram
  matrix as 1.1.0 sums it (float32 block products, float64 sum) and exactly in float64;
  reported are the largest relative difference and the principal angles between the 10
  components of both. 1 and ``os.cpu_count()`` workers must give bit-identical components.
- **Search:** SIF-10 with ``sv_dtype`` float32, float16 and int8; for 1,000 fixed query
  rows, recall@10 of ``most_similar`` against float32, and how many neighbors in the
  float32 top 100 reach a cosine of 0.5, 0.7 and 0.9 (what ``min_similarity`` keeps).

    .venv-baseline/bin/python benchmarks/baseline/numerics.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np

import fse
from fse import SIF, CorpusFile, Vectors
from fse.models.utils import components_from_gram, gram_matrix

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = Path.home() / ".cache" / "fse-quality" / "wikitext-103-b08601e0.txt"


def principal_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angles in degrees between the row spaces of two orthonormal bases."""
    s = np.linalg.svd(a.astype(np.float64) @ b.astype(np.float64).T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def neighbors(model, queries, topn):
    return np.array(
        [
            [i for i, _ in model.sv.most_similar(int(q), topn=topn + 1) if i != q][:topn]
            for q in queries
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--sentences", type=int, default=1_000_000)
    parser.add_argument("--vectors", default="paranmt-300")
    args = parser.parse_args()
    vectors = Vectors.from_pretrained(args.vectors, mmap="r")
    out: dict = {
        "fse": fse.__version__,
        "machine": platform.platform(),
        "vectors": args.vectors,
        "sentences": args.sentences,
        "load_average": os.getloadavg()[0],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "corpus.txt"
        with args.corpus.open(encoding="utf-8") as src, path.open("w", encoding="utf-8") as dst:
            dst.writelines(itertools.islice(src, args.sentences))
        corpus = CorpusFile(str(path))

        raw = SIF(vectors, components=0, lang_freq="en", workers=os.cpu_count())
        raw.train(corpus)
        x = np.asarray(raw.sv.vectors)
        gram32 = gram_matrix(x, np.float32)
        gram64 = gram_matrix(x, np.float64)
        n = len(x)
        _, c32 = components_from_gram(gram32, n, 10, np.float32)
        _, c64 = components_from_gram(gram64, n, 10, np.float64)
        out["gram_max_relative_difference"] = float(
            np.abs(gram32 - gram64).max() / np.abs(gram64).max()
        )
        out["component_angles_degrees"] = principal_angles(c32, c64).tolist()

        models = {}
        for dtype in ("float32", "float16", "int8"):
            model = SIF(
                vectors, components=10, lang_freq="en", sv_dtype=dtype, workers=os.cpu_count()
            )
            model.train(corpus)
            models[dtype] = model
        one = SIF(vectors, components=10, lang_freq="en", workers=1)
        one.train(corpus)
        out["workers_compared"] = [1, os.cpu_count()]
        out["components_bit_identical_across_workers"] = bool(
            np.array_equal(
                one.svd_res[1].view(np.uint32), models["float32"].svd_res[1].view(np.uint32)
            )
        )
        queries = np.random.default_rng(0).choice(n, 1000, replace=False)
        reference = neighbors(models["float32"], queries, 100)
        out["recall_at_10"] = {}
        for dtype in ("float16", "int8"):
            got = neighbors(models[dtype], queries, 10)
            hits = [len(set(g) & set(r[:10])) for g, r in zip(got, reference, strict=True)]
            out["recall_at_10"][dtype] = float(np.mean(hits) / 10)
        sv = models["float32"].sv
        q = sv.get_vectors()[queries]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        top = sv.get_vectors()[reference.ravel()].reshape(len(queries), 100, -1)
        top /= np.linalg.norm(top, axis=2, keepdims=True)
        cos = np.einsum("qd,qkd->qk", q, top)
        out["top100_cosine_at_least"] = {
            str(t): float((cos >= t).sum(axis=1).mean()) for t in (0.5, 0.7, 0.9)
        }
        np.savez_compressed(
            HERE / "results" / f"neighbors-{fse.__version__}.npz",
            queries=queries.astype(np.int32),
            top10=reference[:, :10].astype(np.int32),
        )
    (HERE / "results" / f"numerics-{fse.__version__}.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
