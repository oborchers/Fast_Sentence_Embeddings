"""Write the 1.1.0 reference fixture in tests/test_data/v1_reference.

Run with fse 1.1.0 from PyPI (see README.md in this directory), not with the repository:

    .venv-baseline/bin/python benchmarks/baseline/make_v1_fixture.py

The fixture holds small random word vectors (plain and FastText), sentences with empty,
unknown-only, long and repeated-word cases, and what 1.1.0 computes from them: the rows
its kernel reads per word vector format, the weights and components it learns, and the
sentence vectors of ``train`` and ``infer`` for Average, SIF, uSIF and Average over
FastText. It also stores the batches of token ids and ngram buckets that the 2.0
contract derives from the same sentences, so the 2.0 tests need no gensim.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

import fse
from fse import SIF, Average, IndexedList, uSIF
from fse.models.average_inner import _kernel_info, ft_ngram_hashes

OUT = Path(__file__).resolve().parents[2] / "tests" / "test_data" / "v1_reference"
OOV_TOKEN = 0xFFFFFFFE
VOCAB, DIM, BUCKET = 400, 12, 500


def word_vectors(rng):
    from gensim.models import FastText, KeyedVectors

    words = [f"w{i}" for i in range(VOCAB)]
    rows = rng.standard_normal((VOCAB, DIM)).astype(np.float32)
    rows[:8] *= np.float32(1e-20)  # tiny rows, for scales and subnormal products
    kv = KeyedVectors(DIM)
    kv.add_vectors(words, rows)
    for rank, word in enumerate(words, start=1):
        kv.set_vecattr(word, "count", 100_000 // rank + 1)
    ft = FastText(vector_size=DIM, min_count=1, bucket=BUCKET, seed=0)
    ft.build_vocab([words[:200]])
    ft.wv.vectors_vocab[:] = rng.standard_normal(ft.wv.vectors_vocab.shape).astype(np.float32)
    ft.wv.vectors_ngrams[:] = rng.standard_normal(ft.wv.vectors_ngrams.shape).astype(np.float32)
    for rank, word in enumerate(ft.wv.index_to_key, start=1):
        ft.wv.set_vecattr(word, "count", 50_000 // rank + 1)
    return kv, ft.wv


def sentences(rng):
    known = [f"w{i}" for i in range(VOCAB)]
    out = [[], ["x1", "x2"], ["w3"] * 7, [known[i] for i in rng.integers(0, VOCAB, 1000)]]
    out.append(["w0", "w1", "w2", "w3"])  # tiny rows only
    while len(out) < 96:
        n = int(rng.integers(0, 15))
        out.append([known[i] if rng.random() > 0.15 else f"x{i}" for i in rng.integers(0, VOCAB, n)])
    return out


def batch(wv, sents, fasttext, max_ngrams):
    """Token ids, offsets and FastText ngram buckets as the 2.0 contract defines them."""
    tokens, offsets, ngrams, ngram_offsets = [], [0], [], [0]
    for sent in sents:
        for word in sent:
            if word in wv.key_to_index:
                tokens.append(wv.key_to_index[word])
            elif fasttext:
                tokens.append(OOV_TOKEN)
                ngrams.extend(ft_ngram_hashes(word, wv.min_n, wv.max_n, wv.bucket, max_ngrams))
                ngram_offsets.append(len(ngrams))
        offsets.append(len(tokens))
    out = {
        "tokens": np.array(tokens, dtype=np.uint32),
        "offsets": np.array(offsets, dtype=np.int64),
    }
    if fasttext:
        out["ngrams"] = np.array(ngrams, dtype=np.uint32)
        out["ngram_offsets"] = np.array(ngram_offsets, dtype=np.int64)
    return out


def main() -> None:
    if fse.__version__ != "1.1.0" or Path(fse.__file__).resolve().is_relative_to(OUT.parents[2]):
        sys.exit(f"run with fse 1.1.0 from PyPI, not {fse.__version__} at {fse.__file__}")
    rng = np.random.default_rng(20261027)
    kv, ftv = word_vectors(rng)
    sents = sentences(rng)
    data = IndexedList(sents)
    arrays: dict[str, np.ndarray] = {
        "kv.vectors": kv.vectors,
        "kv.counts": np.array([kv.get_vecattr(w, "count") for w in kv.index_to_key], dtype=np.int64),
        "ft.vectors_vocab": ftv.vectors_vocab,
        "ft.vectors_ngrams": ftv.vectors_ngrams,
        "ft.counts": np.array([ftv.get_vecattr(w, "count") for w in ftv.index_to_key], dtype=np.int64),
    }
    configs = {
        "Average": (Average, kv, {}),
        "SIF": (SIF, kv, {"components": 2}),
        "uSIF": (uSIF, kv, {"components": 2, "length": 11}),
        "AverageFT": (Average, ftv, {}),
    }
    for name, (cls, wv, kwargs) in configs.items():
        fasttext = name.endswith("FT")
        for dtype in ("float32", "float16", "int8"):
            model = cls(wv, wv_dtype=dtype, **kwargs)
            for key, value in batch(wv, sents, fasttext, model.batch_ngrams).items():
                arrays[f"batch.{'ft' if fasttext else 'kv'}.{key}"] = value
            model.train(data)
            prefix = f"{name}.{dtype}"
            arrays[f"{prefix}.train"] = np.asarray(model.sv.vectors, dtype=np.float32)
            arrays[f"{prefix}.infer"] = model.infer(data)
            arrays[f"{prefix}.weights"] = model.word_weights
            # The float32 rows are the word vectors above.
            words, scales, ngram_rows, ngram_scales = model._word_arrays()
            if dtype != "float32":
                arrays[f"{prefix}.rows"] = np.asarray(words)
                if scales is not None:
                    arrays[f"{prefix}.scales"] = np.asarray(scales)
                if fasttext:
                    arrays[f"{prefix}.ngram_rows"] = np.asarray(ngram_rows)
                    if ngram_scales is not None:
                        arrays[f"{prefix}.ngram_scales"] = np.asarray(ngram_scales)
            if name in ("SIF", "uSIF"):
                arrays[f"{prefix}.components"] = model.svd_res[1].astype(np.float32)
                if name == "uSIF":
                    arrays[f"{prefix}.component_weights"] = model.svd_weights.astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "outputs.npz", **arrays)
    (OUT / "sentences.json").write_text(json.dumps(sents), encoding="utf-8")
    # The vocabulary order of 1.1.0's gensim, which other gensim versions may not repeat.
    words = {"kv": kv.index_to_key, "ft": ftv.index_to_key}
    (OUT / "words.json").write_text(json.dumps(words), encoding="utf-8")
    from threadpoolctl import threadpool_info

    manifest = {
        "fse": fse.__version__,
        "python": platform.python_version(),
        "machine": platform.platform(),
        "kernel": _kernel_info(),
        "versions": {p: version(p) for p in ("numpy", "scipy", "gensim")},
        "blas": [{k: i.get(k) for k in ("internal_api", "version")} for i in threadpool_info()],
        "sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(OUT.iterdir())
            if f.name != "MANIFEST.json"
        },
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {sum(f.stat().st_size for f in OUT.iterdir()):,} bytes")


if __name__ == "__main__":
    main()
