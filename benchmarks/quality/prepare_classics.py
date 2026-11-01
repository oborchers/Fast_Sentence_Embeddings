"""Prepare plain paranmt or SIF-10 classics, retaining FastText's native OOV buckets.

SIF is fitted only on all 859,346 cached WikiText-103 training paragraphs.
The recipe's float32 block Gram products and float64 accumulation are unchanged.
The temporary weighted and folded arrays are disk-backed, with bounded blocks.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_artifacts import file_digest  # isort: skip
from compare_inputs import identity  # isort: skip
from quality_coordination import task_reservation  # isort: skip
from classic_corpus import batches, fit_corpus, frequencies  # isort: skip
from classic_math import components, fold_table, gram_of_batches, reference_means, sif_weights  # isort: skip
from classic_sources import ALL_SOURCES, FIT_LINES, FIT_REPO, FIT_REVISION, SEED, stem  # isort: skip
from preparation_catalog import register  # isort: skip


def source_model(name: str):
    from gensim.models import KeyedVectors
    from huggingface_hub import snapshot_download

    folder = Path(
        snapshot_download(
            f"fse/{name}",
            revision=ALL_SOURCES[name],
            allow_patterns=[f"{name}.model", f"{name}.model.*.npy", "README.md"],
            max_workers=2,
        )
    )
    return KeyedVectors.load(str(folder / f"{name}.model"), mmap="r")


def mmap_array(path: Path, shape):
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def weighted_model(kv, counts, temporary: Path):
    import fse
    from fse.gensim import from_gensim

    source = from_gensim(kv)
    words = kv.index_to_key
    weights = sif_weights(words, counts)
    path = temporary / "weighted.npy"
    table = mmap_array(path, source.stored_vectors.shape)
    for start in range(0, len(table), 16384):
        table[start : start + 16384] = (
            source.stored_vectors[start : start + 16384] * weights[start : start + 16384, None]
        )
    params = source.fasttext
    ngrams = getattr(kv, "vectors_ngrams", None) if params is not None else None
    del source
    gc.collect()
    encoder = fse.Encoder.from_arrays(
        table, words=words, fasttext=params, ngrams=ngrams, tokenizer="external", normalize=False
    )
    del table
    path.unlink()  # Encoder.from_arrays copied these owned temporary rows.
    return encoder, weights


def fit_basis(encoder, words, corpus: Path, workers: int):
    if encoder.word_vectors.fasttext is not None:
        # The real fse FastText path: vocabulary rows for known words, OOV ngrams otherwise.
        rows = (encoder.encode(tokens, workers=workers) for tokens in batches(corpus))
    else:
        lookup = {word: index for index, word in enumerate(words)}
        rows = (
            reference_means(encoder.word_vectors.stored_vectors, lookup, tokens)
            for tokens in batches(corpus)
        )
    return components(gram_of_batches(rows))


def folded_arrays(encoder, basis, temporary: Path):
    source = encoder.word_vectors
    vector_path = temporary / "folded-vocab.npy"
    rows = mmap_array(vector_path, source.stored_vectors.shape)
    fold_table(source.stored_vectors, basis, rows)
    ngram_path, ngrams = None, None
    if source.fasttext is not None:
        ngram_path = temporary / "folded-ngrams.npy"
        ngrams = mmap_array(ngram_path, source.stored_ngrams.shape)
        fold_table(source.stored_ngrams, basis, ngrams)
    rows.flush()
    if ngrams is not None:
        ngrams.flush()
    return rows, ngrams, source.fasttext, [p for p in (vector_path, ngram_path) if p is not None]


def fit_classic(name, folder, cache, workers, record):
    import fse

    kv = source_model(name)
    words = list(kv.index_to_key)
    corpus = fit_corpus(cache)
    counts, _ = frequencies(corpus)
    with tempfile.TemporaryDirectory(prefix="sif-fit-", dir=cache) as temporary:
        directory = Path(temporary)
        encoder, weights = weighted_model(kv, counts, directory)
        basis = fit_basis(encoder, words, corpus, workers)
        rows, ngrams, params, intermediates = folded_arrays(encoder, basis, directory)
        state = folder / f"{stem(name)}-fit.npz"
        np.savez(state, weights=weights, components=basis)
        record["fit"] = {
            "corpus": FIT_REPO,
            "revision": FIT_REVISION,
            "split": "train",
            "paragraphs": FIT_LINES,
            "tokenized_corpus_xxh3": file_digest(corpus),
            "seed": SEED,
            "components": 10,
            "a": 1e-3,
            "solver": "scipy.linalg.eigh(driver=evd), no randomized fit",
            "gram": "float32 block product, float64 accumulation, blocks of 16384",
            "tokenizer": "cached run_mteb.tokenize NLTK rules, external at inference",
            "state_xxh3": file_digest(state),
            "recipe": "recipes/01-sif-usif.ipynb",
            "fasttext": "fse from_gensim vocabulary rows and native OOV ngrams, max_ngrams=40"
            if params
            else None,
        }
        # Free the weighted owned tables before constructing another owned table.
        # FastText has both a vocabulary table and an equally large ngram table.
        del encoder, kv
        gc.collect()
        encoder = fse.Encoder.from_arrays(
            rows,
            words=words,
            fasttext=params,
            ngrams=ngrams,
            tokenizer="external",
            normalize=True,
        )
        if np.shares_memory(rows, encoder.word_vectors.stored_vectors):
            raise RuntimeError("finished table unexpectedly retains its temporary mapping")
        del rows, ngrams
        for path in intermediates:
            path.unlink()
    return encoder


def prepare(name: str, folder: Path, cache: Path, workers: int, dtypes: list[str], proxy=False):
    import fse
    from fse.gensim import from_gensim

    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{stem(name)}-{dtype}.safetensors" for dtype in dtypes]
    if any(path.exists() for path in paths):
        raise FileExistsError("preparation refuses existing output artifacts")
    record = {
        **identity(),
        "source": f"fse/{name}",
        "revision": ALL_SOURCES[name],
        "recipe": "plain mean" if name == "paranmt-300" else "SIF-10",
        "load_path": "KeyedVectors.load(mmap='r') -> fse.gensim.from_gensim -> Encoder",
        "timing_proxy": bool(proxy),
        "fit": None,
    }
    if name == "paranmt-300" or proxy:
        kv = source_model(name)
        encoder = fse.Encoder.from_word_vectors(
            from_gensim(kv),
            tokenizer="external" if proxy and name != "paranmt-300" else "fse_words@1",
        )
        if proxy and name != "paranmt-300":
            record["recipe"] = "same-shape timing proxy for SIF-10, no fit"
    else:
        encoder = fit_classic(name, folder, cache, workers, record)
    if name == "paranmt-300" or proxy:
        del kv
    parent = encoder.fingerprint
    record["fitted_float32_fingerprint"] = parent
    for dtype, path in zip(dtypes, paths, strict=True):
        current = encoder if dtype == "float32" else encoder.quantize(dtype)
        current.save(path)
        register(folder, path, record, parent)
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=sorted(ALL_SOURCES), required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/fse-quality")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float32", "float16", "int8"], default=["float32"]
    )
    parser.add_argument("--coordination", type=Path)
    parser.add_argument("--timing-proxy", action="store_true")
    args = parser.parse_args()
    with task_reservation(args.coordination):
        prepare(args.name, args.folder, args.cache, args.workers, args.dtypes, args.timing_proxy)


if __name__ == "__main__":
    main()
