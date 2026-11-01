"""Build classic fse 2.0 folders locally. This tool never writes to the Hub.

Run one source per process, under the coordinator's heavy-work lock. Install
benchmarks/hub-classics-requirements.txt alongside the project's environment.
Only the explicitly pinned, maintainer-owned 1.x source copies are unpickled.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
from hub_classic_corpus import batches
from hub_classic_evidence import digest
from hub_classic_math import components, fold_table, gram_of_batches, wordfreq_weights
from hub_classic_sources import DTYPES, FIT_LINES, SOURCES

import fse
from fse._encoder._format import Source
from fse.gensim import from_gensim

LICENSES = {
    "glove-wiki-gigaword-300": "pddl",
    "word2vec-google-news-300": "other",
    "paranmt-300": "unknown",
    "paragram-300-sl999": "unknown",
    "fasttext-crawl-subwords-300": "cc-by-sa-3.0",
}


def source_model(name: str):
    from gensim.models import KeyedVectors
    from huggingface_hub import snapshot_download

    folder = Path(
        snapshot_download(
            f"fse/{name}",
            revision=SOURCES[name],
            token=False,
            allow_patterns=[f"{name}.model", f"{name}.model.*.npy", "README.md"],
            max_workers=2,
        )
    )
    return KeyedVectors.load(str(folder / f"{name}.model"), mmap="r")


def weighted_encoder(kv, weights, temporary: Path):
    source = from_gensim(kv)
    path = temporary / "weighted.npy"
    table = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=source.stored_vectors.shape
    )
    for start in range(0, len(table), 16384):
        table[start : start + 16384] = (
            source.stored_vectors[start : start + 16384] * weights[start : start + 16384, None]
        )
    params = source.fasttext
    ngrams = getattr(kv, "vectors_ngrams", None) if params is not None else None
    del source
    gc.collect()
    encoder = fse.Encoder.from_arrays(
        table,
        words=kv.index_to_key,
        fasttext=params,
        ngrams=ngrams,
        tokenizer="external",
        normalize=False,
        oov_weight=float(weights.max()) if params is not None else None,
    )
    del table
    path.unlink()
    return encoder


def fit_state(name, kv, selection, output, temporary):
    """Use the measured corpus winner and wordfreq weights for every vector source."""
    chosen = json.loads(selection.read_text())
    corpus = Path(chosen["corpus_path"])
    receipt = dict(chosen["fit"])
    receipt["selection_candidate_fingerprint"] = receipt.pop("candidate_fingerprint")
    receipt["source_repo"] = f"fse/{name}"
    receipt["selection_xxh3"] = digest(selection)
    if version("wordfreq") != receipt["wordfreq_version"]:
        raise ValueError("wordfreq version differs from the selected preparation recipe")
    if digest(corpus) != receipt["corpus_xxh3"]:
        raise ValueError("selected component-fit corpus hash mismatch")
    weights = wordfreq_weights(kv.index_to_key)
    encoder = weighted_encoder(kv, weights, temporary)
    row_count = 0

    def encoded_batches():
        nonlocal row_count
        for tokens in batches(corpus):
            row_count += len(tokens)
            yield encoder.encode(tokens, workers=2)

    basis = components(gram_of_batches(encoded_batches()))
    if row_count != FIT_LINES:
        raise ValueError(f"unexpected component-fit rows: {row_count}")
    state = output / f"{name}-fit.npz"
    np.savez(state, weights=weights, components=basis)
    receipt.update({"state_xxh3": digest(state), "source_revision": SOURCES[name]})
    return state, receipt


def fold_encoder(encoder, basis, temporary):
    source = encoder.word_vectors
    paths, arrays = [], []
    tables = [source.stored_vectors]
    if source.fasttext is not None:
        tables.append(source.stored_ngrams)
    for index, table in enumerate(tables):
        path = temporary / f"folded-{index}.npy"
        target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=table.shape)
        fold_table(table, basis, target)
        target.flush()
        paths.append(path)
        arrays.append(target)
    params = source.fasttext
    del source, encoder
    gc.collect()
    return paths, arrays, params


def provenance(encoder, name):
    content = dataclasses.replace(
        encoder._content,
        source=Source("gensim", f"fse/{name}", SOURCES[name]),
        license=LICENSES[name],
    )
    return type(encoder)._new(content, encoder.word_vectors)


def write_variants(encoder, name, method, output):
    encoder = provenance(encoder, name)
    records = []
    for dtype in DTYPES:
        suffix = "" if dtype == "float32" else f"-{dtype}"
        repo = f"{name}-{method}{suffix}"
        folder = output / repo
        current = encoder if dtype == "float32" else encoder.quantize(dtype)
        current.save_pretrained(folder, language=["en"], model_name=f"fse/{repo}")
        # A full reload checks the folder's config, fingerprints and tensor checksums.
        loaded = fse.Encoder.from_pretrained(folder, mmap=True)
        if loaded.fingerprint != current.fingerprint:
            raise ValueError(f"round-trip fingerprint mismatch: {folder}")
        record = {
            "repo": f"fse/{repo}",
            "method": method,
            "dtype": dtype,
            "source": f"fse/{name}",
            "revision": SOURCES[name],
            "fingerprint": loaded.fingerprint,
            "float32_fingerprint": encoder.fingerprint,
            "files": {
                p.name: {"bytes": p.stat().st_size, "xxh3": digest(p)}
                for p in sorted(folder.iterdir())
            },
        }
        records.append(record)
        (output / f"{repo}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            json.dumps({"built": record["repo"], "fingerprint": record["fingerprint"]}), flush=True
        )
        del loaded, current
        gc.collect()
    return records


def prepare(name, output, selection):
    if shutil.disk_usage(output).free < 25 * 2**30:
        raise RuntimeError("less than 25 GiB free; build stopped")
    if any(
        (output / f"{name}-{method}{suffix}").exists()
        for method in ("raw", "sif10")
        for suffix in ("", "-float16", "-int8")
    ):
        raise FileExistsError("source outputs already exist; never overwrite finished models")
    kv = source_model(name)
    encoder = fse.Encoder.from_word_vectors(from_gensim(kv), tokenizer="fse_words@1")
    write_variants(encoder, name, "raw", output)
    del encoder
    gc.collect()
    with tempfile.TemporaryDirectory(prefix=f"{name}-fit-", dir=output) as temporary:
        directory = Path(temporary)
        state, receipt = fit_state(name, kv, selection, output, directory)
        with np.load(state) as fit:
            weights, basis = fit["weights"], fit["components"]
        encoder = weighted_encoder(kv, weights, directory)
        words = list(kv.index_to_key)
        paths, arrays, params = fold_encoder(encoder, basis, directory)
        del encoder, kv
        gc.collect()
        encoder = fse.Encoder.from_arrays(
            arrays[0],
            words=words,
            fasttext=params,
            ngrams=arrays[1] if params is not None else None,
            tokenizer="fse_words@1",
            normalize=True,
            oov_weight=float(weights.max()) if params is not None else None,
        )
        del arrays
        for path in paths:
            path.unlink()
        write_variants(encoder, name, "sif10", output)
    (output / f"{name}-fit.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=sorted(SOURCES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args.name, args.output, args.selection)


if __name__ == "__main__":
    main()
