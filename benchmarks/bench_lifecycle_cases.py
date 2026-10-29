#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The cases of bench_lifecycle.py, each run in a child process of its own.

    python benchmarks/bench_lifecycle_cases.py '{"side": "v2", "kind": "load", ...}'

prints the measurements of one case as one line of JSON. bench_lifecycle.py starts it
with the Python of fse 2.0 (``side`` v2) or of fse 1.1.0 (v1), so this file imports
only the standard library and numpy at the top. Both sides do the same work inside
the same timers:

- the word frequencies of ``SIFWeights``: the wordfreq "en" counts that 1.1's
  ``lang_freq="en"`` induces, computed once by ``v1_counts`` (untimed), so 2.0 pays for
  them (or for the copy of the word vectors ``lang_freq`` makes) in no timer;
- the same tokens: the corpus is tokenized, so 2.0 uses the external tokenizer, which
  splits at whitespace like 1.1; list inputs are the same token lists on both sides,
  made before the timer;
- the same output: 2.0 with ``normalize=False``, and infer to float32 as 1.1 does;
- the same model: Average on both sides. 2.0 trains nothing, and its ``SIFWeights`` (the
  SIF word weights of the same counts folded into the table, ``v2_finished``, untimed)
  removes no components, so it is no SIF: it runs on 2.0 alone and is never compared
  with 1.1's SIF;
- the same queries: vectors of stored rows, read before (``v2_queries``) by a process
  of their own, searched with their own row among the hits on both sides.

Every returned object is kept until after its timestamp and the peak RSS.
"""

from __future__ import annotations

import itertools
import json
import os
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SIF_ALPHA = 1e-3  # the alpha of SIF in 1.1
INFER_SIZES = (1, 1_000, 100_000)
SEARCHES, BATCH = 20, 64
K = 10  # neighbours of every search on both sides
DOMAIN = 2**31 - 1  # the cumulative count 1.1's lang_freq scales frequencies to


def _peak() -> int:
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def _timed(fn):
    """The seconds of ``fn()`` and its result, which is released only by the caller."""
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def _p50_ms(fn, rounds: int) -> float:
    times = []
    for _ in range(rounds):
        seconds, result = _timed(fn)
        times.append(seconds)
        del result
    return 1e3 * statistics.median(times)


def _tokens(path: str, n: int | None = None) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return [line.split() for line in itertools.islice(f, n)]


def _keep(made: Path, kept: Path) -> None:
    """Keep the first store or model of a model and dtype for the query and infer cases."""
    if not kept.exists():
        os.replace(made, kept)


def _bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _paths(c: dict) -> dict[str, Path]:
    tmp, name = Path(c["tmp"]), f"{c.get('model')}-{c.get('dtype')}"
    return {
        "wv": tmp / f"{c['vectors']}.safetensors",
        "counts": tmp / "counts-en.npy",
        "store": tmp / f"v2-{name}.store",
        "encoder": tmp / f"v2-{c.get('model')}.encoder.safetensors",
        "model": tmp / f"v1-{name}",
        "queries": tmp / f"queries-{name}.npy",
    }


# --- 2.0 ------------------------------------------------------------------------------


def v2_prepare(c: dict) -> dict:
    """Convert the gensim model of the hub cache once to 2.0 word vectors (untimed)."""
    from gensim.models import KeyedVectors
    from huggingface_hub import snapshot_download

    from fse.gensim import from_gensim

    folder = Path(snapshot_download(f"fse/{c['vectors']}", local_files_only=True))
    kv = KeyedVectors.load(str(folder / f"{c['vectors']}.model"), mmap="r")
    if not _paths(c)["wv"].exists():
        from_gensim(kv).save(_paths(c)["wv"])
    return {"words": kv.index_to_key[:: len(kv) // 7]}


def v2_finished(wv, model: str, counts_path, *, tokenizer: str, normalize: bool = False):
    """The finished encoder of ``model`` on the word vectors ``wv``: Average the table as it
    is, SIFWeights the table with the SIF weights alpha / (alpha + p(w)) of the counts
    folded into its rows (float32, then quantized to the dtype of ``wv``), without the
    removal of components, so it is no SIF. Untimed: 2.0 encodes finished models."""
    import fse

    if model == "SIFWeights":
        counts = np.load(counts_path).astype(np.float64)
        weights = SIF_ALPHA / (SIF_ALPHA + counts / counts.sum())
        rows = wv.get_vectors(np.arange(len(wv))) * weights.astype(np.float32)[:, None]
        table = fse.WordVectors.from_arrays(
            rows, vocab_bytes=np.asarray(wv.vocab_bytes), vocab_offsets=np.asarray(wv.vocab_offsets)
        )
        wv = table if wv.dtype == "float32" else table.quantize(wv.dtype)
    return fse.Encoder.from_word_vectors(wv, tokenizer=tokenizer, normalize=normalize)


def v2_load(c: dict) -> dict:
    import fse

    path = _paths(c)["wv"]
    seconds, wv = _timed(lambda: fse.WordVectors.load(path, mmap=c["kind"] == "load_mmap"))
    return {"seconds": seconds, "peak": _peak(), "words": len(wv)}


def _v2_store(encoder, texts, c: dict, run: Path) -> dict:
    """Into a ShardSink up to its last flush (in encode), then the encoder saved."""
    from fse import ShardSink  # imported outside the timer

    def write():
        with ShardSink(run / "store", encoder, dtype=c["dtype"]) as sink:
            encoder.encode(texts, workers=c["workers"], sink=sink)

    encode_s, _ = _timed(write)
    save_s, _ = _timed(lambda: encoder.save(run / "encoder.safetensors"))
    # What reopening needs: the store and the encoder, which holds its table.
    out = {
        "encode_s": encode_s,
        "save_s": save_s,
        "bytes_on_disk": _bytes(run),
        "vector_bytes": _bytes(run / "store"),
    }
    _keep(run / "store", _paths(c)["store"])
    _keep(run / "encoder.safetensors", _paths(c)["encoder"])
    return out


def v2_encode(c: dict) -> dict:
    import fse

    wv = fse.WordVectors.load(_paths(c)["wv"], mmap=True)
    finished = v2_finished(wv, c["model"], _paths(c)["counts"], tokenizer="external")
    out: dict = {"peak_before": _peak()}
    # The token lists, made before the timer, as 1.1 gets them; encode reads them.
    texts = _tokens(c["corpus"]) if c["input"] == "list" else Path(c["corpus"])
    out["build_s"], encoder = _timed(
        lambda: fse.Encoder.from_word_vectors(
            finished.word_vectors, tokenizer="external", normalize=False
        )
    )
    with tempfile.TemporaryDirectory(dir=c["tmp"]) as run:
        if c["target"] == "memory":
            rows_s, rows = _timed(
                lambda: encoder.encode(texts, dtype=c["dtype"], workers=c["workers"])
            )
            out.update(encode_s=rows_s, save_s=0.0, peak=_peak())
            del rows
        else:
            out.update(_v2_store(encoder, texts, c, Path(run)), peak=_peak())
    out["total_s"] = out["build_s"] + out["encode_s"] + out["save_s"]
    return out


def v2_queries(c: dict) -> dict:
    """The query vectors, rows of the store read once (untimed, before the query case)."""
    import fse

    with fse.VectorStore.open(_paths(c)["store"]) as store:
        ids = np.random.default_rng(0).integers(0, len(store), 1 + SEARCHES + BATCH)
        np.save(
            _paths(c)["queries"],
            np.stack([store.get_vectors(start=int(i), stop=int(i) + 1)[0] for i in ids]),
        )
    return {"queries": len(ids)}


def v2_query(c: dict) -> dict:
    from fse import VectorStore  # imported outside the timer

    queries = np.load(_paths(c)["queries"])
    out: dict = {}
    out["open_s"], store = _timed(lambda: VectorStore.open(_paths(c)["store"]))
    out["first_search_s"], _ = _timed(lambda: store.search(queries[0], k=K))
    single = iter(queries[1 : 1 + SEARCHES])
    out["search_1_ms"] = _p50_ms(lambda: store.search(next(single), k=K), SEARCHES)
    batch = queries[1 + SEARCHES :]
    for mode in ("auto", "exact", "blas"):
        out[f"search_{BATCH}_{mode}_ms"] = _p50_ms(
            lambda m=mode: store.search(batch, k=K, mode=m), 3
        )
    out[f"search_{BATCH}_ms"] = out[f"search_{BATCH}_auto_ms"]
    rng = np.random.default_rng(1)
    for m in (1, 1000):
        weights = rng.standard_normal((m, store.dim)).astype(np.float32)
        out[f"score_{m}_s"], _ = _timed(lambda weights=weights: store.score(weights))
    out["peak"] = _peak()
    return out


def v2_infer(c: dict) -> dict:
    import fse

    encoder = fse.Encoder.load(_paths(c)["encoder"])  # it holds its table
    tokens = _tokens(c["corpus"], max(INFER_SIZES))
    out: dict = {}
    for n in (n for n in INFER_SIZES if n <= len(tokens)):  # all but in a short corpus
        batch = tokens[:n]
        out[f"infer_{n}_ms"] = _p50_ms(
            lambda b=batch: encoder.encode(b, dtype="float32", workers=c["workers"]), 5
        )
    out["peak"] = _peak()
    return out


# --- 1.1 ------------------------------------------------------------------------------


def v1_counts(c: dict) -> dict:
    """The counts 1.1's lang_freq="en" induces (``_induce_frequencies``), once, untimed."""
    from wordfreq import get_frequency_dict

    from fse import Vectors

    vectors = Vectors.from_pretrained(c["vectors"], mmap="r")
    freq = get_frequency_dict("en", wordlist="best")
    default = int(1e-8 * DOMAIN)
    counts = [int(freq[w] * DOMAIN) if w in freq else default for w in vectors.index_to_key]
    np.save(_paths(c)["counts"], np.array(counts, dtype=np.int64))
    return {"words": vectors.index_to_key[:: len(vectors) // 7]}


def v1_load(c: dict) -> dict:
    from fse import Vectors

    mmap = "r" if c["kind"] == "load_mmap" else None
    seconds, vectors = _timed(lambda: Vectors.from_pretrained(c["vectors"], mmap=mmap))
    return {"seconds": seconds, "peak": _peak(), "words": len(vectors)}


def _v1_save(model, c: dict) -> dict:
    with tempfile.TemporaryDirectory(dir=c["tmp"]) as run:
        folder = Path(run) / "model"
        folder.mkdir()
        save_s, _ = _timed(lambda: model.save(str(folder / "model")))
        out = {"save_s": save_s, "bytes_on_disk": _bytes(folder)}
        # The sentence vectors and their scales, which 1.1 saves inside its model file.
        scales = model.sv.scales
        out["vector_bytes"] = model.sv.vectors.nbytes + (0 if scales is None else scales.nbytes)
        _keep(folder, _paths(c)["model"])
    return out


def v1_encode(c: dict) -> dict:
    import fse

    vectors = fse.Vectors.from_pretrained(c["vectors"], mmap="r")
    sentences = fse.CorpusFile(c["corpus"])
    if c["input"] == "list":
        sentences = fse.IndexedList(_tokens(c["corpus"]))
    cls = getattr(fse, c["model"])
    out: dict = {"peak_before": _peak()}
    out["construct_s"], model = _timed(
        lambda: cls(vectors, workers=c["workers"], sv_dtype=c["dtype"])
    )
    out["train_s"], _ = _timed(lambda: model.train(sentences))
    out["save_s"] = 0.0
    if c["target"] == "store":
        out.update(_v1_save(model, c))
    out["peak"] = _peak()
    out["total_s"] = out["construct_s"] + out["train_s"] + out["save_s"]
    return out


def v1_query(c: dict) -> dict:
    import fse

    queries = np.load(_paths(c)["queries"])
    out: dict = {}
    path = str(_paths(c)["model"] / "model")
    out["open_s"], model = _timed(lambda: getattr(fse, c["model"]).load(path))
    sv = model.sv
    # A vector as query: 1.1 excludes no row, as 2.0 (an id would exclude itself).
    out["first_search_s"], _ = _timed(lambda: sv.most_similar(queries[0], topn=K))
    single = iter(queries[1 : 1 + SEARCHES])
    out["search_1_ms"] = _p50_ms(lambda: sv.most_similar(next(single), topn=K), SEARCHES)
    # No batch search in 1.1: the 64 queries one after the other.
    batch = queries[1 + SEARCHES :]
    out[f"search_{BATCH}_ms"] = _p50_ms(lambda: [sv.most_similar(q, topn=K) for q in batch], 3)
    rng = np.random.default_rng(1)
    for m in (1, 1000):
        weights = rng.standard_normal((m, sv.vector_size)).astype(np.float32).T
        out[f"score_{m}_s"], _ = _timed(lambda weights=weights: sv.score(weights))
    out["peak"] = _peak()
    return out


def v1_infer(c: dict) -> dict:
    import fse

    model = getattr(fse, c["model"]).load(str(_paths(c)["model"] / "model"))
    # The requested workers (1.1 limits BLAS threads with them); infer itself runs in
    # one thread in 1.1.
    model.workers = c["workers"]
    tokens = _tokens(c["corpus"], max(INFER_SIZES))
    out: dict = {"workers": model.workers}
    for n in (n for n in INFER_SIZES if n <= len(tokens)):
        batch = fse.IndexedList(tokens[:n])
        out[f"infer_{n}_ms"] = _p50_ms(lambda b=batch: model.infer(b), 5)
    out["peak"] = _peak()
    return out


def _v2_hub_into(encoder, texts, c: dict) -> float:
    """Encode into a new store up to its last flush; the seconds."""
    from fse import ShardSink  # imported outside the timer

    with tempfile.TemporaryDirectory(dir=c["tmp"]) as run:

        def write():
            with ShardSink(Path(run) / "store", encoder, dtype=c["dtype"]) as sink:
                encoder.encode(texts, workers=c["workers"], sink=sink)

        return _timed(write)[0]


def v2_hub(c: dict) -> dict:
    """A saved encoder (a finished table, as on the hub) loaded and encoding the corpus,
    into memory or into a store."""
    import fse

    out: dict = {"peak_before": _peak()}
    texts = _tokens(c["corpus"]) if c["input"] == "list" else Path(c["corpus"])
    out["load_s"], encoder = _timed(lambda: fse.Encoder.load(_paths(c)["encoder"]))
    if c["target"] == "store":
        out["encode_s"] = _v2_hub_into(encoder, texts, c)
    else:
        out["encode_s"], rows = _timed(
            lambda: encoder.encode(texts, dtype=c["dtype"], workers=c["workers"])
        )
        out["peak"] = _peak()  # with the rows held, as 1.1 holds them in its model
        del rows
    out.setdefault("peak", _peak())
    out["total_s"] = out["load_s"] + out["encode_s"]
    return out


CHILDREN = {
    ("v2", "prepare"): v2_prepare,
    ("v2", "load"): v2_load,
    ("v2", "load_mmap"): v2_load,
    ("v2", "encode"): v2_encode,
    ("v2", "queries"): v2_queries,
    ("v2", "query"): v2_query,
    ("v2", "infer"): v2_infer,
    ("v2", "hub"): v2_hub,
    ("v1", "counts"): v1_counts,
    ("v1", "load"): v1_load,
    ("v1", "load_mmap"): v1_load,
    ("v1", "encode"): v1_encode,
    ("v1", "query"): v1_query,
    ("v1", "infer"): v1_infer,
}


def main() -> None:
    import fse

    config = json.loads(sys.argv[1])
    result = CHILDREN[config["side"], config["kind"]](config)
    print(json.dumps({**result, "fse": fse.__version__}))


if __name__ == "__main__":
    main()
