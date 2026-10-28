#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The lifecycle of fse 2.0 against fse 1.1.0 on real data: load, fit, encode, search, infer.

The 2.0 counterpart of the 1.1 lifecycle benchmark (``bench_lifecycle.py`` at 4931f5e,
its numbers in benchmarks/baseline/README.md, "Lifecycle"), on glove-wiki-gigaword-300
from the local Hugging Face cache and the WikiText paragraphs (one per line). Every case
runs in a fresh child process, so the peak RSS (``ru_maxrss``) belongs to that case; the
load average is recorded before every case. Cases:

- ``load``, ``load_mmap``: the word vectors; 2.0 reads its safetensors file, which it
  converts once from the gensim model in --tmp (untimed), 1.1 ``Vectors.from_pretrained``;
- ``encode``: the whole corpus, from the file (2.0 its native file source, 1.1
  ``CorpusFile``) or from a list in memory (2.0 str, 1.1 an ``IndexedList`` of token
  lists), into a store (2.0 ``ShardSink`` up to its last flush, 1.1 ``train`` and
  ``save``) or into memory. 2.0 times ``fit`` (SIF: components of the corpus) and
  ``encode``, 1.1 the constructor (quantizing, ``lang_freq="en"``) and ``train`` (its scan
  and components included); ``total`` is their sum;
- ``query``: open the store (1.1 ``load`` of the saved model), the first search, the warm
  search p50 of 1 query and of a batch of 64 (2.0 ``mode="auto"``, also ``"exact"`` and
  ``"blas"``; 1.1 has no batch and answers the 64 one by one), and ``score`` of 1 and
  1,000 weight vectors;
- ``infer``: ``encode`` (1.1 ``infer``) of 1, 1,000 and 100,000 paragraphs from a list in
  memory, p50 of five calls.

--quick (the default) runs the prioritized subset of ``_quick`` in about ten minutes,
--full every combination of --models, --dtypes, --workers, --inputs and --targets; query
and infer cases follow the models, dtypes and worker counts that have a store.

    uv run python benchmarks/bench_lifecycle.py --python-v1 .venv-baseline/bin/python \\
        --tmp /path/with/space --json lifecycle.json
    uv run python benchmarks/bench_lifecycle.py --full --python-v1 .venv-baseline/bin/python \\
        --tmp /path/with/space --json lifecycle-full.json

The children of --python-v1 run this file with fse 1.1.0, so it imports only the standard
library and numpy at the top. Results are only comparable on an idle machine.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

MODELS = ("SIF", "Average")
DTYPES = ("float32", "float16", "int8")
INPUTS = ("file", "list")
TARGETS = ("store", "memory")
COMPONENTS = 10  # SIF-10, as the 1.1 baseline
INFER_SIZES = (1, 1_000, 100_000)
SEARCHES, BATCH = 20, 64
FIELDS = ("model", "dtype", "workers", "input", "target")


def _peak() -> int:
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _p50_ms(fn, rounds: int) -> float:
    return 1e3 * statistics.median(_timed(fn) for _ in range(rounds))


def _lines(path: str, n: int | None = None) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in itertools.islice(f, n)]


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
        "store": tmp / f"v2-{name}.store",
        "encoder": tmp / f"v2-{c.get('model')}.encoder.safetensors",
        "model": tmp / f"v1-{name}",
    }


# --- 2.0 children ---------------------------------------------------------------------


def v2_prepare(c: dict) -> dict:
    """Convert the gensim model of the hub cache once to 2.0 word vectors (untimed)."""
    target = _paths(c)["wv"]
    if not target.exists():
        from gensim.models import KeyedVectors
        from huggingface_hub import snapshot_download

        from fse.gensim import from_gensim

        folder = Path(snapshot_download(f"fse/{c['vectors']}", local_files_only=True))
        kv = KeyedVectors.load(str(folder / f"{c['vectors']}.model"), mmap="r")
        from_gensim(kv).save(target)
    return {"bytes_on_disk": target.stat().st_size}


def v2_load(c: dict) -> dict:
    import fse

    path = _paths(c)["wv"]
    seconds = _timed(lambda: fse.WordVectors.load(path, mmap=c["kind"] == "load_mmap"))
    return {"seconds": seconds, "peak": _peak()}


def _v2_encode_into(encoder, texts, c: dict, run: Path) -> dict:
    import fse

    if c["target"] == "memory":
        seconds = _timed(lambda: encoder.encode(texts, dtype=c["dtype"], workers=c["workers"]))
        return {"encode_s": seconds}
    with fse.ShardSink(run / "store", encoder, dtype=c["dtype"]) as sink:
        seconds = _timed(lambda: encoder.encode(texts, workers=c["workers"], sink=sink))
    out = {"encode_s": seconds, "bytes_on_disk": _bytes(run / "store")}
    _keep(run / "store", _paths(c)["store"])
    return out


def v2_encode(c: dict) -> dict:
    import fse

    wv = fse.WordVectors.load(_paths(c)["wv"], mmap=True)
    texts = Path(c["corpus"]) if c["input"] == "file" else _lines(c["corpus"])
    out: dict = {"peak_before": _peak()}
    start = time.perf_counter()
    if c["model"] == "SIF":
        encoder = fse.SIF(components=COMPONENTS).fit(wv, texts, workers=c["workers"])
    else:
        encoder = fse.Average().fit(wv)
    out["fit_s"] = time.perf_counter() - start
    with tempfile.TemporaryDirectory(dir=c["tmp"]) as run:
        out.update(_v2_encode_into(encoder, texts, c, Path(run)))
        out["peak"] = _peak()
        if not _paths(c)["encoder"].exists():
            encoder.save(_paths(c)["encoder"])
    out["total_s"] = out["fit_s"] + out["encode_s"]
    return out


def v2_query(c: dict) -> dict:
    import fse

    out: dict = {}
    start = time.perf_counter()
    store = fse.VectorStore.open(_paths(c)["store"])
    out["open_s"] = time.perf_counter() - start
    ids = np.random.default_rng(0).integers(0, len(store), 1 + SEARCHES + BATCH)
    queries = np.stack([store.vectors(int(i), int(i) + 1)[0] for i in ids])
    out["first_search_s"] = _timed(lambda: store.search(queries[0], k=10))
    single = iter(queries[1 : 1 + SEARCHES])
    out["search_1_ms"] = _p50_ms(lambda: store.search(next(single), k=10), SEARCHES)
    batch = queries[1 + SEARCHES :]
    for mode in ("auto", "exact", "blas"):
        out[f"search_{BATCH}_{mode}_ms"] = _p50_ms(lambda m=mode: store.search(batch, mode=m), 5)
    out[f"search_{BATCH}_ms"] = out[f"search_{BATCH}_auto_ms"]
    rng = np.random.default_rng(1)
    for m in (1, 1000):
        weights = rng.standard_normal((m, store.dim)).astype(np.float32)
        out[f"score_{m}_s"] = _timed(lambda weights=weights: store.score(weights))
    out["peak"] = _peak()
    return out


def v2_infer(c: dict) -> dict:
    import fse

    # SIF folds its weights and components into the encoder's word vectors; Average does
    # not change them, so its encoder needs them.
    wv = None if c["model"] == "SIF" else fse.WordVectors.load(_paths(c)["wv"], mmap=True)
    encoder = fse.Encoder.load(_paths(c)["encoder"], wv)
    lines = _lines(c["corpus"], max(INFER_SIZES))
    out: dict = {}
    for n in (n for n in INFER_SIZES if n <= len(lines)):  # all but in a short corpus
        batch = lines[:n]
        out[f"infer_{n}_ms"] = _p50_ms(
            lambda batch=batch: encoder.encode(batch, dtype=c["dtype"], workers=c["workers"]), 5
        )
    out["peak"] = _peak()
    return out


# --- 1.1 children -----------------------------------------------------------------------


def v1_load(c: dict) -> dict:
    from fse import Vectors

    mmap = "r" if c["kind"] == "load_mmap" else None
    seconds = _timed(lambda: Vectors.from_pretrained(c["vectors"], mmap=mmap))
    return {"seconds": seconds, "peak": _peak()}


def v1_encode(c: dict) -> dict:
    import fse

    vectors = fse.Vectors.from_pretrained(c["vectors"], mmap="r")
    sentences = fse.CorpusFile(c["corpus"])
    if c["input"] == "list":
        sentences = fse.IndexedList([line.split() for line in _lines(c["corpus"])])
    out: dict = {"peak_before": _peak()}
    kwargs = {"components": COMPONENTS} if c["model"] == "SIF" else {}
    cls = getattr(fse, c["model"])
    start = time.perf_counter()
    model = cls(vectors, workers=c["workers"], sv_dtype=c["dtype"], lang_freq="en", **kwargs)
    out["construct_s"] = time.perf_counter() - start
    out["train_s"] = _timed(lambda: model.train(sentences))
    out["save_s"] = 0.0
    if c["target"] == "store":
        with tempfile.TemporaryDirectory(dir=c["tmp"]) as run:
            (Path(run) / "model").mkdir()
            out["save_s"] = _timed(lambda: model.save(str(Path(run) / "model" / "model")))
            out["bytes_on_disk"] = _bytes(Path(run) / "model")
            _keep(Path(run) / "model", _paths(c)["model"])
    out["peak"] = _peak()
    out["total_s"] = out["construct_s"] + out["train_s"] + out["save_s"]
    return out


def v1_query(c: dict) -> dict:
    import fse

    out: dict = {}
    start = time.perf_counter()
    model = getattr(fse, c["model"]).load(str(_paths(c)["model"] / "model"))
    out["open_s"] = time.perf_counter() - start
    sv = model.sv
    ids = np.random.default_rng(0).integers(0, len(sv), 1 + SEARCHES + BATCH)
    out["first_search_s"] = _timed(lambda: sv.most_similar(int(ids[0]), topn=10))
    single = iter(ids[1 : 1 + SEARCHES])
    out["search_1_ms"] = _p50_ms(lambda: sv.most_similar(int(next(single)), topn=10), SEARCHES)
    batch = ids[1 + SEARCHES :]
    # No batch search in 1.1: the 64 queries one after the other.
    out[f"search_{BATCH}_ms"] = _p50_ms(lambda: [sv.most_similar(int(i)) for i in batch], 3)
    rng = np.random.default_rng(1)
    for m in (1, 1000):
        weights = rng.standard_normal((sv.vector_size, m)).astype(np.float32)
        out[f"score_{m}_s"] = _timed(lambda weights=weights: sv.score(weights))
    out["peak"] = _peak()
    return out


def v1_infer(c: dict) -> dict:
    import fse
    from fse import IndexedList

    model = getattr(fse, c["model"]).load(str(_paths(c)["model"] / "model"))
    lines = _lines(c["corpus"], max(INFER_SIZES))
    out: dict = {}
    for n in (n for n in INFER_SIZES if n <= len(lines)):  # all but in a short corpus
        batch = IndexedList([line.split() for line in lines[:n]])
        out[f"infer_{n}_ms"] = _p50_ms(lambda batch=batch: model.infer(batch), 5)
    out["peak"] = _peak()
    return out


CHILDREN = {
    ("v2", "prepare"): v2_prepare,
    ("v2", "load"): v2_load,
    ("v2", "load_mmap"): v2_load,
    ("v2", "encode"): v2_encode,
    ("v2", "query"): v2_query,
    ("v2", "infer"): v2_infer,
    ("v1", "load"): v1_load,
    ("v1", "load_mmap"): v1_load,
    ("v1", "encode"): v1_encode,
    ("v1", "query"): v1_query,
    ("v1", "infer"): v1_infer,
}


def child(config: dict) -> dict:
    import fse

    result = CHILDREN[config["side"], config["kind"]](config)
    return {**result, "fse": fse.__version__}


# --- the grid -------------------------------------------------------------------------


def _quick(args) -> list[tuple]:
    """The prioritized subset: both models at the most workers for every dtype, SIF
    float32 and int8 into a store at the other worker counts, SIF float32 from a list."""
    top = max(args.workers)
    cases = [(m, d, top, "file", t) for m in MODELS for d in DTYPES for t in TARGETS]
    cases += [("SIF", d, w, "file", "store") for d in ("float32", "int8") for w in args.workers]
    cases += [("SIF", "float32", top, "list", t) for t in TARGETS]
    return list(dict.fromkeys(cases))


def _wanted(case: tuple, args) -> bool:
    allowed = (args.models, args.dtypes, args.workers, args.inputs, args.targets)
    return all(value in among for value, among in zip(case, allowed, strict=True))


def grid(args) -> list[dict]:
    """Every case in order: loads, encodes, then queries and inference of the stores."""
    if args.full:
        combos = list(
            itertools.product(args.models, args.dtypes, args.workers, args.inputs, args.targets)
        )
    else:
        combos = [case for case in _quick(args) if _wanted(case, args)]
    encodes = [{"kind": "encode", **dict(zip(FIELDS, case, strict=True))} for case in combos]
    stores = [e for e in encodes if e["target"] == "store"]
    queries = {(e["model"], e["dtype"]): None for e in stores}
    infers = {(e["model"], e["dtype"], e["workers"]): None for e in stores}
    return [
        {"kind": "load"},
        {"kind": "load_mmap"},
        *encodes,
        *({"kind": "query", "model": m, "dtype": d} for m, d in queries),
        *({"kind": "infer", "model": m, "dtype": d, "workers": w} for m, d, w in infers),
    ]


# --- the parent -----------------------------------------------------------------------


def _run(python: str, config: dict) -> dict:
    # Offline: 1.1 resolves the hub cache through snapshot_download, which would otherwise
    # ask the hub inside the measurement.
    done = subprocess.run(
        [python, __file__, "--child", json.dumps(config)],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        raise SystemExit(f"{config} failed:\n{done.stderr[-3000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def _measured(python: str, config: dict) -> dict:
    load = os.getloadavg()[0]  # before the case
    return {**_run(python, config), "load_average": load}


# The metrics of every kind, as (label, field of 2.0, field of 1.1); None where a side
# has none.
METRICS = {
    "load": [("seconds", "seconds", "seconds"), ("peak MiB", "peak", "peak")],
    "encode": [
        ("fit / construct s", "fit_s", "construct_s"),
        ("encode / train s", "encode_s", "train_s"),
        ("save s", None, "save_s"),
        ("total s", "total_s", "total_s"),
        ("peak MiB", "peak", "peak"),
        ("disk MB", "bytes_on_disk", "bytes_on_disk"),
    ],
    "query": [
        ("open s", "open_s", "open_s"),
        ("first search s", "first_search_s", "first_search_s"),
        ("search 1 p50 ms", "search_1_ms", "search_1_ms"),
        (f"search {BATCH} p50 ms", f"search_{BATCH}_ms", f"search_{BATCH}_ms"),
        (f"  {BATCH} exact ms", f"search_{BATCH}_exact_ms", None),
        (f"  {BATCH} blas ms", f"search_{BATCH}_blas_ms", None),
        ("score 1 s", "score_1_s", "score_1_s"),
        ("score 1000 s", "score_1000_s", "score_1000_s"),
        ("peak MiB", "peak", "peak"),
    ],
    "infer": [(f"infer {n} p50 ms", f"infer_{n}_ms", f"infer_{n}_ms") for n in INFER_SIZES]
    + [("peak MiB", "peak", "peak")],
}
METRICS["load_mmap"] = METRICS["load"]


def _value(result: dict | None, field: str | None) -> float | None:
    if result is None or field is None or field not in result:
        return None
    value = result[field]
    if field == "peak":
        return value / 2**20
    return value / 1e6 if field == "bytes_on_disk" else value


def _cell(value: float | None) -> str:
    return f"{value:>11,.3f}" if value is not None else f"{'':>11}"


def _print_row(row: dict, cpus: int) -> None:
    case = row["case"]
    name = " ".join(str(case[k]) for k in FIELDS if k in case)
    load = max(side["load_average"] for side in (row["v2"], row["v1"]) if side)
    warning = f"  # warning: load above {cpus / 4:g}" if load > cpus / 4 else ""
    print(f"{case['kind']} {name}  (load {load:.1f}){warning}", flush=True)
    for label, f2, f1 in METRICS[case["kind"]]:
        a, b = _value(row["v2"], f2), _value(row["v1"], f1)
        ratio = f"{a / b:>9.2f}" if a is not None and b else ""
        print(f"  {label:<20}{_cell(a)}{_cell(b)}{ratio}", flush=True)


def run(args) -> list[dict]:
    from fse._cpus import usable_cpus

    cpus = usable_cpus()
    base = {"vectors": args.vectors, "tmp": str(args.tmp), "corpus": str(args.corpus)}
    _run(sys.executable, {**base, "side": "v2", "kind": "prepare"})
    print(f"  {'':<20}{'2.0':>11}{'1.1':>11}{'2.0/1.1':>9}")
    rows = []
    for case in grid(args):
        config = {**base, **case}
        v2 = _measured(sys.executable, {**config, "side": "v2"})
        v1 = _measured(args.python_v1, {**config, "side": "v1"}) if args.python_v1 else None
        rows.append({"case": case, "v2": v2, "v1": v1})
        _print_row(rows[-1], cpus)
    return rows


def _corpus(args, tmp: Path) -> Path:
    """The corpus, or its first --sentences lines as a file of their own."""
    if args.sentences is None:
        return args.corpus.expanduser()
    out = tmp / "corpus.txt"
    with (
        args.corpus.expanduser().open(encoding="utf-8") as src,
        out.open("w", encoding="utf-8") as dst,
    ):
        dst.writelines(itertools.islice(src, args.sentences))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vectors", default="glove-wiki-gigaword-300")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("~/.cache/fse-quality/wikitext-103-b08601e0.txt"),
        help="one paragraph per line",
    )
    parser.add_argument("--sentences", type=int, help="only the first lines of the corpus")
    parser.add_argument("--python-v1", help="a Python with fse 1.1.0, for the 1.1 side")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 10])
    parser.add_argument("--inputs", nargs="+", choices=INPUTS, default=list(INPUTS))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    grid_size = parser.add_mutually_exclusive_group()
    grid_size.add_argument("--quick", action="store_true", help="the prioritized subset (default)")
    grid_size.add_argument("--full", action="store_true", help="every combination")
    parser.add_argument("--tmp", type=Path, help="where vectors, stores and models are written")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child(json.loads(args.child))))
        return

    import fse

    print(
        f"# {platform.platform()} | {os.cpu_count()} logical CPUs | Python "
        f"{platform.python_version()} | fse {fse.__version__} | load {os.getloadavg()[0]:.1f}"
    )
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp_name:
        tmp = Path(tmp_name)
        args.corpus, args.tmp = _corpus(args, tmp), tmp
        rows = run(args)
    if args.json:
        result = {"config": {k: str(v) for k, v in vars(args).items()}, "rows": rows}
        result["machine"], result["python"] = platform.platform(), platform.python_version()
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
