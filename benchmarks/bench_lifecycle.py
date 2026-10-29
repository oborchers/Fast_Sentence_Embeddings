#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The lifecycle of fse 2.0 against fse 1.1.0 on real data: load, encode, search, infer.

The 2.0 counterpart of the 1.1 lifecycle benchmark (``bench_lifecycle.py`` at 4931f5e,
its numbers in benchmarks/baseline/README.md, "Lifecycle"), on glove-wiki-gigaword-300
from the local Hugging Face cache and the tokenized WikiText paragraphs (one per line).
Both sides do the same work in the same timers (bench_lifecycle_cases.py says how).
Every case runs in a fresh child process, so the peak RSS (``ru_maxrss``) belongs to it,
as --pairs alternating pairs (2.0 then 1.1, then 1.1 then 2.0); the raw values, their
median and the load average before every run are kept. Before every run its inputs
(word vectors, gensim model, corpus, and the stores and models of earlier cases) are read
once, so every number is one with a warm page cache. Cases:

- ``load``, ``load_mmap``: the word vectors; 2.0 reads its safetensors file, converted
  once from the gensim model (untimed), 1.1 ``Vectors.from_pretrained``;
- ``encode``: the whole corpus, from the file (2.0 its native file source, 1.1
  ``CorpusFile``) or from token lists in memory, into a store or into memory. 2.0 times
  building the encoder of its finished table (``Encoder.from_word_vectors``; the table of
  ``SIFWeights`` is made before, untimed, as 2.0 trains nothing), ``encode`` (of the token lists or the
  file; a store: the ``ShardSink`` from its creation to its last flush) and saving the
  encoder; 1.1 the constructor, ``train`` (its scan and components) and ``save``. Only
  the totals compare the same work, so only they get a ratio; the bytes count what
  reopening needs (2.0: the store and the encoder, which holds its table), "vectors MB" only the
  sentence vectors (2.0 the store, 1.1 their arrays, which it pickles into its model);
- ``query``: open the store (1.1 ``load`` of the saved model), the first search, the warm
  search p50 of 1 query and of a batch of 64 (2.0 ``mode="auto"``, also ``"exact"`` and
  ``"blas"``; 1.1 has no batch and answers the 64 one by one), ``score`` of 1 and 1,000
  weight vectors;
- ``infer``: 1, 1,000 and 100,000 paragraphs as token lists to float32, p50 of five calls
  (1.1 ``infer`` runs in one thread whatever its workers);
- ``hub``: 2.0 only, the SIFWeights model as it comes from the hub, saved before
  (untimed): 2.0 loads the encoder of its finished table and encodes the whole corpus
  (from the file, or token lists made before the timer) into memory or into a store up
  to its last flush.

--quick (the default) runs the prioritized subset of ``_quick`` in about ten minutes,
--full every combination of --models, --dtypes, --workers, --inputs and --targets; query
cases follow the models and dtypes with a store, infer cases those of float32.

    uv run python benchmarks/bench_lifecycle.py --python-v1 .venv-baseline/bin/python \\
        --tmp /path/with/space --json lifecycle.json

--python-v1 is a Python with fse 1.1.0 and wordfreq, which also computes the SIF word
frequencies of both sides. Results are only comparable on an idle machine.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from bench_lifecycle_cases import BATCH, INFER_SIZES, K

MODELS = ("SIFWeights", "Average")
# The models both sides run. SIFWeights (the SIF word weights folded into the table, no
# components removed) is no SIF, so it runs on 2.0 alone and is never compared with 1.1.
V1_MODELS = ("Average",)
DTYPES = ("float32", "float16", "int8")
INPUTS = ("file", "list")
TARGETS = ("store", "memory")
FIELDS = ("model", "dtype", "workers", "input", "target")
CASES = Path(__file__).with_name("bench_lifecycle_cases.py")


# --- the grid -------------------------------------------------------------------------


def _quick(args) -> list[tuple]:
    """The prioritized subset: both models at the most workers for float32 and int8 into a
    store, SIFWeights float32 also into memory, at the other worker counts and from a list."""
    top = max(args.workers)
    cases = [(m, d, top, "file", "store") for m in MODELS for d in ("float32", "int8")]
    cases.append(("SIFWeights", "float32", top, "file", "memory"))
    cases += [("SIFWeights", "float32", w, "file", "store") for w in args.workers]
    cases.append(("SIFWeights", "float32", top, "list", "store"))
    return list(dict.fromkeys(cases))


def _wanted(case: tuple, args) -> bool:
    allowed = (args.models, args.dtypes, args.workers, args.inputs, args.targets)
    return all(value in among for value, among in zip(case, allowed, strict=True))


def _hubs(encodes: list[dict], stores: list[dict]) -> dict:
    """The (dtype, workers) of the hub cases: those of the SIFWeights encodes, whose saved
    models (from such an encode into a store of the dtype) the hub cases load."""
    hubs = {(e["dtype"], e["workers"]): None for e in encodes if e["model"] == "SIFWeights"}
    saved = {e["dtype"] for e in stores if e["model"] == "SIFWeights"}
    if missing := sorted({d for d, _ in hubs} - saved):
        raise SystemExit(
            f"the hub cases of {', '.join(missing)} load the models a SIFWeights encode into a "
            "store saves; add --targets store"
        )
    return hubs


def grid(args) -> list[dict]:
    """Every case in order: loads, encodes, then queries and inference of the stores."""
    if args.full:
        combos = itertools.product(
            args.models, args.dtypes, args.workers, args.inputs, args.targets
        )
    else:
        combos = [case for case in _quick(args) if _wanted(case, args)]
    encodes = [{"kind": "encode", **dict(zip(FIELDS, case, strict=True))} for case in combos]
    if not encodes:
        raise SystemExit(
            "no encode case of the quick grid matches --models, --dtypes, --workers, "
            "--inputs and --targets; pass --full for every combination of them"
        )
    stores = [e for e in encodes if e["target"] == "store"]
    queries = {(e["model"], e["dtype"]): None for e in stores}
    infers = {(e["model"], e["workers"]): None for e in stores if e["dtype"] == "float32"}
    hubs = _hubs(encodes, stores)
    return [
        {"kind": "load"},
        {"kind": "load_mmap"},
        *encodes,
        *({"kind": "query", "model": m, "dtype": d} for m, d in queries),
        *({"kind": "infer", "model": m, "dtype": "float32", "workers": w} for m, w in infers),
        *(
            {
                "kind": "hub",
                "model": "SIFWeights",
                "dtype": d,
                "workers": w,
                "input": i,
                "target": t,
            }
            for (d, w) in hubs
            for i in args.inputs
            for t in args.targets
        ),
    ]


# --- running --------------------------------------------------------------------------


def _run(python: str, config: dict) -> dict:
    # Offline: 1.1 resolves the hub cache through snapshot_download, which would otherwise
    # ask the hub inside the measurement.
    done = subprocess.run(
        [python, str(CASES), json.dumps(config)],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        raise SystemExit(f"{config} failed:\n{done.stderr[-3000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def _median(runs: list[dict]) -> dict:
    numbers = (int, float)
    return {
        k: statistics.median(r[k] for r in runs)
        for k, v in runs[0].items()
        if isinstance(v, numbers) and not isinstance(v, bool)
    }


def _pairs(config: dict, pythons: dict[str, str], pairs: int, warm: list[Path]) -> dict:
    """``pairs`` runs of every side, alternating which goes first, each with its inputs in
    the page cache."""
    runs: dict[str, list[dict]] = {side: [] for side in pythons}
    for pair in range(pairs):
        for side in sorted(pythons, reverse=pair % 2 == 0):  # v2 first, then v1 first
            _warm([*warm, *_written(Path(config["tmp"]))])
            load = os.getloadavg()[0]  # before the run
            runs[side].append(
                {**_run(pythons[side], {**config, "side": side}), "load_average": load}
            )
    return {side: {"median": _median(r), "runs": r} for side, r in runs.items()}


def _prepare(args, base: dict) -> None:
    """The word vectors of 2.0 and the wordfreq counts of SIF, once and untimed."""
    words = _run(sys.executable, {**base, "side": "v2", "kind": "prepare"})["words"]
    if args.python_v1:
        v1_words = _run(args.python_v1, {**base, "side": "v1", "kind": "counts"})["words"]
        if v1_words != words:
            raise SystemExit("the vocabularies of 2.0 and 1.1 are not in the same order")
    elif "SIFWeights" in args.models:
        raise SystemExit(
            "SIFWeights needs the wordfreq counts of --python-v1 (fse 1.1.0, wordfreq)"
        )


def _warm(paths: list[Path]) -> None:
    """Read the files once, so the next case starts with them in the page cache."""
    for path in paths:
        with path.open("rb") as f:
            while f.read(1 << 24):
                pass


def _written(tmp: Path) -> list[Path]:
    """The kept stores, models and query vectors of earlier cases, which later cases read."""
    return sorted(p for p in tmp.rglob("*") if p.is_file() and p.name != "corpus.txt")


def _hub_files(vectors: str) -> list[Path]:
    """The files of the gensim model in the hub cache, which 1.1 loads."""
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(f"fse/{vectors}", local_files_only=True))
    return [f for f in folder.iterdir() if f.name.startswith(vectors)]


def _identity(path: Path) -> dict:
    from fse._container._hash import Hasher

    hasher, lines = Hasher(), 0
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            hasher.update(block)
            lines += block.count(b"\n")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "lines": lines,
        "xxh3": hasher.checksum(),
    }


# --- the table ------------------------------------------------------------------------

# Every kind's metrics as (label, field of 2.0, field of 1.1, ratio); None where a side has
# none, and a ratio only where both sides do the same work.
_PEAK = ("peak MiB", "peak", "peak", True)
METRICS = {
    "load": [("seconds", "seconds", "seconds", True), _PEAK],
    "encode": [
        ("build / construct s", "build_s", "construct_s", False),
        ("encode / train s", "encode_s", "train_s", False),
        ("save s", "save_s", "save_s", False),
        ("total s", "total_s", "total_s", True),
        _PEAK,
        ("disk MB", "bytes_on_disk", "bytes_on_disk", True),
        ("vectors MB", "vector_bytes", "vector_bytes", True),
    ],
    "query": [
        ("open s", "open_s", "open_s", True),
        ("first search s", "first_search_s", "first_search_s", True),
        ("search 1 p50 ms", "search_1_ms", "search_1_ms", True),
        (f"search {BATCH} p50 ms", f"search_{BATCH}_ms", f"search_{BATCH}_ms", True),
        (f"  {BATCH} exact ms", f"search_{BATCH}_exact_ms", None, False),
        (f"  {BATCH} blas ms", f"search_{BATCH}_blas_ms", None, False),
        ("score 1 s", "score_1_s", "score_1_s", True),
        ("score 1000 s", "score_1000_s", "score_1000_s", True),
        _PEAK,
    ],
    "infer": [(f"infer {n} p50 ms", f"infer_{n}_ms", f"infer_{n}_ms", True) for n in INFER_SIZES]
    + [_PEAK],
    "hub": [  # 2.0 only
        ("load s", "load_s", None, False),
        ("encode s", "encode_s", None, False),
        ("total s", "total_s", None, False),
        _PEAK,
    ],
}
METRICS["load_mmap"] = METRICS["load"]


def _value(side: dict | None, field: str | None) -> float | None:
    if side is None or field is None or field not in side["median"]:
        return None
    value = side["median"][field]
    if field == "peak":
        return value / 2**20
    return value / 1e6 if field.endswith("bytes") or field == "bytes_on_disk" else value


def _cell(value: float | None) -> str:
    return f"{value:>11,.3f}" if value is not None else f"{'':>11}"


def _print_row(row: dict, cpus: int) -> None:
    case, v2, v1 = row["case"], row.get("v2"), row.get("v1")
    name = " ".join(str(case[k]) for k in FIELDS if k in case)
    load = max(r["load_average"] for side in (v2, v1) if side for r in side["runs"])
    warning = f"  # warning: load above {cpus / 4:g}" if load > cpus / 4 else ""
    print(f"{case['kind']} {name}  (load up to {load:.1f}){warning}", flush=True)
    for label, f2, f1, comparable in METRICS[case["kind"]]:
        a, b = _value(v2, f2), _value(v1, f1)
        ratio = f"{a / b:>9.2f}" if comparable and a is not None and b else ""
        print(f"  {label:<20}{_cell(a)}{_cell(b)}{ratio}", flush=True)


def run(args, base: dict, warm: list[Path]) -> list[dict]:
    from fse._cpus import usable_cpus

    cpus = usable_cpus()
    pythons = {"v2": sys.executable, **({"v1": args.python_v1} if args.python_v1 else {})}
    print(f"  {'':<20}{'2.0':>11}{'1.1':>11}{'2.0/1.1':>9}  (medians of {args.pairs} pairs)")
    rows = []
    for case in grid(args):
        config = {**base, **case}
        shared = case["kind"].startswith("load") or case.get("model") in V1_MODELS
        sides = pythons if shared else {"v2": pythons["v2"]}
        if case["kind"] == "query":  # the query vectors, in a process of their own
            _run(sys.executable, {**config, "side": "v2", "kind": "queries"})
        rows.append({"case": case, **_pairs(config, sides, args.pairs, warm)})
        _print_row(rows[-1], cpus)
    return rows


def _corpus(args, tmp: Path) -> Path:
    """The corpus, or its first --sentences lines as a file of their own."""
    if args.sentences is None:
        return args.corpus
    out = tmp / "corpus.txt"
    with args.corpus.open(encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        dst.writelines(itertools.islice(src, args.sentences))
    return out


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {value}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vectors", default="glove-wiki-gigaword-300")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("~/.cache/fse-quality/wikitext-103-b08601e0.txt"),
        help="one tokenized paragraph per line",
    )
    parser.add_argument("--sentences", type=_positive, help="only the first lines of the corpus")
    parser.add_argument("--python-v1", help="a Python with fse 1.1.0 and wordfreq")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--workers", type=_positive, nargs="+", default=[1, 4, 10])
    parser.add_argument("--inputs", nargs="+", choices=INPUTS, default=list(INPUTS))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--pairs", type=_positive, default=2, help="alternating runs of each side")
    grid_size = parser.add_mutually_exclusive_group()
    grid_size.add_argument("--quick", action="store_true", help="the prioritized subset (default)")
    grid_size.add_argument("--full", action="store_true", help="every combination")
    parser.add_argument("--tmp", type=Path, help="where vectors, stores and models are written")
    parser.add_argument("--json", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.corpus = args.corpus.expanduser()
    grid(args)  # an empty grid fails before the preparation
    import fse

    print(
        f"# {platform.platform()} | {os.cpu_count()} logical CPUs | Python "
        f"{platform.python_version()} | fse {fse.__version__} | load {os.getloadavg()[0]:.1f}"
    )
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp_name:
        tmp = Path(tmp_name)
        used = _corpus(args, tmp)
        base = {"vectors": args.vectors, "tmp": str(tmp), "corpus": str(used)}
        corpus = {"source": _identity(args.corpus), "read": _identity(used)}
        print(f"# corpus {args.corpus}: {corpus['read']['lines']:,} lines read", flush=True)
        if corpus["read"]["lines"] < K and any(c["kind"] == "query" for c in grid(args)):
            raise SystemExit(f"the query cases search for {K} rows: read at least {K} lines")
        _prepare(args, base)
        rows = run(args, base, [*_hub_files(args.vectors), used])
    if args.json:
        result = {"config": {k: str(v) for k, v in vars(args).items()}, "corpus": corpus}
        result.update(machine=platform.platform(), python=platform.python_version(), rows=rows)
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
