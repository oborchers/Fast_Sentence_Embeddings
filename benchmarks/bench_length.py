#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Encoding throughput of fse 2.0 by text length: sentences, paragraphs, documents.

The same WikiText tokens cut into texts of different lengths (bench_length_corpus.py),
encoded with one frozen model per configuration (2.0: finished tables, ``SIFWeights``
with the SIF word weights of the wordfreq counts folded in but no components removed, so
it is no SIF and is never compared with 1.1, Average as is, and Average with normalize as
the control with a step; 1.1: Average, the only model both sides share), on
glove-wiki-gigaword-300. Every case runs in a fresh child (bench_length_cases.py), so its
peak RSS belongs to it; a child does one warm-up and --repeats timed encodes and reports
their median, minimum and maximum. Before every run its inputs are read once (a warm page
cache) and the load average is recorded. Tables: texts/s, tokens/s and known tokens/s,
the time ratio 2.0/1.1 (below 1 is faster), the peak RSS and the chunks the pipeline cuts;
then the worker scaling efficiency (tokens/s at the most workers over that many times
tokens/s at 1) and the least-squares fit ``t = a * texts + b * tokens`` over the synthetic
lengths (descriptive at one budget), and at 1 worker over three budgets
``t = c + a * texts + b * tokens`` with its residuals.

Modes:

- ``prepare``: the word vectors, the wordfreq counts (of --python-v1, fse 1.1.0 and
  wordfreq), the corpora with their manifests and the default models, once and untimed;
- ``compare``: 2.0 against 1.1, alternating --pairs, on sentences, paragraphs,
  documents and synthetic 20, 500 and 10,000 with Average float32 into memory at 1 and
  10 workers and int8 into a store at 10 (--workers and --models change it);
- ``matrix``: 2.0 only, the quick grid (see ``matrix_cases``) or with --full every
  combination; --wv-dtype, --chunk-bytes and --tokenizer are the ablations (the baseline
  of --chunk-bytes is ``--chunk-bytes auto``, through the same entry of the pipeline);
- ``verify``: the full check (xxh3) of what one profile case reads, and its model made if
  missing, before the profiled process starts;
- ``profile``: one 2.0 case in this process, encoding in a loop for --seconds, for
  ``make profile-cpu``, ``profile-mem`` and ``profile-rss``; it checks its files only
  against the stamps of ``verify`` (their size and time), so no hashing is profiled.

Throughputs count the tokens of the tokenizer that ran (--tokenizer, ``fse_words@1`` for
the raw corpora), from the manifest; compare refuses the raw corpora, which 1.1 would
split at whitespace.

    uv run --with pyarrow python benchmarks/bench_length.py prepare \\
        --python-v1 .venv-baseline/bin/python --tokens 20000000
    uv run python benchmarks/bench_length.py compare --python-v1 .venv-baseline/bin/python \\
        --tmp /path/with/space --json length-compare.json
    uv run python benchmarks/bench_length.py matrix --tmp /path/with/space \\
        --json length-matrix.json
    uv run python benchmarks/bench_length.py matrix --wv-dtype int8 \\
        --lengths sentences documents
    uv run python benchmarks/bench_length.py verify --length documents --model SIFWeights \\
        --dtype int8 --target store
    make profile-cpu PROFILE_DIR=profiles/documents-int8 RUN="benchmarks/bench_length.py \\
        profile --length documents --model SIFWeights --dtype int8 --target store --seconds 20"

``make profile-mem`` and ``profile-rss`` take the same RUN, with one PROFILE_DIR per
configuration and length; ``verify`` takes the same flags as ``profile``. --tokens and
--seed select the prepared corpora (the default 20,000,000 tokens of the first articles;
``all`` for every article). Results are only comparable on an idle machine.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import bench_length_corpus as corpora
from bench_length_cases import BATCH_DOCUMENTS
from bench_length_files import (
    Prepared,
    ensure_model,
    ensure_vectors,
    model_files,
    run_child,
    vocabulary_words,
)
from bench_length_report import print_analysis, print_header, print_row
from bench_lifecycle import _hub_files, _identity, _median, _warm

MODES = ("prepare", "compare", "matrix", "verify", "profile")
MODELS = ("SIFWeights", "Average", "AverageNorm")
V1_MODELS = ("Average",)  # SIFWeights is no SIF: it removes no components
DTYPES = ("float32", "float16", "int8")
TARGETS = ("memory", "store")
COMPARE = (
    "sentences",
    "paragraphs",
    "documents",
    *(corpora.synthetic(n) for n in (20, 500, 10_000)),
)
CORE = (*corpora.REAL, *(corpora.synthetic(n) for n in corpora.SYNTHETIC), *corpora.SKEW)
FIT_CONFIG = ("SIFWeights", "float32", "store", 1)


# --- the grids ------------------------------------------------------------------------


def _case(corpus: str, model: str, dtype: str, target: str, workers: int) -> dict:
    return {
        "kind": "encode",
        "corpus": corpus,
        "model": model,
        "dtype": dtype,
        "target": target,
        "workers": workers,
    }


def compare_cases(args) -> list[dict]:
    top = max(args.workers)
    configs = [(m, "float32", "memory", w) for m in args.models for w in args.workers]
    configs += [(m, "int8", "store", top) for m in args.models]
    return [_case(c, *config) for c in args.lengths for config in configs]


def _quick(top: int, workers: list[int]) -> list[tuple]:
    configs = [("Average", "float32", "memory", top), ("SIFWeights", "float32", "memory", top)]
    configs += [("SIFWeights", "float32", "store", top), ("SIFWeights", "int8", "store", top)]
    return configs + [("SIFWeights", "float32", "store", w) for w in workers if w != top]


def _full(args) -> list[dict]:
    """Every combination of --models, --dtypes, --targets and --workers on the core."""
    configs = itertools.product(args.models, args.dtypes, args.targets, args.workers)
    return [_case(c, *config) for config in configs for c in CORE]


def _extra(top: int) -> list[dict]:
    """The permuted real corpora (locality), Average with normalize on the real ones (the
    cost of a step), the smaller synthetic budgets at 1 worker (the fit), the raw text
    with fse_words@1 and the strings case."""
    cases = [_case(c, "SIFWeights", "float32", "memory", top) for c in corpora.PERMUTED]
    cases += [_case(c, "AverageNorm", "float32", "memory", top) for c in corpora.REAL]
    cases += [_case(c, *FIT_CONFIG) for c in corpora.FIT if c not in CORE]
    cases += [_case(c, "SIFWeights", "float32", "memory", top) for c in corpora.RAW]
    strings = _case("documents", "SIFWeights", "float32", "memory", top)
    return [*cases, *({**strings, "kind": "strings", "batch": b} for b in ("long", "batch32"))]


def matrix_cases(args) -> list[dict]:
    """The quick grid: the core corpora (real, synthetic, skew) in the six configurations
    of ``_quick``, then ``_extra``; --full: every combination on the core."""
    top = max(args.workers)
    if args.full:
        cases = _full(args)
    else:
        cases = [_case(c, *config) for c in CORE for config in _quick(top, args.workers)]
        cases += _extra(top)
    return [c for c in cases if c["corpus"] in args.lengths]


# --- the paths ------------------------------------------------------------------------


def _tokenizer(args, corpus: str) -> str:
    return "fse_words@1" if corpus in corpora.RAW else args.tokenizer


def _v2_key(args, case: dict) -> tuple[str, str, str]:
    return case["model"], _tokenizer(args, case["corpus"]), args.wv_dtype


def _v1_key(case: dict) -> tuple[str, str]:
    return case["model"], case["dtype"]


# --- running --------------------------------------------------------------------------


def _strings_input(ctx: Prepared, case: dict, tmp: Path) -> Path:
    """The texts of a strings case, selected in this process: the longest document or the
    first BATCH_DOCUMENTS documents, one per line."""
    path = tmp / f"strings-{case['batch']}.txt"
    if not path.exists():
        lines = ctx.corpus("documents").read_text(encoding="utf-8").splitlines()
        texts = [max(lines, key=len)] if case["batch"] == "long" else lines[:BATCH_DOCUMENTS]
        path.write_text("\n".join(texts) + "\n", encoding="utf-8")
    return path


def side_config(ctx: Prepared, case: dict, side: str, tmp: Path) -> dict:
    args = ctx.args
    config = {
        **case,
        "side": side,
        "corpus_name": case["corpus"],
        "corpus": str(ctx.corpus(case["corpus"])),
        "vectors": args.vectors,
        "repeats": args.repeats,
        "tmp": str(tmp),
    }
    if case["kind"] == "strings":
        config["texts"] = str(_strings_input(ctx, case, tmp))
    if side == "v1":
        return {**config, "v1_model": str(ctx.v1_model(_v1_key(case)))}
    key = _v2_key(args, case)
    config.update(tokenizer=key[1], wv_dtype=key[2])
    # --chunk-bytes: the pipeline entry, with "auto" its default chunks (the baseline of
    # the ablation through the same entry).
    chunk = args.chunk_bytes
    config.update(entry="encode" if chunk is None else "pipeline")
    config["chunk_bytes"] = None if chunk in (None, "auto") else chunk
    return {**config, "encoder": str(ctx.v2_model(key)), "wv": str(ctx.wv(key[2]))}


def _inputs(config: dict) -> list[Path]:
    """The files a case reads: its texts and the model (2.0: the encoder holds its table)."""
    paths = [Path(config.get("texts", config["corpus"]))]
    if config["side"] == "v1":
        return paths + model_files(Path(config["v1_model"])) + _hub_files(config["vectors"])
    return [*paths, Path(config["encoder"])]


def _sides(case: dict, pythons: dict[str, str]) -> list[str]:
    compared = case["kind"] == "encode" and case["model"] in V1_MODELS
    return [s for s in pythons if s == "v2" or compared]


def run_pairs(ctx: Prepared, case: dict, pythons: dict[str, str], tmp: Path) -> dict:
    """--pairs runs of every side, alternating which goes first, each warm."""
    sides = _sides(case, pythons)
    runs: dict[str, list[dict]] = {side: [] for side in sides}
    for pair in range(ctx.args.pairs):
        for side in sorted(sides, reverse=pair % 2 == 0):  # v2 first, then v1 first
            config = side_config(ctx, case, side, tmp)
            _warm(_inputs(config))
            load = os.getloadavg()[0]  # before the run
            runs[side].append({**run_child(pythons[side], config), "load_average": load})
    return {side: {"median": _median(r), "runs": r} for side, r in runs.items()}


def _models_of(ctx: Prepared, cases: list[dict], pythons: dict[str, str]) -> None:
    """Check every corpus and model (making the missing ones) before anything runs."""
    for case in cases:
        ctx.stats(case["corpus"], _tokenizer(ctx.args, case["corpus"]))
        ensure_model(ctx, "v2", _v2_key(ctx.args, case), pythons["v2"])
        if "v1" in _sides(case, pythons):
            ensure_model(ctx, "v1", _v1_key(case), pythons["v1"])


def run_grid(ctx: Prepared, cases: list[dict], pythons: dict[str, str]) -> list[dict]:
    from fse._cpus import usable_cpus

    _models_of(ctx, cases, pythons)
    args = ctx.args
    print(f"# word vectors {args.wv_dtype}, tokenizer {args.tokenizer} (raw: fse_words@1)")
    print(
        f"# chunk bytes {args.chunk_bytes or 'auto'} ({'pipeline' if args.chunk_bytes else 'encode'})"
    )
    print_header(args.pairs, args.repeats)
    rows = []
    with tempfile.TemporaryDirectory(dir=ctx.args.tmp) as tmp:
        for case in cases:
            row = {"case": case, **run_pairs(ctx, case, pythons, Path(tmp))}
            row["manifest"] = ctx.stats(case["corpus"], _tokenizer(ctx.args, case["corpus"]))
            rows.append(row)
            print_row(row, usable_cpus())
    print_analysis(rows, max(ctx.args.workers))
    return rows


# --- the modes ------------------------------------------------------------------------


def _prepared(ctx: Prepared) -> bool:
    try:
        for name in corpora.CORPORA:
            ctx.manifest(name)
    except SystemExit:
        ctx.manifests.clear()
        return False
    return True


def _corpora(ctx: Prepared) -> None:
    chosen = corpora.first(corpora.articles(), ctx.args.tokens)
    context = {"generator": ctx.generator, "sources": ctx.sources()}
    context.update(vocabulary=vocabulary_words(ctx.wv_float32))
    context["vocabulary_fingerprint"] = ctx.vocabulary()
    corpora.prepare(ctx.folder, chosen, context, ctx.artifact)


def prepare(args) -> None:
    ctx = Prepared(args, _identity)
    ensure_vectors(ctx, "float32")
    base = {"vectors": args.vectors, "tmp": str(args.cache)}
    words = run_child(sys.executable, {**base, "side": "v2", "kind": "prepare"})["words"]
    if args.python_v1:
        v1 = run_child(args.python_v1, {**base, "side": "v1", "kind": "counts"})["words"]
        ctx.forget(ctx.counts)
        if v1 != words:
            raise SystemExit("the vocabularies of 2.0 and 1.1 are not in the same order")
    elif not ctx.counts.exists():
        raise SystemExit(
            "SIFWeights needs the wordfreq counts of --python-v1 (fse 1.1.0, wordfreq)"
        )
    if _prepared(ctx):
        print(f"# the corpora in {ctx.folder} are prepared", flush=True)
    else:
        _corpora(ctx)
    for model in MODELS:
        ensure_model(ctx, "v2", (model, "external", "float32"), sys.executable)
    ensure_model(ctx, "v2", ("SIFWeights", "fse_words@1", "float32"), sys.executable)
    for dtype in ("float32", "int8") if args.python_v1 else ():
        ensure_model(ctx, "v1", ("Average", dtype), args.python_v1)


def _write_json(args, rows: list[dict]) -> None:
    if args.json:
        result = {"config": {k: str(v) for k, v in vars(args).items()}, "rows": rows}
        result.update(machine=platform.platform(), python=platform.python_version())
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")


def compare(args) -> None:
    if not args.python_v1:
        raise SystemExit("compare needs --python-v1, a Python with fse 1.1.0")
    if (args.wv_dtype, args.chunk_bytes, args.tokenizer) != ("float32", None, "external"):
        raise SystemExit("--wv-dtype, --chunk-bytes and --tokenizer are ablations of matrix")
    if raw := sorted(set(args.lengths) & set(corpora.RAW)):
        # 1.1 splits at whitespace, 2.0 tokenizes them with fse_words@1.
        raise SystemExit(f"{', '.join(raw)}: the raw corpora are not tokenized alike in 1.1")
    pythons = {"v2": sys.executable, "v1": args.python_v1}
    _write_json(args, run_grid(Prepared(args, _identity), compare_cases(args), pythons))


def matrix(args) -> None:
    cases = matrix_cases(args)
    if not cases:
        raise SystemExit("no case of the grid matches --lengths")
    _write_json(args, run_grid(Prepared(args, _identity), cases, {"v2": sys.executable}))


def _profiled(args) -> dict:
    return _case(args.length, args.model, args.dtype, args.target, max(args.workers))


def verify(args) -> None:
    """The full check of everything the profile case reads (hashed and stamped), its
    statistics and model made if missing, before the profiled process starts."""
    ctx = Prepared(args, _identity)
    case = _profiled(args)
    ctx.stats(args.length, _tokenizer(args, args.length))
    path = ensure_model(ctx, "v2", _v2_key(args, case), sys.executable)
    print(f"# verified {ctx.corpus(args.length)} and {path}", flush=True)


def profile(args) -> None:
    """One 2.0 case in this process, its encode in a loop for --seconds; its files are
    checked against the stamps of ``verify`` (size and time), not hashed again."""
    from bench_length_cases import v2_encoder, v2_run

    ctx = Prepared(args, _identity, verify=False)
    case = _profiled(args)
    stats = ctx.stats(args.length, _tokenizer(args, args.length))
    key = _v2_key(args, case)
    wv, model = ctx.wv(key[2]), ctx.v2_model(key)
    if not (ctx.made_of(wv, ctx.wv_meta(key[2])) and ctx.made_of(model, ctx.model_meta("v2", key))):
        raise SystemExit(f"the model {'-'.join(key)} is stale or missing: run verify first")
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        config = side_config(ctx, case, "v2", Path(tmp))
        encoder = v2_encoder(config)
        times: list[float] = []
        start = time.perf_counter()
        while not times or time.perf_counter() - start < args.seconds:
            times.append(v2_run(encoder, config))
    best = min(times)
    print(
        f"{args.length} {'-'.join(key)} {args.dtype} {args.target} {case['workers']} workers: "
        f"{len(times)} encodes, min {best:.3f} s, {stats['tokens'] / best / 1e6:.1f} M tokens/s"
    )


# --- the arguments --------------------------------------------------------------------


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {value}")
    return value


def _tokens(text: str) -> int | None:
    return None if text == "all" else _positive(text)


def _chunk_bytes(text: str) -> int | str:
    return text if text == "auto" else _positive(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--vectors", default="glove-wiki-gigaword-300")
    parser.add_argument("--tokens", type=_tokens, default=20_000_000, help="a number or all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=Path, default=corpora.CACHE / "length")
    parser.add_argument("--python-v1", help="a Python with fse 1.1.0 and wordfreq")
    parser.add_argument("--lengths", nargs="+", choices=corpora.CORPORA)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--workers", type=_positive, nargs="+")
    parser.add_argument("--pairs", type=_positive, help="alternating runs (compare 2, else 1)")
    parser.add_argument("--repeats", type=_positive, default=3, help="timed encodes per child")
    parser.add_argument("--full", action="store_true", help="matrix: every combination")
    parser.add_argument("--wv-dtype", choices=DTYPES, default="float32")
    parser.add_argument(
        "--chunk-bytes", type=_chunk_bytes, help="auto (the default size) or bytes per chunk"
    )
    parser.add_argument("--tokenizer", default="external", choices=("external", "fse_words@1"))
    profiled = parser.add_argument_group("profile")
    profiled.add_argument("--length", choices=corpora.CORPORA, default="documents")
    profiled.add_argument("--model", choices=MODELS, default="SIFWeights")
    profiled.add_argument("--dtype", choices=DTYPES, default="int8")
    profiled.add_argument("--target", choices=TARGETS, default="store")
    profiled.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--tmp", type=Path, help="where stores are written")
    parser.add_argument("--json", type=Path)
    return parser


def _defaults(args) -> None:
    compared = args.mode == "compare"
    args.cache = args.cache.expanduser()
    args.lengths = args.lengths or list(COMPARE if compared else corpora.CORPORA)
    args.models = args.models or (["Average"] if compared else list(MODELS))
    workers = {"compare": [1, 10], "profile": [10], "verify": [10]}.get(args.mode, [1, 4, 10])
    args.workers = args.workers or workers
    args.pairs = args.pairs or (2 if compared else 1)


def main() -> None:
    args = _parser().parse_args()
    _defaults(args)
    import fse

    print(
        f"# {platform.platform()} | {os.cpu_count()} logical CPUs | Python "
        f"{platform.python_version()} | fse {fse.__version__} | load {os.getloadavg()[0]:.1f}",
        flush=True,
    )
    modes = {"prepare": prepare, "compare": compare, "matrix": matrix, "verify": verify}
    {**modes, "profile": profile}[args.mode](args)


if __name__ == "__main__":
    main()
