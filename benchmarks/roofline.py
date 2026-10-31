#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The throughput bounds of fse for one encoder, one corpus and one machine, and the
measured throughput as a share of each (PERFORMANCE.md explains the model).

measure: the dataset profile (texts, bytes, tokens, distinct rows) and the seconds of
every stage at one worker; then ``Encoder.encode`` at 1 and N workers in paired,
alternating rounds (a round with a load above ``--max-load`` before or after is
discarded), with the summed wall and thread CPU time of the tasks at each N; then the
bounds of roofline_model.py and the measured shares. ``--json`` keeps the numbers.
compare: two or more table dtypes end to end on one encoder and corpus, paired and
alternating in one process (roofline_dtype.py), as the time ratio to the first.
probe: the attainable rate of reading a float32 table far larger than the caches, by
numpy's sum and by fse's pooling kernel on consecutive and on random rows, in 1 to N
threads (roofline_probe.py).
show: the report of a saved ``measure --json`` on another hardware profile (the stage
times stay those of the machine that measured them).

    uv run python benchmarks/roofline.py measure --model potion-base-8M --corpus massive
    uv run python benchmarks/roofline.py measure --model static-retrieval-mrl-en-v1 \\
        --dim 1024 --corpus wikitext --workers 8 10 --json out.json
    uv run python benchmarks/roofline.py compare --model static-retrieval-mrl-en-v1 \\
        --dim 1024 --corpus massive --dtypes float32 int8 --workers 1 8
    uv run python benchmarks/roofline.py probe --workers 1 4 8 10
    uv run python benchmarks/roofline.py show out.json --hardware m4-pro-14
    uv run python benchmarks/roofline.py _encode potion-base-8M none float32 massive 1 40

The last form only encodes in a loop (model, dim, dtype, corpus, workers, seconds), for
``make profile-cpu RUN=...``. Corpora and models come from the local Hugging Face cache
(bench_wordpiece.py), nothing is downloaded. Keep a run to a few minutes (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import roofline_model as rm
from bench_pretrained import COMMITS
from bench_wordpiece import texts
from roofline_dtype import SpecEncoder, compare, lines
from roofline_probe import probe
from roofline_report import report

# The snapshots of the local cache that PERFORMANCE.md measured.
MODELS = {
    "potion-base-8M": ("minishlab/potion-base-8M", "bf8b056651a2c21b8d2565580b8569da283cab23"),
    "potion-base-32M": ("minishlab/potion-base-32M", "1e5a03f8eeb2c98b928fbbd846f22f816360919f"),
    "static-retrieval-mrl-en-v1": (
        "sentence-transformers/static-retrieval-mrl-en-v1",
        COMMITS["sentence-transformers/static-retrieval-mrl-en-v1"],
    ),
}


def _load() -> float:
    return os.getloadavg()[0]


def _seconds(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _best(fn, repeats: int) -> float:
    return min(_seconds(fn) for _ in range(repeats))


def pretrained(name: str, dim: int | None):
    """The pretrained encoder ``name`` at its pinned snapshot, cut to ``dim`` (the
    sentence-transformers model only)."""
    from fse import Encoder

    repo, commit = MODELS[name]
    options = {} if dim is None else {"truncate_dim": dim}
    return Encoder.from_pretrained(repo, revision=commit, local_files_only=True, **options)


def encoder(name: str, dim: int | None, dtype: str):
    """``pretrained``, its table stored as ``dtype``; another dtype than the model's runs
    as a ``SpecEncoder`` (roofline_dtype.py: Encoder refuses it for these tokenizers)."""
    enc = pretrained(name, dim)
    return enc if dtype == enc.word_vectors.dtype else SpecEncoder(enc, dtype)


class TaskClock:
    """Wraps ``Pipeline._work`` of one pipeline: the summed wall and thread CPU seconds of
    its tasks (list.append is atomic under the GIL, so the workers need no lock)."""

    def __init__(self, pipeline):
        self.wall: list[float] = []
        self.cpu: list[float] = []
        work = pipeline._work

        def timed(*args):
            wall, cpu = time.perf_counter(), time.thread_time()
            try:
                return work(*args)
            finally:
                self.cpu.append(time.thread_time() - cpu)
                self.wall.append(time.perf_counter() - wall)

        pipeline._work = timed

    def take(self) -> tuple[float, float, int]:
        """The summed wall and CPU seconds and the number of tasks since the last take."""
        out = (sum(self.wall), sum(self.cpu), len(self.wall))
        self.wall.clear()
        self.cpu.clear()
        return out


class Stages:
    """The stages of one encoder on one corpus at one worker, run as the runtime runs them."""

    def __init__(self, enc, sentences: list[str]):
        from fse._text import new_workspace

        self.enc, self.sentences = enc, sentences
        self.pipeline = enc._pipeline("cpu")
        self.clock = TaskClock(self.pipeline)
        self.source = self.join()
        self.row_bytes = self.pipeline._row_bytes("float32")
        self.tasks = self.plan(1)
        self.workspace = new_workspace()
        self.batches = self.tokenize()
        self.out = np.empty((len(sentences), enc.output_dim), np.float32)

    def join(self):
        from fse._text import StringSource

        return StringSource(self.sentences)

    def plan(self, workers: int) -> list:
        return list(self.pipeline.plan_tasks(self.source, workers, None, self.row_bytes)[0].tasks)

    def tokenize(self) -> list:
        return [t.batch(self.pipeline.lookup, self.workspace) for t in self.tasks]

    def _views(self):
        for b in self.batches:
            yield b, self.out[b.first_row : b.first_row + len(b.offsets) - 1]

    def pool(self) -> None:
        kernel = self.pipeline.prepared.state.kernel
        for b, view in self._views():
            kernel.pool(b, view)

    def normalize(self) -> None:
        from fse import _native

        steps = self.pipeline.prepared.steps
        for b, view in self._views() if steps else ():
            _native.postprocess_pooled(view, steps, b.first_row)

    def infer(self) -> None:
        """The join and ``infer`` at one worker, as ``Encoder.encode`` runs them: the
        joined buffer is in the caches when the tasks read it."""
        self.pipeline.infer(self.join(), workers=1)

    def encode(self, workers: int):
        return self.enc.encode(self.sentences, workers=workers)

    def dataset(self, corpus: str) -> rm.Dataset:
        tokens = sum(
            len(b.tokens) + (0 if b.ngrams is None else len(b.ngrams)) for b in self.batches
        )
        ids = np.unique(np.concatenate([b.tokens for b in self.batches]))
        nbytes = sum(len(s.encode("utf-8")) for s in self.sentences)
        return rm.Dataset(corpus, len(self.sentences), nbytes, int(tokens), len(ids))

    def times(self, repeats: int) -> rm.Stages:
        """Seconds per stage, each the fastest of ``repeats`` rounds that run every stage
        once in turn; the rests are differences of the fastest times, so they can be
        slightly negative within the noise."""
        self.encode(1)  # warm: the table's pages and the caches
        stages = {
            "join": self.join,
            "plan": lambda: self.plan(1),
            "tokenize": self.tokenize,
            "pool": self.pool,
            "normalize": self.normalize,
            "encode": lambda: self.encode(1),
        }
        b = dict.fromkeys([*stages, "infer", "tasks"], float("inf"))
        for _ in range(repeats):
            for name, fn in stages.items():
                b[name] = min(b[name], _seconds(fn))
            self.clock.take()
            infer = _seconds(self.infer)
            if infer < b["infer"]:
                b["infer"], b["tasks"] = infer, self.clock.take()[0]
        return rm.Stages(
            join=b["join"],
            plan=b["plan"],
            caller=b["infer"] - b["join"] - b["plan"] - b["tasks"],
            encode_rest=b["encode"] - b["infer"],
            tokenize=b["tokenize"],
            pool=b["pool"],
            normalize=b["normalize"],
            task_rest=b["tasks"] - b["tokenize"] - b["pool"] - b["normalize"],
        )


def _round(stages: Stages, order: list[int], passes: int) -> dict[int, tuple]:
    """Per worker count the fastest of ``passes`` encodes, with the task time of that pass."""
    got = {}
    for w in order:
        best = (float("inf"), 0.0, 0.0, 0)
        for _ in range(passes):
            stages.clock.take()
            seconds = _seconds(lambda w=w: stages.encode(w))
            best = min(best, (seconds, *stages.clock.take()))
        got[w] = best
    return got


def paired(stages: Stages, workers: list[int], rounds: int, passes: int, max_load: float) -> dict:
    """``Encoder.encode`` per worker count, each count once per round in alternating order;
    rounds with a load above ``max_load`` before or after are discarded (at most
    ``3 * rounds`` attempts). Medians over the kept rounds."""
    kept: list[dict[int, tuple]] = []
    loads, discarded = [], 0
    for attempt in range(3 * rounds):
        if len(kept) == rounds:
            break
        before = _load()
        got = _round(stages, workers if attempt % 2 == 0 else workers[::-1], passes)
        after = _load()
        if max(before, after) > max_load:
            discarded += 1
            continue
        loads.append((round(before, 2), round(after, 2)))
        kept.append(got)
    return {"rounds": len(kept), "discarded": discarded, "loads": loads, **_medians(kept, workers)}


def _medians(kept: list[dict[int, tuple]], workers: list[int]) -> dict:
    """Per worker count the medians of the kept rounds, and the range of their seconds."""
    if not kept:
        return {key: {} for key in ("seconds", "task_wall", "task_cpu", "tasks", "range")}
    out: dict = {}
    for i, key in enumerate(("seconds", "task_wall", "task_cpu", "tasks")):
        out[key] = {w: statistics.median(r[w][i] for r in kept) for w in workers}
    seconds = {w: [r[w][0] for r in kept] for w in workers}
    out["range"] = {w: (min(v), max(v)) for w, v in seconds.items()}
    return out


def measure(args) -> dict:
    enc = encoder(args.model, args.dim, args.dtype)
    stages = Stages(enc, texts(args.corpus))
    wv = enc.word_vectors
    model = rm.Model(args.model, enc.output_dim, wv.dtype, len(wv), bool(enc.steps))
    times = stages.times(args.repeats)
    plans = {w: _best(lambda w=w: stages.plan(w), args.repeats) for w in args.workers}
    run = paired(stages, sorted({1, *args.workers}), args.rounds, args.passes, args.max_load)
    return {
        "model": asdict(model),
        "dataset": asdict(stages.dataset(args.corpus)),
        "stages": asdict(times),
        "plan_at": plans,
        "measured": run,
        "machine": platform.platform(),
        "hardware": args.hardware,
    }


def _hardware(name: str, ratio: float | None) -> rm.Hardware:
    profiles = rm.load_profiles()
    if name not in profiles:
        raise SystemExit(f"no hardware profile {name!r}; known: {', '.join(profiles)}")
    hw = profiles[name]
    return hw if ratio is None else replace(hw, e_core_ratio=ratio, e_core_source="--e-core-ratio")


def _compare(args) -> None:
    """The dtypes of ``--dtypes`` end to end on one encoder and corpus, paired."""
    sentences = texts(args.corpus)
    enc = pretrained(args.model, args.dim)
    tokens = Stages(SpecEncoder(enc, args.dtypes[0]), sentences).dataset(args.corpus).tokens
    w = (args.workers, args.rounds, args.passes, args.max_load)
    results = compare(enc, sentences, args.dtypes, *w)
    print("\n".join(lines(results, args.dtypes, tokens)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


def _encode_loop(argv: list[str]) -> None:
    """Encode at ``workers`` for ``seconds``, for the profilers."""
    name, dim, dtype, corpus, workers, seconds = argv
    stages = Stages(encoder(name, None if dim == "none" else int(dim), dtype), texts(corpus))
    end = time.perf_counter() + float(seconds)
    while time.perf_counter() < end:
        stages.encode(int(workers))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--model", choices=MODELS, required=True)
    m.add_argument("--dim", type=int, default=None, help="truncate_dim (sentence-transformers)")
    m.add_argument("--dtype", choices=rm.ITEMSIZE, default="float32")
    m.add_argument("--corpus", choices=("wikitext", "massive"), required=True)
    m.add_argument("--workers", type=int, nargs="+", default=[8])
    m.add_argument(
        "--repeats", type=int, default=15, help="rounds of all stages; the fastest counts"
    )
    m.add_argument("--rounds", type=int, default=5)
    m.add_argument(
        "--passes", type=int, default=5, help="per round and worker count; the fastest counts"
    )
    m.add_argument("--max-load", type=float, default=5.0)
    m.add_argument("--hardware", default="m1-pro-10")
    m.add_argument("--e-core-ratio", type=float, default=None)
    m.add_argument("--json", default=None)
    p = sub.add_parser("probe")
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 10])
    p.add_argument("--table-gib", type=float, default=1.0)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=5)
    c = sub.add_parser("compare")
    c.add_argument("--model", choices=MODELS, required=True)
    c.add_argument("--dim", type=int, default=None)
    c.add_argument("--corpus", choices=("wikitext", "massive"), required=True)
    c.add_argument("--dtypes", nargs="+", choices=rm.ITEMSIZE, default=["float32", "int8"])
    c.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    c.add_argument("--rounds", type=int, default=5)
    c.add_argument("--passes", type=int, default=5)
    c.add_argument("--max-load", type=float, default=5.0)
    c.add_argument("--json", default=None)
    s = sub.add_parser("show")
    s.add_argument("result")
    s.add_argument("--hardware", default=None)
    s.add_argument("--e-core-ratio", type=float, default=None)
    s.add_argument("--workers", type=int, nargs="*", default=[], help="more worker counts")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_encode":
        return _encode_loop(argv[1:])
    args = _parser().parse_args(argv)
    load = _load()
    print(
        f"# {platform.platform()} | load {load:.1f}"
        + (" (high: timings are noisy)" if load > 2 else "")
    )
    if args.command == "probe":
        probe(args.workers, args.table_gib, args.dim, args.repeats)
        return None
    if args.command == "compare":
        return _compare(args)
    if args.command == "measure":
        result = measure(args)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=1)
    else:
        with open(args.result, encoding="utf-8") as f:
            result = json.load(f)
    print(
        "\n".join(
            report(
                result,
                _hardware(args.hardware or result["hardware"], args.e_core_ratio),
                tuple(getattr(args, "workers", ()) if args.command == "show" else ()),
            )
        )
    )
    return None


if __name__ == "__main__":
    main()
