#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Where the workers of a run idle, as a timeline (PERFORMANCE.md, "Where the workers
idle"). The executor is observed from this process only (runtime_timeline_probe.py): the
start and end of every task, and the time of the calling thread in the task iterator, in
``reserve``, waiting for the oldest future and in ``commit``. From these the idle area of
the workers, ``slots * (last end - first submit) - the wall time of all tasks``, is split
exactly by cause (runtime_timeline_attr.py): tail, ramp, head-of-line blocking, producer,
sink and hand-over. Per thread only differences in the rate are shown; they do not say
which core type ran a thread.

measure: the split per worker count for one encoder and corpus (the first ``--sentences``
texts of it, for a short run), medians over rounds of passes; a round with a load above
``--max-load`` before or after is discarded.
overhead: plain calls against traced ones, and a second plain call as the control, paired
and alternating: the cost of the observation and the noise between equal calls.
ablate: A1 chunk, A2 window, A3 pool, A4 sink (runtime_timeline_ablate.py).

    uv run --with pyarrow python benchmarks/runtime_timeline.py measure \\
        --model potion-base-8M --corpus wikitext --workers 1 4 8 10 --json out.json
    uv run python benchmarks/runtime_timeline.py overhead --model potion-base-8M \\
        --corpus massive --workers 8
    uv run python benchmarks/runtime_timeline.py ablate chunk --model potion-base-8M \\
        --corpus massive --workers 8

Models and corpora come from the local Hugging Face cache (roofline.py), nothing is
downloaded. Keep a run to a few minutes (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys

from bench_harness import load
from roofline import MODELS, encoder
from runtime_timeline_ablate import ABLATIONS, rss_children, rss_probe
from runtime_timeline_probe import Recorder, tracing
from runtime_timeline_run import HEADER, line, series, timed


def overhead(pipeline, sentences, workers: int, rounds: int, passes: int, max_load: float) -> dict:
    """Plain call A, plain call B (the control) and a traced call, each the fastest of
    ``passes`` per round, rounds in rotating order; the ratios to A, round by round."""
    rec = Recorder()
    ratios: dict[str, list[float]] = {"B": [], "traced": []}
    loads, discarded = [], 0
    for attempt in range(3 * rounds):
        if len(ratios["B"]) == rounds:
            break
        order = ["A", "B", "traced"]
        order = order[attempt % 3 :] + order[: attempt % 3]
        before = load()
        best = {n: _best(pipeline, sentences, workers, passes, n == "traced", rec) for n in order}
        after = load()
        if max(before, after) > max_load:
            discarded += 1
            continue
        loads.append((round(before, 2), round(after, 2)))
        for n, values in ratios.items():
            values.append(best[n] / best["A"])
    return {
        "workers": workers,
        "rounds": len(ratios["B"]),
        "discarded": discarded,
        "loads": loads,
        "ratios": ratios,
    }


def _best(pipeline, sentences, workers: int, passes: int, traced: bool, rec: Recorder) -> int:
    if not traced:
        return min(timed(pipeline, sentences, workers) for _ in range(passes))
    with tracing(rec):
        best = min(timed(pipeline, sentences, workers) for _ in range(passes))
    rec.runs.clear()
    rec.infers.clear()
    return best


def overhead_lines(r: dict) -> list[str]:
    out = [f"workers {r['workers']}: {r['rounds']} rounds, {r['discarded']} discarded"]
    for name, label in (("B", "plain B / plain A (noise)"), ("traced", "traced / plain A")):
        v = r["ratios"][name]
        if v:
            out.append(
                f"  {label:<28} median {statistics.median(v):.4f}  range {min(v):.4f} to {max(v):.4f}"
            )
    return out


def _cell(args):
    from bench_wordpiece import texts

    enc = encoder(args.model, args.dim, args.dtype)
    sentences = texts(args.corpus)
    sentences = sentences[: args.sentences] if args.sentences else sentences
    return enc._pipeline("cpu"), sentences


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--dim", type=int, default=None, help="truncate_dim (sentence-transformers)")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--corpus", choices=("wikitext", "massive"), required=True)
    p.add_argument("--sentences", type=int, default=0, help="only the first N texts (a short run)")
    p.add_argument("--workers", type=int, nargs="+", default=[8])
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--max-load", type=float, default=5.0)
    p.add_argument("--json", default=None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    _common(sub.add_parser("measure"))
    o = sub.add_parser("overhead")
    _common(o)
    o.add_argument("--passes", type=int, default=5)
    a = sub.add_parser("ablate")
    a.add_argument("which", choices=[*ABLATIONS, "all"])
    _common(a)
    a.add_argument("--rss", action="store_true", help="A2: the peak RSS per window, in children")
    r = sub.add_parser("_rss")
    _common(r)
    r.add_argument("--factor", default="1")
    return parser


def _write(path: str | None, data: object) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)


def _header(args) -> None:
    sentences = f", first {args.sentences} texts" if args.sentences else ""
    print(
        f"# {platform.platform()} | load {load():.1f} | {args.model} {args.dim or ''} "
        f"{args.corpus}{sentences}"
    )


def _measure(args, pipeline, sentences) -> list:
    print(HEADER)
    results = []
    for w in args.workers:
        results.append(series(pipeline, sentences, w, args.rounds, args.max_load))
        print(line(results[-1]), flush=True)
    return results


def _overhead(args, pipeline, sentences) -> list:
    results = []
    for w in args.workers:
        results.append(overhead(pipeline, sentences, w, args.rounds, args.passes, args.max_load))
        print("\n".join(overhead_lines(results[-1])), flush=True)
    return results


def _ablate(args, pipeline, sentences) -> list:
    results = []
    for w in args.workers:
        for name in ABLATIONS if args.which == "all" else [args.which]:
            lines, result = ABLATIONS[name](pipeline, sentences, w, args.rounds, args.max_load)
            results.append({"workers": w, "ablation": name, **result})
            print(f"workers {w}\n" + "\n".join(lines), flush=True)
        if args.rss:
            print("\n".join(rss_children(_child_args(args, w), w)), flush=True)
    return results


COMMANDS = {"measure": _measure, "overhead": _overhead, "ablate": _ablate}


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    pipeline, sentences = _cell(args)
    if args.command == "_rss":
        factor = None if args.factor == "unbounded" else float(args.factor)
        print(json.dumps(rss_probe(pipeline, sentences, args.workers[0], factor, 5)))
        return
    _header(args)
    _write(args.json, COMMANDS[args.command](args, pipeline, sentences))


def _child_args(args, workers: int) -> list[str]:
    out = ["--model", args.model, "--corpus", args.corpus, "--workers", str(workers)]
    out += ["--dtype", args.dtype]
    out += ["--dim", str(args.dim)] if args.dim else []
    return out + (["--sentences", str(args.sentences)] if args.sentences else [])


if __name__ == "__main__":
    main()
