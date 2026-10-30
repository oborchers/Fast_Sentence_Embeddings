#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Encoding Arrow string columns against the same texts as a Python list (the gate of
Arrow input, ARCHITECTURE.md "Text sources").

The same sentences, from the WikiText lines (``wiki``) or cut to at most 12 bytes
(``short``, all of them inline views), are encoded by one encoder (``fse_words@1``, a
table of the words of the corpus) as:

- ``list``: a list of str, made by ``to_pylist()`` before the measurement (the
  conversion is reported apart, ``to_pylist_s``, and not counted: the list path gets the
  advantage);
- ``string``, ``large_string``, ``string_view``: pyarrow arrays of that type;
- ``polars``: a Polars Series (string_view, several chunks);
- ``ipc``: a column of 1,000-row chunks read memory-mapped from an Arrow IPC file, as
  Hugging Face datasets hold it;
- ``file``: the lines of a text file (``FileSource``), the pace an Arrow column should
  reach.

Every run is a fresh child process. ``throughput`` alternates the list and each Arrow
kind in ``--pairs`` pairs per worker count and gives the median and spread of the
pairwise ratios of texts/s (Arrow over list), each child timing the best of ``--repeat``
calls. ``memory`` runs each kind once per worker count and gives the peak RSS growth of
the call above the RSS after the input exists and the encoder is loaded, sampled every
millisecond in a thread (so it is no high-water mark of the preparation).

    uv run python benchmarks/bench_arrow.py throughput --data wiki short --workers 1 10
    uv run python benchmarks/bench_arrow.py memory --data wiki --workers 1 10

Keep a comparison to a few minutes (AGENTS.md); the script warns about a high load.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import itertools
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

CORPUS = Path("~/.cache/fse-quality/wikitext-103-b08601e0.txt")
DIM = 128
ARROW_KINDS = ["string", "large_string", "string_view", "polars", "ipc", "file"]
GATE_KINDS = ["string", "large_string", "string_view", "polars"]


def current_rss() -> int:
    """The resident bytes of this process now."""
    if sys.platform == "darwin":

        class RusageInfo(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
                (name, ctypes.c_uint64)
                for name in (
                    "user",
                    "system",
                    "idle",
                    "interrupts",
                    "pageins",
                    "wired",
                    "resident",
                    "footprint",
                    "start",
                    "exit",
                )
            ]  # struct rusage_info_v0 of <libproc.h>, whole: the call writes all of it

        info = RusageInfo()
        if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 0, ctypes.byref(info)):
            raise OSError("proc_pid_rusage failed")
        return int(info.resident)
    with open("/proc/self/statm", encoding="ascii") as statm:
        return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class Peak:
    """The highest RSS while it runs, sampled every millisecond in a thread."""

    def __enter__(self):
        self.peak, self._stop = current_rss(), threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(0.001):
            self.peak = max(self.peak, current_rss())

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, current_rss())


def sentences(data: str, n: int) -> list[str]:
    with open(CORPUS.expanduser(), encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in itertools.islice(f, n)]
    if data == "wiki":
        return lines
    short = []
    for line in lines:  # whole words up to 12 bytes; every row an inline view
        words, out = line.split(), ""
        for w in words:
            joined = f"{out} {w}" if out else w
            if len(joined.encode()) > 12:
                break
            out = joined
        short.append(out)
    return short


def encoder_of(lines: list[str]):
    import numpy as np

    import fse

    words = sorted({w.lower() for line in lines[:100_000] for w in line.split()})
    vectors = np.random.default_rng(0).standard_normal((len(words), DIM), dtype=np.float32)
    return fse.Encoder.from_word_vectors(fse.WordVectors.from_arrays(vectors, words=words))


def make_input(kind: str, lines: list[str], tmp: str):
    """The input of ``kind`` and the seconds of ``to_pylist`` for the list."""
    import pyarrow as pa

    if kind == "list":
        array = pa.array(lines)
        start = time.perf_counter()
        out = array.to_pylist()
        return out, time.perf_counter() - start
    if kind in ("string", "large_string", "string_view"):
        return pa.array(lines, type=getattr(pa, kind)()), None
    if kind == "polars":
        import polars as pl

        parts = [pl.Series(lines[i : i + 100_000]) for i in range(0, len(lines), 100_000)]
        return pl.concat(parts, rechunk=False), None
    if kind == "ipc":
        path = os.path.join(tmp, "column.arrow")
        schema = pa.schema([("text", pa.string())])
        with pa.OSFile(path, "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
            for i in range(0, len(lines), 1000):
                writer.write_batch(pa.record_batch([pa.array(lines[i : i + 1000])], schema=schema))
        return pa.ipc.open_file(pa.memory_map(path)).read_all()["text"], None
    path = Path(tmp) / "lines.txt"
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path, None


def child(c: dict) -> dict:
    lines = sentences(c["data"], c["sentences"])
    encoder = encoder_of(lines)
    encoder.encode(lines[:2000], workers=c["workers"])  # load everything lazy first
    with tempfile.TemporaryDirectory(dir=c.get("tmp")) as tmp:
        texts, to_pylist = make_input(c["kind"], lines, tmp)
        n = len(lines)
        del lines
        gc.collect()
        if c["mode"] == "memory":
            base = current_rss()
            with Peak() as peak:
                rows = encoder.encode(texts, workers=c["workers"])
            return {"growth": peak.peak - base, "rows": len(rows), "base": base}
        best = float("inf")
        for _ in range(c["repeat"]):
            start = time.perf_counter()
            rows = encoder.encode(texts, workers=c["workers"])
            best = min(best, time.perf_counter() - start)
            del rows
        return {"seconds": best, "texts_per_s": n / best, "to_pylist_s": to_pylist}


def run(config: dict) -> dict:
    done = subprocess.run(
        [sys.executable, __file__, "_child", json.dumps(config)],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        raise SystemExit(f"{config} failed:\n{done.stderr[-3000:]}")
    return {**json.loads(done.stdout.strip().splitlines()[-1]), "load": os.getloadavg()[0]}


def _pairs(args, data: str, workers: int, kind: str) -> list[dict]:
    """``--pairs`` runs of the list and of ``kind``, alternating which goes first."""
    runs = []
    for pair in range(args.pairs):
        got = {}
        for side in ["list", kind] if pair % 2 == 0 else [kind, "list"]:
            got[side] = run(
                {
                    "mode": "throughput",
                    "data": data,
                    "workers": workers,
                    "kind": side,
                    "sentences": args.sentences,
                    "repeat": args.repeat,
                    "tmp": args.tmp,
                }
            )
        runs.append(got)
    return runs


def _summary(data: str, workers: int, kind: str, runs: list[dict]) -> dict:
    ratios = [g[kind]["texts_per_s"] / g["list"]["texts_per_s"] for g in runs]
    return {
        "data": data,
        "workers": workers,
        "kind": kind,
        "ratios": ratios,
        "median_ratio": statistics.median(ratios),
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "list_texts_per_s": statistics.median(g["list"]["texts_per_s"] for g in runs),
        "arrow_texts_per_s": statistics.median(g[kind]["texts_per_s"] for g in runs),
        "to_pylist_s": statistics.median(g["list"]["to_pylist_s"] for g in runs),
        "load": max(g[s]["load"] for g in runs for s in g),
    }


def throughput(args) -> list[dict]:
    out = []
    for data, workers, kind in itertools.product(args.data, args.workers, args.kinds):
        r = _summary(data, workers, kind, _pairs(args, data, workers, kind))
        print(
            f"{data:5} {workers:2} workers {kind:12} ratio {r['median_ratio']:.3f} "
            f"[{r['min_ratio']:.3f}, {r['max_ratio']:.3f}]  arrow {r['arrow_texts_per_s']:,.0f}/s"
            f"  list {r['list_texts_per_s']:,.0f}/s  load {r['load']:.1f}",
            flush=True,
        )
        out.append(r)
    return out


def memory(args) -> list[dict]:
    out = []
    for data, workers, kind in itertools.product(args.data, args.workers, ["list", *args.kinds]):
        c = {
            "mode": "memory",
            "data": data,
            "workers": workers,
            "kind": kind,
            "sentences": args.sentences,
            "tmp": args.tmp,
        }
        r = {**c, **run(c)}
        print(
            f"{data:5} {workers:2} workers {kind:12} peak growth {r['growth'] / 2**20:8.1f} MiB",
            flush=True,
        )
        out.append(r)
    return out


def main(argv=None):
    if argv is None and len(sys.argv) > 2 and sys.argv[1] == "_child":
        print(json.dumps(child(json.loads(sys.argv[2]))))
        return
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["throughput", "memory"])
    parser.add_argument("--data", nargs="+", default=["wiki", "short"], choices=["wiki", "short"])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 10])
    parser.add_argument("--kinds", nargs="+", default=GATE_KINDS, choices=ARROW_KINDS)
    parser.add_argument("--sentences", type=int, default=400_000)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--tmp", default=None, help="the directory of temporary files")
    parser.add_argument("--json", help="write the results here")
    args = parser.parse_args(argv)
    if os.getloadavg()[0] > 5:
        print(f"warning: load average {os.getloadavg()[0]:.1f}; numbers will be noisy", flush=True)
    results = throughput(args) if args.mode == "throughput" else memory(args)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
