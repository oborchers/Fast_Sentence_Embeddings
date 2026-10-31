#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""ShardSink with write and hash in one thread: a measurement, with no change of src.

``FusedShardSink`` is the ShardSink of fse with its hash thread folded into the write
thread by two proxies, so a commit is one piece of work (write, then hash) and the shard
is handed on to the publisher after it, in the same thread. It keeps the order of the
split sink and one queue of at most 16 pieces and 64 MiB, the budget a merged design has
(a plain ``_hash = _io`` would count every commit twice against both limits). It is a
measurement of the whole merge (one thread, one queue) and a conservative approximation of
it: it keeps the ``written`` counter, its condition and ``wait_written``, which a merged
design drops.

Modes (paired runs of fresh children, alternating, the load before and after each run,
rounds with a load above --max-load discarded and made again; sides ``split``, ``fused``
and ``aa``, a second ``split`` that is the noise of the same run):

- ``cells``: the cells of the plan, ``bench_store.py`` children (synthetic, 10 workers) for
  float32, float16 and int8 in shards of the default size and of --small-rows, and the 1.33
  GB store of ``bench_length`` (sentences, SIF float32) when its cache is prepared;
- ``hasher``: the time of ``Hasher`` on 1.33 GB in pieces of 12 MB, once;
- ``child``: one run (used by the others).

    uv run python benchmarks/bench_store_fused.py cells --tmp DIR --json OUT.json
    uv run python benchmarks/bench_store_fused.py hasher

Every run checks the store it wrote: ``VectorStore.open`` and ``verify``, and a digest of
all its files, which the fused and the A/A run must equal that of the split run of their
round. A cell passes the rule fixed before the measurement when the median of fused/split
wall times is at most 1.02 and not above the upper end of the A/A ratios, not every
pair is slower, and the median peak RSS is at most 5 % above that of split.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import bench_harness as harness

SIDES = ("split", "fused", "aa")  # aa: split again
MAX_RATIO, MAX_RSS = 1.02, 1.05
PIECE = 12 * 2**20
FILES = ("vectors.npy", "inv_norms.npy", "scales.npy", "manifest.json")
CACHE = Path("~/.cache/fse-quality/length").expanduser()


def _peak() -> int:
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def _writer():
    """The ShardWriter class (fse 1.1.0, which the scripts also run under, has none)."""
    from fse._store._shard import ShardWriter

    return ShardWriter


class _Proxy:
    """A stand-in for a WriterThread: everything it does not change goes to ``io``."""

    def __init__(self, io):
        self._io = io

    def __getattr__(self, name):
        return getattr(self._io, name)


class _Hash(_Proxy):
    """In place of the hash thread: a commit is already done by the write piece, and
    everything else (the hand-on of a finished shard) goes to the write thread."""

    def submit(self, work, *args, nbytes=0, report=True):
        if getattr(work, "__func__", None) is not _writer().commit:
            self._io.submit(work, *args, nbytes=nbytes, report=report)


class _Write(_Proxy):
    """In place of the write thread: a write is the write and then the commit (hash)."""

    def submit(self, work, *args, nbytes=0, report=True):
        if getattr(work, "__func__", None) is _writer().write:
            writer = work.__self__

            def work(begin, codes):
                writer.write(begin, codes)
                writer.commit(begin, codes)

        self._io.submit(work, *args, nbytes=nbytes, report=report)


def _fused_class():
    """The FusedShardSink class (it needs fse, so it is made on first use)."""
    import fse

    class FusedShardSink(fse.ShardSink):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._hash.stop()  # the extra thread ends, idle
            io = self._io
            self._hash, self._io = _Hash(io), _Write(io)

    return FusedShardSink


def make_sink(kind: str, *args, **kwargs):
    """A ``fse.ShardSink`` (``split``) or a ``FusedShardSink`` (``fused``)."""
    import fse

    cls = fse.ShardSink if kind == "split" else _fused_class()
    return cls(*args, **kwargs)


def check_store(path: Path) -> dict:
    """Open and verify the store, and digest its files (all shards, in order)."""
    import fse
    from fse._container._hash import Hasher

    fse.VectorStore.open(path).verify()
    digest = Hasher()
    for entry in sorted(path.iterdir()):
        if entry.name.startswith("."):
            continue
        for name in ("",) if entry.is_file() else FILES:
            file = entry / name if name else entry
            if file.exists():
                digest.update(file.name.encode())
                digest.update(file.read_bytes())
    return {"digest": digest.checksum()}


# --- children -----------------------------------------------------------------------


def child_length(args) -> dict:
    """The 1.33 GB case: the sentences of bench_length, SIF float32, into a new store."""
    from bench_lifecycle_cases import v2_finished
    from safetensors.numpy import load_file

    import fse

    root = Path(next(CACHE.glob("*-20000000-s0")) if args.corpus is None else args.corpus)
    corpus = root / "sentences.txt"
    # The cached word vectors are a pre-release file (a counts tensor 2.0 no longer has).
    tensors = load_file(CACHE / "glove-wiki-gigaword-300.safetensors")
    wv = fse.WordVectors.from_arrays(
        tensors["vectors"],
        vocab_bytes=tensors["vocab.bytes"],
        vocab_offsets=tensors["vocab.offsets"],
    )
    encoder = v2_finished(wv, "SIFWeights", CACHE / "counts-en.npy", tokenizer="external")
    out: dict = {}
    for timed in (False, True):  # a warm-up, then the run
        with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
            store = Path(tmp) / "store"
            start = time.perf_counter()
            with make_sink(args.sink, store, encoder, dtype="float32") as sink:
                encoder.encode(corpus, workers=args.workers, sink=sink)
            seconds = time.perf_counter() - start
            if timed:
                out = {"seconds": seconds, "peak": _peak()} | check_store(store)
    return out


def child_hasher(_args) -> dict:
    import numpy as np

    from fse._container._hash import Hasher

    rows = np.random.default_rng(0).standard_normal((1_105_490, 300)).astype(np.float32)
    flat = rows.reshape(-1).view(np.uint8)
    hasher = Hasher()
    start = time.perf_counter()
    for begin in range(0, len(flat), PIECE):
        hasher.update(flat[begin : begin + PIECE])
    seconds = time.perf_counter() - start
    return {"bytes": flat.nbytes, "seconds": seconds, "gb_per_s": flat.nbytes / seconds / 1e9}


# --- parent ----------------------------------------------------------------------------


def _store_cmd(args, cell: dict, side: str) -> list[str]:
    """The bench_store child of a synthetic cell; side ``aa`` is split."""
    here = Path(__file__).with_name("bench_store.py")
    cmd = [sys.executable, str(here), "child", "--child", cell["child"], "--tmp", args.tmp]
    cmd += ["--dtype", cell["dtype"], "--workers", str(args.workers)]
    cmd += ["--sentences", str(cell["sentences"]), "--kind", "shards", "--check"]
    cmd += ["--sink", "fused" if side == "fused" else "split"]
    if cell.get("shard_rows"):
        cmd += ["--shard-rows", str(cell["shard_rows"])]
    return cmd


def _length_cmd(args, side: str) -> list[str]:
    cmd = [sys.executable, __file__, "child", "--child", "length", "--tmp", args.tmp]
    return [*cmd, "--workers", str(args.workers), "--sink", side if side == "fused" else "split"]


def _seconds(result: dict) -> float:
    return result.get("seconds", result.get("run_s", 0.0))


def judge(done: harness.Rounds) -> dict:
    """The medians and the verdict of one cell against the rule."""
    pairs = len(done.kept)
    fused = done.ratios("fused", "split", _seconds)
    noise = done.ratios("aa", "split", _seconds)
    rss = harness.median([r["peak"] for r in done.results("fused")]) / max(
        1.0, harness.median([r["peak"] for r in done.results("split")])
    )
    same = all(
        r["fused"]["digest"] == r["split"]["digest"] == r["aa"]["digest"] for r in done.by_round()
    )
    median = harness.median(fused)
    upper = max([1.0, *noise])
    verdict = (
        pairs >= 3
        and same
        and median <= MAX_RATIO
        and median <= upper
        and not all(r > 1 for r in fused)
        and rss <= MAX_RSS
    )
    return {
        "pairs": pairs,
        "discarded": len(done.discarded),
        "split_s": [round(_seconds(r), 3) for r in done.results("split")],
        "fused_s": [round(_seconds(r), 3) for r in done.results("fused")],
        "aa_s": [round(_seconds(r), 3) for r in done.results("aa")],
        "fused_over_split": [round(r, 3) for r in fused],
        "aa_over_split": [round(r, 3) for r in noise],
        "median_ratio": round(median, 3),
        "rss_ratio": round(rss, 3),
        "loads": [(round(a, 1), round(b, 1)) for a, b in done.loads],
        "bit_equal": same,
        "within_rule": verdict,
    }


def cells(args) -> list[dict]:
    plan = []
    for dtype in args.dtypes:
        for label, rows in (("large", None), ("small", args.small_rows)):
            plan.append(
                {
                    "name": f"{dtype} {label}",
                    "child": "v2_write",
                    "dtype": dtype,
                    "sentences": args.sentences,
                    "shard_rows": rows,
                }
            )
    plan.append(
        {
            "name": "sync int8 1024 rows",
            "child": "sync",
            "dtype": "int8",
            "sentences": 50_000,
            "shard_rows": args.small_rows,
            "pairs": 2,
        }
    )
    if not args.skip_length and any(CACHE.glob("*-20000000-s0")):
        plan.append({"name": "1.33 GB sentences SIF float32", "length": True})
    out = []
    for cell in plan:

        def run_one(side, cell=cell):
            cmd = _length_cmd(args, side) if cell.get("length") else _store_cmd(args, cell, side)
            return harness.run_child(cmd, label=cell["name"], timeout=900)

        done = harness.rounds(
            SIDES,
            run_one,
            cell.get("pairs", args.pairs),
            max_load=args.max_load,
            budget_s=args.budget,
        )
        row = {"cell": cell["name"]} | judge(done)
        print(json.dumps(row), flush=True)
        out.append(row)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["cells", "hasher", "child"])
    parser.add_argument("--child", choices=["length", "hasher"])
    parser.add_argument("--sink", choices=["split", "fused"], default="split")
    parser.add_argument("--tmp", default=tempfile.gettempdir())
    parser.add_argument("--corpus", help="the prepared bench_length cache of the corpus")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--sentences", type=int, default=500_000)
    parser.add_argument("--small-rows", type=int, default=1024)
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float16", "int8"])
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--max-load", type=float, default=5.0)
    parser.add_argument("--budget", type=float, default=240.0, help="seconds per cell")
    parser.add_argument("--skip-length", action="store_true")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    if args.mode == "child":
        print(json.dumps({"length": child_length, "hasher": child_hasher}[args.child](args)))
        return
    print(f"# load {harness.load():.1f}", flush=True)
    result = child_hasher(args) if args.mode == "hasher" else cells(args)
    if args.mode == "hasher":
        print(json.dumps(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
