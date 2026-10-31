#!/usr/bin/env python
"""Measure store sync costs in fresh processes, without changing production code.

C0 probes dirty files/directories from a durable state. C1 seals every batch.
C2 writes synthetic rows. C3 writes the cached 20M-token SIF corpus, after warm-up.
All write timers start on an identically initialised, durable store. ``deferred``
is an optimistic lower bound; only ``flush_model`` enters the bulk decision.
Use --tmp and --json in the coordinator's scratchpad. Hold the shared bench.lock
externally; --budget is shared across every cell, side and discarded round.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import bench_harness as harness
import bench_store as store_bench
import bench_store_sync_run as run

SIDES = ("full", "fsync", "none", "deferred", "flush_model", "aa")


def evidence(rounds, side, threshold):
    """Symmetric A/A rule, with a minimum of three complete valid pairs."""
    if len(rounds) < 3:
        return {"pairs": len(rounds), "passes": None}
    noise = [r["aa"]["seconds"] / r["full"]["seconds"] for r in rounds]
    ratios = [r[side]["seconds"] / r["full"]["seconds"] for r in rounds]
    bound = min(min(noise), 1 / max(noise), 1)
    same = all(r[side]["digest"] == r["full"]["digest"] == r["aa"]["digest"] for r in rounds)
    return {
        "pairs": len(rounds),
        "ratios": ratios,
        "aa": noise,
        "noise_lower": bound,
        "median": statistics.median(ratios),
        "equal": same,
        "passes": same and statistics.median(ratios) <= threshold and max(ratios) < bound,
    }


def decision(cells):
    """Apply the pre-registered B/S rule. Missing evidence never counts as a miss."""
    by_name = {c["name"]: c["kept"] for c in cells}
    bulk = [evidence(by_name.get(n, []), "flush_model", 0.9) for n in ("C2-default", "C3")]
    stream = evidence(by_name.get("C1-10000", []), "none", 0.5)
    if any(b["passes"] for b in bulk):
        outcome = "tiers in 2.0"
    elif any(b["passes"] is None for b in bulk) or stream["passes"] is None:
        outcome = "not decided"
    elif stream["passes"]:
        outcome = "full + none in 2.0, flush in 2.1"
    else:
        outcome = "2.1"
    return {"outcome": outcome, "bulk": bulk, "stream": stream}


def plan():
    cells = [
        {"name": f"C1-{batch}", "sentences": n, "batch": batch, "dtype": "float32"}
        for batch, n in ((1000, 200_000), (10000, 200_000), (65536, 524_288))
    ]
    cells += [
        {"name": name, "sentences": 250_000, "dtype": dtype, "shard_rows": rows}
        for name, dtype, rows in (
            ("C2-default", "float32", None),
            ("C2-small-f32", "float32", 16384),
            ("C2-small-i8", "int8", 16384),
        )
    ]
    cells.append({"name": "C3", "length": True, "dtype": "float32"})
    return cells


def prepare(args):
    import fse

    if args.length:
        from bench_lifecycle_cases import v2_finished
        from bench_store_fused import CACHE
        from safetensors.numpy import load_file

        roots = sorted(CACHE.glob("*-20000000-s0"))
        root = Path(args.corpus) if args.corpus else roots[0]
        tensors = load_file(CACHE / "glove-wiki-gigaword-300.safetensors")
        wv = fse.WordVectors.from_arrays(
            tensors["vectors"],
            vocab_bytes=tensors["vocab.bytes"],
            vocab_offsets=tensors["vocab.offsets"],
        )
        encoder = v2_finished(wv, "SIFWeights", CACHE / "counts-en.npy", tokenizer="external")
        return encoder, root / "sentences.txt"
    sentences = [" ".join(tokens) for tokens in store_bench._corpus(args.sentences)]
    wv = fse.WordVectors.from_arrays(
        store_bench._rows(), words=[f"w{i}" for i in range(store_bench.VOCAB)]
    )
    encoder = fse.Encoder.from_word_vectors(wv, normalize=False, tokenizer="external")
    encoder.encode(sentences[:1000], dtype=args.dtype)
    return encoder, sentences


def encode(args, encoder, texts, sink):
    if not args.batch:
        encoder.encode(texts, workers=args.workers, sink=sink)
        return []
    batches = []
    begin = time.perf_counter()
    for _ in encoder.encode_batches(
        (texts[i : i + args.batch] for i in range(0, len(texts), args.batch)),
        workers=args.workers,
        sink=sink,
    ):
        now = time.perf_counter()
        batches.append(now - begin)
        begin = now
    return batches


def measure(args, encoder, texts):
    import fse
    from fse._store._sink import default_shard_rows

    class SealingSink(fse.ShardSink):
        def publish(self):
            self.seal()
            return super().publish()

    rows = args.shard_rows
    if args.side == "large4":
        rows = 4 * default_shard_rows(encoder.output_dim, args.dtype)
    if args.side == "single":
        rows = sum(1 for _ in texts.open()) if isinstance(texts, Path) else len(texts)
    trace = run.Trace()
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp, trace.recording():
        path = Path(tmp) / "store"
        # Close an empty sink durably, identical on every side, outside the write timer.
        start = time.perf_counter()
        fse.ShardSink(path, encoder, dtype=args.dtype, shard_rows=rows).close()
        initialisation = {"seconds": time.perf_counter() - start, "syncs": trace.snapshot()}
        trace.syncs.clear()
        cls = SealingSink if args.batch else fse.ShardSink
        sink = cls(path, encoder, dtype=args.dtype, shard_rows=rows)
        batches = []
        with run.ablation(args.side), trace.endpoints(sink):
            start = time.perf_counter()
            try:
                if args.side == "flush_model":
                    run.marker(path, 0)  # before any shard can be renamed
                batches = encode(args, encoder, texts, sink)
                trace.events["encode_return"] = time.perf_counter()
            finally:
                sink.close()
            trace.events["close_end"] = time.perf_counter()
            if args.side in {"deferred", "flush_model"}:
                run.sync_tree(path, model=args.side == "flush_model", rows=sink.durable_rows)
            end = time.perf_counter()
        syncs = trace.snapshot()
        peak = store_bench._peak()
        checked = run.checked(path)  # verification and digests outside write timer
    last = trace.events["last_commit"]
    close = trace.events["close_end"]
    publish_tail = sum(max(0, b - max(a, last)) for a, b in trace.publications)
    result = {
        "seconds": end - start,
        "peak": peak,
        "initialisation": initialisation,
        "syncs": syncs,
        "events": {k: v - start for k, v in trace.events.items()},
        "finishes": [[a - start, b - start] for a, b in trace.finishes],
        "tail_seconds": close - last,
        "tail_publish_seconds": publish_tail,
        "tail_wait_seconds": max(0, close - last - publish_tail),
        **checked,
    }
    if batches:
        result["batch_p50_s"] = statistics.median(batches)
        result["batch_p90_s"] = statistics.quantiles(batches, n=10)[-1]
    return result


def child(args):
    encoder, texts = prepare(args)
    if args.length:
        measure(args, encoder, texts)  # same-side warm-up, cleaned up before timed store
    return measure(args, encoder, texts)


def dirty_probe(target, file_fd, path):
    if target == "file":
        os.pwrite(file_fd, b"y" * 65536, 0)
    else:
        entry = path / "entry"
        entry.unlink() if entry.exists() else entry.touch()


def probe_samples(operations, durable, file_fd, dir_fd, path):
    values = {}
    for _ in range(50):
        for name, operation in operations.items():
            for target, descriptor in (("file", file_fd), ("dir", dir_fd)):
                durable(file_fd)
                durable(dir_fd)
                dirty_probe(target, file_fd, path)
                begin = time.perf_counter()
                operation(descriptor)
                values.setdefault(f"{name}_{target}", []).append(time.perf_counter() - begin)
    return values


def probe(args):
    import fcntl

    operations = {"fsync": os.fsync}
    if sys.platform == "darwin":
        operations |= {
            "full": lambda fd: fcntl.fcntl(fd, fcntl.F_FULLFSYNC),
            "barrier": lambda fd: fcntl.fcntl(fd, 85),
        }
    durable = operations.get("full", os.fsync)
    values = {}
    with (
        tempfile.TemporaryDirectory(dir=args.tmp) as tmp,
        (Path(tmp) / "data").open("w+b") as file,
    ):
        file.write(b"x" * 65536)
        file.flush()
        fd = os.open(tmp, os.O_RDONLY)
        try:
            values = probe_samples(operations, durable, file.fileno(), fd, Path(tmp))
        finally:
            os.close(fd)
    return {
        k: {
            "p50_ms": 1000 * statistics.median(v),
            "p90_ms": 1000 * statistics.quantiles(v, n=10)[-1],
        }
        for k, v in values.items()
    }


def valid_round(round_, sides, limit):
    return set(round_) == set(sides) and all(
        r["before"] < limit and r["after"] < limit for r in round_.values()
    )


def child_command(args, cell, side):
    cmd = [
        sys.executable,
        __file__,
        "child",
        "--tmp",
        args.tmp,
        "--side",
        side,
        "--workers",
        str(args.workers),
    ]
    for key, value in cell.items():
        if key == "name" or value is None:
            continue
        cmd += [f"--{key.replace('_', '-')}"] + ([] if value is True else [str(value)])
    if args.corpus:
        cmd += ["--corpus", args.corpus]
    return cmd


def run_cell(args, cell, deadline):
    sides = list(SIDES)
    if not cell.get("batch"):
        sides += ["large4", "single"]
    if cell.get("length"):
        sides += ["pend1g", "pend1g_n"]
    kept, discarded = [], []
    attempt = 0
    while len(kept) < args.pairs and time.monotonic() < deadline:
        round_ = {}
        for side in harness.order(sides, attempt):
            before = harness.load()
            if (
                before >= args.max_load
                or time.monotonic() >= deadline
                or (Path(args.tmp) / "pause-local.marker").exists()
            ):
                break
            cmd = child_command(args, cell, side)
            result = harness.run_child(cmd, timeout=max(1, deadline - time.monotonic()))
            round_[side] = result | {"before": before, "after": harness.load()}
            print(json.dumps({"cell": cell["name"], "side": side, **round_[side]}), flush=True)
        (kept if valid_round(round_, sides, args.max_load) else discarded).append(round_)
        if not round_:
            time.sleep(min(10, max(0, deadline - time.monotonic())))
        attempt += 1
    return {"name": cell["name"], "kept": kept, "discarded": discarded}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["cells", "child", "probe"])
    p.add_argument("--tmp", required=True)
    p.add_argument("--json")
    p.add_argument(
        "--side", choices=[*SIDES, "large4", "single", "pend1g", "pend1g_n"], default="full"
    )
    p.add_argument("--length", action="store_true")
    p.add_argument("--corpus")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--sentences", type=int, default=250_000)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--shard-rows", type=int)
    p.add_argument("--dtype", choices=["float32", "float16", "int8"], default="float32")
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--max-load", type=float, default=5)
    p.add_argument(
        "--budget", type=float, default=1200, help="shared seconds, reserve 600 for profiles"
    )
    p.add_argument("--only", nargs="+")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode != "cells":
        result = child(args) if args.mode == "child" else probe(args)
    else:
        deadline = time.monotonic() + args.budget
        cells = []
        result = {"python": sys.version, "platform": platform.platform(), "cells": cells}
        for cell in plan():
            if args.only and cell["name"] not in args.only:
                continue
            cells.append(run_cell(args, cell, deadline))
            result["decision"] = decision(cells)
            if args.json:
                Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
