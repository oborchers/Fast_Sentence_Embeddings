#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The resident memory of a command and all its child processes over time.

Runs the command, samples the RSS of its whole process tree every --interval seconds
with ``ps`` (no dependency; benchmarks start child processes, so the tree is what
counts), and prints the peak, the time of the peak and the final value; --csv writes
every sample. Memory-mapped file pages count as far as the system counts them as
resident.

    uv run python tools/rss.py --csv rss.csv -- python benchmarks/bench_store.py write
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time


def tree_rss(root: int) -> int:
    """The RSS in bytes of ``root`` and every process below it."""
    out = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, check=True
    )
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in out.stdout.splitlines():
        pid, ppid, kib = (int(field) for field in line.split())
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kib * 1024
    total, todo = 0, [root]
    while todo:
        pid = todo.pop()
        total += rss.get(pid, 0)
        todo.extend(children.get(pid, []))
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--csv")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command to measure after --")
    start = time.perf_counter()
    process = subprocess.Popen(command)
    samples: list[tuple[float, int]] = []
    while process.poll() is None:
        samples.append((time.perf_counter() - start, tree_rss(process.pid)))
        time.sleep(args.interval)
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["seconds", "rss_bytes"])
            writer.writerows(samples)
    if samples:
        when, peak = max(samples, key=lambda s: s[1])
        print(
            f"# rss: peak {peak / 2**20:.0f} MiB at {when:.2f} s, last {samples[-1][1] / 2**20:.0f} MiB, "
            f"{len(samples)} samples over {samples[-1][0]:.2f} s",
            file=sys.stderr,
        )
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
