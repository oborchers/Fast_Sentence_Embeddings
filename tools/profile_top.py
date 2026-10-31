#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The functions that take the time in a samply profile, in the terminal.

Reads a profile that ``samply record --save-only --unstable-presymbolicate -o p.json``
wrote (``make profile-cpu`` does) with its ``p.syms.json`` and prints, over all threads
and for the busiest threads, the functions with the most samples of their own (self) and
in their call stacks (total), so the hot functions of a benchmark can be named without
opening a browser.

    uv run python tools/profile_top.py profiles/cpu.json --top 25 --threads 4
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter
from pathlib import Path


class Symbols:
    """The symbol tables of the libraries of a profile, by library name."""

    def __init__(self, syms: dict):
        strings = syms["string_table"]
        self.tables: dict[str, tuple[list[int], list[tuple[int, int, str]]]] = {}
        for lib in syms["data"]:
            entries = sorted(
                (e["rva"], e["size"], strings[e["symbol"]]) for e in lib["symbol_table"]
            )
            self.tables[lib["debug_name"]] = ([e[0] for e in entries], entries)

    def name(self, lib: str, address: int) -> str:
        starts, entries = self.tables.get(lib, ([], []))
        i = bisect.bisect_right(starts, address) - 1
        if i >= 0 and address < entries[i][0] + max(entries[i][1], 1):
            return entries[i][2]
        return f"{lib}+{address:#x}"


def thread_counts(thread: dict, libs: list[dict], symbols: Symbols) -> tuple[Counter, Counter, int]:
    """Self and total samples per function name of one thread, and its samples."""
    frames, funcs, stacks = thread["frameTable"], thread["funcTable"], thread["stackTable"]
    resources = thread["resourceTable"]
    names = []
    for f in range(frames["length"]):
        resource = funcs["resource"][frames["func"][f]]
        lib = libs[resources["lib"][resource]]["debugName"] if resource >= 0 else "?"
        names.append(symbols.name(lib, frames["address"][f]))
    own, total = Counter(), Counter()
    samples = [s for s in thread["samples"]["stack"] if s is not None]
    for stack in samples:
        own[names[stacks["frame"][stack]]] += 1
        seen = set()
        while stack is not None:
            name = names[stacks["frame"][stack]]
            if name not in seen:
                total[name] += 1
                seen.add(name)
            stack = stacks["prefix"][stack]
    return own, total, len(samples)


def _print(title: str, own: Counter, total: Counter, n: int, top: int) -> None:
    print(f"\n== {title}: {n} samples")
    print(f"   {'self':>6} {'total':>6}  function")
    for name, count in own.most_common(top):
        print(f"   {100 * count / n:5.1f}% {100 * total[name] / n:5.1f}%  {name[:110]}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("profile")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--threads", type=int, default=3, help="the busiest threads shown alone")
    args = parser.parse_args(argv)
    path = Path(args.profile)
    profile = json.loads(path.read_text(encoding="utf-8"))
    symbols = Symbols(
        json.loads(path.with_name(path.stem + ".syms.json").read_text(encoding="utf-8"))
    )
    all_own, all_total, all_n = Counter(), Counter(), 0
    per_thread = []
    for thread in profile["threads"]:
        own, total, n = thread_counts(thread, profile["libs"], symbols)
        all_own.update(own)
        all_total.update(total)
        all_n += n
        per_thread.append((n, thread.get("name", "?"), own, total))
    _print("all threads", all_own, all_total, max(all_n, 1), args.top)
    for n, name, own, total in sorted(per_thread, key=lambda t: -t[0])[: args.threads]:
        _print(f"thread {name}", own, total, max(n, 1), min(args.top, 12))


if __name__ == "__main__":
    main()
