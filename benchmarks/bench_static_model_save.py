#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Save and reload of ``fse.StaticModel`` (``save_pretrained``, ``from_pretrained``) for a
model of the Hub: wall time of each, and bit identity of the rows after the reload.

    uv run python benchmarks/bench_static_model_save.py [repo] [--repeat N]

Not a hot path (no row is computed by it): it is run under ``make profile-cpu``,
``profile-mem`` and ``profile-rss`` to name where the time and the memory go.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import fse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", default="minishlab/potion-base-32M")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    model = fse.StaticModel.from_pretrained(args.repo)
    model.max_length = 64
    texts = ["the quick brown fox jumps over the lazy dog"] * 64
    want = model.encode(texts).tobytes()
    size = model.embedding.nbytes / 2**20
    print(f"{args.repo}: {len(model.tokens)} tokens, {model.dim} dimensions, {size:.1f} MiB table")
    for _ in range(args.repeat):
        with tempfile.TemporaryDirectory() as tmp:
            start = time.perf_counter()
            folder = model.save_pretrained(Path(tmp) / "saved")
            saved = time.perf_counter()
            again = fse.StaticModel.from_pretrained(folder)
            loaded = time.perf_counter()
            assert again.encode(texts).tobytes() == want
            print(f"save {1e3 * (saved - start):7.1f} ms   reload {1e3 * (loaded - saved):7.1f} ms")


if __name__ == "__main__":
    main()
