#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Tables of another dtype for roofline.py: an encoder's stages 1 to 4 with its table
stored as float16 or int8, and the paired comparison of two dtypes end to end.

``Encoder`` refuses float16 and int8 tables for wordpiece@1 and unigram@1
(``fse._encoder._format_tokenizer.check_tokenizer_table``: model2vec computes the mean of a
float16 table in float16, so only float32 keeps the bound of its comparison), while the
runtime and the kernels take them. ``SpecEncoder`` therefore runs the encoder's spec with
the quantized table on a ``Pipeline`` of its own: ``encode`` is ``Pipeline.infer`` of a
``StringSource``, which is what ``Encoder.encode`` does for a list of str, without its
argument checks. A comparison runs every dtype this way, float32 included.
"""

from __future__ import annotations

import os
import statistics
import time


class SpecEncoder:
    """The parts of an ``Encoder`` that roofline.py uses, with the table as ``dtype``."""

    def __init__(self, enc, dtype: str):
        from fse._runtime import EncodeSpec, Pipeline

        spec = enc._spec
        wv = spec.word_vectors
        self.word_vectors = wv if dtype == wv.dtype else wv.quantize(dtype)
        self.output_dim = enc.output_dim
        self.steps = enc.steps
        self._cpu = Pipeline(
            EncodeSpec(self.word_vectors, spec.weights, spec.steps, spec.tokenizer)
        )

    def _pipeline(self, backend: str):
        if backend != "cpu":
            raise ValueError("SpecEncoder runs on the cpu backend only")
        return self._cpu

    def encode(self, texts, workers: int | None = None):
        from fse._text import StringSource

        return self._cpu.infer(StringSource(texts), workers=workers)


def _best(fn, passes: int) -> float:
    best = float("inf")
    for _ in range(passes):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _round(
    encoders: dict, sentences: list[str], workers: int, order: list[str], passes: int
) -> dict:
    return {d: _best(lambda d=d: encoders[d].encode(sentences, workers), passes) for d in order}


def compare(
    enc,
    sentences: list[str],
    dtypes: list[str],
    workers: list[int],
    rounds: int,
    passes: int,
    max_load: float,
) -> list[dict]:
    """Per worker count, the seconds of each dtype in rounds of alternating order (the best
    of ``passes`` each), rounds above ``max_load`` before or after discarded, and the
    median and range of ``seconds of dtypes[0] / seconds of dtype``: above 1 the dtype is
    faster than the first."""
    encoders = {d: SpecEncoder(enc, d) for d in dtypes}
    for e in encoders.values():
        e.encode(sentences, 1)  # warm
    results = []
    for w in workers:
        kept, discarded = [], 0
        for attempt in range(3 * rounds):
            if len(kept) == rounds:
                break
            before = os.getloadavg()[0]
            got = _round(
                encoders, sentences, w, dtypes if attempt % 2 == 0 else dtypes[::-1], passes
            )
            if max(before, os.getloadavg()[0]) > max_load:
                discarded += 1
                continue
            kept.append(got)
        results.append(_summary(kept, dtypes, w, discarded))
    return results


def _summary(kept: list[dict], dtypes: list[str], workers: int, discarded: int) -> dict:
    out = {"workers": workers, "rounds": len(kept), "discarded": discarded}
    if not kept:
        return out
    base = dtypes[0]
    for d in dtypes:
        ratios = [r[base] / r[d] for r in kept]
        out[d] = {
            "seconds": statistics.median(r[d] for r in kept),
            "ratio": statistics.median(ratios),
            "range": (min(ratios), max(ratios)),
        }
    return out


def lines(results: list[dict], dtypes: list[str], tokens: int) -> list[str]:
    out = []
    for r in results:
        head = f"workers {r['workers']}: {r['rounds']} rounds ({r['discarded']} discarded)"
        if r["rounds"] == 0:
            out.append(head)
            continue
        for d in dtypes:
            x = r[d]
            head += f"; {d} {tokens / x['seconds'] / 1e6:.1f} M tokens/s"
            if d != dtypes[0]:
                low, high = x["range"]
                head += f", {dtypes[0]} time / {d} time {x['ratio']:.3f} ({low:.3f} to {high:.3f})"
        out.append(head)
    return out
