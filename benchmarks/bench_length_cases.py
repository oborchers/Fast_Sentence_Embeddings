#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The cases of bench_length.py, each run in a child process of its own.

    python benchmarks/bench_length_cases.py '{"side": "v2", "kind": "encode", ...}'

prints the measurements of one case as one line of JSON. bench_length.py starts it with
the Python of fse 2.0 (``side`` v2) or of fse 1.1.0 (v1), so this file imports only the
standard library, numpy and bench_lifecycle_cases (the same) at the top. Every path comes
from the parent in the config. The cases:

- ``model``: untimed, the frozen model every length is encoded with: 2.0 builds the
  finished encoder (SIFWeights: the table with the SIF weights of the wordfreq counts
  folded in and no components removed, ``bench_lifecycle_cases.v2_finished``; Average
  with or without normalize) and saves it; 1.1 trains its model on ``paragraphs`` and
  saves it without sentence vectors, as a model from the hub comes;
- ``encode``: one warm-up and ``repeats`` timed encodes of the corpus file, the median,
  minimum and maximum: 2.0 ``Encoder.encode(path)`` into memory or into a ``ShardSink``
  up to its last flush; 1.1 ``train(update=True)`` on a ``CorpusFile`` (its bulk path)
  and for a store ``sv.save``, its model loaded again before every encode (untimed), as
  ``update=True`` appends rows. 2.0 also reports the chunks the pipeline cuts (untimed,
  with its own chunk parameters) and an estimate of the Workspace of the worker that
  encodes the longest text;
- ``strings``: 2.0 only, the texts the parent selected (the longest document, or the
  first 32 documents) as strings, each in a child of its own: the p50 latency and the
  peak RSS;
- ``prepare`` and ``quantize``: untimed, the word vectors.

The peak RSS (``ru_maxrss``) is that of the whole child, warm-up included.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np
from bench_lifecycle_cases import (
    _p50_ms,
    _peak,
    _timed,
    v1_counts,
    v2_finished,
    v2_prepare,
)

BATCH_DOCUMENTS = 32
# The bytes Workspace grows to for a text of n bytes (TextLookup._sentence): cand 2n and
# starts and ends 8n each.
WORKSPACE_PER_BYTE = 2 + 8 + 8


def _summary(times: list[float]) -> dict:
    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "times": times,
    }


# --- 2.0 ------------------------------------------------------------------------------


def _save(artifact, path: Path) -> None:
    """Save into a new file next to ``path``, then replace it."""
    tmp = path.with_suffix(".part.safetensors")
    tmp.unlink(missing_ok=True)
    artifact.save(tmp)
    os.replace(tmp, path)


def v2_quantize(c: dict) -> dict:
    """The float32 word vectors quantized to ``wv_dtype`` (untimed)."""
    import fse

    wv = fse.WordVectors.load(c["wv_float32"], mmap=True)
    _save(wv.quantize(c["wv_dtype"]), Path(c["wv"]))
    return {"words": len(wv)}


def v2_model(c: dict) -> dict:
    """Build and save the finished encoder (untimed)."""
    import fse

    wv = fse.WordVectors.load(c["wv"], mmap=True)
    normalize = c["model"] == "AverageNorm"
    encoder = v2_finished(
        wv, c["model"], c["counts"], tokenizer=c["tokenizer"], normalize=normalize
    )
    _save(encoder, Path(c["encoder"]))
    return {"fingerprint": encoder.fingerprint}


def v2_encoder(c: dict):
    import fse

    return fse.Encoder.load(c["encoder"])  # it holds its table


def _into_memory(encoder, path: Path, c: dict):
    if c["entry"] == "encode":
        return encoder.encode(path, dtype=c["dtype"], workers=c["workers"])
    # The ablation of --chunk-bytes (and its baseline "auto"): the pipeline with
    # chunk_bytes, which Encoder.encode does not take (it calls the same infer).
    from fse._text_file import FileSource

    with FileSource(path) as source:
        return encoder._pipeline("cpu").infer(
            source, workers=c["workers"], dtype=c["dtype"], chunk_bytes=c["chunk_bytes"]
        )


def _into_sink(encoder, path: Path, sink, c: dict) -> None:
    if c["entry"] == "encode":
        encoder.encode(path, workers=c["workers"], sink=sink)
        return
    from fse._text_file import FileSource

    with FileSource(path) as source:
        encoder._pipeline("cpu").infer(
            source, workers=c["workers"], sink=sink, chunk_bytes=c["chunk_bytes"]
        )
    sink.flush()


def v2_run(encoder, c: dict) -> float:
    """The seconds of one encode of the corpus, into memory (the rows held until after the
    timer) or into a new store up to its last flush."""
    path = Path(c["corpus"])
    if c["target"] == "memory":
        seconds, rows = _timed(lambda: _into_memory(encoder, path, c))
        del rows
        return seconds
    from fse import ShardSink  # imported outside the timer

    with tempfile.TemporaryDirectory(dir=c["tmp"]) as run:

        def write():
            with ShardSink(Path(run) / "store", encoder, dtype=c["dtype"]) as sink:
                _into_sink(encoder, path, sink, c)

        return _timed(write)[0]


def _percentile(values: list[int], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


# The lines of the task plan that ``_chunks`` repeats, as the source has them: if one is
# gone, the plan changed and ``_chunks`` must follow (there is no function to call).
PLAN = {
    "Pipeline.infer": (
        "inverse = wants_inverse_norms(sink)",
        "row_bytes = self._row_bytes(dtype, inverse)",
        "window = max(1, min(2 * workers, WINDOW_BYTES // row_bytes))",
        "task_rows = max(1, min(self.task_rows, WINDOW_BYTES // (window * row_bytes)))",
        "chunk_rows=task_rows,",
    ),
    "plan": (
        "inline = source.nbytes <= INLINE_MAX_BYTES",
        "workers=1 if inline else workers, chunk_bytes=chunk_bytes, chunk_rows=chunk_rows",
    ),
}


def check_plan() -> None:
    """Fail loudly if the task plan of the runtime is no longer the one ``_chunks`` uses."""
    import inspect

    from fse._runtime import _pipeline, _tasks

    functions = {"Pipeline.infer": _pipeline.Pipeline.infer, "plan": _tasks.plan}
    for name, lines in PLAN.items():
        source = " ".join(inspect.getsource(functions[name]).split())
        if missing := [line for line in lines if line not in source]:
            raise RuntimeError(f"{name} changed its task plan ({missing}): update _chunks")


def _chunks(encoder, c: dict) -> dict:
    """The chunks of the corpus as the pipeline cuts them, untimed: the window, the rows
    per task and the inline limit of Pipeline.infer and plan (checked by ``check_plan``),
    with their constants imported. There is no per-task timing; these explain the
    balance."""
    from fse import ShardSink
    from fse._runtime._pipeline import WINDOW_BYTES
    from fse._runtime._tasks import INLINE_MAX_BYTES
    from fse._text_file import FileSource

    check_plan()
    pipeline = encoder._pipeline("cpu")
    inverse = c["target"] == "store" and ShardSink.inverse_norms
    row_bytes = pipeline._row_bytes(c["dtype"], inverse)
    window = max(1, min(2 * c["workers"], WINDOW_BYTES // row_bytes))
    task_rows = max(1, min(pipeline.task_rows, WINDOW_BYTES // (window * row_bytes)))
    with FileSource(Path(c["corpus"])) as source:
        inline = source.nbytes <= INLINE_MAX_BYTES
        scan = source.scan(
            workers=1 if inline else c["workers"],
            chunk_bytes=c.get("chunk_bytes"),
            chunk_rows=task_rows,
        )
    rows = [chunk.n_sentences for chunk in scan.chunks]
    sizes = [chunk.end - chunk.start for chunk in scan.chunks]
    return {
        "chunks": len(rows),
        "chunk_rows_p50": _percentile(rows, 50),
        "chunk_rows_max": max(rows, default=0),
        "chunk_bytes_p50": _percentile(sizes, 50),
        "chunk_bytes_max": max(sizes, default=0),
        "chunks_per_worker": len(rows) / c["workers"],
        "task_rows": task_rows,
        "window": window,
    }


def _workspace(chunks: dict, c: dict) -> dict:
    """An estimate, not a measurement (the Workspace does not expose its capacity): the
    buffers of the one worker that encodes the longest text, for that text and the tokens
    and offsets of the largest chunk (at most a token per 2 bytes), without the reserve
    of doubling."""
    with open(c["corpus"], "rb") as f:
        longest = max((len(line) for line in f), default=0)
    estimate = (
        WORKSPACE_PER_BYTE * longest
        + 4 * chunks["chunk_bytes_max"] // 2
        + 8 * (chunks["chunk_rows_max"] + 1)
    )
    return {"longest_text_bytes": longest, "workspace_estimate_bytes": estimate}


def v2_encode(c: dict) -> dict:
    out: dict = {"peak_before": _peak()}
    encoder = v2_encoder(c)
    v2_run(encoder, c)  # the warm-up
    out.update(_summary([v2_run(encoder, c) for _ in range(c["repeats"])]))
    out["peak"] = _peak()
    chunks = _chunks(encoder, c)
    return {**out, **chunks, **_workspace(chunks, c)}


def v2_strings(c: dict) -> dict:
    """The texts the parent selected (the longest document, or the first documents) as
    strings: the p50 latency and the peak RSS of this child."""
    encoder = v2_encoder(c)
    texts = Path(c["texts"]).read_text(encoding="utf-8").splitlines()
    out: dict = {"texts": len(texts), "tokens": sum(len(t.split()) for t in texts)}
    out["peak_before"] = _peak()
    encoder.encode(texts, workers=c["workers"])  # the warm-up
    out["p50_ms"] = _p50_ms(
        lambda: encoder.encode(texts, dtype="float32", workers=c["workers"]), max(5, c["repeats"])
    )
    out["peak"] = _peak()
    return out


# --- 1.1 ------------------------------------------------------------------------------


def v1_model(c: dict) -> dict:
    """Train the 1.1 model on the paragraphs and save it without its sentence vectors
    (untimed), as a model from the hub comes."""
    import fse

    vectors = fse.Vectors.from_pretrained(c["vectors"], mmap="r")
    model = getattr(fse, c["model"])(vectors, workers=c["workers"], sv_dtype=c["dtype"])
    model.train(fse.CorpusFile(c["paragraphs"]))
    sv = model.sv
    sv.vectors = np.zeros((0, sv.vector_size), sv.vectors.dtype)
    if sv.scales is not None:
        sv.scales = np.zeros(0, sv.scales.dtype)
    folder = Path(c["v1_model"])
    folder.mkdir(parents=True, exist_ok=True)
    model.save(str(folder / "model"))
    return {"rows": len(sv.vectors)}


def _v1_run(c: dict) -> float:
    import fse

    model = getattr(fse, c["model"]).load(str(Path(c["v1_model"]) / "model"))  # untimed
    model.workers = c["workers"]
    sentences = fse.CorpusFile(c["corpus"])
    seconds, _ = _timed(lambda: model.train(sentences, update=True))
    if c["target"] == "store":
        with tempfile.TemporaryDirectory(dir=c["tmp"]) as run:
            seconds += _timed(lambda: model.sv.save(str(Path(run) / "sv")))[0]
    return seconds


def v1_encode(c: dict) -> dict:
    out: dict = {"peak_before": _peak()}
    _v1_run(c)  # the warm-up
    out.update(_summary([_v1_run(c) for _ in range(c["repeats"])]))
    out["peak"] = _peak()
    return out


CHILDREN = {
    ("v2", "prepare"): v2_prepare,
    ("v2", "quantize"): v2_quantize,
    ("v2", "model"): v2_model,
    ("v2", "encode"): v2_encode,
    ("v2", "strings"): v2_strings,
    ("v1", "counts"): v1_counts,
    ("v1", "model"): v1_model,
    ("v1", "encode"): v1_encode,
}


def main() -> None:
    import fse

    config = json.loads(sys.argv[1])
    result = CHILDREN[config["side"], config["kind"]](config)
    print(json.dumps({**result, "fse": fse.__version__}))


if __name__ == "__main__":
    main()
