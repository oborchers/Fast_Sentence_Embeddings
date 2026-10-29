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

- ``model``: untimed, the frozen model every length is encoded with: 2.0 fits its recipe
  (SIF-10 on ``paragraphs`` with the wordfreq counts, Average with or without normalize)
  and saves the encoder (SIF folded); 1.1 trains its model on ``paragraphs`` and saves it
  without sentence vectors, as a model from the hub comes (its weights and components);
- ``encode``: one warm-up and ``repeats`` timed encodes of the corpus file, the median,
  minimum and maximum: 2.0 ``Encoder.encode(path)`` into memory or into a ``ShardSink``
  up to its last flush; 1.1 ``train(update=True)`` on a ``CorpusFile`` (its bulk path)
  and for a store ``sv.save``, its model loaded again before every encode (untimed), as
  ``update=True`` appends rows. 2.0 also reports the chunks the pipeline cuts (untimed,
  with its own chunk parameters) and an estimate of the Workspace of its longest text;
- ``strings``: 2.0 only, the longest document and the first 32 documents as strings, the
  p50 latency and the peak RSS.

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
from bench_lifecycle_cases import COMPONENTS, _p50_ms, _peak, _timed, v1_counts, v2_prepare

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


def _recipe(c: dict):
    import fse

    if c["model"] == "SIF":
        return fse.SIF(components=COMPONENTS, normalize=False, tokenizer=c["tokenizer"])
    return fse.Average(normalize=c["model"] == "AverageNorm", tokenizer=c["tokenizer"])


def v2_model(c: dict) -> dict:
    """Fit and save the encoder (untimed); the quantized word vectors first if missing."""
    import fse

    if not Path(c["wv"]).exists():  # only quantized vectors are made here
        source = fse.WordVectors.load(c["wv_float32"], mmap=True)
        source.quantize(c["wv_dtype"]).save(c["wv"])
    wv = fse.WordVectors.load(c["wv"], mmap=True)
    recipe = _recipe(c)
    if c["model"] == "SIF":
        counts = np.load(c["counts"])
        encoder = recipe.fit(wv, Path(c["paragraphs"]), counts=counts)
    else:
        encoder = recipe.fit(wv)
    tmp = Path(c["encoder"]).with_suffix(".part.safetensors")
    tmp.unlink(missing_ok=True)
    encoder.save(tmp)
    os.replace(tmp, c["encoder"])
    return {"folded": encoder.folded, "fingerprint": encoder.fingerprint}


def v2_encoder(c: dict):
    import fse

    path = c["encoder"]
    if c["model"] == "SIF":  # folded: the encoder holds its word vectors
        return fse.Encoder.load(path)
    return fse.Encoder.load(path, fse.WordVectors.load(c["wv"], mmap=True))


def _into_memory(encoder, path: Path, c: dict):
    if c.get("chunk_bytes") is None:
        return encoder.encode(path, dtype=c["dtype"], workers=c["workers"])
    from fse._text_file import FileSource  # the ablation: the pipeline with chunk_bytes

    with FileSource(path) as source:
        return encoder._pipeline("cpu").infer(
            source, workers=c["workers"], dtype=c["dtype"], chunk_bytes=c["chunk_bytes"]
        )


def _into_sink(encoder, path: Path, sink, c: dict) -> None:
    if c.get("chunk_bytes") is None:
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


def _chunks(encoder, c: dict) -> dict:
    """The chunks of the corpus as the pipeline cuts them (Pipeline.infer: the window, the
    rows per task, the inline limit), untimed. There is no per-task timing; these explain
    the balance."""
    from fse import ShardSink
    from fse._runtime._pipeline import WINDOW_BYTES
    from fse._runtime._tasks import INLINE_MAX_BYTES
    from fse._text_file import FileSource

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
    """An estimate of the Workspace buffers (they double on demand and are not exposed):
    per worker those of the longest text and the tokens and offsets of the largest chunk
    (at most a token per 2 bytes)."""
    with open(c["corpus"], "rb") as f:
        longest = max((len(line) for line in f), default=0)
    per_worker = (
        WORKSPACE_PER_BYTE * longest
        + 4 * chunks["chunk_bytes_max"] // 2
        + 8 * (chunks["chunk_rows_max"] + 1)
    )
    return {"longest_text_bytes": longest, "workspace_bytes": c["workers"] * per_worker}


def v2_encode(c: dict) -> dict:
    out: dict = {"peak_before": _peak()}
    encoder = v2_encoder(c)
    v2_run(encoder, c)  # the warm-up
    out.update(_summary([v2_run(encoder, c) for _ in range(c["repeats"])]))
    out["peak"] = _peak()
    chunks = _chunks(encoder, c)
    return {**out, **chunks, **_workspace(chunks, c)}


def v2_strings(c: dict) -> dict:
    """The longest document and a batch of the first documents as strings."""
    encoder = v2_encoder(c)
    lines = Path(c["corpus"]).read_text(encoding="utf-8").splitlines()
    texts = {"long": [max(lines, key=len)], f"batch{BATCH_DOCUMENTS}": lines[:BATCH_DOCUMENTS]}
    del lines
    out: dict = {"long_tokens": len(texts["long"][0].split()), "peak_before": _peak()}
    for name, batch in texts.items():
        encoder.encode(batch, workers=c["workers"])  # the warm-up

        def call(b=batch):
            return encoder.encode(b, dtype="float32", workers=c["workers"])

        out[f"{name}_ms"] = _p50_ms(call, max(5, c["repeats"]))
    out["peak"] = _peak()
    return out


# --- 1.1 ------------------------------------------------------------------------------


def v1_model(c: dict) -> dict:
    """Train the 1.1 model on the paragraphs and save it without its sentence vectors
    (untimed), as bench_lifecycle_cases.v1_hub_model does for SIF."""
    import fse

    vectors = fse.Vectors.from_pretrained(c["vectors"], mmap="r")
    kwargs = {}
    if c["model"] == "SIF":
        vectors.expandos["count"] = np.load(c["counts"])  # instead of lang_freq
        kwargs["components"] = COMPONENTS
    model = getattr(fse, c["model"])(vectors, workers=c["workers"], sv_dtype=c["dtype"], **kwargs)
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
