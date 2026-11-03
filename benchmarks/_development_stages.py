"""Tokenizer and pooling stages through existing text/backend contracts."""

from __future__ import annotations

from pathlib import Path

from _development_process import checked
from _measurement import document, sha256


def inputs(job):
    from fse import Encoder

    artifact = job["artifact"]
    corpus = job["corpus"]
    model = Encoder.load(checked(artifact["path"], artifact["sha256"]), mmap=True)
    data = document(checked(corpus["path"], corpus["sha256"]))
    source = data["timing_texts"]
    if not source or job["texts"] % len(source):
        raise ValueError("timing rows must repeat complete corpus cycles")
    texts = source * (job["texts"] // len(source))
    return model, texts


def tokenizer(model, texts):
    from fse._text import Chunk, StringSource, make_lookup, new_workspace

    source = StringSource(texts)
    source.scan(workers=1)
    lookup = make_lookup(model.word_vectors, model._spec.tokenizer)
    workspace = new_workspace()
    chunk = Chunk(0, len(texts), 0, len(texts))
    return lambda: source.encode(chunk, lookup, workspace)


def pooling(model, job):
    import numpy as np

    from fse._backends._cpu import CpuBackend
    from fse._contract import Batch

    record = job["batch"]
    offsets = np.load(
        checked(record["offsets"]["path"], record["offsets"]["sha256"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    tokens = np.load(
        checked(record["tokens"]["path"], record["tokens"]["sha256"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    if len(offsets) != job["texts"] + 1:
        raise ValueError("fixed pooling batch row count differs")
    batch = Batch(0, offsets, tokens)
    backend = CpuBackend()
    prepared = backend.prepare(model._spec.pool_input(), ())
    output = np.empty((job["texts"], model.output_dim), dtype=np.float32)

    def operation():
        backend.rows(batch, prepared, output)
        return output

    return operation


def prepare_batch(job):
    import numpy as np

    model, texts = inputs(job)
    batch = tokenizer(model, texts)()
    result = {}
    for name in ("offsets", "tokens"):
        path = Path(job["work"]) / (job["tag"] + "-" + name + ".npy")
        if path.exists():
            raise FileExistsError(path)
        np.save(path, getattr(batch, name), allow_pickle=False)
        result[name] = dict(path=str(path), sha256=sha256(path))
    return result
