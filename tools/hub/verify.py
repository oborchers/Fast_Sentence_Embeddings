"""Runtime checks for a downloaded classic model and its recorded 512-sentence probe."""

from __future__ import annotations

import re
import subprocess

import numpy as np
from scipy.stats import pearsonr

import fse
from fse._container._hash import checksum

STS_COMMIT = "fc3cbfb0509f56dfe2374c4c94fd68024f8810d6"
STS_CHECKSUM = "xxh3-128:139bba1402050737ae635ecf8498280b"


def runtime(folder, entry, *, revision=None, cache=None):
    card = (folder / "README.md").read_text()
    blocks = re.findall(r"```python\r?\n(.*?)\r?\n```", card, flags=re.DOTALL)
    if blocks != [entry["python_example"], entry["folded_head_python_example"]]:
        raise ValueError("downloaded card examples differ from the recorded examples")
    source = folder if revision is None else entry["repo"]
    hub = {} if revision is None else {"revision": revision, "token": False, "cache_dir": cache}
    encoder = fse.Encoder.from_pretrained(source, mmap=True, **hub)
    metadata = entry["metadata"]
    if (
        encoder.fingerprint != metadata["fingerprint"]
        or encoder.output_dim != metadata["dimension"]
        or len(encoder.word_vectors) != metadata["vocabulary_size"]
        or encoder.word_vectors.stored_vectors.dtype.name != metadata["dtype"]
    ):
        raise ValueError("readback runtime metadata differs")
    raw = fse.Encoder.from_pretrained(source, mmap=True, normalize=False, **hub)
    raw_vectors = raw.encode(["A sentence."], workers=1)
    if raw_vectors.shape != (1, encoder.output_dim) or not np.isfinite(raw_vectors).all():
        raise ValueError("normalize=False example failed")
    # The Hub mirror rounds some labels. Keep the publication's original fixture.
    fixture = subprocess.check_output(["git", "show", f"{STS_COMMIT}:evaluation/sts-test.csv"])
    if checksum(fixture) != STS_CHECKSUM:
        raise ValueError("historical STS-B probe fixture checksum differs")
    rows = [line.split("\t") for line in fixture.decode().splitlines()]
    probe = entry["probe"]
    selected = [rows[index] for index in probe["pair_indices"]]
    texts = [row[5] for row in selected] + [row[6] for row in selected]
    vectors = encoder.encode(texts, workers=2)
    if (
        vectors.shape != (probe["sentences"], encoder.output_dim)
        or vectors.dtype != np.dtype("float32")
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("readback probe shape, dtype or finite values differ")
    actual = checksum(vectors)
    if actual != probe["vectors_xxh3"]:
        raise ValueError("readback probe differs bit for bit")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    half = len(selected)
    similarities = (normalized[:half] * normalized[half:]).sum(axis=1)
    pearson = float(pearsonr([float(row[4]) for row in selected], similarities).statistic * 100)
    if probe["tolerance_points"] != 0.001 or abs(pearson - probe["pearson_x100"]) > 0.001:
        raise ValueError("readback Pearson differs from the recorded probe")
    return {
        "vectors_xxh3": actual,
        "pearson_x100": pearson,
        "bitwise_ok": True,
        "default_and_normalize_false_load_ok": True,
    }
