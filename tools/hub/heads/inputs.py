"""Pinned public sources, deterministic sampling and bounded local work."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("civil-comments.json")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(path=CONFIG):
    """Only the reviewed protocol is supported; changes require a new recipe version."""
    config = read(path)
    if config != read(CONFIG):
        raise ValueError("configuration differs from the reviewed civil-comments-head@1 protocol")
    return config


def external(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("data and models must be outside the checkout")
    return path


class Budget:
    """Check at safe operation boundaries, including every vectorization batch."""

    def __init__(self, work, config):
        self.work, self.config = work, config
        stamp = work / "started.json"
        if not stamp.exists():
            write(stamp, {"unix": time.time()})
        self.started = read(stamp)["unix"]

    def remaining(self):
        now = time.time()
        local = time.localtime(now)
        cutoff = time.mktime((*local[:3], 18, 0, 0, *local[6:]))
        return min(self.config["budget_seconds"] - (now - self.started), cutoff - now)

    def rss(self):
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value * (1 if sys.platform == "darwin" else 1024)

    @contextmanager
    def monitor(self):
        """A memory overrun terminates the process, preserving its incomplete files."""
        stop = threading.Event()

        def watch():
            while not stop.wait(0.1):
                if self.rss() > self.config["maximum_rss_bytes"]:
                    os.write(
                        2,
                        b"inconclusive: memory watchdog exceeded configured RSS limit; no retry\n",
                    )
                    os._exit(75)

        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()

    def admit(self, encoder, train_rows, validation_rows, *, inplace=False):
        table = encoder._spec.pool_input().vectors
        matrix = train_rows * encoder.output_dim * 8
        # Includes scaler's temporary variance arrays, validation, table and 2 GiB
        # for Python texts, targets, tokenizer, optimizer and transient imports.
        estimate = (
            (1 if inplace else 3) * matrix
            + validation_rows * encoder.output_dim * 8
            + table.nbytes
            + 2 * 1024**3
        )
        receipt = {
            "estimated_peak_bytes": estimate,
            "limit_bytes": self.config["maximum_rss_bytes"],
            "inplace_scaling": inplace,
        }
        if estimate > self.config["maximum_rss_bytes"]:
            raise RuntimeError("inconclusive: memory plan exceeds 8 GiB")
        self.check()
        return receipt

    def check(self):
        if self.remaining() <= 0 or time.localtime().tm_hour >= 18:
            raise RuntimeError("inconclusive: wall-time budget or 18:00 cutoff reached")
        if shutil.disk_usage(self.work).free < self.config["minimum_free_bytes"]:
            raise RuntimeError("inconclusive: fewer than 40 GiB free")
        # macOS reports bytes, Linux KiB. This recipe runs on either platform.
        if self.rss() > self.config["maximum_rss_bytes"]:
            raise RuntimeError("inconclusive: peak RSS exceeded configured RSS limit")


def source(work, spec, filename, dataset=False):
    """Reuse existing immutable public snapshots read-only, otherwise download locally."""
    kind = "datasets" if dataset else "models"
    cached = (
        Path.home()
        / ".cache/huggingface/hub"
        / (kind + "--" + spec["repo"].replace("/", "--"))
        / "snapshots"
        / spec["revision"]
        / filename
    )
    if cached.is_file():
        return cached
    return Path(
        hf_hub_download(
            spec["repo"],
            filename,
            revision=spec["revision"],
            repo_type="dataset" if dataset else "model",
            token=False,
            cache_dir=work / "hub",
        )
    )


def sample(rows, count, seed):
    """Smallest SHA-256(seed:position), then original row order (hash-sample@1)."""
    if not 0 < count <= rows:
        raise ValueError("invalid sample count")
    hashes = [hashlib.sha256(f"{seed}:{i}".encode()).digest() for i in range(rows)]
    order = sorted(range(rows), key=hashes.__getitem__)[:count]
    return np.array(sorted(order), dtype=np.int64)


def digest_positions(positions):
    return hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes()).hexdigest()


def coverage(y):
    positive = y.sum(axis=0)
    if ((positive == 0) | (positive == len(y))).any():
        raise ValueError("every label needs positives and negatives; undefined metrics abort")
    return {"rows": len(y), "positive": positive.tolist(), "negative": (len(y) - positive).tolist()}


def vectors(encoder, texts, path, config, budget):
    """Disk-backed float64 vectors avoid an extra full-size copy during sklearn fitting."""
    first = encoder.encode(texts[:1])
    result = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float64, shape=(len(texts), first.shape[1])
    )
    for start in range(0, len(texts), config["batch_rows"]):
        budget.check()
        stop = min(start + config["batch_rows"], len(texts))
        result[start:stop] = encoder.encode(texts[start:stop])
    result.flush()
    return result
