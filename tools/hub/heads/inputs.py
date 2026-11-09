"""Pinned public sources, deterministic sampling and bounded local work."""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
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
        return self.config["budget_seconds"] - (time.time() - self.started)

    def check(self):
        if self.remaining() <= 0 or time.localtime().tm_hour >= 18:
            raise RuntimeError("inconclusive: wall-time budget or 18:00 cutoff reached")
        if shutil.disk_usage(self.work).free < self.config["minimum_free_bytes"]:
            raise RuntimeError("inconclusive: fewer than 40 GiB free")
        # macOS reports bytes, Linux KiB. This recipe runs on either platform.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss *= 1 if sys.platform == "darwin" else 1024
        if rss > self.config["maximum_rss_bytes"]:
            raise RuntimeError("inconclusive: peak RSS exceeded 8 GiB")


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


def prepare(work, config, budget):
    for split in ("train", "validation"):
        dataset(work, config, split, budget)
    for spec in config["encoders"]:
        folder = work / spec["name"]
        folder.mkdir()
        for filename in spec["files"]:
            budget.check()
            shutil.copyfile(source(work, spec, filename), folder / filename)
        budget.check()


def dataset(work, config, split, budget):
    if split not in ("train", "validation", "test"):
        raise ValueError("unknown split")
    count = 2 if split == "train" else 1
    columns = ["text", *config["labels"]] + (["severe_toxicity"] if split == "train" else [])
    texts, targets = [], []
    omitted_positive = 0
    for shard in range(count):
        budget.check()
        name = f"data/{split}-{shard:05d}-of-{count:05d}.parquet"
        table = pq.read_table(source(work, config["dataset"], name, True), columns=columns)
        if split == "train":
            omitted_positive += int(
                (table["severe_toxicity"].to_numpy() >= config["target_threshold"]).sum()
            )
        texts.extend(table["text"].to_pylist())
        targets.append(np.column_stack([table[label].to_numpy() for label in config["labels"]]))
    values = np.concatenate(targets)
    if split == "train":
        write(
            work / "excluded-labels.json",
            {"severe_toxicity": {"positive": omitted_positive, "train_rows": len(texts)}},
        )
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("targets must be finite annotator fractions in [0, 1]")
    return texts, values >= config["target_threshold"]


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
