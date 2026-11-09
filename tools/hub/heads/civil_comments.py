"""Civil Comments adapter: pinned parquet splits and annotator-fraction targets."""

from __future__ import annotations

import shutil

import numpy as np
import pyarrow.parquet as pq
from inputs import source, write


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
