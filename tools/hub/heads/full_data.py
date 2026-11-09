"""Stream the full pinned training split into one float32 working matrix."""

from __future__ import annotations

import gc

import numpy as np
import pyarrow.parquet as pq
from inputs import coverage, source, write
from sklearn.preprocessing import StandardScaler

import fse


def raw_vectors(parent, work, config, budget):
    encoder = fse.Encoder.from_pretrained(parent / "bekko", normalize=False)
    shape = (config["train_rows"], config["dimension"])
    path = work / "train.npy"
    initial = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    del initial
    target = np.empty((shape[0], len(config["labels"])), dtype=bool)
    offset = 0
    for shard in range(2):
        filename = f"data/train-{shard:05d}-of-00002.parquet"
        parquet = pq.ParquetFile(source(parent, config["dataset"], filename, True))
        columns = ["text", *config["labels"]]
        for batch in parquet.iter_batches(
            batch_size=config["batch_rows"], columns=columns, use_threads=False
        ):
            budget.check()
            texts = batch.column("text").to_pylist()
            stop = offset + len(texts)
            if stop > shape[0]:
                raise ValueError("training input exceeds the declared row count")
            values = np.column_stack([batch.column(label).to_numpy() for label in config["labels"]])
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("training annotator fractions are invalid")
            target[offset:stop] = values >= config["target_threshold"]
            encoded = encoder.encode(texts)
            # Releasing the mapping each batch keeps embedding RSS independent of row count.
            mapped = np.load(path, mmap_mode="r+")
            mapped[offset:stop] = encoded
            mapped.flush()
            del mapped, encoded, texts, values
            offset = stop
        parquet.close()
        del parquet
    if offset != shape[0]:
        raise ValueError("training input does not match the declared row count")
    del encoder
    gc.collect()
    np.save(work / "targets.npy", target)
    write(work / "coverage.json", coverage(target))
    budget.check()


def standardize(work, config, budget):
    path = work / "train.npy"
    scaler = StandardScaler(copy=True)
    mapped = np.load(path, mmap_mode="r+")
    if mapped.dtype != np.float32:
        raise ValueError("full training working matrix must stay float32")
    for start in range(0, len(mapped), config["batch_rows"]):
        budget.check()
        scaler.partial_fit(mapped[start : start + config["batch_rows"]])
    for start in range(0, len(mapped), config["batch_rows"]):
        budget.check()
        scaler.transform(mapped[start : start + config["batch_rows"]], copy=False)
    mapped.flush()
    del mapped
    gc.collect()
    budget.check()
    return scaler
