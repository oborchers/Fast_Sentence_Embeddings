"""Seeded WikiText train passage inputs, prepared only on the Linux pod."""

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from compare_artifacts import file_digest
from compare_inputs import DATASETS
from pod_large_inputs import write_shuffled
from text_lengths import Lengths, batches, counter, token_counts

SEED = 0
LABEL = "sampled and shuffled WikiText-103 passages"


def sampled_indices(source_count, count, seed=SEED):
    if source_count < 1 or count < 1:
        raise ValueError("source and sampled input must be nonempty")
    generator = np.random.default_rng(seed)
    indices = generator.integers(source_count, size=count, dtype=np.int64)
    generator.shuffle(indices)
    return indices


def source_passages(cache):
    if sys.platform != "linux":
        raise RuntimeError("large benchmark inputs are prepared only on the Linux pod")
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    target = cache / "wikitext-train-passages.txt"
    if target.exists():
        return read_source(target)
    repo, revision = DATASETS["wikitext"]
    snapshot = Path(
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["wikitext-103-raw-v1/train-*.parquet"],
            max_workers=2,
        )
    )
    with target.open("x", encoding="utf-8") as stream:
        for path in sorted((snapshot / "wikitext-103-raw-v1").glob("train-*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=4096):
                for text in batch.column(0).to_pylist():
                    text = re.sub(r"\s+", " ", text).strip()
                    if text and not text.startswith("=") and len(text.split()) >= 5:
                        stream.write(text + "\n")
    return read_source(target)


def read_source(target):
    from pod4_memory import admit

    size = target.stat().st_size
    admit(size * 32, "complete source passage decoding and string headers")
    with target.open(encoding="utf-8") as stream:
        return [line.rstrip("\n") for line in stream]


def sampled_lengths(source, indices):
    model, tokenizer = counter()
    weights = np.bincount(indices, minlength=len(source))
    lengths, offset = Lengths(), 0
    for batch in batches(source):
        for text, tokens in zip(batch, token_counts(model, batch), strict=True):
            weight = int(weights[offset])
            if weight:
                lengths.add_weighted_counts(len(text.split()), len(text), tokens, weight)
            offset += 1
    return {
        **lengths.record(),
        "tokenizer": tokenizer,
        "statistics_method": "exact weighted full sampled input",
    }


def write_input(folder, source, count, seed=SEED):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"passages-{count}-seed{seed}.txt"
    receipt = path.with_suffix(".identity.json")
    if path.exists() or receipt.exists():
        if not path.exists() or not receipt.exists():
            raise ValueError("incomplete sampled input preparation")
        record = json.loads(receipt.read_text())
        if record["checksum"] != file_digest(path):
            raise ValueError("cached sampled input differs from its receipt")
        return path, record
    expected = count * (sum(len(text.encode("utf-8")) + 1 for text in source) / len(source))
    if shutil.disk_usage(folder).free < expected + 25 * 1024**3:
        raise OSError("sampled input would leave less than 25 GiB free")
    from pod4_memory import admit

    admit(count * 24, "sample-index arrays and unique index workspace")
    indices = sampled_indices(len(source), count, seed)
    selected = np.unique(indices)
    distinct = len({source[int(index)] for index in selected})
    digest = write_shuffled(path, source, indices)
    record = {
        **sampled_lengths(source, indices),
        "texts": count,
        "input_digest": digest,
        "checksum": file_digest(path),
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "dataset": DATASETS["wikitext"][0],
        "dataset_revision": DATASETS["wikitext"][1],
        "split": "train",
        "input_label": LABEL,
        "seed": seed,
        "sampling": "uniform source positions with replacement, followed by seeded shuffle",
        "source_passages": len(source),
        "distinct_source_texts": len(set(source)),
        "distinct_input_texts": distinct,
        "distinct_sampled_positions": len(selected),
        "repetition_factor": count / distinct,
        "numpy_version": np.__version__,
        "preprocessing": "collapse whitespace, skip headings, at least five whitespace words",
    }
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return path, record
