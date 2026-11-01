"""Large, bounded-memory speed demonstration for recipe 00, not a benchmark."""

import json
import os
import platform
import re
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from itertools import islice
from pathlib import Path

from _helpers import ROOT

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def sentence_file(recipe, limit=2_000_000):
    """Download real WikiText-103 train rows and split at punctuation plus whitespace."""
    output = recipe.work / "wiki-sentences.txt"
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for count, sentence in enumerate(wiki_sentences(recipe), 1):
            stream.write(sentence + "\n")
            if count == limit:
                return output, count
    raise ValueError(f"Only {count} sentence chunks found; requested {limit}")


def wiki_sentences(recipe):
    """Yield non-heading chunks from downloaded parquet parts."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for part in range(2):
        path = hf_hub_download(
            "Salesforce/wikitext",
            f"wikitext-103-raw-v1/train-{part:05d}-of-00002.parquet",
            repo_type="dataset",
            revision=WIKITEXT_REVISION,
            cache_dir=recipe.work / "wiki-download",
            token=False,
        )
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["text"]):
            for text in batch.column(0).to_pylist():
                yield from split_sentences(text)


def split_sentences(text):
    text = " ".join(text.split())
    if text and not text.startswith("="):
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if len(sentence.split()) >= 5:
                yield sentence


def text_batches(path, size=65536):
    """Stream the same prepared texts for each setting without duplicating the corpus."""
    with path.open(encoding="utf-8") as stream:
        while lines := list(islice(stream, size)):
            yield [line.rstrip("\n") for line in lines]


@contextmanager
def timing_lock(work):
    """Use the coordinator's bench.lock when supplied, or a private local lock."""
    lock = Path(os.environ.get("FSE_RECIPE_BENCH_LOCK", str(work / "bench.lock")))
    lock.mkdir()  # A held lock raises rather than starting a competing timed run.
    owner = lock / "owner"
    owner.write_text(f"dr195-tutorials PID={os.getpid()} {datetime.now().isoformat()}\n")
    try:
        yield
    finally:
        owner.unlink()
        lock.rmdir()


def provenance():
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return {
        "machine": platform.platform(),
        "CPU": cpu,
        "Python": platform.python_version(),
        "fse_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "date": datetime.now().astimezone().isoformat(),
        "load_note": "measured on a MacBook in everyday use, not a controlled benchmark",
        "dataset_revision": WIKITEXT_REVISION,
        "dtype": "float32",
        "batch_rows": 65536,
        "scope": "file reading, Python batch creation and encoding; no vector storage",
    }


def measure(encoder, path, count, work):
    """One warmed run per setting. Never choose the best of several runs."""
    workers_all = os.process_cpu_count() or 1
    results = []
    info = provenance()
    with timing_lock(work):
        for workers in (1, workers_all):
            warmup = next(text_batches(path, size=10000))
            encoder.encode(warmup, workers=workers)
            load_before = os.getloadavg()[0]
            start = time.perf_counter()
            seen = sum(
                batch.n_rows
                for batch in encoder.encode_batches(text_batches(path), workers=workers)
            )
            seconds = time.perf_counter() - start
            load_after = os.getloadavg()[0]
            if seen != count:
                raise AssertionError(f"encoded {seen} rows, expected {count}")
            results.append(
                {
                    "setting": "1 worker" if workers == 1 else f"All cores ({workers} workers)",
                    "workers": workers,
                    "sentences": seen,
                    "seconds": seconds,
                    "sentences_per_second": seen / seconds,
                    "sentences_per_minute": seen / seconds * 60,
                    "load_before": load_before,
                    "load_after": load_after,
                }
            )
    print(json.dumps(info, indent=2))
    print(json.dumps(results, indent=2))
    return info, results


def read_hits(path, ids):
    wanted = set(map(int, ids))
    hits = {}
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                hits[index] = line.rstrip("\n")
                if len(hits) == len(wanted):
                    break
    return hits
