"""Private input preparation and memory accounting for the speed entry."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

from _measurement import append, atomic, sha256


def configure(workers, native_threads=None):
    native_threads = workers if native_threads is None else native_threads
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        os.environ[name] = str(native_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if native_threads > 1 else "false"
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < workers:
            raise ValueError("requested workers exceed available CPU affinity")
        os.sched_setaffinity(0, cpus[:workers])


def check_output(reference, other, exact):
    import numpy as np

    if reference.shape != other.shape or reference.dtype != other.dtype:
        raise ValueError("complete output shape or dtype differs")
    largest = 0.0
    for start in range(0, len(reference), 4096):
        a, b = reference[start : start + 4096], other[start : start + 4096]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite output")
        equal = (
            np.array_equal(a.view(np.uint8), b.view(np.uint8))
            if exact
            else np.allclose(a, b, rtol=3e-5, atol=3e-6)
        )
        if not equal:
            raise ValueError("full output comparison failed")
        largest = max(largest, float(np.max(np.abs(a - b), initial=0)))
    return largest


def parquet_texts(part):
    for batch in part.iter_batches(columns=["text"], batch_size=4096):
        for text in batch.column(0).to_pylist():
            text = " ".join(text.split())
            if not text.startswith("=") and len(text.split()) >= 5:
                yield text


def memory_plan(job, input_bytes, model_bytes, limit):
    n, d, w = job["texts"], job["dimensions"], job["workers"]
    components = {
        "controller_libraries_reserve": (2 if job.get("functional_only") else 8) * 1024**3,
        "python_texts_and_list": input_bytes * 4 + n * 72,
        "joined_utf8_or_tokenizer_buffers": input_bytes,
        "output_and_concatenation": n * d * 4 * 2,
        "worker_buffers": w * 64 * 1024**2,
        "model_copies": model_bytes * (w + 1),
        "sample_buffers": min(n, 10000) * d * 4 * 3,
    }
    total = sum(components.values())
    return {
        "components": components,
        "estimated_peak_bytes": total,
        "limit_bytes": limit,
        "safety_fraction": 0.85,
        "admitted": total <= limit * 0.85,
    }


def collect_sentences(source, text, functional):
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 5]
    source.extend(sentences[: max(0, 1024 - len(source))] if functional else sentences)
    return len(sentences)


def corpus_sources(contract, cache):
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    spec = contract["corpus"]
    folder = Path(
        snapshot_download(
            spec["repo"],
            revision=spec["revision"],
            repo_type="dataset",
            cache_dir=cache,
            allow_patterns=["wikitext-103-raw-v1/train-*.parquet"],
        )
    )
    wanted = {r["source_position"]: r for r in contract["unknown_cases"]}
    source, mandatory, paragraphs, sentences = [], {}, 0, 0
    parts = sorted((folder / spec["configuration"]).glob("train-*.parquet"))
    texts = (text for part in parts for text in parquet_texts(pq.ParquetFile(part)))
    for text in texts:
        if paragraphs in wanted:
            if hashlib.sha256(text.encode()).hexdigest() != wanted[paragraphs]["text_sha256"]:
                raise ValueError("mandatory unknown-token source changed")
            mandatory[paragraphs] = text
        paragraphs += 1
        sentences += collect_sentences(source, text, contract.get("functional_only", False))
    if (
        paragraphs != spec["expected_paragraphs"]
        or sentences != spec["expected_source_positions"]
        or set(mandatory) != set(wanted)
    ):
        raise ValueError("pinned sentence corpus or mandatory cases changed")
    return source, [mandatory[r["source_position"]] for r in contract["unknown_cases"]]


def write_input(path, source, mandatory, count, seed, token_counts=None, admission=None):
    import numpy as np

    if count < len(mandatory) or path.exists():
        raise ValueError("fresh positive input required")
    rng = np.random.Generator(np.random.PCG64(seed))
    positions = rng.integers(len(source), size=count - len(mandatory), dtype=np.int64)
    rng.shuffle(positions)
    sample = np.sort(
        np.r_[
            np.arange(len(mandatory)),
            rng.choice(
                np.arange(len(mandatory), count), min(10000, count) - len(mandatory), replace=False
            ),
        ]
    )
    if token_counts is not None:
        atomic(
            path.with_suffix(".sample-texts.json"), [source[int(positions[int(i)])] for i in sample]
        )
    with path.open("xb") as stream:
        for text in mandatory:
            stream.write(text.encode() + b"\n")
        for start in range(0, len(positions), 100000):
            check_preparation(admission)
            for position in positions[start : start + 100000]:
                stream.write(source[int(position)].encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    unique_source = len(set(source) | set(mandatory))
    weights = np.bincount(positions, minlength=len(source))
    unique_input = len({source[int(i)] for i in np.flatnonzero(weights)} | set(mandatory))
    word_lengths = np.fromiter((len(text.split()) for text in source), dtype=np.int64)
    character_lengths = np.fromiter((len(text) for text in source), dtype=np.int64)
    word_total = int(weights @ word_lengths) + sum(len(text.split()) for text in mandatory)
    character_total = int(weights @ character_lengths) + sum(len(text) for text in mandatory)
    identity = {
        "mean_tokenizer_ids_per_text": int(weights @ token_counts) / count
        if token_counts is not None
        else None,
        "token_definition": "POTION tokenizer ids after character/token limit, including unknown ids, no special ids",
        "mean_words_per_text": word_total / count,
        "mean_characters_per_text": character_total / count,
        "distinct_source_texts": unique_source,
        "distinct_input_texts": unique_input,
        "repetition_factor": count / unique_source,
        "texts": count,
        "seed": seed,
        "sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "source_positions": len(source),
        "sampling": "sampled and shuffled WikiText-103 sentences",
        "mandatory_prefix_note": "unchanged historical paragraph UNK cases" if mandatory else None,
        "mandatory_prefix_rows": len(mandatory),
        "sample_rows": sample.tolist(),
    }
    atomic(path.with_suffix(".json"), identity)
    return identity


def stream_budget(work, cell, source):
    input_bytes = cell["texts"] * sum(len(t.encode()) + 1 for t in source) / len(source)
    required = int(input_bytes * 1.2 + cell["texts"] * 264 + 5 * 1024**3)
    if shutil.disk_usage(work).free < required:
        raise OSError("not enough free disk for headline input and one output")
    limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    if not cell.get("functional_only") and (limit == "max" or int(limit) * 0.85 < 20 * 1024**3):
        raise MemoryError("headline needs a bounded memory allowance of at least 26 GB")


def remove_headline(work, output, files):
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if output.is_symlink() or not output.resolve().is_relative_to(work.resolve()):
        raise ValueError("headline cleanup ownership changed")
    if actual != {r["path"] for r in files}:
        raise ValueError("headline output inventory changed before cleanup")
    if any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("headline cleanup contains a symlink")
    shutil.rmtree(output)
    descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def check_preparation(admission):
    if admission is not None and not admission(120):
        raise TimeoutError("input preparation budget or stop signal exhausted")


def headline_rounds(context, cell, path, sample, identity, launch, cleanup):
    rows, ledger = [], context["ledger"]
    try:
        for number in range(cell["rounds"]):
            if not ledger.admit(120):
                break
            row = launch(context, cell, "fse", number, path, sample, identity)
            rows.append(row)
            output = Path(row["sample_file"])
            remove_headline(context["work"], output, row["files"])
            append(
                ledger.root / "cleanup.jsonl",
                dict(
                    round=number,
                    completed_at=time.time(),
                    disk_after=shutil.disk_usage(context["work"])._asdict(),
                ),
            )
        cleanup(
            context,
            [path, path.with_suffix(".json"), sample, path.with_suffix(".sample-texts.json")],
        )
        if len({json.dumps(row["identity"], sort_keys=True) for row in rows}) > 1:
            raise ValueError("headline runtime identity changed between rounds")
    except Exception as error:
        return rows, repr(error)
    return rows, None
