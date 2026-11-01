"""Reproducible repeated/shuffled files, prepared once and streamed by timed children."""

import json
from pathlib import Path

import numpy as np
from compare_artifacts import file_digest
from text_lengths import Lengths, batches, counter, token_counts
from wiki_chunks import RECIPE_SOURCE_COMMIT, REVISION, chunks

SEED = 0
MATRIX_ROWS = 10_000_000
SHOWCASE_ROWS = 100_000_000


def shuffled_indices(source_count, count, seed):
    if source_count < 1:
        raise ValueError("cannot repeat an empty source corpus")
    indices = np.arange(count, dtype=np.int64) % source_count
    np.random.default_rng(seed).shuffle(indices)
    return indices


def repeated_lengths(source, model, count):
    lengths = Lengths()
    cycles, remainder = divmod(count, len(source))
    offset = 0
    for batch in batches(source):
        for text, tokens in zip(batch, token_counts(model, batch), strict=True):
            weight = cycles + (offset < remainder)
            lengths.add_weighted_counts(len(text.split()), len(text), tokens, weight)
            offset += 1
    return lengths.record()


def write_shuffled(path, source, indices):
    from fse._container._hash import Hasher

    digest = Hasher()
    with path.open("xb") as stream:
        for number, index in enumerate(indices, 1):
            data = source[int(index)].encode("utf-8")
            stream.write(data + b"\n")
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)
            if number % 1_000_000 == 0:
                print("input_file_rows_written", number, flush=True)
    return digest.checksum()


def prepare_one(folder: Path, count: int):
    folder.mkdir(parents=True)
    print("stage_start full-wikitext-source", flush=True)
    source = list(chunks())
    if not source:
        raise ValueError("no qualifying WikiText train chunks")
    if count < len(source):
        raise ValueError("requested input cannot contain every qualifying source chunk")
    distinct = len(set(source))
    print("stage_end full-wikitext-source", len(source), "distinct", distinct, flush=True)
    model, tokenizer = counter()
    lengths = repeated_lengths(source, model, count)
    indices = shuffled_indices(len(source), count, SEED)
    name = "repeated-shuffled-" + str(count // 1_000_000) + "m"
    path = folder / (name + ".txt")
    digest = write_shuffled(path, source, indices)
    record = {
        **lengths,
        "input_digest": digest,
        "file": path.name,
        "file_bytes": path.stat().st_size,
        "checksum": file_digest(path),
        "dataset": "Salesforce/wikitext",
        "dataset_revision": REVISION,
        "split": "train",
        "recipe_helper_commit": RECIPE_SOURCE_COMMIT,
        "chunk_rule": "whitespace collapse; skip headings; split punctuation followed by whitespace; at least five words",
        "qualifying_source_chunks": len(source),
        "distinct_source_texts": distinct,
        "input_label": "repeated and shuffled input",
        "source_positions_repeated": count > len(source),
        "seed": SEED,
        "shuffle": "numpy default_rng(seed).shuffle(arange(count) % source_chunk_count)",
        "numpy_version": np.__version__,
        "repetition_factor": count / len(source),
        "repetition_factor_distinct_texts": count / distinct,
        "distinct_input_texts": distinct if count >= len(source) else None,
        "tokenizer": tokenizer,
        "statistics_method": "exact source-position multiplicities; shuffle preserves the length histograms",
        "text_result_cache": "none in fse; native WordPiece has no sentence-result cache",
        "rust_text_cache": "no sentence-result cache in pinned model.rs / fast_wordpiece.rs",
        "rust_cache_audit_source": "https://github.com/MinishLab/model2vec-rs/blob/19f58bef6ce145f2bc08de7016498f28d521feea/src/model.rs",
        "os_page_cache": "not cleared; input checksum verification warms filesystem caches before timing",
    }
    (folder / (name + ".identity.json")).write_text(json.dumps(record, indent=2) + "\n")
    return record
