"""Count every speed corpus without timing, under the shared heavy-job lock."""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_inputs import corpus_identity, identity, texts  # isort: skip
from compare_artifacts import file_digest  # isort: skip
from compare_runtime import verify_backend_identity  # isort: skip
from prepare_native_classifier import inputs  # isort: skip
from quality_coordination import task_reservation  # isort: skip
from text_lengths import counter, measure_lengths  # isort: skip
from wiki_chunks import RECIPE_SOURCE_COMMIT, REVISION, chunks  # isort: skip


def source_rows(cache):
    for name in ("wikitext", "massive"):
        original = texts(name)
        provenance = corpus_identity(name, original)
        yield name + " encode", original, provenance
        canonical = [row.replace("\r", " ").replace("\n", " ") for row in original]
        yield (
            name + " pipeline",
            canonical,
            {
                **provenance,
                "original_input_digest": provenance["input_digest"],
                "preprocessing": "CR/LF each replaced with one space",
            },
        )
    yield (
        "fse 101 unique 2M train chunks",
        islice(chunks(), 2_000_000),
        {
            "dataset": "Salesforce/wikitext",
            "dataset_revision": REVISION,
            "recipe_helper_commit": RECIPE_SOURCE_COMMIT,
            "split": "train",
            "rule": "first 2M punctuation chunks, min five whitespace words, no artificial repetition",
        },
    )
    for suite in ("setfit", "fineweb"):
        for name, data, provenance in inputs(suite, cache / suite, None):
            yield (
                suite + "/" + name + " evaluation",
                data[2],
                evaluation_provenance(suite, provenance),
            )


def evaluation_provenance(suite, provenance):
    if suite == "setfit":
        return {**provenance, "split": "test", "training_selection": provenance["split"]}
    return provenance


def markdown(result, json_path):
    lines = [
        "# Speed corpus lengths",
        "",
        f"Exact counts from `{json_path}`. Words use whitespace splitting. Characters are Unicode code points.",
        "POTION tokens follow pinned model truncation and unknown-token removal. No throughput is measured here.",
        "",
        "| Corpus | Texts | Words mean / median | Characters mean / median | POTION tokens mean / median |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["corpora"]:
        values = [
            f"{row[key]['mean']:.2f} / {row[key]['median']:g}"
            for key in ("words", "characters", "potion_tokens")
        ]
        lines.append("| " + " | ".join([row["name"], f"{row['texts']:,}", *values]) + " |")
    return "\n".join(lines) + "\n"


def run(args):
    model, provenance = counter()
    methods = [
        Path(__file__),
        Path(__file__).with_name("text_lengths.py"),
        Path(__file__).with_name("wiki_chunks.py"),
    ]
    method_hashes = {path.name: file_digest(path) for path in methods}
    result = {
        "complete": False,
        "method_files": method_hashes,
        "format": "speed-corpus-lengths-v1",
        "tokenizer": provenance,
        "runtime": identity("model2vec"),
        "speed_claim": False,
        "corpora": [],
    }
    for name, rows, source in source_rows(args.cache):
        print("counting_corpus", name, flush=True)
        row = {**source, **measure_lengths(rows, model), "name": name}
        if name == "fse 101 unique 2M train chunks" and row["texts"] != 2_000_000:
            raise ValueError("recipe 00 source has fewer than 2M qualifying chunks")
        result["corpora"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        args.guide.write_text(markdown(result, args.output.name))
    verify_backend_identity("model2vec", result["runtime"])
    if method_hashes != {path.name: file_digest(path) for path in methods}:
        raise ValueError("corpus counting method changed during the job")
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--coordination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    args = parser.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("counting must run nice 10 under heavy.lock")
    with task_reservation(args.coordination):
        run(args)


if __name__ == "__main__":
    main()
