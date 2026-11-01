"""Assemble the local classic-model report from completed, measured artifacts only."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from hub_classic_sources import DTYPES, SOURCES
from prepare_hub_classics import digest


def load(path):
    return json.loads(path.read_text())


def checked_record(models, stem):
    record = load(models / f"{stem}.json")
    folder = models / stem
    if {p.name for p in folder.iterdir()} != set(record["files"]):
        raise ValueError(f"file inventory mismatch: {folder}")
    for file, spec in record["files"].items():
        path = folder / file
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["xxh3"]:
            raise ValueError(f"file size or checksum mismatch: {path}")
    return record


def inventory(models):
    records = []
    for name, method, dtype in product(SOURCES, ("raw", "sif10"), DTYPES):
        suffix = "" if dtype == "float32" else f"-{dtype}"
        records.append(checked_record(models, f"{name}-{method}{suffix}"))
    if len(records) != 30:
        raise ValueError("expected 30 finished folders")
    return records


def comparison_text(selection):
    lines = [
        "## Component-fit corpus comparison",
        "",
        "| Corpus | STS-B Pearson x100 | STS-B Spearman x100 | NanoBEIR nDCG@10 x100 | Mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selection["candidates"]:
        lines.append(
            f"| {row['corpus']} | {row['pearson']:.5f} | {row['spearman']:.5f} | {row['nanobeir_ndcg_at_10']:.5f} | {row['selection_mean']:.5f} |"
        )
    lines += [
        "",
        f"Selected corpus: **{selection['selected']}**.",
        "",
        "All three scores have equal weight. NanoBEIR is the macro mean of all 13 "
        "retrieval tasks. Components use training paragraphs only. Evaluation scores "
        "selected the corpus, so they are not untouched holdout estimates.",
        "",
        "```json",
        json.dumps(selection["fit"], indent=2),
        "```",
        "",
    ]
    return lines


def verification_text(models):
    lines = [
        "## Comparison with fse 1.x",
        "",
        "Pearson x100 on all 1,379 STS-B test pairs. Controlled preprocessing follows "
        "the 1.x notebook. Historical SIF fits components on the test texts solely "
        "as a diagnostic and is never shipped. Recall@10 uses the 2,758 sentence "
        "rows, excludes self, and breaks ties by row id.",
        "",
        "| Source | Method | Dtype | 1.x | 2.0 | Difference, points | vs float32, points | Recall@10 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    explanations = []
    for name in SOURCES:
        rows = load(models / f"{name}-verification.json")["rows"]
        historical = next(r for r in rows if r["method"] == "sif10-historical-test-fit")
        for row in rows:
            change = row["difference_1x_points"]
            parent = f"{row['difference_float32_points']:+.5f}" if "repo" in row else "n/a"
            recall = f"{row['recall_at_10']:.6f}" if "repo" in row else "n/a"
            lines.append(
                f"| {name} | {row['method']} | {row['dtype']} | {row['baseline_1x']:.2f} | {row['pearson']:.5f} | {change:+.5f} | {parent} | {recall} |"
            )
            if abs(change) > 0.5:
                explanations.append(
                    f"- {name}, {row['method']}, {row['dtype']}: {change:+.5f} points. Historical wordfreq/test-fit diagnostic: {historical['pearson']:.5f} against {historical['baseline_1x']:.2f}. The shipped model instead uses components from the selected separate training corpus. Folding components into word and ngram tables also changes rounding relative to post-pooling removal. See the per-source verification record for the default tokenizer score."
                )
    lines += ["", "### Differences above 0.5 points", "", *(explanations or ["None."]), ""]
    return lines


def write_report(models, comparison, output):
    records = inventory(models)
    selection = load(comparison / "selection.json")
    lines = [
        "# Classic word-vector preparation report",
        "",
        "DR-157 (Ausgewählte Modelle auf den Hugging Face Hub laden). "
        "All model folders are local. No Hugging Face upload or write was performed.",
        "",
        "Draft PR: [#201 (2.0: Prepare the classic word-vector models for the Hub)]"
        "(https://github.com/oborchers/Fast_Sentence_Embeddings/pull/201).",
        "",
        *comparison_text(selection),
        *verification_text(models),
        "## Built folders",
        "",
        "Checksums were recomputed over every file before this report was written. "
        "The JSON manifest beside each folder lists all file sizes and checksums.",
        "",
        "| Intended repository | Model bytes | Model xxh3-128 | Encoder fingerprint |",
        "|---|---:|---|---|",
    ]
    for record in records:
        model = record["files"]["model.safetensors"]
        lines.append(
            f"| {record['repo']} | {model['bytes']} | `{model['xxh3']}` | `{record['fingerprint']}` |"
        )
    lines += [
        "",
        "## Provenance and cards",
        "",
        f"Finished folders: `{models}`. Card previews: `{output.parent / 'cards'}`.",
        "",
        "The existing fse 2.0 card template supports generic vector-source provenance, "
        "license, language, base_model and int8-only quantized relation. Offline card "
        "enrichment adds original source tags, the SIF fit dataset, licensing context, "
        "fit hashes and measured dtype comparisons. No core runtime dependency or "
        "persistence-schema change is needed for wordfreq. fse.source pins each "
        "original vector repository and revision. Each SIF card and preparation.json "
        "record corpus, revision, seed, wordfreq version and fitted-state checksum.",
        "",
        "Licenses: GloVe pddl; Google News word2vec other/research-only; fastText "
        "cc-by-sa-3.0 with Facebook attribution and ShareAlike; ParaNMT and Paragram "
        "unknown with preservation and removal-on-request notices. See each card's "
        "linked primary sources. The old fse source repositories remain unchanged.",
        "",
        "No timings, load-time comparisons or speed claims were made on this Mac.",
        "",
        "## Delivery checks and open questions",
        "",
        "Repository gates and independent code/card review must be recorded here "
        "after they finish. A complete artifact inventory alone does not establish "
        "completion of those checks.",
        "",
        "FastText publication remains deferred pending DR-146 (Über FastText in 2.0 "
        "entscheiden (12f)). All publication requires the maintainer's separate "
        "approval. The superseded initial GloVe prototypes in the parent folder "
        "are excluded from this inventory.",
        "",
    ]
    output.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.models, args.comparison, args.output)


if __name__ == "__main__":
    main()
