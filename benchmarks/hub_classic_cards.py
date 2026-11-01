"""Enrich the existing fse 2.0 card template with measured classic-model facts.

No core template or persistence schema changes are needed. fse.source identifies
the original vector copy; preparation.json and the card document the offline fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_hub_classics import LICENSES, digest

from fse._encoder import _card, _folder

SOURCE_TAGS = {
    "glove-wiki-gigaword-300": "glove",
    "word2vec-google-news-300": "word2vec",
    "paranmt-300": "paragram",
    "paragram-300-sl999": "paragram",
    "fasttext-crawl-subwords-300": "fasttext",
}
TERMS = {
    "glove-wiki-gigaword-300": (
        "The pretrained vectors are released under the [Public Domain Dedication and License]"
        "(https://opendatacommons.org/licenses/pddl/1-0/), as stated by the "
        "[GloVe authors](https://github.com/stanfordnlp/GloVe). "
        "The Apache-2.0 license of their implementation is separate from the vector terms."
    ),
    "word2vec-google-news-300": (
        "The [original word2vec project](https://code.google.com/archive/p/word2vec/) "
        "states that the project is 'released for research purposes only'. "
        "The card therefore uses `license: other`. Apache-2.0 covers the word2vec code, "
        "not these pretrained vectors. No broader permission is asserted here."
    ),
    "paranmt-300": (
        "The authors did not state a license for these vectors in the release materials "
        "we inspected. `license: unknown` does not grant permission. This conversion is "
        "kept for preservation. See [ParaNMT](https://github.com/jwieting/para-nmt-50m) "
        "and the [paper](https://aclanthology.org/P18-1042/). "
        "Its training data, [CzEng 1.6](https://ufal.mff.cuni.cz/czeng/czeng16), is freely "
        "available only for non-commercial research. This does not establish a license "
        "for the derived vectors. Rights holders can request removal through the "
        "[repository issues](https://github.com/oborchers/Fast_Sentence_Embeddings/issues)."
    ),
    "paragram-300-sl999": (
        "The authors did not state a license for these vectors in the release materials "
        "we inspected. `license: unknown` does not grant permission. This conversion is "
        "kept for preservation. See [From Paraphrase Database to Compositional Paraphrase "
        "Model and Back](https://aclanthology.org/Q15-1025/). Rights holders can request "
        "removal through the [repository issues]"
        "(https://github.com/oborchers/Fast_Sentence_Embeddings/issues)."
    ),
    "fasttext-crawl-subwords-300": (
        "The source vectors are by Facebook, from [fastText English word vectors]"
        "(https://fasttext.cc/docs/en/english-vectors.html), under "
        "[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). "
        "All six derived repositories retain CC BY-SA 3.0, including ShareAlike. "
        "Changes: conversion to fse safetensors, sentence mean and normalization, "
        "SIF weighting and component folding for `sif10`, and storage rounding for "
        "float16 or int8. Facebook does not endorse this conversion."
    ),
}


def front_matter(template, name, fit):
    header, body = template.split("\n---\n", 1)
    header += "\n" + "\n".join(
        f"- {json.dumps(tag)}" for tag in ("gensim", SOURCE_TAGS[name], "word-embeddings")
    )
    if fit is not None:
        header += "\ndatasets:\n- " + json.dumps(fit["corpus"])
    if name == "word2vec-google-news-300":
        header += '\nlicense_name: "word2vec-google-news-research-only"'
        header += '\nlicense_link: "https://code.google.com/archive/p/word2vec/"'
    return header + "\n---\n" + body


def fit_text(fit):
    return [
        "## Offline SIF preparation",
        "",
        "Word weights are a/(a+p(w)), with a=0.001. Frequencies match fse 1.x "
        "`lang_freq='en'`: English wordfreq best-list frequencies, converted to integer "
        "counts over 2**31-1 with a 1e-8 default, then normalized in float32.",
        "",
        f"- wordfreq version: {fit['wordfreq_version']}",
        f"- component corpus: `{fit['corpus']}` at `{fit['revision']}`",
        f"- training paragraphs: {fit['paragraphs']}; seed: {fit['seed']}",
        f"- tokenized corpus checksum: `{fit['corpus_xxh3']}`",
        f"- fitted weights and components checksum: `{fit['state_xxh3']}`",
        "",
        "Ten uncentered components were fitted on that training corpus, never on "
        "the evaluation texts, then folded into the weighted table. Quantization "
        "follows folding. GloVe selected this corpus by the mean of STS-B Pearson, "
        "STS-B Spearman and NanoBEIR macro nDCG@10, each scaled to 0-100. Those "
        "benchmarks guided corpus selection, so their scores are not untouched "
        "holdout estimates. `preparation.json` records the full fit receipt.",
        "",
        "Frequency-data attribution: [wordfreq](https://github.com/rspeer/wordfreq) "
        "by Robyn Speer. Its code is Apache-2.0; its frequency data includes "
        "CC BY-SA 4.0 material and source-specific credits in "
        "[NOTICE.md](https://github.com/rspeer/wordfreq/blob/master/NOTICE.md). "
        "Preparation transforms these frequencies into weights and folds them into "
        "the table. The source-vector license field does not waive those data terms.",
        "",
    ]


def measured_text(record, result, rows):
    by_key = {(r["method"], r["dtype"]): r for r in rows if "repo" in r}
    raw, sif = (by_key[(method, "float32")]["pearson"] for method in ("raw", "sif10"))
    return [
        "## Choosing a method and storage format",
        "",
        f"On the 1,379 STS-B test pairs, raw float32 scored {raw:.5f} Pearson x100 "
        f"and SIF-10 float32 scored {sif:.5f}. Use raw for a plain mean of the "
        "source vectors. Use SIF-10 to apply the fixed frequency weighting and "
        "training-corpus component removal measured here. Evaluate the choice on your own data.",
        "",
        "Use float32 to retain the prepared table without storage rounding. "
        "Use float16 or int8 when the measured size reduction and neighbour changes "
        "below meet your needs. fse computes all three in float32.",
        "",
        "| This model | Measured value |",
        "|---|---:|",
        f"| Model file bytes | {record['files']['model.safetensors']['bytes']} |",
        f"| STS-B Pearson x100 | {result['pearson']:.5f} |",
        f"| Difference from same-method float32, points | {result['difference_float32_points']:+.5f} |",
        f"| Recall@10 against same-method float32 | {result['recall_at_10']:.6f} |",
        f"| Default tokenizer Pearson x100 | {result['default_tokenizer_pearson']:.5f} |",
        "",
        "The controlled STS-B comparison uses the 1.x notebook's NLTK preprocessing "
        "through an external-tokenizer view of this same table. The default-tokenizer "
        "row uses the saved `fse_words@1` tokenizer on raw text. Recall@10 compares "
        "each of the 2,758 STS-B sentence rows against all other rows, excludes self, "
        "and resolves equal similarities by row index. No load-time or speed claim is made.",
        "",
    ]


def render(folder, record, verification, fit):
    metadata, shapes = _folder._header(folder / "model.safetensors")
    name = record["source"].split("/")[1]
    template = _card.card(
        metadata,
        shapes,
        license=LICENSES[name],
        language=["en"],
        model_name=record["repo"],
        location=record["repo"],
    ).decode()
    text = front_matter(template, name, fit)
    text = text.replace("pip install fse", 'pip install "fse>=2.0.0"')
    text = text.replace("## Usage\n", "## Usage\n\nThis model requires the fse 2.0 API.\n")
    result = next(r for r in verification["rows"] if r.get("repo") == record["repo"])
    text += "\n" + "\n".join(measured_text(record, result, verification["rows"]))
    if fit is not None:
        text += "\n" + "\n".join(fit_text(fit))
    text += "\n## Source terms and attribution\n\n" + TERMS[name] + "\n"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    args = parser.parse_args()
    args.cards.mkdir(parents=True, exist_ok=True)
    for folder in sorted(
        p for p in args.models.iterdir() if p.is_dir() and (p / "config.json").exists()
    ):
        record_path = args.models / f"{folder.name}.json"
        record = json.loads(record_path.read_text())
        name = record["source"].split("/")[1]
        verification = json.loads((args.models / f"{name}-verification.json").read_text())
        fit = (
            json.loads((args.models / f"{name}-fit.json").read_text())
            if record["method"] == "sif10"
            else None
        )
        text = render(folder, record, verification, fit)
        (folder / "README.md").write_text(text)
        (args.cards / f"{folder.name}.md").write_text(text)
        if fit is not None:
            (folder / "preparation.json").write_text(json.dumps(fit, indent=2) + "\n")
        record["files"] = {
            p.name: {"bytes": p.stat().st_size, "xxh3": digest(p)} for p in sorted(folder.iterdir())
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
