"""Render the prepared classifier card exclusively from the recorded receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(record):
    if record["status"] != "complete":
        raise ValueError("incomplete runs cannot produce a release-ready model card")
    config, selected = record["config"], record["selection"]["selected"]
    spec = next(s for s in config["encoders"] if s["name"] == selected["encoder"])
    sample = record["selection"]["sample"]
    omitted = record["selection"]["excluded_labels"]["severe_toxicity"]
    lines = [
        "---",
        "license: mit",
        "language: [en]",
        "library_name: fse",
        "pipeline_tag: text-classification",
        "datasets: [google/civil_comments]",
        f"base_model: {spec['repo']}",
        "tags: [multilabel, toxicity, linear-classifier]",
        "---",
        "",
        f"# {record['model_name']}",
        "",
        "An English multilabel text classifier trained outside fse with scikit-learn. "
        "fse only imports and runs the fitted head. The standalone folder includes its "
        "tokenizer and folded score table. The complete folder also retains the encoder vectors.",
        "",
        "## Requirements and usage",
        "",
        "Requires fse from the recorded master revision below, or fse 2.1 when released. "
        "fse 2.0.0 cannot load this multilabel artifact. This prepared model has not been uploaded.",
        "",
        "```python",
        "import fse",
        'model = fse.Classifier.from_pretrained("/path/to/standalone")',
        'scores = model.predict_proba(["A calm discussion of the proposed policy."])',
        'mask = model.predict(["A calm discussion of the proposed policy."])',
        "print(model.classes_)",
        "```",
        "",
        "Labels, in output order: " + ", ".join(f"`{s}`" for s in config["labels"]) + ".",
        "",
        "## Targets and training",
        "",
        "[Civil Comments](https://huggingface.co/datasets/google/civil_comments) contains "
        "English news-site comments from 2015 to 2017, annotated by crowd workers. "
        "[TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/civil_comments) "
        "defines the source targets as fractions of annotators who assigned each attribute. "
        f"Training binarizes those fractions with >= {config['target_threshold']}. "
        f"Inference uses scores strictly > {config['prediction_threshold']}. Equality is excluded.",
        "",
        f"The severe_toxicity label was removed because only {omitted['positive']:,} of the "
        f"{omitted['train_rows']:,} training rows are positive at >= {config['target_threshold']}. "
        "The retained labels are trained independently.",
        "",
        f"- Dataset revision: `{config['dataset']['revision']}` (CC0-1.0).",
        f"- Encoder: `{spec['repo']}` at `{spec['revision']}` (MIT), normalize=False.",
        f"- fse source revision: `{record['environment']['git_revision']}`.",
        f"- Selected variant: {selected['variant']}; training rows: {selected['train_rows']:,}.",
        f"- Small training sample: {sample['rows']:,} rows, seed {config['seed']}, {sample['method']}.",
        f"- Sample positions SHA256 (little-endian int64): `{sample['sha256_little_endian_int64']}`.",
        f"- StandardScaler, then OneVsRestClassifier with LogisticRegression (lbfgs, C={selected['C']}, "
        f"max_iter={config['max_iter']}, class_weight={config['class_weight']}, n_jobs=1).",
        f"- scikit-learn {record['environment']['scikit-learn']}; NumPy {record['environment']['numpy']}; "
        f"Python {record['environment']['python']}.",
        "",
        f"Each encoder uses the best validation macro ROC AUC over C={config['C']}; ties use "
        f"the smaller C. Bekko replaces POTION only with a gain of at least {config['bekko_minimum_gain']}. "
        "The official validation split is used in full. Training keeps natural class proportions. "
        "Balanced class weights change the fitting loss. These scores are model outputs for "
        "binary targets, not calibrated probabilities of toxicity or annotator agreement.",
        "",
        "Full training is admitted only for POTION after the small comparison passes, with "
        "sufficient budget and memory. It replaces the small head only if its validation "
        "macro ROC AUC is no lower. No test value guides any model, size, C or threshold choice.",
        "",
        "## Evaluation",
        "",
        f"One evaluation on the untouched official test split ({record['test']['rows']:,} rows), "
        "after selection. All test scores come from the saved standalone fse classifier. "
        "Average precision is the non-interpolated summary of the precision-recall curve, "
        "not trapezoidal PR area. Precision, recall and F1 use the fixed inference threshold. "
        "Undefined precision or F1 is recorded as zero. Missing positive or negative classes abort the run.",
        "",
        "| Label | Positive | ROC AUC | Average precision | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, values in record["test"]["per_label"].items():
        numbers = [str(values["positive"])] + [
            f"{values[k]:.6f}"
            for k in ("roc_auc", "average_precision", "precision", "recall", "f1")
        ]
        lines.append("| " + label + " | " + " | ".join(numbers) + " |")
    macro = record["test"]["macro"]
    lines += [
        "| Macro | | "
        + " | ".join(
            f"{macro[k]:.6f}" for k in ("roc_auc", "average_precision", "precision", "recall", "f1")
        )
        + " |",
        "",
        "## Limits and intended use",
        "",
        "Use for assisted screening and research with human review. Do not use for automatic "
        "moderation without human oversight. The source is historical English news discussion. "
        "Other languages, domains and current usage have not been evaluated. Short static "
        "embeddings can miss context, negation, quotation and intent. Rare labels need particular care.",
        "",
        "Identity-related bias is a known concern in this corpus "
        "([Borkan et al.](https://arxiv.org/abs/1903.04561)). "
        "The primary-label Hugging Face configuration used here has no identity subgroup "
        "annotations. No subgroup bias metrics or calibration study were run. "
        "The model does not establish whether a statement is fair, illegal or harmful. "
        "Evaluate thresholds on your own labeled validation data before use.",
        "",
        "Labels with fewer than the configured minimum test positives are marked unreliable "
        "in metrics.json. Unreliable labels in this run: "
        + (", ".join(record["unreliable_labels"]) or "none")
        + ".",
        "",
        "## License and evidence",
        "",
        "The fitted head is MIT, as is the selected encoder. Source data is CC0-1.0. "
        "metrics.json records selection, per-label evaluation, package versions and correctness "
        "checks. SHA256SUMS.json identifies all files in both prepared folders. "
        "No runtime or throughput claim is made.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.metrics.read_text())))


if __name__ == "__main__":
    main()
