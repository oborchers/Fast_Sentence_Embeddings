"""Render the frozen two-label SGD model card from its completed metrics receipt."""

from __future__ import annotations


def table(result):
    lines = [
        "| Label | Positive | ROC AUC | AP | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in result["per_label"].items():
        values = " | ".join(
            f"{row[k]:.6f}" for k in ("roc_auc", "average_precision", "precision", "recall", "f1")
        )
        lines.append(f"| {label} | {row['positive']} | {values} |")
    row = result["macro"]
    values = " | ".join(
        f"{row[k]:.6f}" for k in ("roc_auc", "average_precision", "precision", "recall", "f1")
    )
    return [*lines, f"| Macro | | {values} |"]


def render(record):
    config, selection = record["config"], record["selection"]
    chosen = selection["selected"]
    validation = selection["validation"]
    spec = next(s for s in config["encoders"] if s["name"] == "bekko")
    excluded = selection["excluded_labels"]["severe_toxicity"]
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
        "An English two-label multilabel classifier for `toxicity` and `insult`, in that order. "
        "Training happened outside fse in scikit-learn. The frozen Bekko encoder was not trained. "
        "fse imports and runs the finished affine head. This preparation has not been uploaded.",
        "",
        "## Requirements and use",
        "",
        "Requires an fse master build with multilabel support, or fse 2.1 when released. "
        "fse 2.0.0 cannot load this artifact. The standalone folder contains the tokenizer "
        "and folded score table. The complete folder also retains encoder vectors.",
        "",
        "```python",
        "import fse",
        "",
        'model = fse.Classifier.from_pretrained("/path/to/standalone")',
        'scores = model.predict_proba(["A calm discussion of the proposed policy."])',
        'mask = model.predict(["A calm discussion of the proposed policy."])',
        "print(model.classes_)",
        "```",
        "",
        "## Data, training and selection",
        "",
        "[Civil Comments](https://huggingface.co/datasets/google/civil_comments) contains "
        "historical English news-site discussion. The "
        "[source targets](https://www.tensorflow.org/datasets/catalog/civil_comments) are "
        "annotator fractions, binarized at >= 0.5 for training and evaluation.",
        "",
        f"- Dataset revision `{config['dataset']['revision']}` (CC0-1.0).",
        f"- Encoder `{spec['repo']}` revision `{spec['revision']}` (MIT), normalize=False.",
        f"- Training rows: {chosen['train_rows']:,}. Exactly {chosen['epochs']} full epochs.",
        "- StandardScaler statistics were fitted incrementally on training vectors only. "
        "The standardized float32 matrix was streamed in batches of 8,192.",
        "- Six independent SGDClassifier estimators used log_loss, L2, inverse-scaling "
        "learning rate (eta0=0.01, power_t=0.5), averaged coefficients, seed 17, "
        "and fixed train-wide balanced class weights N/(2 * class_count).",
        f"- Selected alpha: {chosen['alpha']:.17g}, from alpha=1/(C*N), C in {{0.1, 1, 10}}.",
        "- Selection used the entire validation split only. The full six-label head's "
        f"macro ROC AUC was {selection['full_macro_auc']:.6f} versus "
        f"{selection['small_macro_auc']:.6f} for the small Bekko head, a gain "
        f"of {selection['gain']:.6f}, exceeding the predeclared 0.01 replacement boundary. An exact tie "
        "between C references 1 and 10 chose the larger alpha, corresponding to C=1.",
        "- After validation, the maintainer chose only the already fitted toxicity and "
        "insult estimators. No refit or new alpha selection followed. The scaler was "
        'folded in float64 and imported through Classifier.from_arrays(link="multilabel").',
        f"- Source revision `{record['environment']['git_revision']}`; "
        f"scikit-learn {record['environment']['scikit-learn']}, "
        f"NumPy {record['environment']['numpy']}, Python {record['environment']['python']}.",
        "",
        "## Thresholds and removed labels",
        "",
        "Each retained threshold maximizes F1 on validation only, choosing the highest "
        "float32 cutoff on a tie. Inference is strictly score > cutoff, excluding equality. "
        "These thresholds were stored before test. Scores are not calibrated probabilities "
        "of toxicity or annotator agreement. No calibration study was performed.",
        "",
        "| Retained label | Stored threshold |",
        "|---|---:|",
    ]
    for label, cutoff in zip(config["labels"], config["prediction_thresholds"], strict=True):
        lines.append(f"| {label} | {cutoff:.17g} |")
    lines += [
        "",
        f"`severe_toxicity` was removed before model selection: only "
        f"{excluded['positive']} of {excluded['train_rows']:,} training rows were positive. "
        "No validation estimate was computed for that removed label.",
        "",
        "The maintainer removed `obscene`, `threat`, `identity_attack` and "
        "`sexual_explicit` after reviewing validation. Their maximum-F1 validation "
        "scores were lower than the retained labels. The table preserves all six "
        "full-head validation results to document that decision. These thresholds "
        "were optimized on this same split and are not independent estimates.",
        "",
    ]
    lines += table(validation["tuned"])
    lines += [
        "",
        "## Official test",
        "",
        f"Exactly one evaluation on the official test split ({record['test']['rows']:,} rows) "
        "followed the frozen two-label decision and threshold persistence. All test "
        "scores came from the saved standalone fse model. No model or threshold "
        "changed afterwards. AP means non-interpolated average precision. ROC AUC "
        "means area under the receiver operating characteristic curve. Undefined "
        "precision/F1 is zero; absent positive or negative classes abort evaluation.",
        "",
    ]
    lines += table(record["test"])
    lines += [
        "",
        "## Limits, license and evidence",
        "",
        "Use for assisted screening and research with human review. Other languages, "
        "domains and current usage were not evaluated. Static embeddings can miss "
        "context, negation, quotation and intent. Identity-related bias is a known "
        "[corpus concern](https://arxiv.org/abs/1903.04561). No identity subgroup "
        "or calibration metrics were measured. This head does not establish legality "
        "or harmfulness, and does not output the five removed labels.",
        "",
        "Both the fitted head and encoder use MIT. Source data is CC0-1.0. "
        "metrics.json records selection, versions, correctness checks and test metrics. "
        "SHA256SUMS.json covers the prepared files. Numerical import bounds and "
        "complete/standalone bit identity passed. Labels with fewer than 100 test "
        "positives are marked unreliable: "
        + (", ".join(record["unreliable_labels"]) or "none")
        + ". "
        "Evidence class: correctness. No Linux comparison or speed claim.",
        "",
    ]
    return "\n".join(lines)
